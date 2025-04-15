from PROSE.queries.query_interface import Agent, TargetedGenerator
import math
import concurrent.futures
from PROSE.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
    count_words,
)
import os
import pandas as pd
from tqdm import tqdm
import argparse

from PROSE.utils.llm_tools import DEFAULT_MODEL, GPT
from PROSE.queries.fast_queries import FastAgent
from PROSE.queries.gen_composite import (
    TagNNGenerator,
    WeightedNNGenerator,
    PreviousBestGenerator,
    ClosestClusterGenerator,
)
from PROSE.datasets.load import get_data_filenames

from ast import literal_eval
import pickle


from collections import defaultdict


llm = GPT(
    model=DEFAULT_MODEL,
)

LOGS_COLS = [
    "model",
    "params",
    "prompt",
    "system_prompt",
    "messages",
    "response",
    "completion",
    "timestamp",
    "system_fingerprint",
    "target_approval",
    "target_budget",
    "generator",
    "agent_id",
    "statement",
    "score",
]

INFO_COLS = [
    "statement",
    "approval_level",
    "budget",
    "statement_cost",
    "num_covered_agents",
    "num_remaining_agents",
    "covered_agent_ids",
    "remaining_agent_ids",
]

INFO_COLS_DRUGS = [
    "statement",
    "approval_level",
    "budget",
    "statement_cost",
    "num_covered_agents",
    "num_remaining_agents",
    "covered_agent_ids",
    "remaining_agent_ids",
    "covered_agent_labels",
    "remaining_agent_labels",
]

STATEMENTS_COLS = [
    "stage",
    "statement",
    "approval_level",
    "budget",
    "statement_cost",
    "generator_name",
    "num_covered_agents",
    "num_remaining_agents",
    "agent_utilities",
    "covered_agent_ids",
    "remaining_agent_ids",
    "covered_agent_data",
    "utility_distribution",
]


def generate_slate(
    agents: list[Agent],
    precompute_generators: list[TargetedGenerator],
    generators: list[TargetedGenerator],
    word_budget: int,
    budget_schedule: list[int],
    approval_levels: list[int],
    full_approval_generation: bool = False,
    full_pool_selection: bool = False,
    num_threads: int = 10,
    verbose: bool = False,
    no_budgets: bool = False,
    dataset: str = "",
    minimum_statement_length: int = 0,
):
    """
    Generate a slate of statements from a list of agents using a list of generators, subject to a word budget and a budget schedule.

    Args:
        agents: List of agents
        precompute_generators: List of generators that are run on all the agents at the start
        generators: List of generators that are run on the unsatisfied agents at each step
        word_budget: Total word budget for the slate.
        budget_schedule: In order, what size budgets to try and generate for. E.g. [200, 150, 100, 50] starts with big statements and decreases their length
        approval_levels: List of approval levels to target, in decreasing order.
        full_approval_generation: If False, at approval level `approval_level`, only generate statements targeted to that approval level (as in simple version of algo). If True, at each step, also try generating for all statements with approval_level between `approval_level` and max(approval_levels) (as in approx query version of algo).
        num_threads: Number of threads to use for LLM queries. It's expensive to enable this
        full_pool_selection: If True, at each step, select the best statement from the entire pool of generated statements. If False, select the best statement from the pool of generated statements at the current budget & approval level. Doesn't cost more to enable this, but might result in less diverse slates.
        no_budgets: If True, we don't take into account statement lengths and costs in our computation

    Returns:
        List of statements generated for the slate.
    """
    # Approval levels must be decreasing
    assert approval_levels == sorted(approval_levels, reverse=True)
    # Budget schedule entries must be smaller than word_budget
    assert all(budget <= word_budget for budget in budget_schedule)

    assert (
        word_budget % len(agents) == 0
    ), "It's recommnded to have word_budget divisible by number of agents (not theoretically necessary, but simplifies experiments)"
    words_per_agent = word_budget // len(agents)
    if not no_budgets:
        assert set(
            [words_per_agent, words_per_agent * 2]
        ).issubset(
            budget_schedule
        ), "It's recommended to have budget_schedule include small numbers, to avoid tiny leftover groups that can't be covered"

    num_agents = len(agents)

    log_dirname = (
        get_base_dir_path()
        / "experiment_logs"
        / f"{get_time_string()}__slate__{word_budget}__{dataset}"
    )
    log_filename = log_dirname / "logs.csv"
    info_filename = log_dirname / "info.csv"
    statements_filename = log_dirname / "statements.csv"
    out_filename = log_dirname / "out.txt"
    obj_output_filename = log_dirname / "obj_output.pkl"
    global_params = log_dirname / "global_params.csv"
    os.makedirs(log_dirname)
    pd.DataFrame(columns=LOGS_COLS).to_csv(log_filename, index=False)
    pd.DataFrame(columns=INFO_COLS_DRUGS).to_csv(info_filename, index=False)
    pd.DataFrame(columns=STATEMENTS_COLS).to_csv(statements_filename, index=False)

    pd.DataFrame(
        [
            {
                "word_budget": word_budget,
                "budget_schedule": budget_schedule,
                "approval_levels": approval_levels,
                "full_approval_generation": full_approval_generation,
                "full_pool_selection": full_pool_selection,
                "no_budgets": no_budgets,
            },
        ]
    ).to_csv(global_params, index=False)

    # Run precompute generators
    precomputed_statements = []
    precomputed_statements_full_data = []
    for generator in tqdm(precompute_generators, desc="Running precompute generators"):
        if no_budgets:
            statement_list, log_list = generator.generate(
                agents=agents, target_budget=-1 * len(agents)
            )
        else:
            statement_list, log_list = generator.generate(agents=agents)
        precomputed_statements.extend(statement_list)
        pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
            log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
        )
        for statement in statement_list:
            precomputed_statements_full_data.append(
                {
                    "stage": "precompute",
                    "statement": statement,
                    "approval_level": None,
                    "budget": None,
                    "statement_cost": count_words(statement),
                    "generator_name": generator.get_name(),
                    "agent_utilities": {},
                }
            )
    # And precompute their approvals (this is a little wasteful)
    precomputed_approvals = [dict() for _ in range(len(precomputed_statements))]
    precomputed_generator_names = [None for _ in range(len(precomputed_statements))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_data = {
            executor.submit(agent.get_approval, statement): {
                "agent_id": agent.get_id(),
                "statement": statement,
                "statement_idx": idx,
            }
            for agent in agents
            for idx, statement in enumerate(precomputed_statements)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_data),
            total=len(future_to_data),
            desc="Precomputing approvals",
        ):
            approval, log_list = future.result()
            data = future_to_data[future]
            for log in log_list:
                log.update(data)
            precomputed_approvals[data["statement_idx"]][data["agent_id"]] = approval
            precomputed_generator_names[data["statement_idx"]] = (
                precomputed_statements_full_data[
                    data["statement_idx"]
                ]["generator_name"]
            )
            precomputed_statements_full_data[data["statement_idx"]]["agent_utilities"][
                data["agent_id"]
            ] = approval
            pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
                log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
            )

    pd.DataFrame(
        precomputed_statements_full_data,
        columns=STATEMENTS_COLS,
    ).to_csv(
        statements_filename,
        mode="a",
        header=False,
        index=False,
        columns=STATEMENTS_COLS,
    )
    assert len(precomputed_statements) == len(precomputed_approvals)

    unsatisfied_agents = agents.copy()
    statement_pool = (
        precomputed_statements.copy()
    )  # List of statements generated so far
    statement_pool_approvals = precomputed_approvals.copy()  # At idx i, the approvals of statement_pool[i] (dict mapping agent_id to approval)
    statement_pool_generator_names = precomputed_generator_names.copy()
    slate = []
    slate_costs = []
    slate_covered_agent_ids = []  # At idx i, the agents covered by slate[i] (list of agent ids)
    slate_approvals = []  # At idx i, the approvals of slate[i] (dict mapping agent_id to approval)

    for approval_level in tqdm(approval_levels, desc="Approval levels"):
        is_last = approval_levels.index(approval_level) == len(approval_levels) - 1

        if verbose:
            print(f"Targeting approval level {approval_level}.")
        if not unsatisfied_agents:
            # If all agents are satisfied, we are done (could be under word budget)
            if verbose:
                print("All agents are covered -- done.")
            break

        for budget in tqdm(budget_schedule, desc="Budgets"):
            if verbose:
                print(f"Targeting budget of {budget}-word statements.")

            if no_budgets:
                number_of_agents_requiring_approval = budget
            # Need (1) budget to fit in remaining budget we have
            # and  (2) enough unsatisfied agents to warrant a statement of this large a budget
            while (
                not no_budgets
                and word_budget - sum(slate_costs) >= budget
                and len(unsatisfied_agents)
                >= math.floor(budget * num_agents / word_budget)
                and (is_last or budget >= minimum_statement_length)
            ) or (
                no_budgets
                and len(unsatisfied_agents) >= number_of_agents_requiring_approval
            ):
                for generator in generators:
                    if generator.is_adaptable():
                        if no_budgets:
                            generator.set_cluster_size(
                                number_of_agents_requiring_approval
                            )
                        else:
                            generator.set_cluster_size(
                                math.floor(budget * num_agents / word_budget)
                            )
                    if isinstance(generator, PreviousBestGenerator):
                        generator.set_previous_statements(statement_pool_approvals)
                # Generate new statements
                if no_budgets:
                    budget = -1
                # Approval levels to target
                gen_approval_levels = (
                    [approval_level]
                    if not full_approval_generation
                    else [level for level in approval_levels if approval_level <= level]
                )

                # Generate statements
                if verbose:
                    print(
                        f"Generating statements for approval level {approval_level} and budget {budget}"
                    )
                new_statements = []
                new_statements_full_data = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_threads
                ) as executor:
                    future_to_data = {
                        executor.submit(
                            generator.generate,
                            agents=unsatisfied_agents,
                            target_approval=approval_level,
                            target_budget=budget,
                        ): {
                            "target_approval": approval_level,
                            "target_budget": budget,
                            "generator": generator.get_name(),
                        }
                        for generator in generators
                        for approval_level in gen_approval_levels
                    }

                    for future in concurrent.futures.as_completed(future_to_data):
                        statements, log_list = future.result()
                        data = future_to_data[future]
                        for log in log_list:
                            log.update(data)
                        new_statements.extend(statements)
                        for statement in statements:
                            new_statements_full_data.append(
                                {
                                    "stage": "generate",
                                    "statement": statement,
                                    "approval_level": approval_level,
                                    "budget": budget,
                                    "statement_cost": count_words(statement),
                                    "generator_name": data["generator"],
                                    "agent_utilities": {},
                                    "utility_distribution": defaultdict(float),
                                }
                            )
                        pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
                            log_filename,
                            mode="a",
                            header=False,
                            index=False,
                            columns=LOGS_COLS,
                        )

                # For each newly generated statement, calculate the approvals of all uncovered agents
                if verbose:
                    print(
                        f"Calculating approvals for {len(new_statements)} new statements"
                    )
                new_statement_approvals = [dict() for _ in range(len(new_statements))]
                new_statement_generator_names = [
                    None for _ in range(len(new_statements))
                ]
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_threads
                ) as executor:
                    future_to_data = {
                        executor.submit(agent.get_approval, statement): {
                            "agent_id": agent.get_id(),
                            "statement": statement,
                            "statement_idx": idx,
                        }
                        for agent in unsatisfied_agents
                        for idx, statement in enumerate(new_statements)
                    }

                    for future in concurrent.futures.as_completed(future_to_data):
                        approval, log_list = future.result()
                        data = future_to_data[future]
                        for log in log_list:
                            log.update(data)
                        new_statement_approvals[data["statement_idx"]][
                            data["agent_id"]
                        ] = approval
                        for x in approval_levels:
                            if approval >= x:
                                new_statements_full_data[data["statement_idx"]][
                                    "utility_distribution"
                                ][x] += 1
                        new_statement_generator_names[data["statement_idx"]] = (
                            new_statements_full_data[
                                data["statement_idx"]
                            ]["generator_name"]
                        )
                        new_statements_full_data[data["statement_idx"]][
                            "agent_utilities"
                        ][data["agent_id"]] = approval
                        pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
                            log_filename,
                            mode="a",
                            header=False,
                            index=False,
                            columns=LOGS_COLS,
                        )
                for entry in new_statements_full_data:
                    entry["utility_distribution"] = dict(
                        sorted(entry["utility_distribution"].items())
                    )
                pd.DataFrame(
                    new_statements_full_data,
                    columns=STATEMENTS_COLS,
                ).to_csv(
                    statements_filename,
                    mode="a",
                    header=False,
                    index=False,
                    columns=STATEMENTS_COLS,
                )

                assert len(new_statements) == len(new_statement_approvals)
                assert len(new_statements) >= 1

                # Add new statements to pool
                if not full_pool_selection:
                    # Reset statement pool each round and only pick from new statements (& precomputed ones?)
                    statement_pool = precomputed_statements.copy()
                    statement_pool_approvals = precomputed_approvals.copy()
                    statement_pool_generator_names = precomputed_generator_names.copy()
                statement_pool.extend(new_statements)
                statement_pool_approvals.extend(new_statement_approvals)
                statement_pool_generator_names.extend(new_statement_generator_names)

                # Pick best statement from pool
                num_covered_agents = []
                for approvals in statement_pool_approvals:
                    num_covered_agents.append(
                        sum(
                            1
                            for agent in unsatisfied_agents
                            if approvals[agent.get_id()] >= approval_level
                        )
                    )
                assert len(num_covered_agents) == len(statement_pool)

                if no_budgets:
                    round_based_lower_bound = 0
                else:
                    # making sure that not too short statemnets are added
                    round_based_lower_bound = (
                        budget_schedule[budget_schedule.index(budget) + 1]
                        if budget_schedule.index(budget) + 1 < len(budget_schedule)
                        else 0
                    )

                # Filter for the statements that have enough support (relative to their budget) to be added
                valid_indices = [
                    i
                    for i in range(len(statement_pool))
                    if count_words(statement_pool[i]) > round_based_lower_bound
                    and math.ceil(
                        count_words(statement_pool[i]) * num_agents / word_budget
                    )
                    <= num_covered_agents[i]
                ]

                if valid_indices:
                    # Pick the best statement from the valid indices
                    best_statement_idx = max(
                        valid_indices, key=lambda i: num_covered_agents[i]
                    )
                if len(valid_indices) == 0 or no_budgets:
                    # Handle the case where no statement meets the condition
                    best_statement_idx = max(
                        range(len(statement_pool)), key=lambda i: num_covered_agents[i]
                    )

                best_statement = statement_pool[best_statement_idx]

                pd.DataFrame(
                    [
                        {
                            "stage": "nominate_best",
                            "statement": best_statement,
                            "approval_level": approval_level,
                            "budget": budget,
                            "statement_cost": count_words(best_statement),
                            "generator_name": statement_pool_generator_names[
                                best_statement_idx
                            ],
                            "agent_utilities": statement_pool_approvals[
                                best_statement_idx
                            ],
                        }
                    ],
                    columns=STATEMENTS_COLS,
                ).to_csv(
                    statements_filename,
                    mode="a",
                    header=False,
                    index=False,
                    columns=STATEMENTS_COLS,
                )

                # See whether statement is good enough to add to slate
                preliminarily_covered_agents = [
                    agent
                    for agent in unsatisfied_agents
                    if statement_pool_approvals[best_statement_idx][agent.get_id()]
                    >= approval_level
                ]

                if no_budgets:
                    deserved_num_agents = number_of_agents_requiring_approval
                else:
                    deserved_num_agents = math.ceil(
                        count_words(best_statement) * num_agents / word_budget
                    )

                if (
                    len(preliminarily_covered_agents) >= deserved_num_agents
                    and count_words(best_statement) > round_based_lower_bound
                ):
                    # Only include the deserved_num_agents agents with the highest approval
                    print(deserved_num_agents, len(preliminarily_covered_agents))
                    preliminarily_covered_agents = sorted(
                        preliminarily_covered_agents,
                        key=lambda agent: statement_pool_approvals[best_statement_idx][
                            agent.get_id()
                        ],
                        reverse=True,
                    )[:deserved_num_agents]

                    # Statement is good enough
                    if verbose:
                        print(
                            f"Adding statement to slate: {best_statement} (approval level {approval_level}, budget {budget}, actual word count {count_words(best_statement)})"
                        )
                    slate.append(best_statement)
                    slate_costs.append(count_words(best_statement))
                    slate_covered_agent_ids.append(
                        [agent.get_id() for agent in preliminarily_covered_agents]
                    )
                    # Also return as much approval data on that statement as possible
                    slate_approvals.append(
                        {
                            agent.get_id(): statement_pool_approvals[
                                best_statement_idx
                            ][agent.get_id()]
                            for agent in agents
                            if agent.get_id()
                            in statement_pool_approvals[best_statement_idx]
                        }
                    )
                    unsatisfied_agents = [
                        agent
                        for agent in unsatisfied_agents
                        if agent not in preliminarily_covered_agents
                    ]
                    statement_pool.pop(best_statement_idx)
                    best_statement_approvals = statement_pool_approvals.pop(
                        best_statement_idx
                    )
                    best_statement_generator_name = statement_pool_generator_names.pop(
                        best_statement_idx
                    )
                    # Log to info.csv
                    dat = {
                        "statement": best_statement,
                        "approval_level": approval_level,
                        "budget": budget,
                        "statement_cost": count_words(best_statement),
                        "num_covered_agents": len(preliminarily_covered_agents),
                        "covered_agent_ids": [
                            agent.get_id() for agent in preliminarily_covered_agents
                        ],
                        "num_remaining_agents": len(unsatisfied_agents),
                        "remaining_agent_ids": [
                            agent.get_id() for agent in unsatisfied_agents
                        ],
                    }
                    dat["covered_agent_labels"] = [
                        str(agent.get_id()) + " : " + str(agent.get_label())
                        for agent in preliminarily_covered_agents
                    ]
                    dat["remaining_agent_labels"] = [
                        agent.get_label() for agent in unsatisfied_agents
                    ]

                    pd.DataFrame(
                        [dat],
                        columns=INFO_COLS_DRUGS,
                    ).to_csv(
                        info_filename,
                        mode="a",
                        header=False,
                        index=False,
                        columns=INFO_COLS_DRUGS,
                    )

                    pd.DataFrame(
                        [
                            {
                                "stage": "add_to_slate",
                                "statement": best_statement,
                                "approval_level": approval_level,
                                "budget": budget,
                                "statement_cost": count_words(best_statement),
                                "generator_name": best_statement_generator_name,
                                "agent_utilities": best_statement_approvals,
                                "num_covered_agents": len(preliminarily_covered_agents),
                                "covered_agent_ids": [
                                    agent.get_id()
                                    for agent in preliminarily_covered_agents
                                ],
                                "num_remaining_agents": len(unsatisfied_agents),
                                "remaining_agent_ids": [
                                    agent.get_id() for agent in unsatisfied_agents
                                ],
                                "covered_agent_data": [
                                    agent.get_data()
                                    for agent in preliminarily_covered_agents
                                ],
                            }
                        ],
                        columns=STATEMENTS_COLS,
                    ).to_csv(
                        statements_filename,
                        mode="a",
                        header=False,
                        index=False,
                        columns=STATEMENTS_COLS,
                    )

                    # Note: this doesn't break the while loop, we will continue trying to generate statements at this combo of (approval_level, budget) until we can't anymore

                else:
                    if verbose:
                        print(
                            f"Statement not good enough to add to slate: {best_statement} (approval level {approval_level}, budget {budget})"
                        )
                    break  # this will break while loop, moving on to next budget

    if unsatisfied_agents:
        print(f"Warning: {len(unsatisfied_agents)} agents left uncovered")
        print(f"Word budget : {word_budget}")
        print(f"Slate total cost: {sum(slate_costs)}")
        print(f"Word budget leftover: {word_budget - sum(slate_costs)}")
        print(
            f"Num words the unsatisfied agents deserve: {word_budget * len(unsatisfied_agents) / num_agents}"
        )

    # Log slate to out.txt
    with open(out_filename, "w") as f:
        f.write("Slate:")
        f.write("\n".join(slate))
        f.write("\n")
        f.write("Slate costs:")
        f.write("\n".join(str(cost) for cost in slate_costs))
        f.write("\n")
        f.write("Slate covered agent ids:")
        f.write("\n".join(str(ids) for ids in slate_covered_agent_ids))
        f.write("\n")
        f.write("Slate approvals:")
        f.write("\n".join(str(approvals) for approvals in slate_approvals))

    # Write slate, slate_costs, slate_covered_agent_ids, slate_approvals to obj_output.pkl
    output_dict = {
        "slate": slate,
        "slate_costs": slate_costs,
        "slate_covered_agent_ids": slate_covered_agent_ids,
        "slate_approvals": slate_approvals,
    }
    with open(obj_output_filename, "wb") as f:
        pickle.dump(output_dict, f)

    return slate, slate_costs, slate_covered_agent_ids, slate_approvals, log_dirname


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "bowlinggreen_41",
            "drugs_80",
            "drugs_obesity_80",
            "drugs_80_extreme_middle",
        ],
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=15,
    )

    parser.add_argument("--no_budgets", action="store_true")

    args = parser.parse_args()

    no_budgets = args.no_budgets

    data_filename, summaries_filename, tags_filename, _ = get_data_filenames(
        args.dataset
    )

    df = pd.read_csv(
        get_base_dir_path() / data_filename,
        index_col=0,
    )
    agents = [
        FastAgent(
            id=row["user_id"],
            data=row["data"],
            prompt_type="agreement_v1",
            specificity_coeff=1,
            specificity_prompt_type="specificity_v1",
            label=row["label"],
        )
        for _, row in df.iterrows()
    ]

    assert len(agents) == len(set([agent.get_id() for agent in agents]))

    # Don't use summaries
    for agent in agents:
        agent.summary = agent.data

    for agent in tqdm(agents, desc="Computing agent embeddings"):
        assert agent.summary
        agent.set_embedding(
            llm.client.embeddings.create(
                input=agent.summary,
                model="text-embedding-3-large",
            )
            .data[0]
            .embedding
        )

    # Load tags
    tags_df = pd.read_csv(get_base_dir_path() / tags_filename)
    tags_df.dropna(subset=["agent_id"], inplace=True)

    for agent in agents:
        indiv_df = tags_df[tags_df["agent_id"] == agent.get_id()]
        assert len(indiv_df) == 1
        tags = indiv_df["response"].values[0]
        tags_dict = literal_eval(tags)
        agent.tags = tags_dict

    if "bowlinggreen" in args.dataset:
        words_per_agent = 4
        word_budget = words_per_agent * len(agents)
        budget_schedule = [
            words_per_agent * i for i in [20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        ]
        approval_levels = [5.5, 5, 4.5, 4, 3.5, 3, 2, 1, 0]
        minimum_statement_length = 8
    else:
        words_per_agent = 2
        word_budget = words_per_agent * len(agents)
        budget_schedule = [
            words_per_agent * i
            for i in [40, 35, 30, 25, 20, 18, 16, 14, 12, 10, 8, 6, 5, 4, 3, 2, 1]
        ]
        approval_levels = [5.5, 5, 4.5, 4, 3.5, 3, 2, 1, 0]
        minimum_statement_length = 10

    if no_budgets:
        number_of_slate_statements = 5
        if not len(agents) % number_of_slate_statements == 0:
            exit(
                "Number of statments needs to divide the number of agents for no budgets"
            )
        word_budget = len(agents)
        budget_schedule = [int(len(agents) / number_of_slate_statements)]
    generators = [
        ClosestClusterGenerator(
            cluster_size=-1,
            prompt_type="v3_very_high_detail",
            seed=0,
            adaptable=True,
            LLM_emb=False,
        ),
        ClosestClusterGenerator(
            cluster_size=-1,
            prompt_type="v3_very_high_detail",
            seed=1,
            adaptable=True,
            LLM_emb=True,
        ),
        PreviousBestGenerator(
            cluster_size=-1,
            prompt_type="v3_very_high_detail",
            seed=2,
        ),
        PreviousBestGenerator(
            cluster_size=-1, prompt_type="v3_very_high_detail", seed=3, temperature=1.0
        ),
        WeightedNNGenerator(
            cluster_size=-1,
            prompt_type="v3_very_high_detail",
            seed=4,
            adaptable=True,
            temperature=0.5,
            LLM_emb=False,
        ),
        WeightedNNGenerator(
            cluster_size=-1,
            prompt_type="v3_very_high_detail",
            seed=5,
            adaptable=True,
            temperature=0.5,
            LLM_emb=True,
        ),
        TagNNGenerator(
            cluster_size=-1,
            prompt_type="v3_very_high_detail",
            seed=6,
            adaptable=True,
            LLM_emb=False,
        ),
        TagNNGenerator(
            cluster_size=-1,
            prompt_type="v3_very_high_detail",
            seed=7,
            adaptable=True,
            LLM_emb=True,
        ),
    ]

    # Generate slate

    slate, slate_costs, slate_covered_agent_ids, slate_approvals, log_dirname = (
        generate_slate(
            agents=agents,
            precompute_generators=[],
            generators=generators,
            word_budget=word_budget,
            budget_schedule=budget_schedule,
            approval_levels=approval_levels,
            full_approval_generation=False,
            full_pool_selection=True,
            num_threads=args.num_threads,
            verbose=True,
            no_budgets=no_budgets,
            dataset=args.dataset,
            minimum_statement_length=minimum_statement_length,
        )
    )

    print(f"Done generating slate, saved to {log_dirname}")
    print(f"Slate: {slate}")
