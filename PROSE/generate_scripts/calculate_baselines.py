from ast import literal_eval
import math
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PROSE.utils.llm_tools import GPT, LLMLog, DEFAULT_MODEL
from tqdm import tqdm
import concurrent.futures
from PROSE.queries.query_interface import Agent

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from PROSE.queries.fast_queries import FastAgent, FastGenerator
from PROSE.queries.verification_disc_query import CoTAgent
import scipy.optimize
from PROSE.utils.helper_functions import (
    count_words,
    get_base_dir_path,
    get_time_string,
)
from PROSE.generate_scripts.generate_slate import LOGS_COLS
from PROSE.datasets.load import get_data_filenames


import argparse

llm = GPT(
    model=DEFAULT_MODEL,
)

# Define the original prompt
CONTEXTLESS_ZERO_SHOT_BASELINE_PROMPT_TEMPLATE = """ 
Your task is to write a *proportional opinion slate* on a particular topic while staying within a word budget. A proportional opinion slate is a collection of opinions that, taken together, give an overview of the general population's opinions on that topic. Morover, the lengths of the opinions should correspond to the relative proportion of people that hold that opinion (hence "proportional" opinion slate). 

**Stylized example:** Suppose 70% of people believe salads are the best dinner, 20% of people believe burgers are the best dinner, and 10% of people believe soup is the best dinner, and the word budget is 50 words. Then, a proportional opinion slate might look like this:
- Salads are the best dinner option by far. They are healthy and tasty. They fulfill all major dietary requirements so anybody can have them. They are also flexible, since many different ingredients can be substituted.
- Burgers make the best dinner because they're tasty and filling.
- Soup makes the best dinner.

**Writing guidelines:** Each statement should be a short, strong opinion that reflects a single, clear stance reflecting a population segment's thoughts. The opinion should sound personal, direct, and conversational, as if written by someone expressing their own thoughts. Avoid summary-style language or listing multiple viewpoints. Finally, do not list word counts next to your statements -- just put the statements and nothing else.

The topic: {topic} 
The word budget: {word_budget} words. Do not write more than {word_budget} words under any circumstances!
""".strip()

# Needs to change if structure of orginal prompt changes
CONTEXTLESS_ZERO_SHOT_BASELINE_PROMPT_TEMPLATE_WITHOUT_BUDGET = "\n".join(
    CONTEXTLESS_ZERO_SHOT_BASELINE_PROMPT_TEMPLATE.split("\n")[:-1]
)


NO_ADDITIONAL_DATA_PROMPT_TEMPLATE = """
Write your proportional summary below. Write each statement on a new line, and use "- " as bullet points. Respect the word limit!
""".strip()

ADDITIONAL_DATA_PROMPT_TEMPLATE = """
To improve the accuracy of your proportional opinion slate, below provided are representative opinions that people have on the topic. You should aim to construct a proportional opinion slate that matches the distribution of these opinions.

{user_opinions_list_str}

Write your proportional summary below. Write each statement on a new line, and use "- " as bullet points. Respect the word limit!
""".strip()


def get_contextless_zero_shot_baseline(
    topic: str, word_budget: int
) -> Tuple[List[str], List[LLMLog]]:
    if word_budget > -1:
        response, _, log = llm.call(
            messages=[
                {
                    "role": "system",
                    "content": CONTEXTLESS_ZERO_SHOT_BASELINE_PROMPT_TEMPLATE.format(
                        topic=topic,
                        word_budget=word_budget,
                    ),
                },
                {
                    "role": "user",
                    "content": NO_ADDITIONAL_DATA_PROMPT_TEMPLATE,
                },
            ]
        )
    else:
        response, _, log = llm.call(
            messages=[
                {
                    "role": "system",
                    "content": CONTEXTLESS_ZERO_SHOT_BASELINE_PROMPT_TEMPLATE_WITHOUT_BUDGET.format(
                        topic=topic
                    ),
                },
                {
                    "role": "user",
                    "content": NO_ADDITIONAL_DATA_PROMPT_TEMPLATE,
                },
            ]
        )

    response = response.strip()

    response_l = [s.removeprefix("- ").strip() for s in response.split("\n")]

    return response_l, [log]


def get_zero_shot_baseline(
    topic: str, word_budget: int, user_opinions: List[str]
) -> Tuple[List[str], List[LLMLog]]:
    user_opinions_list_str = "\n".join(["- " + s for s in user_opinions])
    if word_budget > -1:
        response, _, log = llm.call(
            messages=[
                {
                    "role": "system",
                    "content": CONTEXTLESS_ZERO_SHOT_BASELINE_PROMPT_TEMPLATE.format(
                        topic=topic,
                        word_budget=word_budget,
                    ),
                },
                {
                    "role": "user",
                    "content": ADDITIONAL_DATA_PROMPT_TEMPLATE.format(
                        user_opinions_list_str=user_opinions_list_str
                    ),
                },
            ]
        )
    else:
        response, _, log = llm.call(
            messages=[
                {
                    "role": "system",
                    "content": CONTEXTLESS_ZERO_SHOT_BASELINE_PROMPT_TEMPLATE_WITHOUT_BUDGET.format(
                        topic=topic
                    ),
                },
                {
                    "role": "user",
                    "content": ADDITIONAL_DATA_PROMPT_TEMPLATE.format(
                        user_opinions_list_str=user_opinions_list_str
                    ),
                },
            ]
        )

    response = response.strip()

    response_l = [s.removeprefix("- ").strip() for s in response.split("\n")]

    return response_l, [log]


def get_clustering_baseline(
    word_budget: int, agents: List[Agent]
) -> Tuple[List[str], List[LLMLog], List[List[str]]]:
    # First, compute all agent embeddings (text embeddings of their text)

    agent_embeddings = []

    for agent in tqdm(agents, desc="Computing agent embeddings"):
        assert agent.summary
        agent_embeddings.append(
            llm.client.embeddings.create(
                input=agent.summary,
                model="text-embedding-3-large",
            )
            .data[0]
            .embedding
        )

    assert len(agent_embeddings) == len(agents)

    scaled_tag_vectors = StandardScaler().fit_transform(agent_embeddings)

    pca = PCA(n_components=5, random_state=0).fit(scaled_tag_vectors)
    tag_vectors_reduced = pca.transform(scaled_tag_vectors)

    cluster = AffinityPropagation(random_state=0).fit(tag_vectors_reduced)

    agent_groups = []
    for cluster_label in set(cluster.labels_):
        agent_group = [
            agent
            for agent, label in zip(agents, cluster.labels_)
            if label == cluster_label
        ]
        agent_groups.append(agent_group)

    assert len(agents) == sum([len(agent_group) for agent_group in agent_groups])

    generator = FastGenerator(
        prompt_type="v3_moderate_detail",
    )

    statements = []
    logs = []

    for agent_group in tqdm(agent_groups, desc="Generating statements"):
        if word_budget > -1:
            statement_list, log_list = generator.generate(
                agents=agent_group,
                target_approval=None,
                target_budget=math.ceil(word_budget * len(agent_group) / len(agents)),
            )
        else:
            statement_list, log_list = generator.generate(
                agents=agent_group,
                target_approval=None,
                target_budget=-1,
            )
        statements.extend(statement_list)
        logs.extend(log_list)

    agent_id_groups = [
        [agent.id for agent in agent_group] for agent_group in agent_groups
    ]

    return statements, logs, agent_id_groups


def compute_agent_id_groups(
    agent_id_to_utilities: Dict[str, List[float]],
    statement_capacities: List[int],
) -> List[List[str]]:
    """
    Given agent_id_to_utilities and statement capacities, compute how the agents should be matched to the statements to maximize utility. (This is when we use some sort of slate generation procedure that doesn't also match agents proportionally to statements, e.g. zero-shot baselines).
    """
    assert sum(statement_capacities) >= len(
        agent_id_to_utilities
    )  # not exact because we round up
    assert (
        len(set([len(utility_list) for utility_list in agent_id_to_utilities.values()]))
        == 1
    )

    num_agents = len(agent_id_to_utilities)
    num_statements = len(statement_capacities)

    utility_matrix = np.zeros((num_agents, sum(statement_capacities)))

    agent_ids = list(agent_id_to_utilities.keys())
    column_index = 0

    for statement_idx, capacity in enumerate(statement_capacities):
        for _ in range(capacity):
            for agent_idx, agent_id in enumerate(agent_ids):
                utility_matrix[agent_idx, column_index] = agent_id_to_utilities[
                    agent_id
                ][statement_idx]
            column_index += 1

    row_indices, col_indices = scipy.optimize.linear_sum_assignment(
        utility_matrix, maximize=True
    )

    agent_id_to_matched_statement_idx = {}
    cumulative_capacities = [0] + list(np.cumsum(statement_capacities))

    for agent_idx, col_idx in zip(row_indices, col_indices):
        agent_id = agent_ids[agent_idx]
        for statement_idx, (start, end) in enumerate(
            zip(cumulative_capacities[:-1], cumulative_capacities[1:])
        ):
            if start <= col_idx < end:
                agent_id_to_matched_statement_idx[agent_id] = statement_idx
                break

    agent_id_groups = []
    for statement_idx in range(num_statements):
        agent_ids = [
            agent_id
            for agent_id, matched_statement_idx in agent_id_to_matched_statement_idx.items()
            if matched_statement_idx == statement_idx
        ]
        agent_id_groups.append(agent_ids)

    assert sum([len(agent_ids) for agent_ids in agent_id_groups]) == num_agents

    return agent_id_groups


def compute_matching_and_approval(
    slate: List[str],
    agents: List[Agent],
    num_threads: int,
) -> Tuple[Dict[str, float], List[LLMLog]]:
    agent_id_to_utilities = {}
    logs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_agent = {
            executor.submit(compute_agent_approvals, agent, slate): agent
            for agent in agents
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_agent), total=len(agents)
        ):
            agent_id, log_list, utilities = future.result()
            agent_id_to_utilities[agent_id] = utilities
            logs.extend(log_list)

    statement_lengths = [count_words(statement) for statement in slate]

    statement_capacities = [
        math.ceil((statement_length / sum(statement_lengths)) * len(agents))
        for statement_length in statement_lengths
    ]

    agent_id_groups = compute_agent_id_groups(
        agent_id_to_utilities, statement_capacities
    )

    agent_id_to_approval = {}
    for statement_idx, agent_id_group in enumerate(agent_id_groups):
        for agent_id in agent_id_group:
            agent_id_to_approval[agent_id] = agent_id_to_utilities[agent_id][
                statement_idx
            ]

    return agent_id_to_approval, logs


def compute_approval_given_matching(
    slate: List[str],
    agents: List[Agent],
    agent_id_groups: List[List[str]],
    num_threads: int,
) -> Tuple[Dict[str, float], List[LLMLog]]:
    agent_id_to_approval = {}

    logs = []

    for statement, agent_id_group in tqdm(
        zip(slate, agent_id_groups), total=len(slate)
    ):
        for agent_id in agent_id_group:
            agent = agent_id_to_agent[agent_id]
            approval, log_list = agent.get_approval(statement)
            agent_id_to_approval[agent_id] = approval
            logs.extend(log_list)

    return agent_id_to_approval, logs


def round_down(x, digits=2):
    return math.floor(x * 10**digits) / 10**digits


def find_bjr_violations(
    agent_id_to_statement_pool_utilities: Dict[str, List[float]],
    agent_id_to_approval: Dict[str, float],
    pool_statement_costs: List[int],
    pool_statements: List[str],
    total_budget: int,
) -> Tuple[List[dict], float]:
    """
    agent_id_to_statement_pool_utilities: maps each agent_id to a list of utilities, each corresponds to the utility that agent gets
                                          for some statement in the pool (e.g., everything generate_slate computed along the way)
    agent_id_to_approval: maps each agent_id to the utility that agent gets for the statement in whatever slate is being considered
    pool_statement_costs: the cost of each statement in the pool
    pool_statements: the statements in the pool (just for logging)
    """

    assert len(agent_id_to_statement_pool_utilities) == len(agent_id_to_approval)

    num_agents = len(agent_id_to_approval)

    # Round down all floats to nearest 0.01
    agent_id_to_approval = {
        agent_id: round_down(approval, digits=2)
        for agent_id, approval in agent_id_to_approval.items()
    }
    attainable_utilities_by_slate = set(agent_id_to_approval.values())
    agent_id_to_statement_pool_utilities = {
        agent_id: [round_down(utility, digits=2) for utility in utilities]
        for agent_id, utilities in agent_id_to_statement_pool_utilities.items()
    }

    num_pool_statements = len(list(agent_id_to_statement_pool_utilities.values())[0])

    bjr_data = []
    bjr_data_concise = []

    for statement_idx in range(num_pool_statements):
        statement_with_violation = False
        # Find all utilities that that statement attained
        attainable_utilities = set()
        for _, utilities in agent_id_to_statement_pool_utilities.items():
            attainable_utilities.add(utilities[statement_idx])
        # and also, all utilities that agents got in the slate
        attainable_utilities.update(attainable_utilities_by_slate)
        attainable_utilities = sorted(list(attainable_utilities))

        utility_midpoints = [
            (attainable_utilities[i] + attainable_utilities[i + 1]) / 2
            for i in range(len(attainable_utilities) - 1)
        ]

        for utility_cutoff in utility_midpoints:
            # Find agents that like statement_idx more (or eq to) than utility_cutoff, and their current
            # match less than utility_cutoff
            coalition = [
                (agent_id, utilities)
                for agent_id, utilities in agent_id_to_statement_pool_utilities.items()
                if utilities[statement_idx] >= utility_cutoff
                and agent_id_to_approval[agent_id] < utility_cutoff
            ]

            required_coalition_size = (
                pool_statement_costs[statement_idx] * num_agents / total_budget
            )

            if len(coalition) >= required_coalition_size:
                is_duplicate = any(
                    entry["statement_idx"] == statement_idx
                    and entry["coalition_size"] == len(coalition)
                    and entry["utility_gap"]
                    == min([agent[1][statement_idx] for agent in coalition])
                    - max([agent_id_to_approval[agent[0]] for agent in coalition])
                    for entry in bjr_data
                )

                entry = {
                    "statement_idx": statement_idx,
                    "statement": pool_statements[statement_idx],
                    "statement_cost": pool_statement_costs[statement_idx],
                    "size_over": len(coalition) / required_coalition_size,
                    "utility_gap": min([agent[1][statement_idx] for agent in coalition])
                    - max([agent_id_to_approval[agent[0]] for agent in coalition]),
                    "utility_cutoff": utility_cutoff,
                    "coalition_size": len(coalition),
                    "coalition_agent_ids": coalition,
                }
                if not statement_with_violation:
                    bjr_data_concise.append(entry)

                if not is_duplicate:
                    bjr_data.append(entry)

                statement_with_violation = True

                # break  # done with this statement

    prop_bjr_violations = len(bjr_data_concise) / num_pool_statements

    return bjr_data, bjr_data_concise, prop_bjr_violations


def compute_agent_approvals(
    agent: Agent, statements: List[str]
) -> Tuple[str, List[LLMLog], List[float]]:
    # helper function for multithreading
    agent_id = agent.get_id()
    utilities = []
    temp_logs = []
    for statement in statements:
        approval, log_list = agent.get_approval(statement)
        temp_logs.extend(log_list)
        utilities.append(approval)
    assert len(utilities) == len(statements)
    return agent_id, temp_logs, utilities


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
        "--slate_dirname",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--disc_query_type",
        choices=["fast", "cot"],
        default="cot",
    )

    parser.add_argument(
        "--bjr_pool_size",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-11-20",
        type=str,
    )
    parser.add_argument(
        "--eval_dirname",
        type=str,
        default=None,
    )

    parser.add_argument("--no_budgets", action="store_true")

    args = parser.parse_args()

    no_budgets = args.no_budgets

    slate_dirname = args.slate_dirname

    data_filename, summaries_filename, tags_filename, topic = get_data_filenames(
        dataset=args.dataset
    )

    df = pd.read_csv(
        get_base_dir_path() / data_filename,
        index_col=0,
    )

    if args.disc_query_type == "fast":
        agents = [
            FastAgent(
                id=row["user_id"],
                data=row["data"],
                prompt_type="agreement_v1",
                specificity_coeff=0,
                specificity_prompt_type="specificity_v1",
            )
            for _, row in df.iterrows()
        ]
    elif args.disc_query_type == "cot":
        agents = [
            CoTAgent(
                id=row["user_id"],
                data=row["data"],
                prompt_type="verification_v1",
                model=args.model,
            )
            for _, row in df.iterrows()
        ]
    else:
        raise NotImplementedError(
            f"unrecognized disc query type={args.disc_query_type}"
        )

    assert len(agents) == len(set([agent.get_id() for agent in agents]))

    for agent in agents:
        agent.summary = agent.data

    # Load tags
    tags_df = pd.read_csv(get_base_dir_path() / tags_filename)
    tags_df.dropna(subset=["agent_id"], inplace=True)

    for agent in agents:
        indiv_df = tags_df[tags_df["agent_id"] == agent.get_id()]
        assert len(indiv_df) == 1
        tags = indiv_df["response"].values[0]
        tags_dict = literal_eval(tags)
        agent.tags = tags_dict

    assert len(agents) == len(set([agent.get_id() for agent in agents]))

    agent_id_to_agent = {agent.get_id(): agent for agent in agents}

    llm = GPT(
        model=DEFAULT_MODEL,
    )

    # Load slate we're benchmarking against
    slate_df = pd.read_csv(get_base_dir_path() / slate_dirname / "info.csv")

    # Compute covered_agent_ids of slate, to make sure it's valid
    # (if it leaves people uncovered, then don't allow it to obe used )
    statements_slate = list(slate_df["statement"].values)
    covered_agent_ids_slate = [
        literal_eval(s) for s in slate_df["covered_agent_ids"].values
    ]
    leftover_agent_ids_slate = literal_eval(slate_df["remaining_agent_ids"].values[-1])
    assert (
        len(leftover_agent_ids_slate) == 0
    ), f"Slate {slate_dirname} leaves {len(leftover_agent_ids_slate)} agents uncovered"
    # covered_agent_ids_slate[-1].extend(leftover_agent_ids_slate) # add extra ones to last slot
    assert sum([len(agent_ids) for agent_ids in covered_agent_ids_slate]) == len(agents)

    slate_global_params = pd.read_csv(
        get_base_dir_path() / slate_dirname / "global_params.csv"
    )

    word_budget = slate_global_params["word_budget"].values[0]

    if no_budgets:
        word_budget = -1

    # Create baseline log dir

    baseline_log_dir = (
        get_base_dir_path()
        / "experiment_logs"
        / f"{get_time_string()}__baseline__{args.dataset}__{word_budget}"
    )
    os.makedirs(baseline_log_dir)
    log_filename = baseline_log_dir / "logs.csv"
    pd.DataFrame(columns=LOGS_COLS).to_csv(log_filename, index=False)

    print("Running baselines...")

    slate1, log_list = get_contextless_zero_shot_baseline(
        topic=topic,
        word_budget=word_budget,
    )
    pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
        log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    )

    slate2, log_list = get_zero_shot_baseline(
        topic=topic,
        word_budget=word_budget,
        user_opinions=[agent.summary for agent in agents],
    )
    pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
        log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    )

    slate3, log_list, agent_id_groups3 = get_clustering_baseline(
        word_budget=word_budget,
        agents=agents,
    )
    pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
        log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    )

    # Save the slates to slates.txt
    with open(baseline_log_dir / "baseline_slates.txt", "w") as f:
        f.write("Contextless zero shot slate:\n")
        f.write("\n".join(slate1))
        f.write("\n\n")
        f.write("Zero shot slate:\n")
        f.write("\n".join(slate2))
        f.write("\n\n")
        f.write("Clustered slate:\n")
        f.write("\n".join(slate3))
        f.write("\n\n")
        print("Written to " + str(baseline_log_dir) + "baseline_slates.txt")

    # Compute utilities

    # For baselines 1 and 2, we need to compute everything,
    # since we're not given a matching a priori

    print("Computing utilities...")

    agent_id_to_approval1, log_list = compute_matching_and_approval(
        slate=slate1,
        agents=agents,
        num_threads=args.num_threads,
    )
    pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
        log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    )

    agent_id_to_approval2, log_list = compute_matching_and_approval(
        slate=slate2,
        agents=agents,
        num_threads=args.num_threads,
    )
    pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
        log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    )

    # For baseline 3, it gives us a matching -- so we can use that one if it's
    # valid, otherwise do a new matching as a fallback

    agent_id_to_approval3, log_list = compute_approval_given_matching(
        slate=slate3,
        agents=agents,
        agent_id_groups=agent_id_groups3,
        num_threads=args.num_threads,
    )
    pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
        log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    )

    # agent_id_to_approval3_1, log_list = compute_matching_and_approval(
    #     slate=slate3,
    #     agents=agents,
    #     num_threads=args.num_threads,
    # )
    # pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
    #    log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    # )

    # Finally, compute the utilities of our slate

    statements_slate = list(slate_df["statement"].values)

    # Save our slate to slate.txt
    with open(baseline_log_dir / "slates.txt", "w") as f:
        f.write("\n".join(statements_slate))
        f.write("\n\n")

    print("Computing utilities for slate...")

    covered_agent_ids_slate = [
        literal_eval(s) for s in slate_df["covered_agent_ids"].values
    ]
    leftover_agent_ids_slate = literal_eval(slate_df["remaining_agent_ids"].values[-1])
    # For now, just add the leftover agents to the last group - maybe eventually match intelligently
    covered_agent_ids_slate[-1].extend(leftover_agent_ids_slate)

    assert sum([len(agent_ids) for agent_ids in covered_agent_ids_slate]) == len(agents)

    agent_id_to_approval_slate, log_list = compute_approval_given_matching(
        slate=statements_slate,
        agents=agents,
        agent_id_groups=covered_agent_ids_slate,
        num_threads=args.num_threads,
    )
    pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
        log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    )

    # agent_id_to_approval_slate_1, log_list = compute_matching_and_approval(
    #    slate=statements_slate,
    #    agents=agents,
    #    num_threads=args.num_threads,
    # )
    # pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
    #    log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
    # )

    assert (
        set(agent_id_to_approval_slate.keys())
        == set(agent_id_to_approval1.keys())
        == set(agent_id_to_approval2.keys())
        == set(agent_id_to_approval3.keys())
    )

    utilities = [
        {
            "agent_id": agent_id,
            "cluster_baseline_utility": agent_id_to_approval3[agent_id],
            # "cluster_baseline_utility_rematched": agent_id_to_approval3_1[agent_id],
            "slate_utility": agent_id_to_approval_slate[agent_id],
            # "slate_utility_rematched": agent_id_to_approval_slate_1[agent_id],
            "zero_shot_utility": agent_id_to_approval2[agent_id],
            "contextless_zero_shot_utility": agent_id_to_approval1[agent_id],
        }
        for agent_id in agent_id_to_agent
    ]

    utilities_df = pd.DataFrame(utilities)

    utilities_df.to_csv(baseline_log_dir / "utilities.csv", index=False)

    utilities_df = pd.read_csv(baseline_log_dir / "utilities.csv")
    agent_id_to_approval_slate = dict(
        zip(utilities_df["agent_id"], utilities_df["slate_utility"])
    )
    agent_id_to_approval1 = dict(
        zip(utilities_df["agent_id"], utilities_df["contextless_zero_shot_utility"])
    )
    agent_id_to_approval2 = dict(
        zip(utilities_df["agent_id"], utilities_df["zero_shot_utility"])
    )
    agent_id_to_approval3 = dict(
        zip(utilities_df["agent_id"], utilities_df["cluster_baseline_utility"])
    )

    ## Now, do BJR calculation

    statements_df = pd.read_csv(get_base_dir_path() / slate_dirname / "statements.csv")
    full_statement_pool = sorted(set(statements_df["statement"]))
    if len(full_statement_pool) >= args.bjr_pool_size:
        statement_pool = random.sample(full_statement_pool, args.bjr_pool_size)
    else:
        print(
            f"Warning: only {len(full_statement_pool)} statements available (bjr_pool_size={args.bjr_pool_size}). Using all of them"
        )
        statement_pool = full_statement_pool
    print(f"Calculating BJR violations using {len(statement_pool)} statements.")

    # Calculate agent utilities of the statements in this pool
    agent_id_to_statement_pool_utilities = {
        agent.get_id(): [None for _ in range(len(statement_pool))] for agent in agents
    }
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_threads
    ) as executor:
        future_to_data = {
            executor.submit(agent.get_approval, statement): {
                "agent_id": agent.get_id(),
                "statement_idx": statement_idx,
            }
            for agent in agents
            for statement_idx, statement in enumerate(statement_pool)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_data), total=len(future_to_data)
        ):
            data = future_to_data[future]
            approval, log_list = future.result()
            agent_id_to_statement_pool_utilities[data["agent_id"]][
                data["statement_idx"]
            ] = approval
            pd.DataFrame(log_list, columns=LOGS_COLS).to_csv(
                log_filename, mode="a", header=False, index=False, columns=LOGS_COLS
            )

    pool_statement_costs = [count_words(statement) for statement in statement_pool]
    slate_global_params = pd.read_csv(
        get_base_dir_path() / slate_dirname / "global_params.csv"
    ).to_dict(orient="records")[0]
    total_budget = slate_global_params["word_budget"]

    # For the slate and each of the baselines, calculate the number of BJR violations
    bjr_data_slate, bjr_data_concise_slate, bjr_prop_violation_slate = (
        find_bjr_violations(
            agent_id_to_statement_pool_utilities,
            agent_id_to_approval_slate,
            pool_statement_costs,
            statement_pool,
            total_budget,
        )
    )

    bjr_data_baseline1, bjr_data_concise_baseline1, bjr_prop_violation_baseline1 = (
        find_bjr_violations(
            agent_id_to_statement_pool_utilities,
            agent_id_to_approval1,
            pool_statement_costs,
            statement_pool,
            total_budget,
        )
    )

    bjr_data_baseline2, bjr_data_concise_baseline2, bjr_prop_violation_baseline2 = (
        find_bjr_violations(
            agent_id_to_statement_pool_utilities,
            agent_id_to_approval2,
            pool_statement_costs,
            statement_pool,
            total_budget,
        )
    )

    bjr_data_baseline3, bjr_data_concise_baseline3, bjr_prop_violation_baseline3 = (
        find_bjr_violations(
            agent_id_to_statement_pool_utilities,
            agent_id_to_approval3,
            pool_statement_costs,
            statement_pool,
            total_budget,
        )
    )

    for data, source_name in zip(
        [
            bjr_data_slate,
            bjr_data_baseline1,
            bjr_data_baseline2,
            bjr_data_baseline3,
        ],
        [
            "slate",
            "contextless_zero_shot",
            "zero_shot",
            "cluster",
        ],
    ):
        for entry in data:
            entry["source"] = source_name

    bjr_results = (
        bjr_data_slate + bjr_data_baseline1 + bjr_data_baseline2 + bjr_data_baseline3
    )

    pd.DataFrame(bjr_results).to_csv(
        get_base_dir_path() / args.eval_dirname / "bjr_results.csv", index=False
    )

    for data, source_name in zip(
        [
            bjr_data_concise_slate,
            bjr_data_concise_baseline1,
            bjr_data_concise_baseline2,
            bjr_data_concise_baseline3,
        ],
        [
            "slate",
            "contextless_zero_shot",
            "zero_shot",
            "cluster",
        ],
    ):
        for entry in data:
            entry["source"] = source_name

    bjr_concise_results = (
        bjr_data_concise_slate
        + bjr_data_concise_baseline1
        + bjr_data_concise_baseline2
        + bjr_data_concise_baseline3
    )

    pd.DataFrame(bjr_concise_results).to_csv(
        get_base_dir_path() / args.eval_dirname / "bjr_concise_results.csv", index=False
    )

    print("Done!")
    print(f"Utilities logged to {baseline_log_dir}/utilities.csv")
    print(f"BJR results logged to {baseline_log_dir}/bjr_results.csv")
    print(f"BJR violation prop of slate = {bjr_prop_violation_slate}")
    print(f"BJR violation prop of baseline1 = {bjr_prop_violation_baseline1}")
    print(f"BJR violation prop of baseline2 = {bjr_prop_violation_baseline2}")
    print(f"BJR violation prop of baseline3 = {bjr_prop_violation_baseline3}")
