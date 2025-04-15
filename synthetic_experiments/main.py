# -*- coding: utf-8 -*-
import random
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np

import multiprocessing
import time
import pandas as pd

import argparse

import csv


from multiprocessing import Pool

import os


from scipy.stats import scoreatpercentile


def tenth_percentile_mean(values):
    """
    Computes the mean of the bottom 10% of a list of values using SciPy.

    Parameters:
        values (list): A list of numerical values.

    Returns:
        float: The mean of the bottom 10% of values.
    """
    if not values:
        raise ValueError("The list of values is empty.")

    # Find the 10th percentile threshold
    percentile_10 = scoreatpercentile(values, 10)

    # Select values less than or equal to the 10th percentile
    bottom_10_percent = [v for v in values if v <= percentile_10]

    # Compute and return the mean of these values
    return np.mean(bottom_10_percent)


#################### Environemnt ####################
class Issue:
    def __init__(self, issue_id, opinions):
        self.issue_id = issue_id
        self.opinions = list(range(1, opinions + 1))

    def utility(self, agent_opinion, statement_opinion):
        return int(len(self.opinions) / 2) - abs(agent_opinion - statement_opinion)


class Agent:
    def __init__(self, agent_id, opinions):
        self.agent_id = agent_id
        self.opinions = opinions  # Dictionary mapping issue_id to opinion
        self._perturbed_cache = {}
        # self.mean=mean

    def utility_from_statement(self, statement):
        utility = 0
        for issue_id, opinion in statement.opinions.items():
            # print(issue_id,utility)
            agent_opinion = self.opinions.get(issue_id, None)
            if agent_opinion is not None:
                issue_utility = statement.issues[issue_id].utility(
                    agent_opinion, opinion
                )
                utility += issue_utility
        return utility

    @staticmethod
    def random_agent(agent_id, issues):
        """
        Generate an agent with random opinions on all the issues.
        :param agent_id: The ID of the agent.
        :param issues: A dictionary of issues.
        :return: A randomly generated Agent object.
        """
        random_opinions = {
            issue_id: random.choice(issue.opinions)
            for issue_id, issue in issues.items()
        }
        return Agent(agent_id=agent_id, opinions=random_opinions)

    @staticmethod
    def random_Gauss(agent_id, issues):
        mean = 1
        random_opinions = {}
        for issue_id, issue in issues.items():
            std_dev = len(issue.opinions) / 4
            weights = np.exp(-0.5 * ((np.array(issue.opinions) - mean) / std_dev) ** 2)
            weights /= weights.sum()  # Normalize

            random_opinions[issue_id] = int(np.random.choice(issue.opinions, p=weights))

        # random_opinions = {issue_id: random.choice(issue.opinions) for issue_id, issue in issues.items()}
        return Agent(agent_id=agent_id, opinions=random_opinions)

    def perturbed_utility(self, statement, error, worst_case=False):
        # Convert statement opinions to a tuple to use as a key for the cache
        statement_key = tuple(sorted(statement.opinions.items()))

        # If this statement has already been processed, return the cached value
        if statement_key in self._perturbed_cache:
            return self._perturbed_cache[statement_key]

        # Compute the original utility
        original_utility = self.utility_from_statement(statement)

        # Apply random perturbation to the total utility
        if worst_case:
            value = random.choice([-1, 1])
            perturbation = value * error
        else:
            perturbation = random.randint(-error, error)
        perturbed_utility = original_utility + perturbation

        # Store the result in the cache
        self._perturbed_cache[statement_key] = perturbed_utility

        return perturbed_utility


class Statement:
    def __init__(self, opinions, issues):
        """
        :param opinions: A dictionary mapping issue_id to opinion for the statement
        :param issues: A dictionary mapping issue_id to Issue objects
        """
        self.opinions = opinions
        self.issues = issues  # Reference to the list of issues

    def cost(self):
        return len(self.opinions)

    def diversity(self):
        print(self.opinions)
        summed_div = 0
        for issue_id, opinion in self.opinions.items():
            options = self.issues[issue_id].opinions
            summed_div += abs((len(options) + 1) / 2 - opinion)

        print(summed_div / len(self.opinions))
        return summed_div / len(self.opinions)


#################### Gen Querry ####################


def find_statements_with_support(
    agents,
    issues,
    max_statement_length,
    utility_th,
    supp_th,
    worst_case=False,
    mW=False,
):
    """
    Finds all statements that have at least supp_th agents that like them with utility at least utility_th.

    :param agents: List of agents.
    :param issues: Dictionary of Issue objects.
    :param max_statement_length: The maximum length of the statement (number of issues it can address).
    :param utility_th: Minimum utility value an agent should have to "like" a statement.
    :param supp_th: Minimum number of agents required to like the statement.
    :return: A list of statements that meet the criteria.
    """
    qualifying_statements = []

    # Generate all combinations of issues (subsets) up to max_statement_length
    if mW:
        rang = [max_statement_length]
    else:
        rang = range(1, max_statement_length + 1)
    for length in rang:
        issue_combinations = itertools.combinations(issues.items(), length)

        for issue_subset in issue_combinations:
            # Extract the issue IDs and their associated opinions from the issue subset
            issue_ids = [issue_id for issue_id, _ in issue_subset]
            available_opinions = [issue.opinions for _, issue in issue_subset]

            # Generate all possible opinion combinations for this subset of issues
            opinions_combinations = itertools.product(*available_opinions)

            for opinions in opinions_combinations:
                # Create a mapping of issue IDs to the corresponding opinions
                statement_opinions = dict(zip(issue_ids, opinions))

                # Create a statement object with the selected issues and opinions
                statement = Statement(opinions=statement_opinions, issues=issues)

                # Count how many agents like this statement (utility >= x)
                agents_liking_statement = 0
                for agent in agents:
                    utility = agent.utility_from_statement(statement)
                    if utility >= utility_th:
                        agents_liking_statement += 1

                # If enough agents like the statement, add it to the qualifying statements
                if agents_liking_statement >= supp_th:
                    if worst_case:
                        qualifying_statements.append(
                            [statement, agents_liking_statement]
                        )
                    else:
                        qualifying_statements.append(statement)
    if worst_case:
        # Find the minimum value of agents_liking_statement
        if qualifying_statements == []:
            return qualifying_statements
        min_value = min(statement[1] for statement in qualifying_statements)

        # Filter the list to only keep elements with the minimum value
        qualifying_statements = [
            statement[0]
            for statement in qualifying_statements
            if statement[1] == min_value
        ]

        return qualifying_statements
    else:
        return qualifying_statements


def find_best_statement(
    agents, issues, max_statement_length, utility_threshold, det=False, mW=False
):
    """
    Finds the statement (of at most max_statement_length) for which the highest number of agents evaluate it above
    the given utility threshold using the real utility function (not perturbed).

    :param agents: List of agents
    :param issues: Dictionary of Issue objects
    :param max_statement_length: The maximum length of the statement (number of issues it can address)
    :param utility_threshold: The utility threshold to evaluate
    :return: The best statement and the number of agents evaluating it above the utility threshold
    """
    best_statement = None
    max_agents_above_threshold = 0

    # Generate all combinations of issues (subsets) up to max_statement_length
    if mW:
        rang = [max_statement_length]
    else:
        rang = range(1, max_statement_length + 1)
    for length in rang:
        issue_combinations = itertools.combinations(issues.items(), length)

        for issue_subset in issue_combinations:
            # Extract the issue IDs and their associated opinions from the issue subset
            issue_ids = [issue_id for issue_id, _ in issue_subset]
            available_opinions = [issue.opinions for _, issue in issue_subset]

            # Generate all possible opinion combinations for this subset of issues
            opinions_combinations = itertools.product(*available_opinions)

            for opinions in opinions_combinations:
                # Create a mapping of issue IDs to the corresponding opinions
                statement_opinions = dict(zip(issue_ids, opinions))

                # Create a statement object with the selected issues and opinions
                statement = Statement(opinions=statement_opinions, issues=issues)

                # Count how many agents evaluate this statement above the threshold
                agents_above_threshold = 0
                for agent in agents:
                    utility = agent.utility_from_statement(statement)
                    if utility >= utility_threshold:
                        agents_above_threshold += 1

                # Check if this is the best statement so far
                if agents_above_threshold > max_agents_above_threshold:
                    max_agents_above_threshold = agents_above_threshold
                    best_statement = statement
    if det:
        return best_statement, max_agents_above_threshold
    else:
        return best_statement


def find_best_statement_app(
    agents,
    issues,
    max_statement_length,
    utility_threshold,
    mu,
    ga,
    de,
    worst_case=False,
    mW=False,
):
    _, max_agents_above_threshold = find_best_statement(
        agents,
        issues,
        math.ceil(mu * max_statement_length),
        utility_threshold,
        det=True,
        mW=mW,
    )
    supp_th = math.ceil(ga * max_agents_above_threshold)
    qualifying_statements = find_statements_with_support(
        agents,
        issues,
        max_statement_length,
        utility_threshold - de,
        supp_th,
        worst_case=worst_case,
        mW=mW,
    )
    if len(qualifying_statements) == 0:
        return None
    else:
        return random.choice(qualifying_statements)


#################### Axiom Checks ####################


def BJR_fullfilled(agents, issues, agent_satisfaction_map, B):
    """
    Finds the statement (of at most max_statement_length) for which the highest number of agents evaluate it above
    the given utility threshold and check if a group of agents likes the statement more than their current one.

    :param agents: List of agents
    :param issues: Dictionary of Issue objects
    :param max_statement_length: The maximum length of the statement (number of issues it can address)
    :param utility_threshold: The utility threshold to evaluate
    :param agent_satisfaction_map: A dictionary mapping agents to their current satisfaction for their assigned statement
    :param B: The budget constraint used to calculate the group size condition
    :return: The best statement and the number of agents evaluating it above the utility threshold
    """
    n = len(agents)  # Total number of agents

    # Generate all combinations of issues (subsets) up to max_statement_length
    for length in range(1, len(issues) + 1):
        issue_combinations = itertools.combinations(issues.items(), length)

        for issue_subset in issue_combinations:
            # Extract the issue IDs and their associated opinions from the issue subset
            issue_ids = [issue_id for issue_id, _ in issue_subset]
            available_opinions = [issue.opinions for _, issue in issue_subset]

            # Generate all possible opinion combinations for this subset of issues
            opinions_combinations = itertools.product(*available_opinions)

            for opinions in opinions_combinations:
                # Create a mapping of issue IDs to the corresponding opinions
                statement_opinions = dict(zip(issue_ids, opinions))

                # Create a statement object with the selected issues and opinions
                statement = Statement(opinions=statement_opinions, issues=issues)

                # Required number of agents based on the group size condition
                required_group_size = (statement.cost() * n) / B

                # Check if there's a value `l` that satisfies the conditions for a sufficiently large group
                for l in range(
                    min_util_per_issue * len(issues),
                    max_util_per_issue * len(issues) + 1,
                ):  # Loop over possible integer values for `l`
                    agents_in_group = []

                    for agent in agents:
                        current_satisfaction = agent_satisfaction_map[agent][0]
                        new_satisfaction = agent.utility_from_statement(statement)

                        # Check if the agent satisfies the condition for the current value of `l`
                        if new_satisfaction >= l and current_satisfaction < l:
                            agents_in_group.append(agent)

                    # If the number of agents in the group is large enough, BJR is violated
                    if len(agents_in_group) >= required_group_size:
                        print("BJR Violated: Group found with l =", l)
                        print(
                            [agent.agent_id for agent in agents_in_group],
                            "New statement:",
                            statement.opinions,
                        )
                        return False
    return True


def approx_BJR_fullfilled(
    agents, issues, agent_satisfaction_map, B, min_util_per_issue, max_util_per_issue
):
    """
    Finds the statement (of at most max_statement_length) for which the highest number of agents evaluate it above
    the given utility threshold and check if a group of agents likes the statement more than their current one.

    :param agents: List of agents
    :param issues: Dictionary of Issue objects
    :param max_statement_length: The maximum length of the statement (number of issues it can address)
    :param utility_threshold: The utility threshold to evaluate
    :param agent_satisfaction_map: A dictionary mapping agents to their current satisfaction for their assigned statement
    :param B: The budget constraint used to calculate the group size condition
    :return: The best statement and the number of agents evaluating it above the utility threshold
    """
    l_c = list(range(0, max_util_per_issue * len(issues) + 1))
    approx_quality = {c: 0 for c in l_c}

    n = len(agents)  # Total number of agents

    min_c_for_no_normal_deviation = 0

    # Generate all combinations of issues (subsets) up to max_statement_length
    for length in range(1, len(issues) + 1):
        issue_combinations = itertools.combinations(issues.items(), length)

        for issue_subset in issue_combinations:
            # Extract the issue IDs and their associated opinions from the issue subset
            issue_ids = [issue_id for issue_id, _ in issue_subset]
            available_opinions = [issue.opinions for _, issue in issue_subset]

            # Generate all possible opinion combinations for this subset of issues
            opinions_combinations = itertools.product(*available_opinions)

            for opinions in opinions_combinations:
                # Create a mapping of issue IDs to the corresponding opinions
                statement_opinions = dict(zip(issue_ids, opinions))

                # Create a statement object with the selected issues and opinions
                statement = Statement(opinions=statement_opinions, issues=issues)

                # Required number of agents based on the group size condition
                required_group_size = math.ceil((statement.cost() * n) / B)

                for c in l_c:
                    # Check if there's a value `l` that satisfies the conditions for a sufficiently large group
                    for l in range(
                        min_util_per_issue * len(issues),
                        max_util_per_issue * len(issues) + 1,
                    ):  # Loop over possible integer values for `l`
                        agents_in_group = []

                        for agent in agents:
                            current_satisfaction = agent_satisfaction_map[agent][0]
                            new_satisfaction = agent.utility_from_statement(statement)

                            # Check if the agent satisfies the condition for the current value of `l`
                            if new_satisfaction >= l and current_satisfaction < l - c:
                                agents_in_group.append(agent)

                        # Check whether we have a bigger mismatch here
                        if (
                            len(agents_in_group) / math.ceil((statement.cost() * n) / B)
                            >= approx_quality[c]
                        ):
                            approx_quality[c] = len(agents_in_group) / math.ceil(
                                (statement.cost() * n) / B
                            )

                        if (
                            len(agents_in_group) / math.ceil((statement.cost() * n) / B)
                            >= 1
                        ):
                            min_c_for_no_normal_deviation = max(
                                min_c_for_no_normal_deviation, c
                            )

    return approx_quality, min_c_for_no_normal_deviation


#################### Slate Generation ####################


def finding_statement(
    version,
    S,
    issues,
    max_statement_length,
    l,
    mu,
    ga,
    de,
    worst_case,
    r,
    error_disc,
    app,
    mW=False,
):
    if version == 0:
        if app:
            a = find_best_statement_app(
                S,
                issues,
                max_statement_length,
                l,
                mu,
                ga,
                de,
                worst_case=worst_case,
                mW=mW,
            )
        else:
            a = find_best_statement(S, issues, max_statement_length, l, mW=mW)
    elif version == 1 or version == 2:
        statements = []
        for ll in range(l, r + 1):
            if app:
                st = find_best_statement_app(
                    S,
                    issues,
                    max_statement_length,
                    ll,
                    mu,
                    ga,
                    de,
                    worst_case=worst_case,
                    mW=mW,
                )
            else:
                st = find_best_statement(S, issues, max_statement_length, ll, mW=mW)
            if st is not None:
                statements.append(st)
        # TODO Is this really a list of lists or a single list?
        # print(statements)
        if len(statements) == 0:
            a = None
        else:
            a = max(
                statements,
                key=lambda statement: sum(
                    agent.perturbed_utility(
                        statement, error_disc, worst_case=worst_case
                    )
                    >= l
                    for agent in S
                ),
            )
    return a


def base_slate_generation(
    agents,
    issues,
    r,
    B,
    min_util_per_issue,
    error_disc=0,
    mu=1,
    ga=1,
    de=0,
    version=0,
    app=True,
    worst_case=False,
    mW=False,
):
    n = len(agents)
    S = agents  # Initialize set S to all agents
    W = []  # Initialize set W to be empty
    l = r  # Set l to the initial threshold value
    agent_satisfaction_map = {}  # Dictionary to map agents to their satisfaction (utility) for the matched statement

    while l >= 0 and len(S) > 0:
        if mW:
            if version == 0 or version == 1:
                j = len(issues) * n / B
            else:
                j = len(issues)

        else:
            if version == 0 or version == 1:
                j = len(issues) * n / B
            else:
                j = len(issues)
            # j = 1
        while j >= 1:
            # print(B,j, l)
            # Dynamically compute the max_statement_length as j * (B / len(S))
            if version == 0 or version == 1:
                max_statement_length = int(j * (B / n))
            else:
                max_statement_length = j

            if max_statement_length - sum(statement.cost() for statement in W) > B:
                j -= 1
                continue

            a = finding_statement(
                version,
                S,
                issues,
                max_statement_length,
                l,
                mu,
                ga,
                de,
                worst_case,
                r,
                error_disc,
                app,
                mW=mW,
            )

            if a is None:
                if mW:
                    break
                else:
                    # j += 1
                    j -= 1

            else:
                # print(a.opinions)
                # Get the subset of agents in S that have utility ≥ l from a
                S_prime = [
                    agent
                    for agent in S
                    if agent.perturbed_utility(a, error_disc, worst_case=worst_case)
                    >= l
                ]

                # If the size of S_prime is large enough, update W and remove agents
                if len(S_prime) >= math.ceil(a.cost() * n / B):
                    S_prime_sorted = sorted(
                        S_prime,
                        key=lambda agent: agent.perturbed_utility(
                            a, error_disc, worst_case=worst_case
                        ),
                        reverse=True,
                    )
                    agents_to_remove = S_prime_sorted[: math.ceil(a.cost() * n / B)]

                    # Store the satisfaction (utility) for each agent
                    for agent in agents_to_remove:
                        agent_satisfaction_map[agent] = [
                            agent.utility_from_statement(a),
                            a,
                        ]

                    # Remove these agents from S
                    S = [agent for agent in S if agent not in agents_to_remove]

                    # Add a to W
                    W.append(a)
                    # print(W)
                else:
                    if mW:
                        break
                    else:
                        # j += 1  # Increment j if S_prime is not large enough
                        j -= 1

        l -= 1  # Decrease the threshold

    for agent in S:
        agent_satisfaction_map[agent] = [0, Statement({}, issues)]

    return W, agent_satisfaction_map  # Return the selected statements


#################### Postprocessing ####################
def heatmap(results, args):
    (
        i,
        num_issues,
        opinions,
        num_agents,
        B,
        max_util_per_issue,
        min_util_per_issue,
        error_disc,
        mu,
        ga,
        de,
        version,
        app,
        worst_case,
        mW,
        name,
        gauss,
    ) = args

    if worst_case:
        ex = "worst"
    else:
        ex = "random"

    if mW:
        ex = ex + "_MW"

    os.makedirs("res/" + name + "/", exist_ok=True)

    # Extract all keys and values
    l_approx = [item[0] for item in results]  # Extracts all the first elements
    l_satisfaction = [item[1] for item in results]  # Extracts all the second elements
    l_lengths = [item[2] for item in results]  # Extracts all the second elements
    l_divs = [item[3] for item in results]  # Extracts all the second elements
    l_min_c = [item[4] for item in results]  # Extracts all the second elements

    ######minCforNoDev
    filename = (
        "res/"
        + name
        + f"/minc_nodev_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.csv"
    )
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([[value] for value in l_min_c])

    # Create a histogram to visualize the distribution of satisfaction scores
    plt.figure(figsize=(10, 6))
    plt.hist(
        l_min_c,
        bins=range(min(l_min_c), max(l_min_c) + 2),
        edgecolor="black",
        align="left",
    )
    plt.title("Distribution of cs; avg cs " + str(sum(l_min_c) / len(l_min_c)))
    plt.xlabel("min c")
    plt.ylabel("Frequency")
    plt.xticks(range(min(l_min_c), max(l_min_c) + 1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.show()

    filename = (
        "res/"
        + name
        + f"/minc_nodev_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.png"
    )
    plt.savefig(filename)

    plt.close()

    ######SATISFACTION
    filename = (
        "res/"
        + name
        + f"/satislist_disc_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.csv"
    )
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(l_satisfaction)

    flat_satisfaction = [score for sublist in l_satisfaction for score in sublist]

    # Create a histogram to visualize the distribution of satisfaction scores
    plt.figure(figsize=(10, 6))
    plt.hist(
        flat_satisfaction,
        bins=range(min(flat_satisfaction), max(flat_satisfaction) + 2),
        edgecolor="black",
        align="left",
    )
    plt.title(
        "Distribution of Satisfaction Scores; avg sat "
        + str(sum(flat_satisfaction) / len(flat_satisfaction))
        + " 10th avg "
        + str(tenth_percentile_mean(flat_satisfaction))
    )
    plt.xlabel("Satisfaction Score")
    plt.ylabel("Frequency")
    plt.xticks(range(min(flat_satisfaction), max(flat_satisfaction) + 1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.show()

    filename = (
        "res/"
        + name
        + f"/satisfaction_disc_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.png"
    )
    plt.savefig(filename)

    plt.close()

    ######LENGTHS
    filename = (
        "res/"
        + name
        + f"/lens_disc_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.csv"
    )
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(l_lengths)

    flat_leng = [leng for sublist in l_lengths for leng in sublist]

    # Create a histogram to visualize the distribution of satisfaction scores
    plt.figure(figsize=(10, 6))
    plt.hist(
        flat_leng,
        bins=range(min(flat_leng), max(flat_leng) + 2),
        edgecolor="black",
        align="left",
    )
    plt.title(
        "Distribution of Statement Lenghts; avg len "
        + str(sum(flat_leng) / len(flat_leng))
    )
    plt.xlabel("Lengths")
    plt.ylabel("Frequency")
    plt.xticks(range(min(flat_leng), max(flat_leng) + 1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.show()

    filename = (
        "res/"
        + name
        + f"/len_disc_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.png"
    )
    plt.savefig(filename)

    plt.close()

    ######DIVS
    filename = (
        "res/"
        + name
        + f"/divs_disc_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.csv"
    )
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(l_divs)

    flat_div = [div for sublist in l_divs for div in sublist]
    print(flat_div)
    # Create a histogram to visualize the distribution of satisfaction scores
    plt.figure(figsize=(10, 6))
    plt.hist(flat_div, edgecolor="black", align="left")
    plt.title(
        "Distribution of Statement Div; avg div " + str(sum(flat_div) / len(flat_div))
    )
    plt.xlabel("Diversity")
    plt.ylabel("Frequency")
    # plt.xticks(range(min(flat_div), max(flat_div) + 1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.show()

    filename = (
        "res/"
        + name
        + f"/div_disc_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.png"
    )
    plt.savefig(filename)

    plt.close()

    #####APPROX

    # l_approx=results[0]
    keys = sorted(l_approx[0].keys())  # Assuming all dicts have the same keys
    all_values = set()
    for d in l_approx:
        all_values.update(d.values())
    all_values = sorted(all_values)

    # Create a 2D array (heatmap) with dimensions [num_values, num_keys]
    heatmap = np.zeros((len(all_values), len(keys)))

    # Map values to indices for the heatmap
    value_to_index = {v: i for i, v in enumerate(all_values)}

    # Populate the heatmap with counts
    for d in l_approx:
        for key, value in d.items():
            value_index = value_to_index[value]
            key_index = keys.index(key)
            heatmap[value_index, key_index] += 1

    # Plot the heatmap
    # Create a DataFrame for CSV export
    df = pd.DataFrame(heatmap, index=all_values, columns=keys)

    # Save the DataFrame as a CSV file
    csv_filename = (
        "res/"
        + name
        + f"/heatmap_error_disc_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.csv"
    )
    df.to_csv(csv_filename)
    print(f"Heatmap data saved to {csv_filename}")

    plt.imshow(heatmap, cmap="Blues", aspect="auto", origin="lower")

    # Add colorbar and labels
    plt.colorbar(label="Count of Lists")
    plt.xticks(ticks=np.arange(len(keys)), labels=keys)
    plt.yticks(ticks=np.arange(len(all_values)), labels=all_values)
    plt.xlabel("c")
    plt.ylabel("d")

    # Annotate each cell with the count (number of occurrences)
    for i in range(len(all_values)):  # Iterate over the y-axis (values)
        for j in range(len(keys)):  # Iterate over the x-axis (keys)
            count = int(heatmap[i, j])  # Get the count for this cell
            plt.text(j, i, str(count), ha="center", va="center", color="black")

    x_value = 2 * error_disc + de
    y_value = 1 / (mu * ga)
    plt.title("no c>" + str(x_value) + " and d>" + str(y_value))
    filename = (
        "res/"
        + name
        + f"/heatmap_error_disc_{error_disc}_mu_{mu}_ga_{ga}_de_{de}_n_{num_agents}_B_{B}_iss_{num_issues}_ops_{opinions}_version_{version}_{gauss}_{ex}.png"
    )
    plt.savefig(filename)


def process_iteration(args):
    (
        i,
        num_issues,
        opinions,
        num_agents,
        B,
        max_util_per_issue,
        min_util_per_issue,
        error_disc,
        mu,
        ga,
        de,
        version,
        app,
        worst_case,
        mW,
        name,
        gauss,
    ) = args
    random.seed(i)

    # Step 1: Create the issues
    issues = {i: Issue(i, opinions) for i in range(1, num_issues + 1)}

    # Step 2: Generate random agents
    if gauss:
        agents = [
            Agent.random_Gauss(agent_id=i, issues=issues) for i in range(num_agents)
        ]
    else:
        agents = [
            Agent.random_agent(agent_id=i, issues=issues) for i in range(num_agents)
        ]

    # Step 3: Run the slate generation algorithm
    r = num_issues * max_util_per_issue
    W, agent_satisfaction_map = base_slate_generation(
        agents,
        issues,
        r,
        B,
        min_util_per_issue,
        error_disc=error_disc,
        mu=mu,
        ga=ga,
        de=de,
        version=version,
        app=app,
        worst_case=worst_case,
        mW=mW,
    )
    # Step 4: Compute the approx quality using approx_BJR_fullfilled
    approx_quality, min_c_for_no_normal_deviation = approx_BJR_fullfilled(
        agents,
        issues,
        agent_satisfaction_map,
        B,
        min_util_per_issue,
        max_util_per_issue,
    )
    print("Generated Slate (Selected Statements):")
    for statement in W:
        print(statement.opinions)

    print("\nAgent Satisfaction Map (Utility derived from each statement):")
    for agent, ma in agent_satisfaction_map.items():
        print(
            f"Agent {agent.agent_id} with {agent.opinions}: Satisfaction = {ma[0]} for statment = {ma[1].opinions} "
        )

    print("End Process", i)

    return [
        approx_quality,
        [value[0] for value in agent_satisfaction_map.values()],
        [len(statement.opinions) for statement in W],
        [statement.diversity() for statement in W],
        min_c_for_no_normal_deviation,
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the program with specified parameters."
    )
    parser.add_argument("--error_disc", type=int, default=0, help="Error discretion")
    parser.add_argument("--mu", type=float, default=1, help="Mu parameter")
    parser.add_argument("--ga", type=float, default=1, help="Ga parameter")
    parser.add_argument("--de", type=float, default=0, help="De parameter")
    parser.add_argument("--version", type=int, default=0, help="Version")
    parser.add_argument("--num_agents", type=int, default=40, help="Number of agents")
    parser.add_argument("--B", type=int, default=20, help="Budget")
    parser.add_argument("--num_issues", type=int, default=2, help="Number of issues")
    parser.add_argument("--opinions", type=int, default=6, help="Number of opinions")
    parser.add_argument(
        "--worst_case", type=int, default=0, help="Worst case error"
    )  # 0 False
    parser.add_argument(
        "--mW", type=int, default=0, help="Multiwinner Voting setup"
    )  # 0 False
    parser.add_argument("--name", type=str, default="name", help="Experiment Name")
    parser.add_argument("--gauss", type=int, default=0, help="Gaussian")  # 0 False
    args = parser.parse_args()

    max_util_per_issue = int(args.opinions / 2)
    min_util_per_issue = max_util_per_issue - (args.opinions - 1)

    if args.mW and not args.B % args.num_issues == 0:
        print("Multiwinner not only full statements possible")
        exit()

    app = True

    if args.worst_case:
        print("Worst Case")
    else:
        print("Average Case")

    # Prepare arguments for multiprocessing
    args_list = [
        (
            i,
            args.num_issues,
            args.opinions,
            args.num_agents,
            args.B,
            max_util_per_issue,
            min_util_per_issue,
            args.error_disc,
            args.mu,
            args.ga,
            args.de,
            args.version,
            app,
            args.worst_case,
            args.mW,
            args.name,
            args.gauss,
        )
        for i in range(100)
    ]

    # print(args_list)

    start_time = time.time()
    l_approx = []

    # Create a Pool with the number of workers equal to the number of CPU cores
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use pool.map to parallelize the process_iteration function across 100 iterations
        print(multiprocessing.cpu_count())
        l_approx = pool.map(process_iteration, args_list)
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    heatmap(l_approx, args_list[0])
