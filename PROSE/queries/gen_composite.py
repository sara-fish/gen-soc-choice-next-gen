import math
from typing import List, Literal, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans

from PROSE.queries.fast_queries import FastGenerator
from random import Random

from PROSE.queries.query_interface import Agent
from PROSE.utils.llm_tools import LLMLog, DEFAULT_MODEL, GPT
import pandas as pd
from scipy.spatial.distance import cdist
import concurrent.futures
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict
from scipy.optimize import linprog
import random

llm = GPT(
    model=DEFAULT_MODEL,
)


class SubsamplingGenerator(FastGenerator):
    def __init__(
        self,
        *,
        sample_size: int,
        openai_seed: int = 0,
        seed: int = 0,
        prompt_type: str = "v1",
        temperature: float = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ):
        super().__init__(
            prompt_type=prompt_type,
            openai_seed=openai_seed,
            temperature=temperature,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        self.sample_size = sample_size
        self.seed = seed  # not the openai seed, other sources of randomness
        self.random = Random(seed)

    def get_name(self) -> str:
        return (
            self.__class__.__name__
            + f"__approval_{self.target_approval}__budget_{self.target_budget}__sample_size_{self.sample_size}"
        )

    def generate(
        self,
        agents: List[Agent],
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ) -> Tuple[List[str], List[LLMLog]]:
        # If target_approval or target_budget is None, use the default values
        if target_approval is None:
            target_approval = self.target_approval
        if target_budget is None:
            target_budget = self.target_budget
        # Subsample agents
        if len(agents) > self.sample_size:
            sampled_agents = self.random.sample(agents, self.sample_size)
        else:
            sampled_agents = agents
        statement_list, log_list = super().generate(
            agents=sampled_agents,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        log_list[-1]["sampled_agent_ids"] = [agent.get_id() for agent in sampled_agents]
        return statement_list, log_list


class PreviousBestGenerator(FastGenerator):
    def __init__(
        self,
        *,
        cluster_size: int,
        openai_seed: int = 0,
        seed: int = 0,
        prompt_type: str = "v1",
        temperature: float = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
        previous_statements=None,
        adaptable: bool = False,
    ):
        super().__init__(
            prompt_type=prompt_type,
            openai_seed=openai_seed,
            temperature=temperature,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        self.seed = seed  # not the openai seed, other sources of randomness
        self.random_state = np.random.RandomState(seed)
        self.adaptable = adaptable
        self.cluster_size = cluster_size
        self.previous_statements = previous_statements

    def get_name(self) -> str:
        return (
            self.__class__.__name__
            + f"__approval_{self.target_approval}__budget_{self.target_budget}__cluster_size_{self.cluster_size}"
        )

    def set_previous_statements(self, statements):
        self.previous_statements = statements

    def is_adaptable(self) -> bool:
        return self.adaptable

    def generate(
        self,
        agents: List[Agent],
        center_agent: Optional[Agent] = None,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ) -> Tuple[str, List[LLMLog]]:
        """
        center_agent: if not None, use this agent as the center of the cluster
        """
        # print(self.previous_statements)

        if self.previous_statements is None or len(self.previous_statements) == 0:
            clustered_agents = agents
        else:
            covered_agents = []
            for approvals in self.previous_statements:
                covered_agents.append(
                    [
                        agent
                        for agent in agents
                        if approvals[agent.get_id()] >= target_approval
                    ]
                )
            clustered_agents = max(covered_agents, key=len)
        statement_list, log_list = super().generate(
            agents=clustered_agents,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        return statement_list, log_list


class ClosestClusterGenerator(FastGenerator):
    """
    Pick random agent. Find nearest neighbors in tag space. Generate statement.
    """

    def __init__(
        self,
        *,
        cluster_size: int,
        openai_seed: int = 0,
        seed: int = 0,
        prompt_type: str = "v1",
        temperature: float = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
        adaptable: bool = False,
        LLM_emb: bool = False,
    ):
        super().__init__(
            prompt_type=prompt_type,
            openai_seed=openai_seed,
            temperature=temperature,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        self.cluster_size = cluster_size
        self.seed = seed  # not the openai seed, other sources of randomness
        self.random_state = np.random.RandomState(seed)
        self.adaptable = adaptable
        self.LLM_emb = LLM_emb

    def is_adaptable(self) -> bool:
        return self.adaptable

    def get_name(self) -> str:
        return (
            self.__class__.__name__
            + f"__approval_{self.target_approval}__budget_{self.target_budget}__cluster_size_{self.cluster_size}"
        )

    def set_cluster_size(self, cluster_size: int):
        self.cluster_size = cluster_size

    def generate(
        self,
        agents: List[Agent],
        center_agent: Optional[Agent] = None,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ) -> Tuple[str, List[LLMLog]]:
        """
        center_agent: if not None, use this agent as the center of the cluster
        """

        random.seed(self.seed)

        for agent in agents:
            assert hasattr(agent, "tags"), "agent.tags must contain dict of tags"
        ## Build df of tag data
        if not self.LLM_emb:
            data = [list(agent.tags.values()) for agent in agents]
        else:
            agent_embeddings = []

            for agent in agents:
                agent_embeddings.append(agent.get_embedding())

            assert len(agent_embeddings) == len(agents)
            try:
                scaled_tag_vectors = StandardScaler().fit_transform(agent_embeddings)

                pca = PCA(n_components=5, random_state=0).fit(scaled_tag_vectors)
                data = pca.transform(scaled_tag_vectors)
            except:
                data = [list(agent.tags.values()) for agent in agents]

        # Preprocessing
        # Compute pairwise distances
        dist_matrix = cdist(data, data, metric="euclidean")

        x = self.cluster_size  # Number of closest agents to find
        clusters = []
        total_distances = []

        # Iterate over all possible starting points
        for start_idx in range(len(agents)):
            selected_indices = [start_idx]  # Start with one agent
            total_distance = 0  # Track total summed distance

            # Iteratively add agents one by one
            while len(selected_indices) < x:
                remaining_indices = [
                    i for i in range(len(agents)) if i not in selected_indices
                ]

                # Find the next agent that minimizes the total summed distance
                best_next = min(
                    remaining_indices,
                    key=lambda i: sum(dist_matrix[i, j] for j in selected_indices),
                )

                total_distance += sum(
                    dist_matrix[best_next, j] for j in selected_indices
                )
                selected_indices.append(best_next)

            # Store the cluster and its total summed distance
            clusters.append(selected_indices)
            total_distances.append(1 / total_distance)

        # Convert total distances into probabilities
        probabilities = np.array(total_distances) / sum(total_distances)

        # print(probabilities)

        # Select a cluster randomly with probability anti-proportional to its total distance
        try:
            chosen_cluster = random.choices(clusters, weights=probabilities, k=1)[0]
            clustered_agents = [agents[i] for i in chosen_cluster]
        except:
            clustered_agents = agents

        statement_list, log_list = super().generate(
            agents=clustered_agents,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        print(statement_list, [agent.get_label() for agent in clustered_agents])
        return statement_list, log_list


class TagNNGenerator(FastGenerator):
    def __init__(
        self,
        *,
        cluster_size: int,
        openai_seed: int = 0,
        seed: int = 0,
        prompt_type: str = "v1",
        temperature: float = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
        adaptable: bool = False,
        LLM_emb: bool = False,
    ):
        super().__init__(
            prompt_type=prompt_type,
            openai_seed=openai_seed,
            temperature=temperature,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        self.cluster_size = cluster_size
        self.seed = seed  # not the openai seed, other sources of randomness
        self.random_state = np.random.RandomState(seed)
        self.adaptable = adaptable
        self.LLM_emb = LLM_emb

    def is_adaptable(self) -> bool:
        return self.adaptable

    def get_name(self) -> str:
        return (
            self.__class__.__name__
            + f"__approval_{self.target_approval}__budget_{self.target_budget}__cluster_size_{self.cluster_size}"
        )

    def set_cluster_size(self, cluster_size: int):
        self.cluster_size = cluster_size

    def generate(
        self,
        agents: List[Agent],
        center_agent: Optional[Agent] = None,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ) -> Tuple[str, List[LLMLog]]:
        """
        center_agent: if not None, use this agent as the center of the cluster
        """
        for agent in agents:
            assert hasattr(agent, "tags"), "agent.tags must contain dict of tags"
        ## Build df of tag data

        if len(agents) == 1:
            agent_idx = 0
        elif center_agent is not None:
            assert center_agent in agents
            agent_idx = agents.index(center_agent)
        else:
            agent_idx = self.random_state.randint(0, len(agents) - 1)

        if not self.LLM_emb:
            # Extract only the tags and agent ID
            data = [{"agent_id": agent.get_id(), **agent.tags} for agent in agents]
            df = pd.DataFrame(data)

            agent_v = df.drop("agent_id", axis=1).iloc[agent_idx].values
            distances = cdist(
                agent_v.reshape(1, -1),
                df.drop("agent_id", axis=1).values,
                metric="euclidean",
            )
            df["distance"] = distances[0]
            agent_ids = (
                df.sort_values("distance").head(self.cluster_size)["agent_id"].tolist()
            )
        else:
            agent_embeddings = [agent.get_embedding() for agent in agents]

            assert len(agent_embeddings) == len(agents)

            try:
                # Standardize the embeddings
                scaled_tag_vectors = StandardScaler().fit_transform(agent_embeddings)

                # Apply PCA
                pca = PCA(n_components=5, random_state=0).fit(scaled_tag_vectors)
                reduced_embeddings = pca.transform(scaled_tag_vectors)
            except:
                reduced_embeddings = np.array(agent_embeddings)

            agent_v = reduced_embeddings[agent_idx].reshape(1, -1)
            distances = cdist(agent_v, reduced_embeddings, metric="euclidean")[0]

            # Get the closest agents based on distance
            sorted_indices = np.argsort(distances)[: self.cluster_size]
            agent_ids = [agents[i].get_id() for i in sorted_indices]

            # Combine the reduced embeddings with agent metadata

        clustered_agents = [agent for agent in agents if agent.get_id() in agent_ids]
        statement_list, log_list = super().generate(
            agents=clustered_agents,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        log_list[-1]["clustered_agent_ids"] = agent_ids
        log_list[-1]["center_agent_id"] = agents[agent_idx].get_id()
        return statement_list, log_list


def find_set_probabilities(sets, epsilon=5e-3):
    elements = sorted(set().union(*sets))  # Unique elements
    num_sets = len(sets)
    num_elements = len(elements)

    # Decision variables: probabilities p_1, ..., p_n and delta
    num_variables = num_sets + 1  # Last variable is delta
    A = []
    b = []

    # Constraints for |P(e) - c| ≤ delta
    element_probabilities = np.zeros((num_elements, num_sets))  # Tracks P(e) structure
    for i, e in enumerate(elements):
        row = np.zeros(num_variables)
        for j, s in enumerate(sets):
            if e in s:
                row[j] = 1
                element_probabilities[i, j] = 1  # Track for later analysis
        row[-1] = -1  # Enforce P(e) ≤ c + delta
        A.append(row)
        b.append(1)

        row_neg = row.copy()
        row_neg[-1] = 1  # Enforce P(e) ≥ c - delta
        A.append(-row_neg)
        b.append(-1)

    # Probability constraint: sum(p) = 1
    A_eq = np.zeros((1, num_variables))
    A_eq[0, :-1] = 1
    b_eq = np.array([1])

    # Bounds: p_i ≥ epsilon (lower bound), delta unrestricted
    bounds = [(epsilon, 1)] * num_sets + [(None, None)]

    # Objective: minimize delta
    c = np.zeros(num_variables)
    c[-1] = 1  # We minimize delta

    # Solve linear program
    result = linprog(
        c=c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if result.success:
        probabilities = result.x[:-1]  # Extract set probabilities

        return probabilities
    else:
        raise ValueError("Linear program failed to find a solution.")


class WeightedNNGenerator(FastGenerator):
    def __init__(
        self,
        *,
        cluster_size: int,
        openai_seed: int = 0,
        seed: int = 0,
        prompt_type: str = "v1",
        temperature: float = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
        adaptable: bool = False,
        LLM_emb: bool = False,
    ):
        super().__init__(
            prompt_type=prompt_type,
            openai_seed=openai_seed,
            temperature=temperature,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        self.cluster_size = cluster_size
        self.seed = seed  # not the openai seed, other sources of randomness
        self.random_state = np.random.RandomState(seed)
        self.adaptable = adaptable
        self.LLM_emb = LLM_emb

    def is_adaptable(self) -> bool:
        return self.adaptable

    def get_name(self) -> str:
        return (
            self.__class__.__name__
            + f"__approval_{self.target_approval}__budget_{self.target_budget}__cluster_size_{self.cluster_size}"
        )

    def set_cluster_size(self, cluster_size: int):
        self.cluster_size = cluster_size

    def generate(
        self,
        agents: List[Agent],
        center_agent: Optional[Agent] = None,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ) -> Tuple[str, List[LLMLog]]:
        for agent in agents:
            assert hasattr(agent, "tags"), "agent.tags must contain dict of tags"
        ## Build df of tag data
        np.random.seed(self.seed)

        if not self.LLM_emb:
            data = [{"agent_id": agent.get_id(), **agent.tags} for agent in agents]
            df = pd.DataFrame(data)
        else:
            agent_embeddings = [agent.get_embedding() for agent in agents]

            assert len(agent_embeddings) == len(agents)

            # Standardize the embeddings
            scaled_tag_vectors = StandardScaler().fit_transform(agent_embeddings)

            # Apply PCA
            try:
                pca = PCA(n_components=5, random_state=0).fit(scaled_tag_vectors)
                reduced_embeddings = pca.transform(scaled_tag_vectors)
            except:
                reduced_embeddings = np.array(agent_embeddings)

        clusters = []
        agent_appearance_count = defaultdict(int)
        for agent_idx in range(len(agents)):
            if not self.LLM_emb:
                agent_v = df.drop("agent_id", axis=1).iloc[agent_idx].values
                distances = cdist(
                    agent_v.reshape(1, -1),
                    df.drop("agent_id", axis=1).values,
                    metric="euclidean",
                )
                # print(distances)
                df["distance"] = distances[0]
                # print(df)
                agent_ids = (
                    df.sort_values("distance")
                    .head(self.cluster_size)["agent_id"]
                    .tolist()
                )
            else:
                agent_v = reduced_embeddings[agent_idx].reshape(1, -1)
                distances = cdist(agent_v, reduced_embeddings, metric="euclidean")[0]
                # Get the closest agents based on distance
                sorted_indices = np.argsort(distances)[: self.cluster_size]
                agent_ids = [agents[i].get_id() for i in sorted_indices]
            clustered_agents = [
                agent.get_id() for agent in agents if agent.get_id() in agent_ids
            ]
            clusters.append(clustered_agents)

            for agent in clustered_agents:
                agent_appearance_count[agent] += 1

        cluster_probabilities = find_set_probabilities(clusters)

        # Sample a cluster based on probabilities
        selected_cluster_index = np.random.choice(
            len(clusters), p=cluster_probabilities
        )

        # Retrieve the corresponding agent IDs
        selected_agent_ids = clusters[selected_cluster_index]

        # Convert agent IDs to actual Agent objects
        selected_cluster = [
            agent for agent in agents if agent.get_id() in selected_agent_ids
        ]

        statement_list, log_list = super().generate(
            agents=selected_cluster,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        log_list[-1]["clustered_agent_ids"] = agent_ids
        log_list[-1]["center_agent_id"] = agents[agent_idx].get_id()
        return statement_list, log_list


class ClusteringGenerator(FastGenerator):
    def __init__(
        self,
        *,
        clustering_method: Literal["kmeans", "affinity_propagation"],
        seed: int,
        num_clusters: Optional[int] = None,
        openai_seed: int = 0,
        prompt_type: str = "v1",
        num_threads: int = 10,
        temperature: float = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ):
        super().__init__(
            prompt_type=prompt_type,
            openai_seed=openai_seed,
            temperature=temperature,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        self.clustering_method = clustering_method
        self.num_clusters = num_clusters
        if self.clustering_method == "kmeans":
            assert self.num_clusters is not None
        self.seed = seed  # not the openai seed, other sources of randomness
        self.random_state = np.random.RandomState(seed)
        self.num_threads = num_threads

    def get_name(self) -> str:
        return (
            self.__class__.__name__
            + f"__approval_{self.target_approval}__budget_{self.target_budget}__{self.clustering_method}__num_clusters_{self.num_clusters}"
        )

    def generate(
        self,
        agents: List[Agent],
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ) -> tuple[str, List[LLMLog]]:
        assert (
            self.num_clusters is None or len(agents) >= self.num_clusters
        ), "Number of agents must be >= num_clusters"
        for agent in agents:
            assert hasattr(agent, "tags"), "agent.tag must contain dict of tags"

        target_budget = target_budget or self.target_budget
        assert target_budget is not None

        data = [list(agent.tags.values()) for agent in agents]

        # Do preprocessing
        scaled_tag_vectors = StandardScaler().fit_transform(data)
        pca = PCA(n_components=5, random_state=self.random_state).fit(
            scaled_tag_vectors
        )
        data_reduced = pca.transform(scaled_tag_vectors)

        if self.clustering_method == "affinity_propagation":
            clustering = AffinityPropagation(random_state=self.random_state).fit(
                data_reduced
            )
        elif self.clustering_method == "kmeans":
            clustering = KMeans(
                n_clusters=self.num_clusters, random_state=self.random_state
            ).fit(data_reduced)
        else:
            raise NotImplementedError(
                f"Clustering method {self.clustering_method} not implemented"
            )

        clusters = clustering.labels_

        statements = []
        logs = []

        if self.num_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_threads
            ) as executor:
                future_to_data = {
                    executor.submit(
                        super().generate,
                        agents=[
                            agent
                            for i, agent in enumerate(agents)
                            if clusters[i] == cluster_num
                        ],
                        target_approval=target_approval,
                        target_budget=math.ceil(
                            target_budget
                            * len([i for i in clusters if i == cluster_num])
                            / len(clusters)
                        ),
                    ): {
                        "cluster_num": cluster_num,
                        "clustered_agent_ids": [
                            agent.get_id()
                            for i, agent in enumerate(agents)
                            if clusters[i] == cluster_num
                        ],
                    }
                    for cluster_num in set(clusters)
                }

                for future in concurrent.futures.as_completed(future_to_data):
                    statement_list, log_list = future.result()
                    log_list[-1]["cluster_num"] = future_to_data[future]["cluster_num"]
                    log_list[-1]["clustered_agent_ids"] = future_to_data[future][
                        "clustered_agent_ids"
                    ]
                    statements.extend(statement_list)
                    logs.extend(log_list)
        else:
            for cluster_num in set(clusters):
                clustered_agents = [
                    agent
                    for i, agent in enumerate(agents)
                    if clusters[i] == cluster_num
                ]
                statement_list, log_list = super().generate(clustered_agents)
                log_list[-1]["cluster_num"] = cluster_num
                log_list[-1]["clustered_agent_ids"] = [
                    agent.get_id() for agent in clustered_agents
                ]
                statements.extend(statement_list)
                logs.extend(log_list)

        return statements, logs
