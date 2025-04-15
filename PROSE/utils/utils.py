import pandas as pd
import os.path
from PROSE.queries_old.opinion_summary_query import create_summary_all_users
import json
from random import sample, randint, shuffle
from PROSE.utils.constants import BASE_DIR
from PROSE.datasets.polis.load_polis_data import PolisDataset


def load_polis(name="bowlinggreen"):
    dataset = PolisDataset(dataset_name=name)
    data = []
    for author_id in dataset.get_good_author_ids():
        votes_df = dataset.get_comments_by_author_id(author_id)
        data.append([author_id, votes_df])

    df = pd.DataFrame(data, columns=["user_id", "essay"])
    return df


def load_survey_data():
    """load the survey data and remove some unnecessary rows and columns."""
    filename = "datasets/chatbot_personalization_data.csv"
    df = pd.read_csv(filename)
    df = df.loc[df.question_type != "reading"]
    df = df.loc[df.sample_type == "generation"]
    df = df.drop(columns=["answer_date", "sample_type"])
    df = df.dropna(axis="columns", how="all")
    return df


def load_summary():
    filename = "./datasets/chatbot_personalization_summary.csv"
    if not os.path.isfile(filename):
        df = load_survey_data()
        create_summary_all_users(df)
    print(filename)
    dfs = pd.read_csv(filename, sep=";")
    return dfs


def subsample_participants(df, n_participants):
    """return the the subdataframe corresponding to n_participants
    randomly selected participants."""
    sampled_user_ids = (
        df["user_id"].drop_duplicates().sample(n=n_participants, random_state=42)
    )
    return df[df["user_id"].isin(sampled_user_ids)]


def subsample_participant_ids(df, n_participants):
    """return the the subdataframe corresponding to n_participants
    randomly selected participants."""
    sampled_user_ids = (
        df["user_id"].drop_duplicates().sample(n=n_participants, random_state=42)
    )
    return sampled_user_ids.to_list()


def load_synthetic_data(topic):
    filename = BASE_DIR / "datasets/" / (topic + "_data.csv")
    print(filename)
    dfs = pd.read_csv(filename, sep=";")
    filename = BASE_DIR / "datasets/" / (topic + "/overview.json")
    with open(filename) as f:
        opinions = json.load(f)
    return dfs, opinions


def write_to_file(name, data):
    with open(name + ".txt", "a") as f:
        print(data, file=f)


def extract_subinstance_single_issue(topic, number_opinions=5, mix_ops=3):
    """
    Generates a subinstance of a single-issue dataset.

    topic: name of dataset
    number_opinions: number of different opinions available on topic
    mix_ops: number of different opinions to be present in the generated subinstance
    """
    df, opinions = load_synthetic_data(topic)
    user_ids_by_opinion = []
    user_ids = df["user_id"].to_list()
    for i in range(1, number_opinions + 1):
        user_ids_by_opinion.append(
            [x for x in user_ids if x.startswith("opinion_" + str(i))]
        )

    user_groups = sample(user_ids_by_opinion, mix_ops)
    print(user_groups)
    user_ids = []
    for li in user_groups:
        n_remaining = randint(1, 4)
        print(n_remaining)
        user_ids.append(sample(li, n_remaining))
    user_ids = [id for group in user_ids for id in group]
    shuffle(user_ids)
    return df, opinions, user_ids


def extract_subinstance(topic, number_agents=5, polis=False):
    """
    Generates a subinstance of multi-issue dataset topic consisting of number_agents many agents.
    """
    if polis:
        df = load_polis(name=topic)
    else:
        df, opinions = load_synthetic_data(topic)
    user_ids = list(set(df["user_id"].to_list()))
    print(user_ids)
    if polis:
        return df, None, sample(user_ids, number_agents)
    else:
        return df, opinions, sample(user_ids, number_agents)
