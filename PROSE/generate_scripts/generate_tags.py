import argparse
import inflect
import pandas as pd

from PROSE.queries.prompts.summarize_prompts import get_tag_fields
from PROSE.queries.fast_queries import FastAgent
from PROSE.queries.tagging import compute_tags_of_agent
from PROSE.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
import os
from tqdm import tqdm
import json

import random
import concurrent.futures
import re

inflect_engine = inflect.engine()


def str_to_singleton_list(s):
    if isinstance(s, str):
        return [s]
    else:
        return s


def postprocess_tag(tag: str) -> str:
    # gpt-4o doesn't like ending tags with periods, so removing them
    tag = tag.strip(".")

    # Replace digits with words, because we do logprob parsing by looking at the digits.
    # Example: 'Supports abortion if the fetus is under 2-weeks' -> 'Supports abortion if the fetus is under two-weeks'
    def _replace(match):
        return inflect_engine.number_to_words(match.group())

    return re.sub(r"\d+", _replace, tag)


def run(args):
    random.seed(args.seed)  # set global randomness
    num_agents = args.num_agents
    num_tags = args.num_tags
    prompt_type = args.prompt_type
    num_threads = args.num_threads
    dataset = args.dataset
    summaries = args.summaries
    chunk_size = args.chunk_size
    topic = (
        dataset.removeprefix("PROSE/datasets/").removesuffix(".csv").replace("/", "__")
    )
    summary_prompt_type = summaries.split("__")[2]
    assert (
        summary_prompt_type[0] == "v" and summary_prompt_type[1].isdigit()
    ), f"Couldn't parse summary prompt type, best attempt was {summary_prompt_type}"

    dirname = (
        get_base_dir_path()
        / "experiment_logs"
        / f"{get_time_string()}_generate_tags_{topic}_{num_tags}"
    )
    os.makedirs(dirname)
    log_path = dirname / "logs.csv"
    logs = []

    df = pd.read_csv(get_base_dir_path() / dataset)

    summaries_df = pd.read_csv(get_base_dir_path() / summaries)

    # Get agents
    agents = []
    for id in df["user_id"].unique():
        user_df = df[df["user_id"] == id]
        assert len(user_df) == 1
        agent = FastAgent(
            id=str(id),
            data=user_df["data"].values[0],
            prompt_type="agreement_v1",
            specificity_prompt_type="specificity_v1",
        )
        agents.append(agent)

    # Endow agents with summaries
    for agent in agents:
        saved_summary = summaries_df[
            summaries_df["agent_id"].astype(str) == agent.get_id()
        ]["response"].values[0]
        agent.summary = saved_summary

    # Subsample agents
    if num_agents:
        agents = random.sample(agents, num_agents)

    # Get tags
    tags = []
    for agent in agents:
        summary = json.loads(agent.summary)
        for tag in get_tag_fields(prompt_type=summary_prompt_type):
            tags.extend(str_to_singleton_list(summary[tag]))

    # Subsample tags
    if num_tags:
        print(f"Subsampling {num_tags} tags from {len(tags)} tags")
        tags = random.sample(list(set(tags)), num_tags)

    # Postprocesses tags
    tags = [postprocess_tag(tag) for tag in tags]

    # For each agent, get values for tags
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_data = {
            executor.submit(
                compute_tags_of_agent,
                agent=agent,
                fields=tags,
                prompt_type=prompt_type,
                chunk_size=chunk_size,
            ): {"agent_id": agent.get_id()}
            for agent in agents
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_data), total=len(future_to_data)
        ):
            response, log_list = future.result()
            logs.append(
                {
                    "response": response,
                    "agent_id": str(future_to_data[future]["agent_id"]),
                }
            )
            logs.extend(log_list)
            pd.DataFrame(logs).to_csv(log_path, index=False)

    print(f"{len(agents)} tags saved to {log_path}")


def parse_args():
    # Use argparse to parse args for dataset_path, holdout (store if true)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/bowlinggreen_41/bowlinggreen_curated_postprocessed.csv",
        help="Path to dataset to use. Must have cols user_id, question_text, statement, json_choices, choice, choice_numeric, text.",
    )

    parser.add_argument(
        "--summaries",
        type=str,
        default="datasets/bowlinggreen_41/20241112-140604__summaries__v1__bowlinggreen_curated_postprocessed/logs.csv",
    )

    parser.add_argument(
        "--num_tags",
        type=int,
        default=None,
        help="If None, do all possible tags. Else, subsample",
    )

    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="If None, do all possible agents. Else, subsample",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="v1",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50,
        help="How many tags to give the LLM to label at once. If >100, LLM hallucinates extra fields. I think actually recently OpenAI made the models worse and you have to do even less than 50",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    run(args)
