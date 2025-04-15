import argparse

import pandas as pd

from PROSE.queries.fast_queries import FastAgent
from PROSE.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
from PROSE.utils.llm_tools import DEFAULT_MODEL
import os
from tqdm import tqdm

import random
import concurrent.futures


def run(args):
    random.seed(args.seed)  # set global randomness
    model = args.model
    num_samples = args.num_samples
    prompt_type = args.prompt_type
    num_threads = args.num_threads
    dataset = args.dataset
    topic = (
        dataset.removeprefix("PROSE/datasets/").removesuffix(".csv").replace("/", "__")
    )

    dirname = (
        get_base_dir_path()
        / "experiment_logs"
        / f"{get_time_string()}__summaries__{prompt_type}__{topic}"
    )
    os.makedirs(dirname)
    log_path = dirname / "logs.csv"
    logs = []

    df = pd.read_csv(get_base_dir_path() / dataset)

    # Get agents
    agents = []
    for id in df["user_id"].unique():
        user_df = df[df["user_id"] == id].reset_index(drop=True)
        assert len(user_df) == 1
        agent = FastAgent(
            id=str(id),
            data=user_df["data"].values[0],
            prompt_type="agreement_v1",
            model=model,
            specificity_prompt_type="specificity_v1",
        )
        agents.append(agent)

    if num_samples:
        agents = random.sample(agents, num_samples)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_data = {
            executor.submit(
                agent.generate_summary, overwrite=True, prompt_type=prompt_type
            ): {"agent_id": agent.get_id()}
            for agent in agents
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_data), total=len(future_to_data)
        ):
            _, log = future.result()
            log[0]["agent_id"] = str(future_to_data[future]["agent_id"])
            logs.extend(log)
            pd.DataFrame(logs).to_csv(log_path, index=False)

    print(f"{len(agents)} summaries saved to {log_path}")


def parse_args():
    # Use argparse to parse args for dataset_path, holdout (store if true)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="PROSE/datasets/bowlinggreen_41/bowlinggreen_curated_postprocessed.csv",
        help="Path to dataset to use. Must have cols user_id and data",
    )

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="If None, do all possible queries. Else, subsample",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="v1",
        help="Prompt type to use for summaries",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
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
