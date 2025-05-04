# [Generative Social Choice: The Next Generation](https://procaccia.info/wp-content/uploads/2025/01/nextgen.pdf)

This repo contains the code and data associated with the paper [Generative Social Choice: The Next Generation](https://procaccia.info/wp-content/uploads/2025/01/nextgen.pdf).

Authors: [Niclas Boehmer](https://www.niclas-boehmer.com/), [Sara Fish](https://sara-fish.github.io/), [Ariel Procaccia](https://procaccia.info/) 

In case you have any questions regarding the code, please reach out to [niclas.boehmer@hpi.de](mailto:niclas.boehmer@hpi.de) and [sfish@g.harvard.edu](mailto:sfish@g.harvard.edu).

# Setup instructions

1. In the folder where this README.md file is located, call `pip install -e .`
2. Install dependencies: `pipenv install`
3. Set the environment variable `OPENAI_API_KEY_GEN_SOC_CHOICE`

# Section 3.4: Validation in Synthetic Environment 

To reproduce the results from Section 3.4 run:
```
python3 synthetic_experiments/main.py
```
All parameter configurations that need to be executed to reproduce the results shown in the paper can be found in `approx_final.slurm`. 

# Section 4: PROSE 

The code for the experiments in Section 4 involving PROSE (PROportional Slate Engine) can be found in `PROSE/`. 

## Overview of code 

- `datasets/`: Datasets used in the PROSE experiments, along with LLM-generated summaries and tags used by the PROSE queries. 
- `generate_scripts/`: The main implementation of PROSE, including slate generation and baseline computation. 
- `paper_results/`: Logs from the results presented in the paper. 
- `queries/`: The main implementation of the queries used in PROSE (discriminative and generative query). 
- `utils/`: Infrastructure for LLM calling and other low-level helper functions.

## Paper reproduction 

### Dataset information 

Further detail on each dataset:
- `bowlinggreen_41`: Derived from the larger `bowling-green.american-assembly` dataset publicly released by Polis ([link](https://github.com/compdemocracy/openData))
- `drugs_80`, `drugs_obesity_80`, `drugs_80_extreme_middle`: Derived from the UCI ML Drug Review dataset publicly released on Kaggle ([link](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018))

Each dataset is accompanied by a `summaries` and `tags` folder, containing LLM-generated summaries and tags used by the PROSE queries. (These can be generated using the scripts `generate_summaries.py` and `generate_tags.py`.)

### Generating slates with PROSE

To reproduce the results go in the folder generate_scripts from the paper and execute: 
```
python3 PROSE/generate_scripts/generate_slate.py --dataset bowlinggreen_41
python3 PROSE/generate_scripts/generate_slate.py --dataset drugs_80
python3 PROSE/generate_scripts/generate_slate.py --dataset drugs_obesity_80
python3 PROSE/generate_scripts/generate_slate.py --dataset drugs_80_extreme_middle
```

### Run baseline comparisons

To reproduce the baseline comparisons using the slates from the paper:
```
python3 PROSE/generate_scripts/calculate_baselines.py --dataset bowlinggreen_41 --slate_dirname PROSE/paper_results/20250129-213651__slate__164__bowlinggreen_41
python3 PROSE/generate_scripts/calculate_baselines.py --dataset drugs_80 --slate_dirname PROSE/paper_results/20250129-203532__slate__160__drugs_80
python3 PROSE/generate_scripts/calculate_baselines.py --dataset drugs_obesity_80 --slate_dirname PROSE/paper_results/20250129-224128__slate__160__drugs_obesity_80
python3 PROSE/generate_scripts/calculate_baselines.py --dataset drugs_80_extreme_middle --slate_dirname PROSE/paper_results/20250129-175440__slate__160__drugs_80_extreme_middle
```

## Running PROSE on other datasets 

PROSE can operate with unstructured and minimalistic user data, making it usable in a wide range of scenarios. If you are interested in running PROSE on a different dataset, follow the below steps. 

### Step 1: Preprocess your dataset 

Format your dataset as a csv with the columns `user_id` (a unique string corresponding to each user) and `data` (a string containing information about that user's opinion). For an example see `bowlinggreen_41/bowlinggreen_curated_postprocessed.csv`. 

### Step 2: Construct compatible agents 

#### Option 1: Following the PROSE experiments from our paper

Our generative query implementations leverage representations of agents in an embedding space. To do so, we first generate structured summaries of each agent's opinion, and then consider these structured summaries jointly to create a common embedding space for the agents. For example, to generate summaries and tags for `bowlinggreen_41`:

```
python3 PROSE/generate_scripts/generate_summaries.py --dataset PROSE/datasets/bowlinggreen_41/bowlinggreen_curated_postprocessed.csv 
python3 PROSE/generate_scripts/generate_tags.py --dataset PROSE/datasets/bowlinggreen_41/bowlinggreen_curated_postprocessed.csv --summaries PROSE/datasets/bowlinggreen_41/20241112-140604__summaries__v1__bowlinggreen_curated_postprocessed/logs.csv
```

Ultimately, our generative query implementations rely on each `Agent` object having the following attributes:
- `id`: unique string ID of agent
- `data`: raw string representation of agent's opinion
- `summary`: raw string representation of agent's opinion to be used in LLM queries
    - In this paper, we set `agent.summary = agent.data`. 
    - For applications where `agent.data` is not appropriate for direct use in LLM queries, one can optionally distill the content of `agent.data` and save it to `agent.summary` 
- (LLM-generated) `tags`: dict mapping each tag attribute (same for all agents) to a numeric score reflecting the extent to which that attribute reflects that agent. (`agent.tags.values()` is then used as that agent's embedding vector)

Thus, as long as each `Agent` object has these four attributes, it is possible to run PROSE using the precise queries we use in our experiments in the paper.

#### Option 2: Constructing custom agents and queries

To construct custom agents and queries, implement subclasses of the interface classes `Agent` and `TargetedGenerator`. Various example query implementations (not all of which are used in the PROSE experiments in the paper) are in `queries/`. 

### Step 3: Generate a slate

#### Option 1: Following the PROSE experiments from our paper 

*If summaries and tags were generated using our scripts:* Edit the method `get_data_filenames` in `datasets/load.py` to return the filenames for the data, summaries, tags, and topic for your use case. Then the script `generate_scripts/generate_slate.py` should work out of the box. 

*If summaries and tags were not generated using our scripts:* In the `if __name__ == "__main__":` part of the file, modify the current code that creates an `Agent` object for each agent to instead create an `Agent` object using your summaries and tags. The rest of the script should work the same. 

#### Option 2: Running PROSE using custom agents and different queries

The function `generate_slate(...)` in `generate_scripts/generate_slate.py` should work as long as you pass it correctly configured `Agent`s and `TargetedGenerator`s. We recommend in this case you write a new script that configures your custom agents and queries, and then calls `generate_slate(...)` to generate a slate using these.
