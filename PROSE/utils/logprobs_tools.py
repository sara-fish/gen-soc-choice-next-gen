from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np


def get_probabilities_from_completion(completion, token_idx):
    """
    Given a GPT completion object, and token_idx of the token we care about, return the actual probabilitiy distribution of that token.
    """
    completion = completion["choices"]
    assert len(completion) == 1
    try:
        logprobs = completion[0]["logprobs"]["content"][token_idx]["top_logprobs"]
        logprobs = pd.Series({entry["token"]: entry["logprob"] for entry in logprobs})
    except IndexError:
        return pd.Series([])
    probs = logprobs.apply(np.exp)
    return probs


def get_token_idx_of_fields_value(
    completion: Any, logprobs_field: str
) -> Optional[int]:
    """
    Given a completion object (assumed to be JSON), and a field name (assumed to be contained in that JSON), return the index of the token corresponding to the value written after that field name (asumed to be a single-token repsonse).

    For example, if output JSON contains {... some stuff..., "score" : 2}, and logprobs_field='score', then this identifies the index of the token containing '2'.

    If this fails, return None.
    """
    # Figure out which token corresponds to logprobs_field
    # To do this: keep chopping off tokens at the end, until logprobs_field is no longer contained in it
    assert logprobs_field in completion["choices"][0]["message"]["content"]
    for field_idx in range(len(completion["choices"][0]["logprobs"]["content"]), 0, -1):
        initial_message = "".join(
            [
                token_logprob["token"]
                for token_logprob in completion["choices"][0]["logprobs"]["content"][
                    :field_idx
                ]
            ]
        )

        if logprobs_field not in initial_message:
            break
    else:
        raise KeyError(f"Could not find key {logprobs_field}")
    # field_idx is the token idx of the last token of logprobs_field

    # Figure out which token corresponds to the value of logprobs_field (should be shortly after)
    for idx, token_logprob in enumerate(
        completion["choices"][0]["logprobs"]["content"][field_idx + 1 :]
    ):
        token = token_logprob["token"]
        try:
            _ = int(token)
            return idx + field_idx + 1
        except ValueError:
            continue  #  Haven't found token with score yet
    return None


def filter_non_integer_idx(series: pd.Series) -> pd.Series:
    numeric_idxs = []
    for idx in series.index:
        try:
            int(idx)
            numeric_idxs.append(idx)
        except ValueError:
            pass
    return series[numeric_idxs]


def compute_score_from_logprobs(
    response_json: dict, completion: Any, score_col: str
) -> Tuple[float, dict]:
    """
    Given response_json (output from LLM in JSON mode), and completion object (output from LLM query), and score_col (column name in response_json that contains the score), compute the score and distribution of score from logprobs.

    Return: score, full_dist
    """
    # Try to extract logprob from score_col and replace it in the response json
    token_idx = get_token_idx_of_fields_value(completion, logprobs_field=score_col)
    if token_idx is not None:
        logprobs = get_probabilities_from_completion(completion, token_idx)
        if not logprobs.empty:
            # If successfully extracted logprobs, replace the score with the EV wrt logprobs
            logprobs = filter_non_integer_idx(logprobs)
            score = float(
                (logprobs.astype(float) * logprobs.index.astype(float)).sum()
                / logprobs.astype(float).sum()
            )
            full_dist = logprobs.to_dict()
            return score, full_dist
    return float(response_json[score_col]), {}
