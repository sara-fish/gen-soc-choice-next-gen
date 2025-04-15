import json
from typing import List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt
from PROSE.utils.logprobs_tools import (
    filter_non_integer_idx,
)
from PROSE.queries.query_interface import Agent
from PROSE.utils.llm_tools import GPT, LLMLog, DEFAULT_MODEL
from PROSE.utils.logprobs_tools import get_probabilities_from_completion

MAX_TRIES = 10  # num passes of tagging before giving up


def get_response_format(fields):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "TaggingSchema",  # Required name
            "schema": {  # This is the correct key to define the actual schema
                "type": "object",
                "properties": {field: {"type": "integer"} for field in fields},
                "required": fields,
            },
        },
    }


def get_tagging_llm(fields):
    return GPT(
        model=DEFAULT_MODEL,
        response_format=get_response_format(fields),  # Dynamically generated schema
        max_tokens=4096,
        openai_seed=0,
        logprobs=True,
        top_logprobs=7,
    )


TAGGING_SYSTEM_PROMPT_V1 = """You will be provided with a user's response to a survey, in which they describe their opinions on a topic in detail. Then you will be provided with a list of aspects of that topic. For each aspect, your task is to rate the extent to which it pertains to the user's response or captures the user's opinion. The scale is as follows:
1. Strongly goes against user's opinion 
2. Goes against user's opinion
3. Somewhat goes against user's opinion
4. Neutral / unknown 
5. Somewhat aligned with user's opinion 
6. Aligned with user's opinion
7. Strongly aligned with user's opinion
Respond in JSON, mapping each aspect to your rating. It is important to copy the EXACT wording of the aspect with no changes.
""".strip()

TAGGING_PROMPT_TEMPLATE_V1 = """ 
**User survey responses:**
{user_data}

**Output instructions:**
For each of the following aspects, rate on the scale from 1-7 how well it pertains to the user's response or captures the user's opinion. 
{list_of_fields}

Respond in JSON, with the above fields as keys and your ratings as values.
""".strip()


def get_system_prompt_from_prompt_type(prompt_type: str) -> str:
    if prompt_type == "v1":
        return TAGGING_SYSTEM_PROMPT_V1
    else:
        raise NotImplementedError


def get_prompt_template_from_prompt_type(prompt_type: str) -> str:
    if prompt_type == "v1":
        return TAGGING_PROMPT_TEMPLATE_V1
    else:
        raise NotImplementedError


def get_valid_ratings_from_prompt_type(prompt_type: str) -> List[str]:
    if prompt_type == "v1":
        return [1, 2, 3, 4, 5, 6, 7]
    else:
        raise NotImplementedError


def get_default_value_from_prompt_type(prompt_type: str) -> int:
    """
    Given a prompt type, get the "default" rating value. Should be the unknown/neutral option. This is used to fill NaN values when tagging is unsuccesful.
    """
    if prompt_type == "v1":
        return 4
    else:
        raise NotImplementedError


def is_int(token: str) -> bool:
    try:
        _ = int(token)
    except ValueError:
        return False
    return True


@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ValueError))
def compute_tags_of_agent(
    agent: Agent,
    fields: List[str],
    prompt_type: str = "v1",
    chunk_size: int = 50,
) -> Tuple[str, List[LLMLog]]:
    """
    chunk_size: Max fields LLM can handle simultaneously (Not a context window limitation -- above 100, LLM starts hallucinating, which makes logprobs annoying)
    """
    assert len(set(fields)) == len(fields), "Fields must be unique"
    ## If fields contain numbers, the logprob parsing will break
    for field in fields:
        assert not any(
            char.isdigit() for char in field
        ), "Field contains digit -- this will break logprob parsing. Replace with text using inflect package"
    unlabeled_fields = fields.copy()
    field_to_tag = dict()
    logs = []
    num_tries = 0
    while unlabeled_fields and num_tries < MAX_TRIES:
        num_tries += 1
        current_fields = unlabeled_fields[:chunk_size]

        prompt = get_prompt_template_from_prompt_type(prompt_type=prompt_type).format(
            user_data=agent.data,
            list_of_fields="\n".join([f"- {field}" for field in current_fields]),
        )
        messages = [
            {
                "role": "system",
                "content": get_system_prompt_from_prompt_type(prompt_type=prompt_type),
            },
            {"role": "user", "content": prompt},
        ]

        # Create tagging_llm using the processed fields
        tagging_llm = get_tagging_llm(current_fields)
        response, completion, log = tagging_llm.call(messages=messages)
        response_json = json.loads(response)

        ### Parse logprobs from completion
        # Get individual tokens from completion
        tokens = [
            completion["choices"][0]["logprobs"]["content"][i]["token"]
            for i in range(len(completion["choices"][0]["logprobs"]["content"]))
        ]
        # Keep only the tokens that are integers
        idx_to_int_token = {
            i: int(token) for i, token in enumerate(tokens) if is_int(token)
        }
        # If there are exactly as many integers as entries in resopnse_json, then each corresponds to a score
        if len(response_json) != len(idx_to_int_token):
            raise ValueError(
                f"Found {len(idx_to_int_token)} integers in logprobs, but {len(response_json)} fields in response_json"
            )

        tag_to_full_dist = dict()
        # For each tag, extract the logprobs, and overwrite the score to be the weighted average
        for tag, (idx, token) in zip(response_json, idx_to_int_token.items()):
            if response_json[tag] != token:
                raise ValueError(
                    "Mismatch between parsed response and token order -- perhaps LLM wrote tags out of order?"
                )
            probs = get_probabilities_from_completion(completion, idx)
            if not probs.empty:
                probs = filter_non_integer_idx(probs)
                probs = probs[
                    probs.index.astype(int).isin(
                        get_valid_ratings_from_prompt_type(prompt_type)
                    )
                ]
                score = float(
                    (probs.astype(float) * probs.index.astype(float)).sum()
                    / probs.astype(float).sum()
                )
                full_dist = probs.to_dict()
            else:
                full_dist = {}
            response_json[tag] = score
            tag_to_full_dist[tag] = full_dist

        # Also log the full distributions just in case
        log["full_dists"] = json.dumps(tag_to_full_dist)

        # Finally, only keep in unlabeled_fields fields the LLM didn't label yet, becuase we repeat
        # until it gets everything
        log["queried_fields"] = json.dumps(unlabeled_fields)
        field_to_tag.update(response_json)
        unlabeled_fields = list(set(unlabeled_fields).difference(response_json.keys()))
        logs.extend([log])

    # LLM may have hallucinated extra fields -- only keep the ones asked for in input
    field_to_tag = json.dumps(
        {
            field: field_to_tag.get(
                field, get_default_value_from_prompt_type(prompt_type)
            )
            for field in fields
        }
    )

    return field_to_tag, logs
