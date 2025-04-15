from PROSE.queries.prompts.fast_disc_query_prompts import (
    LLM_APPROVAL_LEVELS_AGREEMENT_V1,
)

## Old prompts from mid 2024

VERIFICATION_LLM_APPROVAL_LEVELS_V1 = {
    6: "the complete details of the user's opinion and reasoning are captured by the statement",
    5: "almost all of the details of the user's opinion and reasoning are captured by the statement, with very minor contradictions",
    4: "around two-thirds of the details of the user's opinion and reasoning are captured by the statement, with few contradictions",
    3: "a majority of the details of the user's opinion and reasoning are captured by the statement, with some omissions / contradictions",
    2: "the user would at best slightly agree with the statement, because the statement only partially captures their opinion and reasoning, or is missing something",
    1: "the statement is either heavily incomplete, or contradicts / is orthogonal to the user's opinion and reasoning",
}

SYSTEM_PROMPT_VERIFICATION_V1 = """ 
You will be provided with survey responses from a user, and a separate statement. Your task is to determine the extent to which the user would agree with that statement, on a scale from {min_approval_level} to {max_approval_level}. Your response should be JSON containing your responses for each step in reasoning. 
""".strip()

PROMPT_VERIFICATION_V1 = """ 
**User description:**
{data}

**Statement:**
{statement}

**Instructions:**

To determine the extent to which the statement fully summarizes the user's opinion and reasoning behind it, follow these steps. Be very concise when addressing each step.

Step 1. Summarize the user's opinions and reasoning on the topic. Include a few concrete examples in your summary. 

Step 2. Explain which aspects of the user's opinion and reasoning the statement fails to touch on.

Step 3. Explain which aspects of the user's opinion and reasoning the statement actively contradicts.

Step 4. Weigh all of your considerations thoughtfully, to determine overall how much the statement summarizes the user's opinion and reasoning. Then select from one of the following {num_approval_levels} choices.
{str_approval_levels} 

**Output instructions:**

Respond in JSON as follows: 
{{
"step1" : <your response to step 1>,
"step2" : <your response to step 2>,
"step3" : <your response to step 3>,
"step4" : <your response to step 4>,
"score" : <your score, a number between {min_approval_level} and {max_approval_level}>
}}
""".strip()


## New prompts from recently

SYSTEM_PROMPT_VERIFICATION_V2 = """
Your task is to determine how much a user would agree with some statement. You will be given information about the user's opinions. First, think carefully about how the user would rate the statement, based on your knowledge of the user's opinions. Then, respond with a number between {min_approval_level} and {max_approval_level}. Your response should be JSON containing your responses for each step in reasoning. 
""".strip()


PROMPT_VERIFICATION_V2 = """
**Rating scale meaning:**
{str_approval_levels}

**Statement:**
{statement}

**User information:**
{data}

Now it is time for you to estimate how the user would rate the statement according to this rating scale. First, answer the following questions:

Step 1. What opinion is the statement expressing?
Step 2. Based on the information you have about the user, what is their opinion on the topic the statement is addressing?
Step 3. Consider each item in the rating scale thoughtfully. Which best captures how the user would rate the statement?

**Output instructions:**

Respond in JSON as follows: 
{{
"step1" : <your response to step 1>,
"step2" : <your response to step 2>,
"step3" : <your response to step 3>,
"score" : <your score, a number between {min_approval_level} and {max_approval_level}>
}}
""".strip()


def get_verification_prompts(prompt_type: str) -> str:
    if prompt_type == "verification_v1":
        return (
            SYSTEM_PROMPT_VERIFICATION_V1,
            PROMPT_VERIFICATION_V1,
        )
    elif prompt_type == "verification_v2":
        return SYSTEM_PROMPT_VERIFICATION_V2, PROMPT_VERIFICATION_V2
    else:
        raise NotImplementedError("Invalid prompt type")


def get_llm_approval_levels_verification(prompt_type: str) -> str:
    if prompt_type == "verification_v1":
        return VERIFICATION_LLM_APPROVAL_LEVELS_V1
    elif prompt_type == "verification_v2":
        return LLM_APPROVAL_LEVELS_AGREEMENT_V1
    else:
        raise NotImplementedError
