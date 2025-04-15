from typing import List
from pydantic import BaseModel


SUMMARIZE_SYSTEM_PROMPT_V1 = """ 
You will be provided with a user's response to a survey, in which they describe their opinions on a topic in detail. Your task is to produce a detailed summary of that user's opinion. Your response should be in JSON according to the format specified below.
""".strip()

SUMMARIZE_PROMPT_TEMPLATE_V1 = """
**User survey responses:**
{user_data}

**Output instructions:**
Complete the following entries:
- most_important_aspects (List[str]): List the aspects of the topic that are most important to the user. Each aspect should be succinct and self-contained, so that it can be understood without context. (For example, instead of writing "Religious beliefs", write "Believes X due to religious beliefs") 
- specific_details (List[str]): List any specific details or examples the user provided to support their opinion. Each detail should be succinct and self-contained, so that it can be understood without context. These details, together with the most_important_aspects above, should be enough to mostly reconstruct the user's opinion.
- user_background (List[str]): List any personal information the user may have divulged that is relevant to their opinion.
- overall_summary (str): Write a detailed 2-3 sentence summary of the user's opinion, taking into account all the context provided above.

Respond in JSON with the fields above filled in.
""".strip()


def get_system_prompt_from_prompt_type(*, prompt_type: str) -> str:
    if prompt_type == "v1":
        return SUMMARIZE_SYSTEM_PROMPT_V1
    else:
        raise NotImplementedError


def get_prompt_template_from_prompt_type(*, prompt_type: str) -> str:
    if prompt_type == "v1":
        return SUMMARIZE_PROMPT_TEMPLATE_V1
    else:
        raise NotImplementedError


class SummaryV1(BaseModel):
    most_important_aspects: list[str]
    specific_details: list[str]
    user_background: list[str]
    overall_summary: str


def get_validator_from_prompt_type(*, prompt_type: str) -> str:
    if prompt_type == "v1":
        return SummaryV1
    else:
        raise NotImplementedError


def get_tag_fields(*, prompt_type: str) -> List[str]:
    if prompt_type == "v1":
        return [
            "most_important_aspects",
            "specific_details",
            "user_background",
        ]
    else:
        raise NotImplementedError
