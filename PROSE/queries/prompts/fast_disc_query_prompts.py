from typing import Dict


LLM_APPROVAL_LEVELS_AGREEMENT_V1 = {
    6: "Perfect match. User agrees 100% with all details in the statement.",
    5: "Near-perfect match. User agrees with 95-99% of the statement, with only minor discrepancies.",
    4: "Substantial agreement. User agrees with about 70-80% of the statement's content.",
    3: "Moderate agreement. User agrees with roughly half of the statement's points.",
    2: "Minimal agreement. User agrees with only a small portion (about 20-30%) of the statement.",
    1: "No agreement. Statement contradicts or is irrelevant to user's opinion.",
}

LLM_SPECIFICITY_LEVELS_V1 = {
    6: "Exhaustive. All specific details from user's opinion are present in the statement.",
    5: "Nearly complete. Statement contains most specific details from user's opinion, with only minor omissions.",
    4: "Partially detailed. About 70-80% of specific details from user's opinion are included.",
    3: "Moderately specific. Roughly half of the specific details in the user's opinion are mentioned.",
    2: "Minimally specific. Only a few (20-30%) of the specific details in the user's opinion are included.",
    1: "Non-specific. Statement is entirely general, without any specific details from user's opinion.",
}

SYSTEM_PROMPT_AGREEMENT_V1 = """
Your task is to determine how much a user would agree with some statement. You will be given information about the user's opinions. Respond with a number between {min_approval_level} and {max_approval_level}.
""".strip()

SYSTEM_PROMPT_SPECIFICTY_V1 = """
Your task is to determine how much detail a statement has. You will be given a statement, and information about the user's opinions. Your task is to determine how many *specific details* from the user's opinion are present in the statement. Respond with a number between {min_approval_level} and {max_approval_level}.
""".strip()

# Doing it in this way to save money via prompt caching
PROMPT_V1 = "Rating scale meaning:\n{str_approval_levels}\n\nStatement:\n{statement}\n\nUser information:\n{data}\n\nNow it is time for you to give the most accurate numerical rating. Write a number between {min_approval_level} and {max_approval_level} and nothing else. Your estimated rating: "
# deliberately leaving space here so next token is the score


def get_llm_approval_levels(prompt_type: str) -> Dict[int, str]:
    if prompt_type == "agreement_v1" or prompt_type == "verification_v1":
        return LLM_APPROVAL_LEVELS_AGREEMENT_V1
    if prompt_type == "specificity_v1":
        return LLM_SPECIFICITY_LEVELS_V1
    else:
        raise ValueError("Invalid prompt type")


def get_system_prompt_disc(prompt_type: str) -> str:
    if prompt_type == "agreement_v1":
        return SYSTEM_PROMPT_AGREEMENT_V1
    elif prompt_type == "specificity_v1":
        return SYSTEM_PROMPT_SPECIFICTY_V1
    else:
        raise ValueError("Invalid prompt type")


def get_prompt_disc(prompt_type: str) -> str:
    if prompt_type == "agreement_v1":
        return PROMPT_V1
    elif prompt_type == "specificity_v1":
        return PROMPT_V1
    else:
        raise ValueError("Invalid prompt type")
