import json
from typing import List
from PROSE.queries.summary_agent import SummaryAgent
from PROSE.utils.llm_tools import GPT, LLMLog
from PROSE.queries.prompts.verification_disc_query_prompts import (
    get_llm_approval_levels_verification,
)
from PROSE.queries.prompts.verification_disc_query_prompts import (
    get_verification_prompts,
)
from PROSE.utils.llm_tools import DEFAULT_MODEL
from PROSE.queries.fast_queries import process_llm_approval_levels
from PROSE.utils.logprobs_tools import compute_score_from_logprobs


class CoTAgent(SummaryAgent):
    def __init__(
        self,
        *,
        id: str,
        data: str,
        prompt_type: str,
        model: str = DEFAULT_MODEL,
        temperature: int = 1,
    ):
        self.id = id
        self.data = data
        self.prompt_type = prompt_type
        self.llm_approval_levels = get_llm_approval_levels_verification(prompt_type)
        self.llm = GPT(
            temperature=temperature,
            model=model,
            response_format={"type": "json_object"},
            logprobs=True,
            top_logprobs=6,
        )
        self.summary = None
        self.summary_llm = GPT(
            model=model,
            temperature=0,
        )
        (
            self.max_approval_level,
            self.min_approval_level,
            self.num_approval_levels,
            self.str_approval_levels,
        ) = process_llm_approval_levels(self.llm_approval_levels)

    def get_id(self) -> str:
        return self.id

    def get_approval(self, statement: str) -> tuple[float, List[LLMLog]]:
        system_prompt, prompt = get_verification_prompts(prompt_type=self.prompt_type)
        system_prompt = system_prompt.format(
            min_approval_level=self.min_approval_level,
            max_approval_level=self.max_approval_level,
        )
        prompt = prompt.format(
            str_approval_levels=self.str_approval_levels,
            statement=statement,
            data=self.data,
            min_approval_level=self.min_approval_level,
            max_approval_level=self.max_approval_level,
            num_approval_levels=self.num_approval_levels,
        )
        response, completion, log = self.llm.call(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
        )
        try:
            response_json = json.loads(response)

            # Default value for score, if we don't use logprobs
            score = float(response_json["score"])

            score, full_dist = compute_score_from_logprobs(
                response_json, completion, "score"
            )

            response_json["score"] = score
            response_json["full_dist"] = full_dist
            log["response"] = json.dumps(response_json)  # update log with new response
        except:
            print(log)
            score = 0

        return score, [log]
