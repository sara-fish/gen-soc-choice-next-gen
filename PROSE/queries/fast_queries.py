from typing import Dict, List, Optional, Union, Tuple
from PROSE.queries.query_interface import Agent, TargetedGenerator
from PROSE.queries.summary_agent import SummaryAgent
from PROSE.utils.llm_tools import GPT, LLMLog
from PROSE.utils.logprobs_tools import (
    get_probabilities_from_completion,
)
from PROSE.utils.logprobs_tools import filter_non_integer_idx
from PROSE.queries.prompts.fast_disc_query_prompts import (
    get_prompt_disc,
    get_system_prompt_disc,
    get_llm_approval_levels,
)
from PROSE.queries.prompts.fast_gen_query_prompts import (
    get_system_prompt_fast_generator,
)
from PROSE.queries.prompts.fast_gen_no_budget_query_prompts import (
    get_system_prompt_fast_no_budget_generator,
)
from PROSE.utils.llm_tools import DEFAULT_MODEL
from PROSE.utils.helper_functions import count_words
import random


def process_llm_approval_levels(
    llm_approval_levels: Union[Dict[str, int], List[int]],
) -> Tuple[int, int, int, str]:
    if isinstance(llm_approval_levels, dict):
        max_approval_level = max(llm_approval_levels.keys())
        min_approval_level = min(llm_approval_levels.keys())
        num_approval_levels = len(llm_approval_levels)
        str_approval_levels = str(llm_approval_levels)
    else:
        max_approval_level = max(llm_approval_levels)
        min_approval_level = min(llm_approval_levels)
        num_approval_levels = len(llm_approval_levels)
        str_approval_levels = str(llm_approval_levels)
    assert min_approval_level < max_approval_level
    assert num_approval_levels > 1
    return (
        max_approval_level,
        min_approval_level,
        num_approval_levels,
        str_approval_levels,
    )


class FastAgent(SummaryAgent):
    def __init__(
        self,
        *,
        id: str,
        data: str,
        prompt_type: str,
        specificity_prompt_type: str,
        model: str = "gpt-4o-mini-2024-07-18",
        # model: str = DEFAULT_MODEL,
        temperature: int = 1,
        specificity_coeff: float = 0,
        label: str = " ",
    ):
        self.embedding = None
        self.id = id
        self.data = data
        self.llm_approval_levels = get_llm_approval_levels(prompt_type)
        self.prompt_type = prompt_type
        self.specificity_prompt_type = specificity_prompt_type
        self.label = label
        self.llm = GPT(
            temperature=temperature,
            model=model,
            logprobs=True,
            top_logprobs=5,
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

        # Process specificity levels
        self.llm_specificity_levels = get_llm_approval_levels(specificity_prompt_type)
        self.specificity_coeff = specificity_coeff
        self.max_specificity_level, self.min_specificity_level, _, _ = (
            process_llm_approval_levels(self.llm_specificity_levels)
        )

    def get_id(self) -> str:
        return self.id

    def get_label(self) -> str:
        return self.label

    def get_data(self):
        return self.data

    def get_embedding(self):
        return self.embedding

    def set_embedding(self, embedding):
        self.embedding = embedding

    def get_approval(self, statement: str) -> tuple[float, List[LLMLog]]:
        specificity, log_list_specificity = self._make_query(
            statement, self.specificity_prompt_type
        )
        approval, log_list_approval = self._make_query(statement, self.prompt_type)
        log_list = log_list_specificity + log_list_approval

        specificity_deduction = (
            self.specificity_coeff
            * (self.max_specificity_level - specificity)
            / (self.max_specificity_level - self.min_specificity_level)
        )

        score = approval - specificity_deduction

        return score, log_list

    def _make_query(
        self, statement: str, prompt_type: str
    ) -> tuple[float, List[LLMLog]]:
        system_prompt = get_system_prompt_disc(prompt_type).format(
            min_approval_level=self.min_approval_level,
            max_approval_level=self.max_approval_level,
        )
        prompt = get_prompt_disc(prompt_type).format(
            data=self.data,
            statement=statement,
            str_approval_levels=self.str_approval_levels,
            min_approval_level=self.min_approval_level,
            max_approval_level=self.max_approval_level,
        )
        response, completion, log = self.llm.call(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1,
        )

        try:
            logprobs = get_probabilities_from_completion(completion, 0)
            assert not logprobs.empty
            logprobs = filter_non_integer_idx(logprobs)
            assert not logprobs.empty
            score = float(
                (logprobs.astype(float) * logprobs.index.astype(float)).sum()
                / logprobs.astype(float).sum()
            )
        except AssertionError:
            print(
                f"Error: could not extract score from completion, response was {response}"
            )
            score = 0

        log["score"] = score

        return score, [log]


class FastGenerator(TargetedGenerator):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        prompt_type: str = "v1",
        temperature: float = 1,
        openai_seed: int = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
    ):
        self.llm = GPT(temperature=temperature, model=model, openai_seed=openai_seed)
        self.prompt_type = prompt_type
        self.target_approval = target_approval
        self.target_budget = target_budget

    def get_name(self) -> str:
        return (
            self.__class__.__name__
            + f"__approval_{self.target_approval}__budget_{self.target_budget}"
        )

    def generate(
        self,
        agents: List[Agent],
        target_approval: float,
        target_budget: int,
    ) -> tuple[list[str], List[LLMLog]]:
        if target_approval is None:
            target_approval = self.target_approval
        if target_budget is None:
            target_budget = self.target_budget
        assert target_budget is not None
        # assert target_approval is not None
        if target_budget > -1:
            system_prompt = get_system_prompt_fast_generator(self.prompt_type).format(
                target_budget=target_budget, target_approval=target_approval
            )
        else:
            system_prompt = get_system_prompt_fast_no_budget_generator(
                self.prompt_type
            ).format(target_approval=target_approval)

        random.shuffle(agents)
        agents_descriptions_str = "\n\n".join(
            [f"User {agent.get_id()}: {agent.get_description()}" for agent in agents]
        )
        prompt = (
            f"User information:\n{agents_descriptions_str}\n\nNow write the opinion."
        )

        response, _, log = self.llm.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        # Parse out <opinion> XML tag from response
        assert "<opinion>" in response and "</opinion>" in response
        statement = response.split("<opinion>")[1].split("</opinion>")[0]
        log["statement"] = statement

        return [statement], [log]


class StrictFastGenerator(FastGenerator):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        prompt_type: str = "v1",
        temperature: float = 1,
        openai_seed: int = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
        num_retries: int = 5,
    ):
        """
        Same as FastGenerator, but retry up to num_retries times if the statement generated is out of budget.
        If it's still out of budget after, then print a warning.
        """
        super().__init__(
            model=model,
            prompt_type=prompt_type,
            temperature=temperature,
            openai_seed=openai_seed,
            target_approval=target_approval,
            target_budget=target_budget,
        )
        self.num_retries = num_retries

    def generate(
        self,
        agents: List[Agent],
        target_approval: float,
        target_budget: int,
    ) -> tuple[list[str], List[LLMLog]]:
        for _ in range(self.num_retries):
            statement_list, log_list = super().generate(
                agents=agents,
                target_approval=target_approval,
                target_budget=target_budget,
            )

            assert len(statement_list) == 1

            word_count = count_words(statement_list[0])
            if word_count <= target_budget:
                return statement_list, log_list

        print(
            f"Warning: after {self.num_retries} attempts, could not generate statement within word budget of {target_budget}. (Generated statement instead has {word_count} words.)"
        )

        return statement_list, log_list
