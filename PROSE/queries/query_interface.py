from abc import ABC, abstractmethod
from typing import List
from PROSE.utils.llm_tools import LLMLog


class Agent(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass

    @abstractmethod
    def get_approval(self, statement: str) -> tuple[float, List[LLMLog]]:
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Return description of agent for use in generative queries (e.g. summary of user survey responses).
        """
        pass


class Generator(ABC):
    @abstractmethod
    def generate(self, agents: List[Agent]) -> tuple[list[str], List[LLMLog]]:
        pass


class TargetedGenerator(ABC):
    @abstractmethod
    def get_name(self) -> str:
        # String representation of generator -- used for logging
        pass

    @abstractmethod
    def generate(
        self,
        agents: List[Agent],
        target_approval: float,
        target_budget: int,
    ) -> tuple[List[str], List[LLMLog]]:
        pass
