from PROSE.utils.llm_tools import LLMLog
from PROSE.queries.query_interface import Agent
from PROSE.queries.query_interface import TargetedGenerator
from PROSE.utils.llm_tools import GPT
from PROSE.queries.prompts.gen_iter_prompts import get_prompts
from PROSE.utils.helper_functions import count_words
from typing import Dict, List


class IterGenerator(TargetedGenerator):
    def __init__(
        self,
        approval_levels: Dict[int, str],
        prompt_type: str = "v1",
        temperature: float = 1,
        max_iterations: int = 5,
    ):
        self.llm = GPT(temperature=temperature, model="gpt-4o-2024-05-13")
        self.prompt_type = prompt_type
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.approval_levels = approval_levels

    def generate(
        self,
        agents: List[Agent],
        target_approval: float,
        target_budget: int,
    ) -> tuple[list[str], List[LLMLog]]:
        logs = []
        agent_descriptions = ""
        for agent in agents:
            agent_descriptions += (
                f"Agent {agent.get_id()}: {agent.get_description()}\n\n"
            )
        num_agents = len(agents)

        approval_levels_str = "\n".join(
            [f"{key}: {value}" for key, value in self.approval_levels.items()]
        )

        assert target_approval in self.approval_levels
        target_approval_text = self.approval_levels[target_approval]

        initial_prompt, response_prompt = get_prompts(prompt_type=self.prompt_type)

        ## Send initial message

        messages = [
            {
                "role": "user",
                "content": initial_prompt.format(
                    num_agents=num_agents,
                    approval_levels_str=approval_levels_str,
                    target_approval=target_approval,
                    target_approval_text=target_approval_text,
                    target_budget=target_budget,
                    agent_descriptions=agent_descriptions,
                ),
            }
        ]

        response, _, log = self.llm.call(messages=messages)
        logs.append(log)

        messages.append({"role": "assistant", "content": response})

        for num_iterations in range(1, self.max_iterations + 1):
            # From response extract text bwetween <text>...</text> tags
            assert "<text>" in response and "</text>" in response
            previous_text = response.split("<text>")[1].split("</text>")[0]

            # previous_text_target_budget = len(previous_text)
            previous_text_target_budget = count_words(previous_text)

            # Get approvals of each agent of previous_text
            agent_id_to_approval = {}
            for agent in agents:
                approval, log = agent.get_approval(statement=previous_text)
                logs.extend(log)
                agent_id_to_approval[agent.get_id()] = approval

            agent_approvals = "\n".join(
                [f"Agent {key}: {value}" for key, value in agent_id_to_approval.items()]
            )

            average_approval = sum(agent_id_to_approval.values()) / num_agents

            min_approval = min(agent_id_to_approval.values())

            print(
                f"Iteration {num_iterations}:\n- Average approval: {average_approval}\n- Min approval: {min_approval}\n- Text: {previous_text}\n- Text length: {previous_text_target_budget}\n- Target length: {target_budget}\n- Target approval: {target_approval}\n- Full LLM response: {response}\n\n"
            )

            if (
                min_approval >= target_approval
                and previous_text_target_budget <= target_budget
            ):
                print(f"Success after {num_iterations} iterations.")
                return [previous_text], logs

            messages.append(
                {
                    "role": "user",
                    "content": response_prompt.format(
                        num_iterations=num_iterations,
                        max_iterations=self.max_iterations,
                        target_budget=target_budget,
                        target_approval=target_approval,
                        target_approval_text=target_approval_text,
                        previous_text=previous_text,
                        previous_text_target_budget=previous_text_target_budget,
                        agent_approvals=agent_approvals,
                    ),
                }
            )

            response, _, log = self.llm.call(messages=messages)
            logs.append(log)

            messages.append({"role": "assistant", "content": response})

        return [previous_text], logs
