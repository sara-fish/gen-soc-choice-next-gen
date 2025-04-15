import openai
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_not_exception_type,
)
import os
from PROSE.utils.helper_functions import (
    get_time_string,
)
from typing import Any, Tuple
from pydantic import BaseModel

MAX_QUERY_RETRIES = 5

Message = dict[str, Any]
MessageList = list[Message]

DEFAULT_MODEL = "gpt-4o-2024-11-20"


class LLMLog(BaseModel):
    # These fields are essential, but some LLMLog objects also have more fields
    model: str
    params: dict[str, Any]
    prompt: str
    system_prompt: str
    messages: MessageList
    response: str
    completion: Any
    timestamp: str
    system_fingerprint: str


class GPT:
    def __init__(self, *, model, openai_seed=0, **params):
        ### Get OpenAI API key and organization

        api_key = os.getenv("OPENAI_API_KEY_GEN_SOC_CHOICE")
        assert api_key, "OpenAI API key must be set in environment variable OPENAI_API_KEY_GEN_SOC_CHOICE"

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.params = params
        self.openai_seed = openai_seed

    @retry(
        stop=stop_after_attempt(MAX_QUERY_RETRIES),
        wait=wait_random_exponential(),
        retry=retry_if_not_exception_type(
            (openai.BadRequestError, AssertionError)
        ),  # this means too many tokens
    )
    def _call_chat(self, messages: MessageList, **params) -> Tuple[str, dict]:
        """
        Make OpenAI API call to a chat style model (e.g. gpt-4-0314, gpt-4-turbo, gpt-4o)

        Return: response (str), completion (openai.Completion object)
        """
        local_params = params.copy()
        openai_seed = local_params.pop("openai_seed", self.openai_seed)
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, seed=openai_seed, **local_params
        )
        response = completion.choices[0].message.content
        return response, completion.to_dict()

    def call(
        self,
        messages: MessageList,
        **local_params,
    ) -> Tuple[str, Any, LLMLog]:
        """
        Make LLM call.
        """
        params = self.params.copy()
        params.update(local_params)

        assert messages

        # messages provided
        response, completion = self._call_chat(messages=messages, **params)

        # For logging, extract prompt and system prompt
        system_prompt, prompt = "", ""
        for message in messages:
            if message["role"] == "system":
                system_prompt += message["content"]
            elif message["role"] == "user":
                prompt += message["content"]

        log = {
            "model": self.model,
            "openai_seed": params.get("openai_seed", self.openai_seed),
            "params": params,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "messages": messages,
            "response": response,
            "completion": completion,
            "system_fingerprint": completion.get("system_fingerprint"),
            "timestamp": get_time_string(),
        }

        return response, completion, log
