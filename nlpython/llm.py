from abc import abstractmethod, ABC
from typing import List, Dict, Type

import openai


class LLMBaseInterface(ABC):
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, model: str, messages: List[Dict], **kwargs):
        raise NotImplementedError


class OpenAIChatInterface(LLMBaseInterface):
    def run(self, model: str, messages: List[Dict], **kwargs):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        model_response = response["choices"][0]["message"]["content"]
        return model_response


class OpenAICompletionInterface(LLMBaseInterface):
    PROMPT_TEMPLATE = (
        "{system_message}\n\n" "Chat History:\n{history}\n\n" "Assistant:"
    )

    def run(self, model: str, messages: List[Dict], **kwargs):
        prompt = self._get_prompt_from_messages(messages)
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            **kwargs,
        )
        model_response = response["choices"][0]["text"]
        return model_response

    def _get_prompt_from_messages(self, messages: List[Dict]):
        system_message = (
            messages[0]["content"]
            if messages and messages[0]["role"] == "system"
            else ""
        )
        if len(system_message) > 0 and len(messages) > 1:
            history = messages[1:]
        else:
            history = messages
        history = "\n".join(
            f"{x['role'].capitalize()}: {x['content']}" for x in history
        )
        prompt = self.PROMPT_TEMPLATE.format(
            system_message=system_message,
            history=history,
        )
        return prompt


SUPPORTED_MODELS = [
    "text-davinci-003",
    "gpt-4",
    "gpt-3.5-turbo",
]
MODEL_TO_API: Dict[str, Type[LLMBaseInterface]] = {
    "text-davinci-003": OpenAICompletionInterface,
    "gpt-4": OpenAIChatInterface,
    "gpt-3.5-turbo": OpenAIChatInterface,
}
