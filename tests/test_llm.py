from unittest import TestCase
from unittest.mock import patch

from nlpython.llm import OpenAIChatInterface, OpenAICompletionInterface


class TestOpenAIChatInterface(TestCase):
    @patch("nlpython.llm.openai.ChatCompletion.create")
    def test_run__openai_chat_completion_is_used(self, mock_completion_create):
        interface = OpenAIChatInterface()
        model = "model"
        messages = [{"role": "system", "content": "hello"}]
        interface.run(model, messages)
        mock_completion_create.assert_called_once_with(
            model=model,
            messages=messages,
        )


class TestOpenAICompletionInterface(TestCase):
    @patch("nlpython.llm.openai.Completion.create")
    def test_run__openai_completion_is_used(self, mock_completion_create):
        interface = OpenAICompletionInterface()
        model = "model"
        messages = [{"role": "system", "content": "hello"}]
        interface.run(model, messages)
        mock_completion_create.assert_called_once()

    @patch("nlpython.llm.openai.Completion.create")
    def test_run__if_no_history_prompt_is_generated(self, mock_completion_create):
        interface = OpenAICompletionInterface()
        model = "model"
        messages = [{"role": "system", "content": "hello"}]
        interface.run(model, messages)
        mock_completion_create.assert_called_once_with(
            model=model,
            prompt="hello\n\nChat History:\n\nAssistant:",
        )

    @patch("nlpython.llm.openai.Completion.create")
    def test_run__if_history_prompt_is_generated(self, mock_completion_create):
        interface = OpenAICompletionInterface()
        model = "model"
        messages = [
            {"role": "system", "content": "hello"},
            {"role": "user", "content": "hi"},
        ]
        interface.run(model, messages)
        mock_completion_create.assert_called_once_with(
            model=model,
            prompt="hello\n\nChat History:\nUser: hi\n\nAssistant:",
        )
