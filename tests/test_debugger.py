from unittest import TestCase
from unittest.mock import patch

from nlpython.debugger import NLPDebugger


class TestNLPDebugger(TestCase):
    def test_init__unsupported_model_raises_value_error(self):
        with self.assertRaises(ValueError):
            NLPDebugger(model_name="unsupported-model")

    @patch("nlpython.debugger.OpenAICompletionInterface.run")
    def test_debug__no_errors_returns_code(self, mock_run):
        mock_run.side_effect = [""]
        debugger = NLPDebugger()
        code = "print('Hello, world!')"
        result = debugger.debug(code)
        self.assertEqual(result, code)

    @patch("nlpython.debugger.OpenAICompletionInterface.run")
    def test_debug__single_error_returns_fixed_code(self, mock_run):
        mock_run.side_effect = [
            "print('Hello, world!')",
            "print('Hello, world!')",
            "",
        ]
        debugger = NLPDebugger()
        code = "print('Hello, world')"
        result = debugger.debug(code)
        self.assertEqual(result, "print('Hello, world!')")

    @patch("nlpython.debugger.OpenAICompletionInterface.run")
    def test_debug__multiple_errors_returns_fixed_code(self, mock_run):
        mock_run.side_effect = [
            "print('Hello, world!')",
            "print('Hello, world!')",
            "",
        ]
        debugger = NLPDebugger()
        code = "print('Hello, world')"
        result = debugger.debug(code)
        self.assertEqual(result, "print('Hello, world!')")

    @patch("nlpython.debugger.OpenAICompletionInterface.run")
    def test_debug__max_iterations_reached_raises_exception(self, mock_run):
        mock_run.side_effect = [
            "print('Hello, world')",
            "print('Hello, world')",
            "print('Hello, world')",
            "print('Hello, world')",
            "print('Hello, world')",
            "print('Hello, world')",
            "print('Hello, world')",
            "print('Hello, world')",
            "print('Hello, world')",
            "print('Hello, world')",
        ]
        debugger = NLPDebugger(max_iterations=10)
        code = "print('Hello, world')"
        with self.assertRaises(Exception):
            debugger.debug(code)
