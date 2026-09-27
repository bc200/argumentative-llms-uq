"""Check provider-specific thinking controls in the OpenAI-compatible adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from llm_managers import OpenAiLlmManager


class ThinkingParameterTests(unittest.TestCase):
    def test_qwen_uses_enable_thinking_false(self):
        completion = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        )
        create = Mock(return_value=completion)
        manager = object.__new__(OpenAiLlmManager)
        manager.model_name = "qwen3.8-max"
        manager.thinking_mode = "disabled"
        manager.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        self.assertEqual(manager.chat_completion("Test"), "OK")
        self.assertEqual(create.call_args.kwargs["extra_body"],
                         {"enable_thinking": False})
        self.assertEqual(manager.last_usage,
                         {"input_tokens": 2, "output_tokens": 1})

    def test_deepseek_keeps_existing_thinking_parameter(self):
        completion = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        )
        create = Mock(return_value=completion)
        manager = object.__new__(OpenAiLlmManager)
        manager.model_name = "deepseek-v4-pro-0813"
        manager.thinking_mode = "disabled"
        manager.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        self.assertEqual(manager.chat_completion("Test"), "OK")
        self.assertEqual(create.call_args.kwargs["extra_body"],
                         {"thinking": {"type": "disabled"}})

    def test_qwen_27b_uses_reasoning_effort_none(self):
        completion = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="N/A"))],
        )
        create = Mock(return_value=completion)
        manager = object.__new__(OpenAiLlmManager)
        manager.model_name = "qwen3.8-27b"
        manager.thinking_mode = "disabled"
        manager.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        self.assertEqual(manager.chat_completion("Test"), "N/A")
        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "none")
        self.assertNotIn("extra_body", create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
