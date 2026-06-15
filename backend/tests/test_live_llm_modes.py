import asyncio
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.routes import AnalyzeRequest, analyze  # noqa: E402


class FakeOpenRouterProvider:
    name = "openrouter"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def complete_json(self, **kwargs):
        return kwargs["fallback"]


class FailingOpenRouterProvider(FakeOpenRouterProvider):
    def complete_json(self, **kwargs):
        raise RuntimeError("simulated OpenRouter failure")


class LiveLLMModeTests(unittest.TestCase):
    def test_judge_only_makes_exactly_one_openrouter_call(self):
        response = self._analyze_with_mode("judge_only")

        self.assertEqual(response.metadata.live_llm_mode, "judge_only")
        self.assertEqual(response.metadata.openrouter_call_count, 1)
        self.assertIn("judge=openrouter", response.metadata.provider_used)
        self.assertIn("specialists=fast-deterministic", response.metadata.provider_used)

    def test_all_agents_makes_multiple_openrouter_calls(self):
        response = self._analyze_with_mode("all_agents")

        self.assertEqual(response.metadata.live_llm_mode, "all_agents")
        self.assertGreaterEqual(response.metadata.openrouter_call_count, 4)
        self.assertIn("specialists=openrouter", response.metadata.provider_used)
        self.assertIn("judge=openrouter", response.metadata.provider_used)

    def test_off_makes_zero_openrouter_calls(self):
        response = self._analyze_with_mode("off")

        self.assertEqual(response.metadata.live_llm_mode, "off")
        self.assertEqual(response.metadata.openrouter_call_count, 0)
        self.assertIn("judge=fast-deterministic", response.metadata.provider_used)

    def test_judge_only_falls_back_when_openrouter_fails(self):
        response = self._analyze_with_mode(
            "judge_only", provider_cls=FailingOpenRouterProvider
        )

        self.assertEqual(response.metadata.live_llm_mode, "judge_only")
        self.assertEqual(response.metadata.openrouter_call_count, 1)
        self.assertEqual(response.metadata.fallback_reason, "simulated OpenRouter failure")
        self.assertTrue(response.consensus)

    def _analyze_with_mode(self, mode, provider_cls=FakeOpenRouterProvider):
        env = {
            "USE_LIVE_LLM": "true",
            "LIVE_LLM_MODE": mode,
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "test/model",
            "OPENROUTER_BASE_URL": "https://openrouter.test/api/v1",
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_API_KEY": "",
            "AZURE_OPENAI_DEPLOYMENT": "",
            "AZURE_OPENAI_API_VERSION": "",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("llm.factory.OpenRouterProvider", provider_cls):
                return asyncio.run(
                    analyze(
                        AnalyzeRequest(
                            question="Should we use AI agents to help review software changes?"
                        )
                    )
                )


if __name__ == "__main__":
    unittest.main()
