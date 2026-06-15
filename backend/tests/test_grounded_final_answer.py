import asyncio
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.routes import AnalyzeRequest, analyze  # noqa: E402


class InvalidThenValidOpenRouterProvider:
    name = "openrouter"
    calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def complete_json(self, **kwargs):
        type(self).calls += 1
        if type(self).calls == 1:
            return {
                "summary": "The answer cites an invalid source first.",
                "recommendation": "Use the sourced decision path.",
                "key_findings": [
                    {"claim": "Invalid citation should be repaired.", "source_ids": ["fake-source"]}
                ],
                "risks_or_limitations": ["Citation validation should catch this."],
                "follow_up_questions": ["Which source supports the claim?"],
                "source_quality": "strong",
                "provider_used": "openrouter",
                "live_llm_mode": "judge_only",
            }
        return {
            "summary": "The answer now uses only retrieved source IDs.",
            "recommendation": "Use the sourced decision path.",
            "key_findings": [
                {
                    "claim": "Governed AI use needs accountable controls.",
                    "source_ids": ["enterprise-nist-ai-rmf"],
                }
            ],
            "risks_or_limitations": ["Evidence is decision support, not a guarantee."],
            "follow_up_questions": ["Who owns the control review?"],
            "source_quality": "partial",
            "provider_used": "openrouter",
            "live_llm_mode": "judge_only",
        }


class GroundedFinalAnswerTests(unittest.TestCase):
    def setUp(self):
        InvalidThenValidOpenRouterProvider.calls = 0

    def test_invalid_citations_are_caught_and_not_returned(self):
        response = self._analyze(
            "Should our company allow employees to use public AI tools for confidential client documents?",
            provider_cls=InvalidThenValidOpenRouterProvider,
        )
        valid_source_ids = {source.source_id for source in response.sources}
        cited_source_ids = {
            source_id
            for finding in response.final_answer.key_findings
            for source_id in finding.source_ids
        }

        self.assertTrue(cited_source_ids)
        self.assertTrue(cited_source_ids <= valid_source_ids)
        self.assertNotIn("fake-source", cited_source_ids)
        self.assertEqual(response.metadata.openrouter_call_count, 2)

    def test_weak_retrieval_produces_weak_source_quality(self):
        response = self._analyze(
            "Should I play through quad pain in tonight's soccer match?",
            live_llm_mode="off",
        )

        self.assertEqual(response.final_answer.source_quality, "weak")
        self.assertEqual(response.metadata.openrouter_call_count, 0)

    def test_final_response_schema_is_stable(self):
        response = self._analyze(
            "Should a research team trust a single LLM grader for evaluating student concept maps?",
            live_llm_mode="off",
        )

        payload = response.final_answer.dict()
        self.assertEqual(
            set(payload),
            {
                "summary",
                "recommendation",
                "key_findings",
                "risks_or_limitations",
                "follow_up_questions",
                "source_quality",
                "provider_used",
                "live_llm_mode",
            },
        )

    def test_no_fake_citations_in_deterministic_fallback(self):
        response = self._analyze(
            "Should medical schools use AI to screen applications?",
            live_llm_mode="off",
        )
        valid_source_ids = {source.source_id for source in response.sources}
        for finding in response.final_answer.key_findings:
            self.assertTrue(set(finding.source_ids) <= valid_source_ids)

    def _analyze(
        self,
        question,
        *,
        live_llm_mode="judge_only",
        provider_cls=InvalidThenValidOpenRouterProvider,
    ):
        env = {
            "USE_LIVE_LLM": "true",
            "LIVE_LLM_MODE": live_llm_mode,
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
                return asyncio.run(analyze(AnalyzeRequest(question=question)))


if __name__ == "__main__":
    unittest.main()
