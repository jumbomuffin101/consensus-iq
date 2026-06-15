import os
import unittest
from unittest.mock import AsyncMock, patch

from grounding.citations import validate_citations
from grounding.openrouter_grounding import apply_optional_openrouter_grounding
from llm.openrouter_client import OpenRouterGroundedClient
from models.reasoning import RetrievedContext, ReasoningState


def _state() -> ReasoningState:
    return ReasoningState(
        question="Should the organization allow public AI tools for confidential data?",
        consensus="Use approved tools with confidentiality controls.",
        reasoning_summary="The deterministic agents found governance and data-risk limits.",
        confidence_score=0.56,
        agreement_score=0.62,
        retrieved_context=[
            RetrievedContext(
                id="enterprise-ai-rmf",
                citation_id="S1",
                title="NIST AI Risk Management Framework",
                source="Azure AI Search / Foundry IQ Search Service",
                url="https://www.nist.gov/itl/ai-risk-management-framework",
                snippet="AI risk management should identify, measure, manage, and govern risk.",
                relevance_score=0.74,
            )
        ],
    )


class GroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_openrouter_not_called_without_api_key(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with patch.object(
                OpenRouterGroundedClient,
                "complete_grounded_json",
                new_callable=AsyncMock,
            ) as mocked_completion:
                result = await apply_optional_openrouter_grounding(_state())

        self.assertFalse(mocked_completion.called)
        self.assertEqual(result.consensus, "Use approved tools with confidentiality controls.")
        self.assertTrue(result.citation_validity.valid)

    async def test_openrouter_called_with_api_key_and_sources(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch.object(
                OpenRouterGroundedClient,
                "complete_grounded_json",
                new_callable=AsyncMock,
                return_value={
                    "consensus": "Use approved tools with confidentiality controls [S1].",
                    "reasoning_summary": "The available source supports governed AI use [S1].",
                },
            ) as mocked_completion:
                result = await apply_optional_openrouter_grounding(_state())

        self.assertTrue(mocked_completion.called)
        self.assertIn("[S1]", result.consensus)
        self.assertTrue(result.citation_validity.valid)

    def test_invalid_citations_are_detected(self) -> None:
        validity = validate_citations("The answer cites [S1] and [S9].", _state().retrieved_context)

        self.assertFalse(validity.valid)
        self.assertEqual(validity.invalid_citations, ["S9"])
        self.assertEqual(validity.available_sources, ["S1"])

    async def test_unsupported_answer_does_not_need_fabricated_citations(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch.object(
                OpenRouterGroundedClient,
                "complete_grounded_json",
                new_callable=AsyncMock,
                return_value={
                    "consensus": "The answer could not be verified from the available sources.",
                    "reasoning_summary": "Retrieved context is insufficient for the requested claim.",
                },
            ):
                result = await apply_optional_openrouter_grounding(_state())

        self.assertTrue(result.citation_validity.valid)
        self.assertEqual(result.citation_validity.invalid_citations, [])

    async def test_openrouter_failure_falls_back_to_deterministic_answer(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch.object(
                OpenRouterGroundedClient,
                "complete_grounded_json",
                new_callable=AsyncMock,
                side_effect=RuntimeError("provider down"),
            ):
                result = await apply_optional_openrouter_grounding(_state())

        self.assertEqual(result.consensus, "Use approved tools with confidentiality controls.")
        self.assertTrue(result.citation_validity.valid)

    async def test_invalid_openrouter_citation_triggers_single_repair(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch.object(
                OpenRouterGroundedClient,
                "complete_grounded_json",
                new_callable=AsyncMock,
                side_effect=[
                    {
                        "consensus": "Use approved tools [S9].",
                        "reasoning_summary": "Unsupported citation [S9].",
                    },
                    {
                        "consensus": "Use approved tools [S1].",
                        "reasoning_summary": "Supported by governance evidence [S1].",
                    },
                ],
            ) as mocked_completion:
                result = await apply_optional_openrouter_grounding(_state())

        self.assertEqual(mocked_completion.call_count, 2)
        self.assertTrue(result.citation_validity.valid)
        self.assertIn("[S1]", result.consensus)


if __name__ == "__main__":
    unittest.main()
