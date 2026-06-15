import logging
import os

from dotenv import load_dotenv

from config import azure_openai_configured, live_llm_enabled, prefer_azure_openai
from grounding.citations import (
    strip_invalid_citations,
    validate_citations,
    validate_state_citations,
)
from llm.openrouter_client import OpenRouterGroundedClient
from models.reasoning import ReasoningState


logger = logging.getLogger("consensus_iq.grounding")


async def apply_optional_openrouter_grounding(state: ReasoningState) -> ReasoningState:
    """Optionally refine the deterministic answer with OpenRouter.

    OPENROUTER_API_KEY is the switch. Without it, ConsensusIQ stays on the
    deterministic/mock path. Any OpenRouter failure returns the deterministic
    state after citation validation.
    """

    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        logger.info("reasoning provider used: mock")
        return validate_state_citations(state)

    if azure_openai_configured() and prefer_azure_openai():
        provider_name = "azure" if live_llm_enabled() else "mock"
        logger.info("reasoning provider used: %s", provider_name)
        return validate_state_citations(state)

    if not state.retrieved_context:
        logger.info("reasoning provider used: mock; no retrieved context to ground")
        return validate_state_citations(state)

    client = OpenRouterGroundedClient.from_env()
    if client is None:
        logger.info("reasoning provider used: mock")
        return validate_state_citations(state)

    deterministic_state = validate_state_citations(state)
    deterministic_answer = {
        "consensus": deterministic_state.consensus,
        "reasoning_summary": deterministic_state.reasoning_summary,
        "confidence_score": deterministic_state.confidence_score,
        "agreement_score": deterministic_state.agreement_score,
        "available_sources": deterministic_state.citation_validity.available_sources,
    }

    try:
        payload = await client.complete_grounded_json(
            question=state.question,
            retrieved_context=state.retrieved_context,
            deterministic_answer=deterministic_answer,
            agent_name="grounded consensus",
        )
    except Exception:
        logger.info("reasoning provider used: mock; OpenRouter fallback applied")
        return deterministic_state

    refined_state = _merge_grounded_payload(deterministic_state, payload)
    validity = validate_citations(
        " ".join([refined_state.consensus, refined_state.reasoning_summary]),
        refined_state.retrieved_context,
    )

    if validity.valid:
        logger.info("reasoning provider used: openrouter")
        return refined_state.copy(update={"citation_validity": validity})

    try:
        payload = await client.complete_grounded_json(
            question=state.question,
            retrieved_context=state.retrieved_context,
            deterministic_answer=deterministic_answer,
            agent_name="grounded consensus",
            strict=True,
        )
        refined_state = _merge_grounded_payload(deterministic_state, payload)
        strict_validity = validate_citations(
            " ".join([refined_state.consensus, refined_state.reasoning_summary]),
            refined_state.retrieved_context,
        )
        if strict_validity.valid:
            logger.info("reasoning provider used: openrouter")
            return refined_state.copy(update={"citation_validity": strict_validity})
        validity = strict_validity
    except Exception:
        logger.info("OpenRouter citation repair failed; sanitizing invalid citations")

    logger.warning(
        "citation validation failures sanitized invalid_citations=%s",
        validity.invalid_citations,
    )
    return refined_state.copy(
        update={
            "consensus": strip_invalid_citations(
                refined_state.consensus, validity.invalid_citations
            ),
            "reasoning_summary": strip_invalid_citations(
                refined_state.reasoning_summary, validity.invalid_citations
            ),
            "citation_validity": validity,
        }
    )


def _merge_grounded_payload(
    state: ReasoningState, payload: dict[str, object]
) -> ReasoningState:
    consensus = payload.get("consensus")
    reasoning_summary = payload.get("reasoning_summary")

    updates: dict[str, object] = {}
    if isinstance(consensus, str) and consensus.strip():
        updates["consensus"] = consensus.strip()
    if isinstance(reasoning_summary, str) and reasoning_summary.strip():
        updates["reasoning_summary"] = reasoning_summary.strip()

    return state.copy(update=updates) if updates else state
