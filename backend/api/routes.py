import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel, Field

from llm.base import LLMUsageTracker
from llm.factory import create_llm_provider, get_llm_provider_selection
from llm.mock import MockLLMProvider
from models.reasoning import (
    AgentOutput,
    Disagreement,
    ExecutionMetadata,
    FinalAnswer,
    RetrievedContext,
)
from reasoning.custom_intake import classify_custom_prompt, deterministic_custom_intake
from reasoning.domain import classify_domain
from reasoning.graph import ACTIVE_REASONING_ORDER, ConsensusReasoningGraph
from retrieval.factory import create_retrieval_provider

router = APIRouter()
logger = logging.getLogger("consensus_iq.api")


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AnalyzeResponse(BaseModel):
    consensus: str
    scenario_label: str = "Custom"
    confidence_score: float = Field(..., ge=0, le=1)
    agreement_score: float = Field(..., ge=0, le=1)
    reasoning_summary: str
    agent_outputs: list[AgentOutput]
    disagreements: list[Disagreement]
    sources: list[RetrievedContext]
    final_answer: FinalAnswer | None = None
    metadata: ExecutionMetadata | None = None


class ProviderStatusResponse(BaseModel):
    llm_provider: str
    retrieval_provider: str
    live_llm_enabled: bool
    live_llm_mode: str
    openrouter_configured: bool
    openrouter_model: str
    openrouter_base_url: str
    active_reasoning_order: list[str]
    azure_search_configured: bool


class ProviderHealthResponse(BaseModel):
    use_live_llm: bool
    live_llm_mode: str
    openrouter_configured: bool
    openrouter_model: str
    openrouter_base_url: str
    active_reasoning_order: list[str]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    selection = get_llm_provider_selection()
    usage_tracker = LLMUsageTracker()
    logger.info("query request received")
    logger.info(
        "OpenRouter config: OPENROUTER_API_KEY present=%s OPENROUTER_MODEL=%s LIVE_LLM_MODE=%s",
        selection.openrouter_api_key_present,
        selection.openrouter_model,
        selection.live_llm_mode,
    )

    deterministic_provider = MockLLMProvider()
    live_provider = create_llm_provider(usage_tracker)
    custom_intake = None
    if classify_domain(request.question) == "custom":
        deterministic_intake = deterministic_custom_intake(request.question)
        custom_intake = (
            classify_custom_prompt(request.question, live_provider)
            if selection.live_llm_mode != "off" and deterministic_intake.confidence < 0.45
            else deterministic_intake
        )
        logger.info(
            "custom prompt intake: domain=%s intent=%s urgency=%s confidence=%s",
            custom_intake.domain,
            custom_intake.intent,
            custom_intake.urgency,
            custom_intake.confidence,
        )
    specialist_provider = deterministic_provider
    judge_provider = deterministic_provider
    if selection.live_llm_mode == "all_agents":
        specialist_provider = live_provider
        judge_provider = live_provider
    elif selection.live_llm_mode == "judge_only":
        judge_provider = live_provider

    logger.info(
        "provider selected: specialists=%s judge=%s",
        specialist_provider.name,
        judge_provider.name,
    )
    if specialist_provider.name == "fast-deterministic" and judge_provider.name == "fast-deterministic":
        logger.info(
            "mock selected reason: %s",
            getattr(live_provider, "selection_reason", selection.mock_reason),
        )

    state = ConsensusReasoningGraph(
        specialist_provider=specialist_provider,
        judge_provider=judge_provider,
    ).invoke(request.question, custom_intake=custom_intake)
    provider_used = (
        f"specialists={specialist_provider.name}; judge={judge_provider.name}"
    )
    fallback_reason = usage_tracker.fallback_reason or getattr(
        live_provider, "selection_reason", ""
    )
    state = state.copy(
        update={
            "final_answer": state.final_answer.copy(
                update={
                    "provider_used": provider_used,
                    "live_llm_mode": selection.live_llm_mode,
                }
            ),
            "metadata": state.metadata.copy(
                update={
                    "provider_used": provider_used,
                    "live_llm_mode": selection.live_llm_mode,
                    "openrouter_call_count": usage_tracker.openrouter_call_count,
                    "fallback_reason": fallback_reason,
                    "custom_intake": custom_intake,
                }
            )
        }
    )
    logger.info(
        "OpenRouter usage: agents=%s call_count=%s fallback_reason=%s",
        usage_tracker.openrouter_agents,
        usage_tracker.openrouter_call_count,
        fallback_reason,
    )
    return AnalyzeResponse(
        consensus=state.consensus,
        scenario_label=state.scenario_label,
        confidence_score=state.confidence_score,
        agreement_score=state.agreement_score,
        reasoning_summary=state.reasoning_summary,
        agent_outputs=state.agent_outputs,
        disagreements=state.disagreements,
        sources=state.retrieved_context,
        final_answer=state.final_answer,
        metadata=state.metadata,
    )


@router.get("/provider-status", response_model=ProviderStatusResponse)
async def provider_status() -> ProviderStatusResponse:
    load_dotenv()
    selection = get_llm_provider_selection()
    azure_search_configured = all(
        os.getenv(name, "").strip()
        for name in [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
        ]
    )

    llm_provider = _display_llm_provider(create_llm_provider().name)
    retrieval_provider = _display_retrieval_provider(create_retrieval_provider().name)
    return ProviderStatusResponse(
        llm_provider=llm_provider,
        retrieval_provider=retrieval_provider,
        live_llm_enabled=selection.use_live_llm,
        live_llm_mode=selection.live_llm_mode,
        openrouter_configured=selection.openrouter_api_key_present,
        openrouter_model=selection.openrouter_model,
        openrouter_base_url=selection.openrouter_base_url,
        active_reasoning_order=ACTIVE_REASONING_ORDER,
        azure_search_configured=azure_search_configured,
    )


@router.get("/health/providers", response_model=ProviderHealthResponse)
async def provider_health() -> ProviderHealthResponse:
    selection = get_llm_provider_selection()
    return ProviderHealthResponse(
        use_live_llm=selection.use_live_llm,
        live_llm_mode=selection.live_llm_mode,
        openrouter_configured=selection.openrouter_api_key_present,
        openrouter_model=selection.openrouter_model,
        openrouter_base_url=selection.openrouter_base_url,
        active_reasoning_order=ACTIVE_REASONING_ORDER,
    )


def _display_llm_provider(provider_name: str) -> str:
    normalized = provider_name.lower()
    if "azure" in normalized:
        return "AzureOpenAI"
    if "openrouter" in normalized:
        return "OpenRouter"
    return "FastDeterministic"


def _display_retrieval_provider(provider_name: str) -> str:
    normalized = provider_name.lower()
    if "azure" in normalized:
        return "Azure AI Search"
    if normalized.startswith("foundry"):
        return "Foundry IQ"
    return "Mock"
