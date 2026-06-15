import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel, Field

from llm.factory import create_llm_provider, get_llm_provider_selection
from models.reasoning import (
    AgentOutput,
    Disagreement,
    ExecutionMetadata,
    RetrievedContext,
)
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
    metadata: ExecutionMetadata | None = None


class ProviderStatusResponse(BaseModel):
    llm_provider: str
    retrieval_provider: str
    live_llm_enabled: bool
    openrouter_configured: bool
    openrouter_model: str
    openrouter_base_url: str
    active_reasoning_order: list[str]
    azure_search_configured: bool


class ProviderHealthResponse(BaseModel):
    openrouter_configured: bool
    openrouter_model: str
    openrouter_base_url: str
    active_reasoning_order: list[str]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    selection = get_llm_provider_selection()
    logger.info("query request received")
    logger.info(
        "OpenRouter config: OPENROUTER_API_KEY present=%s OPENROUTER_MODEL=%s",
        selection.openrouter_api_key_present,
        selection.openrouter_model,
    )

    provider = create_llm_provider()
    logger.info("provider selected: %s", provider.name)
    if provider.name == "fast-deterministic":
        logger.info(
            "mock selected reason: %s",
            getattr(provider, "selection_reason", selection.mock_reason),
        )

    state = ConsensusReasoningGraph(provider).invoke(request.question)
    return AnalyzeResponse(
        consensus=state.consensus,
        scenario_label=state.scenario_label,
        confidence_score=state.confidence_score,
        agreement_score=state.agreement_score,
        reasoning_summary=state.reasoning_summary,
        agent_outputs=state.agent_outputs,
        disagreements=state.disagreements,
        sources=state.retrieved_context,
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
        live_llm_enabled=selection.live_llm_enabled,
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
