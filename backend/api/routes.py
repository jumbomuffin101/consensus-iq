import os

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel, Field

from grounding.openrouter_grounding import apply_optional_openrouter_grounding
from llm.factory import create_llm_provider
from models.reasoning import (
    AgentOutput,
    CitationValidity,
    Disagreement,
    ExecutionMetadata,
    RetrievedContext,
)
from reasoning.graph import analyze_question
from retrieval.factory import create_retrieval_provider

router = APIRouter()


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
    citation_validity: CitationValidity | None = None
    metadata: ExecutionMetadata | None = None


class ProviderStatusResponse(BaseModel):
    llm_provider: str
    retrieval_provider: str
    live_llm_enabled: bool
    openrouter_configured: bool
    azure_search_configured: bool


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    state = analyze_question(request.question)
    state = await apply_optional_openrouter_grounding(state)
    return AnalyzeResponse(
        consensus=state.consensus,
        scenario_label=state.scenario_label,
        confidence_score=state.confidence_score,
        agreement_score=state.agreement_score,
        reasoning_summary=state.reasoning_summary,
        agent_outputs=state.agent_outputs,
        disagreements=state.disagreements,
        sources=state.retrieved_context,
        citation_validity=state.citation_validity,
        metadata=state.metadata,
    )


@router.get("/provider-status", response_model=ProviderStatusResponse)
async def provider_status() -> ProviderStatusResponse:
    load_dotenv()
    live_llm_enabled = os.getenv("USE_LIVE_LLM", "false").strip().lower() == "true"
    openrouter_configured = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    azure_search_configured = all(
        os.getenv(name, "").strip()
        for name in [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
        ]
    )

    llm_provider = _display_llm_provider(create_llm_provider().name)
    if openrouter_configured and llm_provider == "FastDeterministic":
        llm_provider = "FastDeterministic + OpenRouterGrounding"
    retrieval_provider = _display_retrieval_provider(create_retrieval_provider().name)
    return ProviderStatusResponse(
        llm_provider=llm_provider,
        retrieval_provider=retrieval_provider,
        live_llm_enabled=live_llm_enabled,
        openrouter_configured=openrouter_configured,
        azure_search_configured=azure_search_configured,
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
