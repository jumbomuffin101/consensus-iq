from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, root_validator

AgentStance = Literal["support", "caution", "alternative", "synthesis"]
DisagreementKind = Literal[
    "conflicting_recommendation", "differing_confidence", "missing_evidence"
]
Severity = Literal["low", "medium", "high"]


class RetrievedContext(BaseModel):
    id: str = ""
    title: str
    source: str
    snippet: str
    url: str = ""
    relevance_score: float = Field(..., ge=0, le=1)
    citation_id: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @root_validator(skip_on_failure=True)
    def default_id_from_citation(cls, values: dict) -> dict:
        if not values.get("id"):
            values["id"] = values.get("citation_id", "")
        return values


class ReasoningTask(BaseModel):
    id: str
    description: str
    owner: str
    priority: Literal["low", "medium", "high"] = "medium"


class AgentOutput(BaseModel):
    agent: str
    role: str
    stance: AgentStance
    recommendation: str
    conclusion: str
    rationale: list[str]
    confidence_score: float = Field(..., ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Disagreement(BaseModel):
    topic: str
    kind: DisagreementKind
    severity: Severity
    positions: list[str]
    suggested_resolution: str


class ConsensusJudgment(BaseModel):
    consensus: str
    confidence_score: float = Field(..., ge=0, le=1)
    agreement_score: float = Field(..., ge=0, le=1)
    reasoning_summary: str


class ExecutionMetadata(BaseModel):
    execution_time_ms: int = 0
    retrieval_time_ms: int = 0
    agent_time_ms: int = 0
    consensus_time_ms: int = 0
    provider_used: str = "fast-deterministic"
    live_llm_mode: str = "off"
    openrouter_call_count: int = 0
    fallback_reason: str = ""


class ReasoningState(BaseModel):
    question: str
    scenario_label: str = "Custom"
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    reasoning_tasks: list[ReasoningTask] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    consensus: str = ""
    confidence_score: float = Field(default=0.0, ge=0, le=1)
    agreement_score: float = Field(default=0.0, ge=0, le=1)
    reasoning_summary: str = ""
    metadata: ExecutionMetadata = Field(default_factory=ExecutionMetadata)

    def upsert_agent_output(self, output: AgentOutput) -> "ReasoningState":
        next_outputs = [
            existing
            for existing in self.agent_outputs
            if existing.agent != output.agent
        ]
        next_outputs.append(output)
        return self.copy(update={"agent_outputs": next_outputs})

    def with_timing(self, field: str, elapsed_ms: int) -> "ReasoningState":
        metadata = self.metadata.copy(update={field: elapsed_ms})
        return self.copy(update={"metadata": metadata})
