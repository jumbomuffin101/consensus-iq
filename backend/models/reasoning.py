from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, root_validator

AgentStance = Literal["support", "caution", "alternative", "synthesis"]
DisagreementKind = Literal[
    "conflicting_recommendation", "differing_confidence", "missing_evidence"
]
Severity = Literal["low", "medium", "high"]
SourceQuality = Literal["strong", "partial", "weak"]
CustomDomain = Literal[
    "clinical_human",
    "pet_health",
    "enterprise_risk",
    "research_eval",
    "finance",
    "legal",
    "education",
    "general_decision",
    "unknown",
]
CustomIntent = Literal[
    "triage",
    "compare_options",
    "evaluate_risk",
    "summarize",
    "plan",
    "diagnose_problem",
    "other",
]
CustomUrgency = Literal["low", "moderate", "high", "emergency_possible"]


class CustomPromptIntake(BaseModel):
    domain: CustomDomain = "general_decision"
    intent: CustomIntent = "other"
    urgency: CustomUrgency = "low"
    missing_information: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    answer_style: str = "Concise practical decision guidance."
    confidence: float = Field(default=0.0, ge=0, le=1)


class KeyFinding(BaseModel):
    claim: str
    source_ids: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    summary: str = ""
    recommendation: str = ""
    key_findings: list[KeyFinding] = Field(default_factory=list)
    risks_or_limitations: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    source_quality: SourceQuality = "weak"
    provider_used: str = "fast-deterministic"
    live_llm_mode: str = "off"


class RetrievedContext(BaseModel):
    id: str = ""
    source_id: str = ""
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
        if not values.get("source_id"):
            values["source_id"] = values.get("id") or values.get("citation_id", "")
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
    custom_intake: CustomPromptIntake | None = None


class ReasoningState(BaseModel):
    question: str
    custom_intake: CustomPromptIntake | None = None
    scenario_label: str = "Custom"
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    reasoning_tasks: list[ReasoningTask] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    consensus: str = ""
    confidence_score: float = Field(default=0.0, ge=0, le=1)
    agreement_score: float = Field(default=0.0, ge=0, le=1)
    reasoning_summary: str = ""
    final_answer: FinalAnswer = Field(default_factory=FinalAnswer)
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
