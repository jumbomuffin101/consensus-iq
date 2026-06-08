from typing import Literal

from pydantic import BaseModel, Field

AgentStance = Literal["support", "caution", "alternative", "synthesis"]
DisagreementKind = Literal[
    "conflicting_recommendation", "differing_confidence", "missing_evidence"
]
Severity = Literal["low", "medium", "high"]


class RetrievedContext(BaseModel):
    id: str
    title: str
    source: str
    excerpt: str
    relevance: float = Field(..., ge=0, le=1)


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


class ReasoningState(BaseModel):
    question: str
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    reasoning_tasks: list[ReasoningTask] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    consensus: str = ""
    confidence_score: float = Field(default=0.0, ge=0, le=1)
    agreement_score: float = Field(default=0.0, ge=0, le=1)
    reasoning_summary: str = ""

    def upsert_agent_output(self, output: AgentOutput) -> "ReasoningState":
        next_outputs = [
            existing
            for existing in self.agent_outputs
            if existing.agent != output.agent
        ]
        next_outputs.append(output)
        return self.copy(update={"agent_outputs": next_outputs})
