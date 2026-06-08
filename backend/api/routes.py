from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.alternatives_agent import alternatives_agent
from agents.consensus_agent import consensus_agent
from agents.evidence_agent import evidence_agent
from agents.planner import planner_agent
from agents.risk_agent import risk_agent
from retrieval.foundry import retrieve_evidence

router = APIRouter()


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AgentOutput(BaseModel):
    agent: str
    role: str
    stance: Literal["support", "caution", "alternative", "synthesis"]
    summary: str
    confidence: float = Field(..., ge=0, le=1)
    evidence_refs: list[str]


class Disagreement(BaseModel):
    topic: str
    positions: list[str]
    severity: Literal["low", "medium", "high"]


class AnalyzeResponse(BaseModel):
    consensus: str
    confidence: float = Field(..., ge=0, le=1)
    agreement_score: float = Field(..., ge=0, le=1)
    agent_outputs: list[AgentOutput]
    disagreements: list[Disagreement]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    evidence = retrieve_evidence(request.question)
    plan = planner_agent(request.question, evidence)

    agent_outputs = [
        risk_agent(request.question, evidence, plan),
        evidence_agent(request.question, evidence, plan),
        alternatives_agent(request.question, evidence, plan),
    ]

    consensus = consensus_agent(request.question, evidence, plan, agent_outputs)
    return AnalyzeResponse(**consensus)
