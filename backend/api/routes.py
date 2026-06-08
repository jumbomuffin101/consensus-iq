from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.reasoning import AgentOutput, Disagreement
from reasoning.graph import analyze_question

router = APIRouter()


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AnalyzeResponse(BaseModel):
    consensus: str
    confidence_score: float = Field(..., ge=0, le=1)
    agreement_score: float = Field(..., ge=0, le=1)
    reasoning_summary: str
    agent_outputs: list[AgentOutput]
    disagreements: list[Disagreement]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    state = analyze_question(request.question)
    return AnalyzeResponse(
        consensus=state.consensus,
        confidence_score=state.confidence_score,
        agreement_score=state.agreement_score,
        reasoning_summary=state.reasoning_summary,
        agent_outputs=state.agent_outputs,
        disagreements=state.disagreements,
    )
