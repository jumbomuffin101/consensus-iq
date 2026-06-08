from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState


class EvidenceAnalystNode:
    """Evaluates evidence quality and supporting rationale.

    Future Azure OpenAI integration point: ground the prompt in retrieved_context
    and require JSON matching AgentOutput.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or MockLLMProvider()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        fallback_output = self._fallback_output(state)
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ Evidence Analyst Agent. Focus on "
                "supporting evidence, justification, and whether the retrieved "
                "context actually supports the recommendation."
            ),
            user_prompt=(
                f"{state_context_payload(state)}\n\n"
                "Return one JSON object with keys: agent, role, stance, "
                "recommendation, conclusion, rationale, confidence_score, "
                "evidence_refs, missing_evidence, limitations."
            ),
            fallback=fallback_output.dict(),
        )
        try:
            output = AgentOutput.parse_obj(payload)
        except Exception:
            output = fallback_output

        return state.upsert_agent_output(output)

    def _fallback_output(self, state: ReasoningState) -> AgentOutput:
        refs = [item.id for item in state.retrieved_context]
        return AgentOutput(
            agent="Evidence Analyst Agent",
            role="Evaluates evidence and separates grounded claims from assumptions.",
            stance="support",
            recommendation="Support a phased evidence-tracked decision.",
            conclusion=(
                "The retrieved context supports moving forward in a measured way, "
                "especially where success criteria and checkpoints are defined."
            ),
            rationale=[
                "The highest-relevance context directly matches the user's question.",
                "Comparable patterns support phased execution with measurable checkpoints.",
                "Risk notes provide conditions that can be incorporated into the final decision.",
            ],
            confidence_score=0.9,
            evidence_refs=refs,
            missing_evidence=[
                "Live source citations from Microsoft Foundry IQ are not connected yet."
            ],
            limitations=[
                "Evidence is mocked, so source authority and recency cannot be scored."
            ],
        )
