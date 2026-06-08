from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState


class RiskAnalystNode:
    """Identifies risks, limitations, and failure modes.

    Future Azure OpenAI integration point: replace the deterministic body with
    a model call that returns the same AgentOutput schema.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or MockLLMProvider()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        fallback_output = self._fallback_output(state)
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ Risk Analyst Agent. Focus only on risks, "
                "limitations, and failure modes. Be precise and evidence-aware."
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
        refs = [item.id for item in state.retrieved_context if item.relevance >= 0.8]
        return AgentOutput(
            agent="Risk Analyst Agent",
            role="Identifies risks, limitations, and failure modes.",
            stance="caution",
            recommendation="Proceed only with explicit risk gates and review criteria.",
            conclusion=(
                "The decision can be reasonable, but the main failure mode is "
                "acting before evidence quality, rollback criteria, and stakeholder "
                "impact are explicit."
            ),
            rationale=[
                "Retrieved risk notes identify unclear rollback criteria as a common failure mode.",
                "Comparable implementations favor phased rollout and checkpoints.",
            ],
            confidence_score=0.82,
            evidence_refs=refs[-2:] if len(refs) >= 2 else refs,
            missing_evidence=[
                "Quantified downside impact for the specific scenario.",
                "Operational owner for monitoring and rollback.",
            ],
            limitations=[
                "Mocked retrieval cannot validate whether all domain-specific hazards were found."
            ],
        )
