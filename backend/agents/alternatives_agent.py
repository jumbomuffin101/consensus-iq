from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState


class AlternativesAnalystNode:
    """Proposes alternative interpretations and approaches.

    Future Azure OpenAI integration point: ask the model for counterarguments,
    exception cases, and alternate plans that still match AgentOutput.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or MockLLMProvider()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        fallback_output = self._fallback_output(state)
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ Alternatives Analyst Agent. Focus on "
                "alternative explanations, exception cases, and different viable "
                "approaches. Cite retrieved context by citation_id in evidence_refs."
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
        refs = [
            item.citation_id
            for item in state.retrieved_context
            if "pattern" in item.title.lower()
        ]
        return AgentOutput(
            agent="Alternative Solutions Agent",
            role="Tests whether other approaches could satisfy the same goal.",
            stance="alternative",
            recommendation="Run a limited pilot before broad commitment.",
            conclusion=(
                "A narrower experiment can preserve optionality while producing "
                "decision-quality evidence for the broader recommendation."
            ),
            rationale=[
                f"{refs[0] if refs else 'S2'} favors phased implementation.",
                "A pilot reduces exposure while validating the highest-impact assumptions.",
            ],
            confidence_score=0.78,
            evidence_refs=refs,
            missing_evidence=[
                "Clear threshold for when the pilot should expand, stop, or change direction."
            ],
            limitations=[
                "Alternative options are mocked and not generated from a live model search."
            ],
        )
