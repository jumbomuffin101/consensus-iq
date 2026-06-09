from agents.prompting import agent_outputs_payload, disagreements_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import ConsensusJudgment, ReasoningState
from reasoning.disagreement import DisagreementDetector


class ConsensusJudgeNode:
    """Synthesizes agent outputs into a transparent final recommendation.

    Future Azure OpenAI integration point: use the model only after deterministic
    disagreement detection has produced auditable inputs for the final judgment.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        detector: DisagreementDetector | None = None,
    ) -> None:
        self.provider = provider or MockLLMProvider()
        self.detector = detector or DisagreementDetector()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        disagreements = self.detector.detect(state.agent_outputs)
        agreement_score = self.detector.calculate_agreement_score(state.agent_outputs)
        avg_confidence = (
            sum(output.confidence_score for output in state.agent_outputs)
            / len(state.agent_outputs)
            if state.agent_outputs
            else 0.0
        )
        confidence_score = round((avg_confidence * 0.7) + (agreement_score * 0.3), 2)

        state_with_disagreements = state.copy(update={"disagreements": disagreements})
        fallback_judgment = ConsensusJudgment(
            consensus=(
                "Consensus recommendation: proceed with a phased, evidence-tracked "
                f"approach to '{state.question}'. The evidence analyst supports the "
                "direction, the risk analyst requires explicit gates, and the "
                "alternatives analyst recommends limiting initial scope before broader "
                "commitment. The recommendation is grounded in the retrieved sources "
                "cited by the specialist agents."
            ),
            confidence_score=confidence_score,
            agreement_score=agreement_score,
            reasoning_summary=self._build_reasoning_summary(
                state_with_disagreements, disagreements
            ),
        )
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ Consensus Judge. Synthesize independent "
                "specialist outputs and the disagreement report into a transparent, "
                "decision-ready recommendation. Preserve uncertainty. Use retrieved "
                "context citations already present in evidence_refs whenever making "
                "evidence-based claims."
            ),
            user_prompt=(
                f"Question: {state.question}\n\n"
                f"Retrieved context:\n{[item.dict() for item in state.retrieved_context]}\n\n"
                f"Agent outputs:\n{agent_outputs_payload(state)}\n\n"
                f"Disagreements:\n{disagreements_payload(state_with_disagreements)}\n\n"
                "Return JSON with keys: consensus, confidence_score, "
                "agreement_score, reasoning_summary. Scores must be numbers "
                "between 0 and 1."
            ),
            fallback=fallback_judgment.dict(),
        )
        try:
            judgment = ConsensusJudgment.parse_obj(payload)
        except Exception:
            judgment = fallback_judgment

        return state.copy(
            update={
                "disagreements": disagreements,
                "consensus": judgment.consensus,
                "confidence_score": judgment.confidence_score,
                "agreement_score": judgment.agreement_score,
                "reasoning_summary": judgment.reasoning_summary,
            }
        )

    def _build_reasoning_summary(self, state: ReasoningState, disagreements: list) -> str:
        task_summary = "; ".join(task.description for task in state.reasoning_tasks)
        disagreement_summary = (
            "; ".join(item.topic for item in disagreements)
            if disagreements
            else "No material disagreements detected."
        )
        return (
            f"Planner decomposed the question into: {task_summary}. Specialist "
            f"agents reviewed risk, evidence, and alternatives independently. "
            f"Detected disagreement areas: {disagreement_summary}."
        )
