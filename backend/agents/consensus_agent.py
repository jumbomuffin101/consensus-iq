from models.reasoning import ReasoningState
from reasoning.disagreement import DisagreementDetector


class ConsensusJudgeNode:
    """Synthesizes agent outputs into a transparent final recommendation.

    Future Azure OpenAI integration point: use the model only after deterministic
    disagreement detection has produced auditable inputs for the final judgment.
    """

    def __init__(self, detector: DisagreementDetector | None = None) -> None:
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

        consensus = (
            "Consensus recommendation: proceed with a phased, evidence-tracked "
            f"approach to '{state.question}'. The evidence analyst supports the "
            "direction, the risk analyst requires explicit gates, and the "
            "alternatives analyst recommends limiting initial scope before broader "
            "commitment."
        )

        return state.copy(
            update={
                "disagreements": disagreements,
                "consensus": consensus,
                "confidence_score": confidence_score,
                "agreement_score": agreement_score,
                "reasoning_summary": self._build_reasoning_summary(
                    state, disagreements
                ),
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
