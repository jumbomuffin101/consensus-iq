from models.reasoning import AgentOutput, ReasoningState


class AlternativesAnalystNode:
    """Proposes alternative interpretations and approaches.

    Future Azure OpenAI integration point: ask the model for counterarguments,
    exception cases, and alternate plans that still match AgentOutput.
    """

    def __call__(self, state: ReasoningState) -> ReasoningState:
        refs = [
            item.id for item in state.retrieved_context if "pattern" in item.title.lower()
        ]
        output = AgentOutput(
            agent="Alternative Solutions Agent",
            role="Tests whether other approaches could satisfy the same goal.",
            stance="alternative",
            recommendation="Run a limited pilot before broad commitment.",
            conclusion=(
                "A narrower experiment can preserve optionality while producing "
                "decision-quality evidence for the broader recommendation."
            ),
            rationale=[
                "Comparable patterns favor phased implementation.",
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
        return state.upsert_agent_output(output)
