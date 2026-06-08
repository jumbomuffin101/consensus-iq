from models.reasoning import ReasoningState, RetrievedContext


class RetrievalNode:
    """Mocked Microsoft Foundry IQ retrieval node.

    Future Azure integration point: replace __call__ with a Foundry IQ query
    that returns RetrievedContext records with source citations and relevance.
    """

    def __call__(self, state: ReasoningState) -> ReasoningState:
        return state.copy(update={"retrieved_context": retrieve_evidence(state.question)})


def retrieve_evidence(question: str) -> list[RetrievedContext]:
    """Mocked Foundry IQ retrieval adapter."""
    normalized = question.strip()
    return [
        RetrievedContext(
            id="IQ-001",
            title="Internal policy and operating context",
            source="Mock Foundry IQ Knowledge Base",
            excerpt=f"Relevant operating constraints and success criteria for: {normalized}",
            relevance=0.92,
        ),
        RetrievedContext(
            id="IQ-002",
            title="Comparable implementation patterns",
            source="Mock Foundry IQ Knowledge Base",
            excerpt=(
                "Prior decisions show phased rollout, measurable checkpoints, "
                "and clear owner assignment reduce execution risk."
            ),
            relevance=0.86,
        ),
        RetrievedContext(
            id="IQ-003",
            title="Risk and mitigation notes",
            source="Mock Foundry IQ Knowledge Base",
            excerpt=(
                "Common risks include insufficient evidence, unclear rollback "
                "criteria, and stakeholder misalignment."
            ),
            relevance=0.81,
        ),
    ]
