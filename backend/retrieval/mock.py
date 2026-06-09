from models.reasoning import RetrievedContext
from retrieval.base import BaseRetrievalProvider


class MockRetrievalProvider(BaseRetrievalProvider):
    """Local citation-ready retrieval provider used when Foundry IQ is unavailable."""

    name = "mock-foundry-iq"

    def retrieve(self, question: str) -> list[RetrievedContext]:
        normalized = question.strip()
        return [
            RetrievedContext(
                citation_id="S1",
                title="Mock Foundry IQ operating context",
                source="Mock Foundry IQ Knowledge Base",
                url="mock://foundry-iq/operating-context",
                snippet=(
                    "Mock source: relevant operating constraints and success "
                    f"criteria for: {normalized}"
                ),
                relevance_score=0.92,
            ),
            RetrievedContext(
                citation_id="S2",
                title="Mock comparable implementation patterns",
                source="Mock Foundry IQ Knowledge Base",
                url="mock://foundry-iq/implementation-patterns",
                snippet=(
                    "Mock source: prior decisions show phased rollout, measurable "
                    "checkpoints, and clear owner assignment reduce execution risk."
                ),
                relevance_score=0.86,
            ),
            RetrievedContext(
                citation_id="S3",
                title="Mock risk and mitigation notes",
                source="Mock Foundry IQ Knowledge Base",
                url="mock://foundry-iq/risk-mitigation",
                snippet=(
                    "Mock source: common risks include insufficient evidence, "
                    "unclear rollback criteria, and stakeholder misalignment."
                ),
                relevance_score=0.81,
            ),
        ]
