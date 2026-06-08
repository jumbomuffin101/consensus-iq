from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    title: str
    source: str
    excerpt: str
    relevance: float


def retrieve_evidence(question: str) -> list[EvidenceItem]:
    """Mocked Foundry IQ retrieval adapter.

    Replace this function with Azure AI Foundry / Foundry IQ retrieval while
    preserving the EvidenceItem contract used by downstream agents.
    """
    normalized = question.strip()
    return [
        EvidenceItem(
            id="IQ-001",
            title="Internal policy and operating context",
            source="Mock Foundry IQ Knowledge Base",
            excerpt=f"Relevant operating constraints and success criteria for: {normalized}",
            relevance=0.92,
        ),
        EvidenceItem(
            id="IQ-002",
            title="Comparable implementation patterns",
            source="Mock Foundry IQ Knowledge Base",
            excerpt="Prior decisions show phased rollout, measurable checkpoints, and clear owner assignment reduce execution risk.",
            relevance=0.86,
        ),
        EvidenceItem(
            id="IQ-003",
            title="Risk and mitigation notes",
            source="Mock Foundry IQ Knowledge Base",
            excerpt="Common risks include insufficient evidence, unclear rollback criteria, and stakeholder misalignment.",
            relevance=0.81,
        ),
    ]
