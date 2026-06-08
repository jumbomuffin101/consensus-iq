from retrieval.foundry import EvidenceItem


def evidence_agent(
    question: str, evidence: list[EvidenceItem], plan: dict[str, object]
) -> dict[str, object]:
    return {
        "agent": "Evidence Analyst Agent",
        "role": "Assesses retrieved evidence and separates grounded claims from assumptions.",
        "stance": "support",
        "summary": (
            "The mocked retrieval set supports a measured recommendation. The "
            "strongest evidence favors a phased rollout with tracked outcomes."
        ),
        "confidence": 0.9,
        "evidence_refs": [item.id for item in evidence],
    }
