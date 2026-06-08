from retrieval.foundry import EvidenceItem


def risk_agent(
    question: str, evidence: list[EvidenceItem], plan: dict[str, object]
) -> dict[str, object]:
    return {
        "agent": "Risk Analyst Agent",
        "role": "Identifies operational, strategic, and evidence-quality risks.",
        "stance": "caution",
        "summary": (
            "The proposal is plausible, but should proceed only with explicit "
            "success metrics, a review checkpoint, and rollback criteria."
        ),
        "confidence": 0.84,
        "evidence_refs": [evidence[1].id, evidence[2].id],
    }
