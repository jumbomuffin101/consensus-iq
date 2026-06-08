from retrieval.foundry import EvidenceItem


def alternatives_agent(
    question: str, evidence: list[EvidenceItem], plan: dict[str, object]
) -> dict[str, object]:
    return {
        "agent": "Alternative Solutions Agent",
        "role": "Tests whether a different approach could satisfy the same goal.",
        "stance": "alternative",
        "summary": (
            "A smaller pilot or limited-scope experiment may produce enough "
            "signal before committing to a broader decision."
        ),
        "confidence": 0.82,
        "evidence_refs": [evidence[1].id],
    }
