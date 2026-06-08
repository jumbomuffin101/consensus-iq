from retrieval.foundry import EvidenceItem


def planner_agent(question: str, evidence: list[EvidenceItem]) -> dict[str, object]:
    return {
        "question": question,
        "objective": "Produce a decision-ready consensus answer grounded in retrieved evidence.",
        "steps": [
            "Validate the question scope.",
            "Extract evidence-backed claims.",
            "Identify operational risks.",
            "Compare alternatives.",
            "Synthesize agreement and disagreement.",
        ],
        "evidence_ids": [item.id for item in evidence],
    }
