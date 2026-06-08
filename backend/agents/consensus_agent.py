from retrieval.foundry import EvidenceItem


def consensus_agent(
    question: str,
    evidence: list[EvidenceItem],
    plan: dict[str, object],
    agent_outputs: list[dict[str, object]],
) -> dict[str, object]:
    avg_confidence = sum(float(output["confidence"]) for output in agent_outputs) / len(
        agent_outputs
    )
    agreement_score = 0.87
    confidence = round((avg_confidence * 0.65) + (agreement_score * 0.35), 2)

    return {
        "consensus": (
            "Consensus recommendation: proceed with a phased, evidence-tracked "
            f"approach to '{question}'. The agents agree that the decision is "
            "directionally supported, but confidence depends on defining measurable "
            "success criteria and reviewing outcomes before full rollout."
        ),
        "confidence": confidence,
        "agreement_score": agreement_score,
        "agent_outputs": agent_outputs,
        "disagreements": [
            {
                "topic": "Rollout scope",
                "positions": [
                    "Evidence Analyst supports moving forward with a structured plan.",
                    "Risk Analyst recommends tighter checkpoints before broad adoption.",
                    "Alternative Solutions Agent prefers a smaller pilot first.",
                ],
                "severity": "medium",
            },
            {
                "topic": "Evidence completeness",
                "positions": [
                    "Current mocked retrieval is enough for an MVP recommendation.",
                    "Production use should require live Foundry IQ citations and source quality scoring.",
                ],
                "severity": "low",
            },
        ],
    }
