from models.reasoning import AgentOutput, ReasoningState


class EvidenceAnalystNode:
    """Evaluates evidence quality and supporting rationale.

    Future Azure OpenAI integration point: ground the prompt in retrieved_context
    and require JSON matching AgentOutput.
    """

    def __call__(self, state: ReasoningState) -> ReasoningState:
        refs = [item.id for item in state.retrieved_context]
        output = AgentOutput(
            agent="Evidence Analyst Agent",
            role="Evaluates evidence and separates grounded claims from assumptions.",
            stance="support",
            recommendation="Support a phased evidence-tracked decision.",
            conclusion=(
                "The retrieved context supports moving forward in a measured way, "
                "especially where success criteria and checkpoints are defined."
            ),
            rationale=[
                "The highest-relevance context directly matches the user's question.",
                "Comparable patterns support phased execution with measurable checkpoints.",
                "Risk notes provide conditions that can be incorporated into the final decision.",
            ],
            confidence_score=0.9,
            evidence_refs=refs,
            missing_evidence=[
                "Live source citations from Microsoft Foundry IQ are not connected yet."
            ],
            limitations=[
                "Evidence is mocked, so source authority and recency cannot be scored."
            ],
        )
        return state.upsert_agent_output(output)
