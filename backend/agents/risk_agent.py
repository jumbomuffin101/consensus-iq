from models.reasoning import AgentOutput, ReasoningState


class RiskAnalystNode:
    """Identifies risks, limitations, and failure modes.

    Future Azure OpenAI integration point: replace the deterministic body with
    a model call that returns the same AgentOutput schema.
    """

    def __call__(self, state: ReasoningState) -> ReasoningState:
        refs = [item.id for item in state.retrieved_context if item.relevance >= 0.8]
        output = AgentOutput(
            agent="Risk Analyst Agent",
            role="Identifies risks, limitations, and failure modes.",
            stance="caution",
            recommendation="Proceed only with explicit risk gates and review criteria.",
            conclusion=(
                "The decision can be reasonable, but the main failure mode is "
                "acting before evidence quality, rollback criteria, and stakeholder "
                "impact are explicit."
            ),
            rationale=[
                "Retrieved risk notes identify unclear rollback criteria as a common failure mode.",
                "Comparable implementations favor phased rollout and checkpoints.",
            ],
            confidence_score=0.82,
            evidence_refs=refs[-2:] if len(refs) >= 2 else refs,
            missing_evidence=[
                "Quantified downside impact for the specific scenario.",
                "Operational owner for monitoring and rollback.",
            ],
            limitations=[
                "Mocked retrieval cannot validate whether all domain-specific hazards were found."
            ],
        )
        return state.upsert_agent_output(output)
