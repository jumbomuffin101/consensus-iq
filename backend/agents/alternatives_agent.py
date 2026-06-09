from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState
from reasoning.domain import bounded_score, build_domain_profile


class AlternativesAnalystNode:
    """Proposes alternative interpretations and approaches.

    Future Azure OpenAI integration point: ask the model for counterarguments,
    exception cases, and alternate plans that still match AgentOutput.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or MockLLMProvider()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        fallback_output = self._fallback_output(state)
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ Alternatives Analyst Agent. Focus on "
                "alternative explanations, exception cases, and different viable "
                "approaches. Cite retrieved context by citation_id in evidence_refs."
            ),
            user_prompt=(
                f"{state_context_payload(state)}\n\n"
                "Return one JSON object with keys: agent, role, stance, "
                "recommendation, conclusion, rationale, confidence_score, "
                "evidence_refs, missing_evidence, limitations."
            ),
            fallback=fallback_output.dict(),
        )
        try:
            output = AgentOutput.parse_obj(payload)
        except Exception:
            output = fallback_output

        return state.upsert_agent_output(output)

    def _fallback_output(self, state: ReasoningState) -> AgentOutput:
        profile = build_domain_profile(state)
        refs = [
            item.citation_id
            for item in state.retrieved_context
            if "pattern" in item.title.lower()
        ]
        domain_content = {
            "clinical": {
                "recommendation": "Consider CT first if MRI is delayed, while preserving the principle of imaging before LP when elevated intracranial risk is plausible.",
                "conclusion": (
                    "The main alternative is not LP-first; it is selecting the fastest appropriate imaging path based on urgency, availability, and contraindications."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S2'} supports adapting the sequence through measurable checkpoints.",
                    "If meningitis is strongly suspected and imaging is delayed, empiric treatment timing may become a parallel concern.",
                ],
                "missing": ["MRI availability, CT access, infection signs, and urgency of antimicrobial therapy."],
            },
            "enterprise": {
                "recommendation": "Use private enterprise AI, redaction workflows, or no-AI handling instead of public AI tools for confidential documents.",
                "conclusion": (
                    "The strongest alternative is a tiered policy: approved enterprise models for low-risk work, human-only handling for restricted client material."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S2'} supports phased implementation and owner assignment.",
                    "A redaction or synthetic-data workflow can preserve productivity without exposing raw client documents.",
                ],
                "missing": ["Data classification scheme and approved secure AI tooling options."],
            },
            "cybersecurity": {
                "recommendation": "Compare containment-first, monitor-only, and shutdown options against blast-radius evidence.",
                "conclusion": (
                    "A monitor-only path may preserve operations but is weak if exposure is active; shutdown may be justified for regulated or high-impact systems."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S2'} supports comparing options through checkpointed evidence.",
                    "Containment choices should vary by data sensitivity and observed attacker activity.",
                ],
                "missing": ["Current indicators of compromise and business-critical system dependencies."],
            },
            "research": {
                "recommendation": "Compare single LLM grading with ensemble grading, human adjudication, and rubric-constrained scoring.",
                "conclusion": (
                    "The practical alternative is hybrid evaluation: use LLMs for scalable preliminary scoring, then audit uncertain or high-impact cases."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S2'} supports staged adoption with measurable checkpoints.",
                    "Multiple graders or human adjudication can reveal model-specific bias and prompt sensitivity.",
                ],
                "missing": ["Cost and latency tradeoff for ensemble or human-reviewed grading."],
            },
            "custom": {
                "recommendation": "Compare at least one conservative, one reversible, and one high-commitment option before deciding.",
                "conclusion": (
                    "A custom question should not inherit a preset recommendation; the alternatives should be derived from the user's decision constraints."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S2'} supports checkpointed alternatives.",
                    "Explicit option comparison prevents premature convergence.",
                ],
                "missing": ["Available options, constraints, and stakeholder tolerance for risk."],
            },
        }[profile.domain]
        confidence = bounded_score(
            0.38 + (profile.evidence_quality * 0.32) - (profile.ambiguity * 0.1) + (0.05 if profile.domain != "custom" else 0)
        )
        return AgentOutput(
            agent="Alternative Solutions Agent",
            role="Tests whether other approaches could satisfy the same goal.",
            stance="alternative",
            recommendation=domain_content["recommendation"],
            conclusion=domain_content["conclusion"],
            rationale=domain_content["rationale"],
            confidence_score=confidence,
            evidence_refs=refs,
            missing_evidence=domain_content["missing"],
            limitations=[
                f"Fallback alternatives are domain-adapted for {profile.domain}; live LLM reasoning can enumerate more options."
            ],
        )
