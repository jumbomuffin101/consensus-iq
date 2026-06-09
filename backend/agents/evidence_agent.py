from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState
from reasoning.domain import bounded_score, build_domain_profile


class EvidenceAnalystNode:
    """Evaluates evidence quality and supporting rationale.

    Future Azure OpenAI integration point: ground the prompt in retrieved_context
    and require JSON matching AgentOutput.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or MockLLMProvider()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        fallback_output = self._fallback_output(state)
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ Evidence Analyst Agent. Focus on "
                "supporting evidence, justification, and whether the retrieved "
                "context actually supports the recommendation. Cite retrieved "
                "context by citation_id in evidence_refs."
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
        refs = [item.citation_id for item in state.retrieved_context]
        domain_content = {
            "clinical": {
                "recommendation": "Favor MRI or equivalent intracranial imaging before LP when focal seizure raises structural concern.",
                "conclusion": (
                    "The supporting rationale is diagnostic sequencing: focal new-onset seizure increases concern for structural pathology, and imaging can reduce LP-related safety risk."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} aligns the question with diagnostic priority.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports checkpointed sequencing rather than immediate invasive testing.",
                    f"{refs[2] if len(refs) > 2 else 'S3'} identifies insufficient evidence and safety criteria as decision risks.",
                ],
                "missing": ["Patient-specific contraindications and local clinical guideline citation."],
            },
            "enterprise": {
                "recommendation": "Support controlled AI use only through approved tools with data-protection terms.",
                "conclusion": (
                    "The evidence supports a governed access model rather than unrestricted public-tool use for confidential documents."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} connects the decision to operating constraints.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports measurable checkpoints and owner assignment.",
                    f"{refs[2] if len(refs) > 2 else 'S3'} highlights stakeholder misalignment if policy is vague.",
                ],
                "missing": ["Current client contract language and vendor data-retention guarantees."],
            },
            "cybersecurity": {
                "recommendation": "Base the recommendation on verified incident scope and compliance evidence.",
                "conclusion": (
                    "The available context supports a controlled response with investigation milestones before final operational decisions."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} provides operating constraints.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports owner assignment and measurable response checkpoints.",
                ],
                "missing": ["Forensic evidence, affected data classification, and applicable notification rules."],
            },
            "research": {
                "recommendation": "Use an LLM grader as one signal, not the sole evaluator, until validity is demonstrated.",
                "conclusion": (
                    "The supporting rationale favors triangulation: concept-map evaluation needs reliability checks, rubric alignment, and bias assessment."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} anchors the decision in success criteria.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports measurable checkpoints before full adoption.",
                    f"{refs[2] if len(refs) > 2 else 'S3'} highlights insufficient evidence as a known risk.",
                ],
                "missing": ["Validation set results, expert-grader comparison, and rubric-level error analysis."],
            },
            "custom": {
                "recommendation": "Use the retrieved context to define criteria before making a broad recommendation.",
                "conclusion": (
                    "The evidence is useful for framing the decision, but more question-specific sources would improve certainty."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} is the highest-relevance retrieved context.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports measured checkpoints.",
                ],
                "missing": ["More domain-specific citations and explicit success metrics."],
            },
        }[profile.domain]
        confidence = bounded_score(
            0.42 + (profile.evidence_quality * 0.45) + (len(refs) * 0.025) - (profile.ambiguity * 0.12)
        )
        return AgentOutput(
            agent="Evidence Analyst Agent",
            role="Evaluates evidence and separates grounded claims from assumptions.",
            stance="support",
            recommendation=domain_content["recommendation"],
            conclusion=domain_content["conclusion"],
            rationale=domain_content["rationale"],
            confidence_score=confidence,
            evidence_refs=refs,
            missing_evidence=domain_content["missing"],
            limitations=[
                f"Fallback sources are mock Foundry IQ-style citations; live sources should verify {profile.primary_focus}."
            ],
        )
