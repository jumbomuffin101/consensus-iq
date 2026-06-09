from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState
from reasoning.domain import bounded_score, build_domain_profile


class RiskAnalystNode:
    """Identifies risks, limitations, and failure modes.

    Future Azure OpenAI integration point: replace the deterministic body with
    a model call that returns the same AgentOutput schema.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or MockLLMProvider()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        fallback_output = self._fallback_output(state)
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ Risk Analyst Agent. Focus only on risks, "
                "limitations, and failure modes. Be precise and evidence-aware. "
                "Cite retrieved context by citation_id in evidence_refs."
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
            if item.relevance_score >= 0.8
        ]
        domain_content = {
            "clinical": {
                "recommendation": "Do not proceed to lumbar puncture until imaging-related safety risks are addressed.",
                "conclusion": (
                    "The major risk is missing an intracranial mass, hemorrhage, or other structural cause of a new focal seizure before LP. "
                    "The safer path is to prioritize diagnostic imaging and urgent stabilization before invasive testing."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} frames the decision around diagnostic priority and patient safety.",
                    f"{refs[-1] if refs else 'S3'} highlights risk controls before irreversible or invasive steps.",
                ],
                "missing": [
                    "Neurologic exam, papilledema status, anticoagulation status, fever, and immunocompromise.",
                    "Local emergency or neurology guideline for imaging before LP.",
                ],
            },
            "enterprise": {
                "recommendation": "Restrict public AI use for confidential client documents until governance and controls exist.",
                "conclusion": (
                    "The principal risks are client confidentiality loss, contractual breach, regulatory exposure, and unclear accountability for model-retained data."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} supports treating the question as a governance and confidentiality decision.",
                    f"{refs[-1] if refs else 'S3'} points to stakeholder and rollback risks if policy is unclear.",
                ],
                "missing": [
                    "Approved vendor list, contractual data-processing terms, and client consent requirements.",
                    "Incident response owner for accidental upload or data leakage.",
                ],
            },
            "cybersecurity": {
                "recommendation": "Prioritize containment and investigation before expanding access or normal operations.",
                "conclusion": (
                    "The main failure mode is acting before scope, exposure, and compliance obligations are understood."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} supports containment-first decision criteria.",
                    f"{refs[-1] if refs else 'S3'} indicates risk of unclear ownership and rollback criteria.",
                ],
                "missing": [
                    "Known blast radius, affected systems, and evidence preservation status.",
                    "Regulatory reporting thresholds and customer notification requirements.",
                ],
            },
            "research": {
                "recommendation": "Do not rely on a single LLM grader without calibration, human review, and bias checks.",
                "conclusion": (
                    "The key risk is measurement error: a single model grader can encode rubric drift, prompt sensitivity, and systematic bias."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} grounds the question in evaluation criteria and quality controls.",
                    f"{refs[-1] if refs else 'S3'} highlights evidence limitations and review checkpoints.",
                ],
                "missing": [
                    "Inter-rater reliability against expert graders.",
                    "Bias analysis across student groups and concept-map styles.",
                ],
            },
            "custom": {
                "recommendation": "Treat the decision as conditional until risk owners and failure criteria are explicit.",
                "conclusion": (
                    "The main risk is overcommitting before the decision criteria, downside exposure, and reversal plan are clear."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} provides the strongest available context.",
                    f"{refs[-1] if refs else 'S3'} points to risk controls and mitigation needs.",
                ],
                "missing": [
                    "Decision owner, success threshold, and rollback criteria.",
                    "Quantified downside if the recommendation is wrong.",
                ],
            },
        }[profile.domain]
        confidence = bounded_score(
            0.52 + (profile.evidence_quality * 0.32) - (profile.ambiguity * 0.18)
        )
        return AgentOutput(
            agent="Risk Analyst Agent",
            role="Identifies risks, limitations, and failure modes.",
            stance="caution",
            recommendation=domain_content["recommendation"],
            conclusion=domain_content["conclusion"],
            rationale=domain_content["rationale"],
            confidence_score=confidence,
            evidence_refs=refs[-2:] if len(refs) >= 2 else refs,
            missing_evidence=domain_content["missing"],
            limitations=[
                f"Fallback reasoning is domain-adapted for {profile.primary_focus}, but live Foundry IQ should verify source completeness."
            ],
        )
