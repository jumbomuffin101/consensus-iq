from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState
from reasoning.domain import bounded_score, build_domain_profile, prompt_injection_risk
from reasoning.general_decision import build_general_decision_frame


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
            agent_name="risk analyst",
        )
        try:
            output = AgentOutput.parse_obj(payload)
        except Exception:
            output = fallback_output

        return state.upsert_agent_output(output)

    def _fallback_output(self, state: ReasoningState) -> AgentOutput:
        profile = build_domain_profile(state)
        question = state.question.lower()
        refs = [
            item.citation_id
            for item in state.retrieved_context
            if item.relevance_score >= 0.35
        ]
        if profile.domain == "clinical" and any(
            term in question for term in ["stroke", "thrombolytic", "aphasia", "weakness"]
        ):
            domain_content = {
                "recommendation": "Do not give thrombolytic therapy until hemorrhage is excluded and contraindications are checked.",
                "conclusion": (
                    "The time window supports urgent thrombolysis evaluation, but the safety risk is catastrophic bleeding if imaging, blood pressure, anticoagulant status, or recent bleeding history is missed."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} frames acute focal deficits as a time-critical stroke workflow.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} identifies contraindications that must be cleared before treatment.",
                ],
                "missing": [
                    "Non-contrast head CT or equivalent imaging result.",
                    "Anticoagulant use, platelet count, blood pressure, glucose, recent surgery, bleeding history, and exact last-known-well.",
                ],
            }
        elif profile.domain == "clinical":
            domain_content = {
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
            }
        elif profile.domain == "cybersecurity":
            domain_content = {
                "recommendation": "Immediately contain the laptop and accounts, preserve evidence, and escalate incident response before taking disciplinary or disclosure decisions.",
                "conclusion": (
                    "The highest-risk failure mode is losing forensic integrity or delaying containment while customer data remains exposed on an unmanaged personal device."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} treats customer data copied to a personal device as a security incident.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports device isolation, account review, and chain-of-custody controls.",
                ],
                "missing": [
                    "Whether the laptop is encrypted, managed, online, or already wiped.",
                    "Exact data fields copied, customer count, downstream sharing, and applicable breach notification thresholds.",
                ],
            }
        elif profile.domain == "research":
            domain_content = {
                "recommendation": "Do not let a single LLM be the sole grader for all essays until reliability, bias, and appeal processes are proven.",
                "conclusion": (
                    "The key risk is invalid assessment: one model can apply a rubric inconsistently, penalize particular writing styles, and leave students without defensible recourse."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} flags construct-irrelevant bias in automated grading.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} identifies prompt sensitivity and rubric drift as reliability risks.",
                ],
                "missing": [
                    "Inter-rater reliability versus trained human graders.",
                    "Bias analysis by course, language background, disability accommodation, and essay genre.",
                ],
            }
        elif profile.domain == "finance":
            domain_content = {
                "recommendation": "Do not invest all savings into one AI stock; preserve liquidity and avoid single-company concentration.",
                "conclusion": (
                    "The dominant risk is concentration: a 19-year-old may have long time horizon, but tuition needs, emergency cash, and a single volatile equity can create avoidable permanent-loss exposure."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} identifies single-stock concentration as a diversifiable risk.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} shows suitability depends on liquidity, debt, and time horizon, not age alone.",
                ],
                "missing": [
                    "Emergency fund, tuition obligations, debt, income stability, and total savings amount.",
                    "Risk tolerance and whether the student can afford a large drawdown.",
                ],
            }
        elif profile.domain == "enterprise":
            if "replace software engineers" in question or "every company" in question:
                domain_content = {
                    "recommendation": "Reject a universal replacement decision; require role-level evidence, accountability, and human review before any workforce automation.",
                    "conclusion": (
                        "The prompt asks for certainty on a broad workforce claim, but the risk profile varies by system criticality, code ownership, compliance, quality assurance, and organizational capability."
                    ),
                    "rationale": [
                        f"{refs[0] if refs else 'S1'} supports governance and accountable owners for AI adoption.",
                        f"{refs[1] if len(refs) > 1 else 'S2'} distinguishes augmentation from role replacement and flags continuity risks.",
                    ],
                    "missing": [
                        "Engineering quality metrics, incident ownership, regulated-system exposure, and security review capacity.",
                        "Evidence that AI agents can maintain, test, and be accountable for the specific software portfolio.",
                    ],
                }
            else:
                domain_content = {
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
                }
        else:
            frame = build_general_decision_frame(state.question)
            domain_content = {
                "recommendation": frame.recommendation,
                "conclusion": frame.key_risk,
                "rationale": [
                    f"Objective: {frame.objective}",
                    frame.evidence_limitation,
                    f"{refs[0] if refs else 'S1'} provides the strongest available decision-support context.",
                ],
                "missing": frame.missing_assumptions,
            }
        if not refs:
            domain_content["rationale"] = [
                "No strong retrieved evidence was available for this prompt.",
                "Risk analysis is therefore based on decision structure, downside exposure, and missing-evidence constraints rather than source grounding.",
            ]
            domain_content["missing"] = [
                "No strong retrieved evidence found.",
                *domain_content["missing"],
            ]

        confidence = bounded_score(
            0.5
            + (profile.evidence_quality * 0.31)
            + (profile.risk_level * 0.04)
            - (profile.ambiguity * 0.16)
            - (0.12 if not refs else 0)
            - prompt_injection_risk(state.question)
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
