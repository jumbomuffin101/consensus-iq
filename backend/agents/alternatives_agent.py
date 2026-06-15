from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState
from reasoning.domain import bounded_score, build_domain_profile, prompt_injection_risk
from reasoning.general_decision import build_general_decision_frame


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
            agent_name="alternatives analyst",
        )
        try:
            output = AgentOutput.parse_obj(payload)
        except Exception:
            output = fallback_output

        return state.upsert_agent_output(output)

    def _fallback_output(self, state: ReasoningState) -> AgentOutput:
        profile = build_domain_profile(state)
        question = state.question.lower()
        refs = [item.citation_id for item in state.retrieved_context]
        if profile.domain == "clinical" and any(
            term in question for term in ["stroke", "thrombolytic", "aphasia", "weakness"]
        ):
            domain_content = {
                "recommendation": "Compare thrombolysis eligibility with thrombectomy evaluation, supportive care, and transfer to a stroke-capable center.",
                "conclusion": (
                    "The viable alternatives are not treatment versus no treatment in the abstract; they depend on imaging, vessel-occlusion suspicion, contraindications, and time-to-treatment logistics."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} supports urgent evaluation rather than delay.",
                    "If large-vessel occlusion is suspected, transfer or thrombectomy evaluation can run alongside thrombolysis screening.",
                ],
                "missing": ["Stroke severity, vascular imaging availability, thrombectomy eligibility, and transfer time."],
            }
        elif profile.domain == "clinical":
            domain_content = {
                "recommendation": "Consider CT first if MRI is delayed, while preserving the principle of imaging before LP when elevated intracranial risk is plausible.",
                "conclusion": (
                    "The main alternative is not LP-first; it is selecting the fastest appropriate imaging path based on urgency, availability, and contraindications."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S2'} supports adapting the sequence through measurable checkpoints.",
                    "If meningitis is strongly suspected and imaging is delayed, empiric treatment timing may become a parallel concern.",
                ],
                "missing": ["MRI availability, CT access, infection signs, and urgency of antimicrobial therapy."],
            }
        elif profile.domain == "cybersecurity":
            domain_content = {
                "recommendation": "Compare device isolation, remote wipe, legal hold, employee interview, customer notification, and credential rotation as separate tracks.",
                "conclusion": (
                    "A single next step is too narrow: containment and evidence preservation should start immediately, while notification decisions wait for scope unless law or contract imposes a clock."
                ),
                "rationale": [
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports forensic handling as its own workstream.",
                    f"{refs[2] if len(refs) > 2 else 'S3'} supports tying disclosure to jurisdiction and exposure evidence.",
                ],
                "missing": ["Whether remote wipe would destroy needed evidence and whether notification deadlines have already started."],
            }
        elif profile.domain == "research":
            domain_content = {
                "recommendation": "Compare single-model grading with human grading, LLM ensemble scoring, rubric-constrained scoring, and audit sampling.",
                "conclusion": (
                    "The strongest alternative is a hybrid design: use LLMs for draft scoring or feedback, then route low-confidence, borderline, or appealed essays to humans."
                ),
                "rationale": [
                    f"{refs[2] if len(refs) > 2 else 'S3'} supports LLM assistance with calibration and human adjudication.",
                    "Multiple graders or audit sampling can reveal model-specific bias and prompt sensitivity.",
                ],
                "missing": ["Cost and latency tradeoff for ensemble or human-reviewed grading."],
            }
        elif profile.domain == "finance":
            domain_content = {
                "recommendation": "Compare a diversified index fund, a small satellite AI-stock position, a cash reserve, and debt or tuition funding before buying.",
                "conclusion": (
                    "The best alternative preserves upside exposure without making one speculative company determine the student's financial resilience."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} supports avoiding uncompensated single-company risk.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports sizing investment around liquidity needs.",
                ],
                "missing": ["Target allocation, expected expenses, and whether the money is needed within one to three years."],
            }
        elif profile.domain == "enterprise":
            if "replace software engineers" in question or "every company" in question:
                domain_content = {
                    "recommendation": "Compare AI-assisted engineering, bounded autonomous agents, vendor tools, and human-led delivery instead of total replacement.",
                    "conclusion": (
                        "The practical alternative is augmentation with measured quality gates: let agents handle constrained tasks while engineers retain architecture, review, accountability, and incident response."
                    ),
                    "rationale": [
                        f"{refs[0] if refs else 'S1'} supports monitored AI use cases with owners.",
                        f"{refs[1] if len(refs) > 1 else 'S2'} supports evaluating continuity and quality before workforce replacement.",
                    ],
                    "missing": ["Which engineering tasks are repetitive, safety-critical, regulated, or customer-facing."],
                }
            else:
                domain_content = {
                    "recommendation": "Use private enterprise AI, redaction workflows, or no-AI handling instead of public AI tools for confidential documents.",
                    "conclusion": (
                        "The strongest alternative is a tiered policy: approved enterprise models for low-risk work, human-only handling for restricted client material."
                    ),
                    "rationale": [
                        f"{refs[0] if refs else 'S1'} supports controlled governance and owner assignment.",
                        "A redaction or synthetic-data workflow can preserve productivity without exposing raw client documents.",
                    ],
                    "missing": ["Data classification scheme and approved secure AI tooling options."],
                }
        else:
            frame = build_general_decision_frame(state.question)
            domain_content = {
                "recommendation": frame.alternative_approaches,
                "conclusion": (
                    f"This custom question should compare concrete options for '{frame.topic}' instead of inheriting a preset recommendation."
                ),
                "rationale": [
                    f"Objective: {frame.objective}",
                    frame.alternative_approaches,
                    f"{refs[0] if refs else 'S2'} supports checkpointed alternatives.",
                ],
                "missing": frame.missing_assumptions,
            }
        if not refs:
            domain_content["rationale"] = [
                "No strong retrieved evidence was available for this prompt.",
                "Alternative analysis remains useful, but options should be validated against better source coverage before acting.",
            ]
            domain_content["missing"] = [
                "No strong retrieved evidence matched this custom prompt.",
                *domain_content["missing"],
            ]

        confidence = bounded_score(
            0.36
            + (profile.evidence_quality * 0.32)
            - (profile.ambiguity * 0.1)
            + (0.06 if profile.domain != "custom" else 0)
            - (0.1 if not refs else 0)
            - (prompt_injection_risk(state.question) * 0.75)
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
