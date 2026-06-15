from agents.prompting import state_context_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import AgentOutput, ReasoningState
from prompts.specialists import EVIDENCE_ANALYST_SYSTEM_PROMPT, SPECIALIST_OUTPUT_INSTRUCTIONS
from reasoning.domain import bounded_score, build_domain_profile, prompt_injection_risk
from reasoning.general_decision import build_general_decision_frame


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
            system_prompt=EVIDENCE_ANALYST_SYSTEM_PROMPT,
            user_prompt=(
                f"{state_context_payload(state)}\n\n"
                f"{SPECIALIST_OUTPUT_INSTRUCTIONS}"
            ),
            fallback=fallback_output.dict(),
            agent_name="evidence analyst",
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
                "recommendation": "Evaluate urgently for thrombolysis, but only after imaging excludes hemorrhage and eligibility criteria are satisfied.",
                "conclusion": (
                    "The strongest evidence signal is time sensitivity: a 45-minute focal deficit presentation is potentially treatable, yet the evidence also requires rapid exclusion of hemorrhage and contraindications."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} supports urgent stroke-team assessment and early-window thrombolysis evaluation.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} grounds the contraindication checklist.",
                    f"{refs[2] if len(refs) > 2 else 'S3'} warns against overconfidence when imaging or onset details are incomplete.",
                ],
                "missing": ["Imaging result, stroke severity score, and contraindication screen."],
            }
        elif profile.domain == "clinical":
            domain_content = {
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
            }
        elif profile.domain == "cybersecurity":
            domain_content = {
                "recommendation": "Treat the event as a likely data exposure and build the next actions around verified scope.",
                "conclusion": (
                    "The supporting evidence favors containment plus investigation: the organization needs facts about copied data, device status, and downstream sharing before deciding notifications and sanctions."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} grounds the event as a customer-data incident.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} supports forensic preservation and access review.",
                    f"{refs[2] if len(refs) > 2 else 'S3'} ties notification decisions to data type, jurisdiction, and exposure evidence.",
                ],
                "missing": ["Forensic timeline, data classification, jurisdiction map, and customer impact estimate."],
            }
        elif profile.domain == "research":
            domain_content = {
                "recommendation": "Use an LLM essay grader only as a calibrated aid, not as the sole grading authority.",
                "conclusion": (
                    "The evidence supports hybrid assessment because essay grading requires construct validity, rubric alignment, and fairness checks that a single model has not demonstrated by default."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} flags validity concerns for single-rater automated grading.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} names reliability checks that should precede replacement of human graders.",
                    f"{refs[2] if len(refs) > 2 else 'S3'} supports LLM use in triage or feedback roles with audits.",
                ],
                "missing": ["Validation set performance, appeals process, and subgroup error analysis."],
            }
        elif profile.domain == "finance":
            domain_content = {
                "recommendation": "Favor diversification and staged investing over committing all savings to a single AI stock.",
                "conclusion": (
                    "The evidence strongly supports avoiding all-in concentration because the student's liquidity needs and downside tolerance are more important than enthusiasm for one sector."
                ),
                "rationale": [
                    f"{refs[0] if refs else 'S1'} supports diversification as protection against unsystematic risk.",
                    f"{refs[1] if len(refs) > 1 else 'S2'} ties suitability to emergency savings, debt, tuition, and time horizon.",
                    f"{refs[2] if len(refs) > 2 else 'S3'} identifies valuation and hype-cycle risk in AI equities.",
                ],
                "missing": ["Current net worth, emergency fund, debt, tuition timeline, and investment objective."],
            }
        elif profile.domain == "enterprise":
            if "replace software engineers" in question or "every company" in question:
                domain_content = {
                    "recommendation": "Use AI agents to augment software work where evidence supports it; do not treat universal replacement as evidence-backed.",
                    "conclusion": (
                        "The evidence supports controlled adoption and operational review, not a one-size-fits-all claim that every company should replace engineers."
                    ),
                    "rationale": [
                        f"{refs[0] if refs else 'S1'} requires accountable owners and monitored AI use cases.",
                        f"{refs[1] if len(refs) > 1 else 'S2'} separates augmentation from role replacement and stresses service continuity.",
                    ],
                    "missing": ["Task-level productivity data, defect rates, security outcomes, and accountability model."],
                }
            else:
                domain_content = {
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
                }
        else:
            frame = build_general_decision_frame(state.question)
            domain_content = {
                "recommendation": frame.recommendation,
                "conclusion": (
                    f"The available evidence can frame '{frame.topic}', but it does not provide prompt-specific validation. "
                    f"{frame.evidence_limitation}"
                ),
                "rationale": [
                    f"Objective: {frame.objective}",
                    f"{refs[0] if refs else 'S1'} is the highest-relevance retrieved context.",
                    frame.evidence_limitation,
                ],
                "missing": frame.missing_assumptions,
            }
        if not refs:
            domain_content["rationale"] = [
                "No strong retrieved evidence was available for this prompt.",
                "Evidence analysis can only identify what would need to be verified before treating the recommendation as grounded.",
            ]
            domain_content["missing"] = [
                "No strong retrieved evidence matched this custom prompt.",
                *domain_content["missing"],
            ]

        confidence = bounded_score(
            0.4
            + (profile.evidence_quality * 0.45)
            + (len(refs) * 0.025)
            - (profile.ambiguity * 0.12)
            - (profile.risk_level * 0.04)
            - (0.14 if not refs else 0)
            - prompt_injection_risk(state.question)
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
                f"Fallback sources are curated public corpus citations; live Foundry IQ sources should verify {profile.primary_focus}."
            ],
        )
