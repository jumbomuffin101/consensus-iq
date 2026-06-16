"""Lightweight policies for custom prompts.

These policies are not retrieval sources. They define safe answer behavior,
missing-information priorities, and style guidance when the prompt falls
outside the curated demo scenarios.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CustomDomainPolicy:
    domain: str
    answer_style: str
    missing_information: list[str]
    urgent_red_flags: list[str]
    guidance: str
    source_policy: str


POLICIES: dict[str, CustomDomainPolicy] = {
    "pet_health": CustomDomainPolicy(
        domain="pet_health",
        answer_style="Safe veterinary triage guidance with concrete next steps.",
        missing_information=[
            "age",
            "duration",
            "eating and drinking",
            "bathroom habits",
            "vomiting or diarrhea",
            "breathing changes",
            "pain or weakness",
            "gum color",
            "collapse or inability to wake normally",
            "medications or possible toxin exposure",
            "baseline activity level",
        ],
        urgent_red_flags=[
            "trouble breathing",
            "collapse",
            "inability to wake normally",
            "pale, blue, gray, or very red gums",
            "repeated vomiting or severe diarrhea",
            "suspected toxin exposure",
            "severe weakness",
            "signs of pain",
            "seizure",
            "bloated abdomen",
        ],
        guidance=(
            "Do not diagnose. Explain when to call a regular veterinarian versus "
            "an emergency vet. Recommend prompt veterinary contact when lethargy "
            "is unusual, prolonged, severe, or paired with red flags."
        ),
        source_policy=(
            "Use pet-health sources only. If none are retrieved, mark source "
            "quality weak and do not cite enterprise, research, or general AI sources."
        ),
    ),
    "clinical_human": CustomDomainPolicy(
        domain="clinical_human",
        answer_style="Medical triage guidance with caution and urgent red flags.",
        missing_information=[
            "age",
            "duration",
            "severity",
            "vital signs or breathing",
            "pain",
            "medications",
            "medical history",
            "new neurologic symptoms",
        ],
        urgent_red_flags=[
            "trouble breathing",
            "chest pain",
            "stroke-like symptoms",
            "fainting",
            "severe pain",
            "confusion",
            "uncontrolled bleeding",
        ],
        guidance="Do not diagnose. Include urgent red flags and advise professional care when symptoms are severe, sudden, worsening, or unexplained.",
        source_policy="Use human medical sources only; otherwise mark source quality weak.",
    ),
    "enterprise_risk": CustomDomainPolicy(
        domain="enterprise_risk",
        answer_style="Concrete risk decision guidance with controls and owners.",
        missing_information=["stakeholders", "data sensitivity", "policy constraints", "owner", "rollback plan"],
        urgent_red_flags=[],
        guidance="Keep decision-support framing but make tradeoffs and controls concrete.",
        source_policy="Enterprise, governance, security, or policy sources are acceptable.",
    ),
    "research_eval": CustomDomainPolicy(
        domain="research_eval",
        answer_style="Research evaluation guidance focused on evidence quality and validity.",
        missing_information=["study design", "sample", "metric", "baseline", "bias checks"],
        urgent_red_flags=[],
        guidance="Keep decision-support framing but identify validity and reliability gaps.",
        source_policy="Research, assessment, or methodology sources are acceptable.",
    ),
    "finance": CustomDomainPolicy(
        domain="finance",
        answer_style="Risk-aware financial education, not personalized financial advice.",
        missing_information=["time horizon", "liquidity needs", "debt", "income stability", "risk tolerance"],
        urgent_red_flags=[],
        guidance="Avoid personalized investment certainty; discuss suitability and risk.",
        source_policy="Use finance education sources only; otherwise mark source quality weak.",
    ),
    "legal": CustomDomainPolicy(
        domain="legal",
        answer_style="Legal issue-spotting with a clear recommendation to consult qualified counsel.",
        missing_information=["jurisdiction", "dates", "contract terms", "parties", "documents"],
        urgent_red_flags=["deadline or court date", "threatened enforcement", "risk of losing rights"],
        guidance="Do not give legal advice; identify issues and urge professional counsel where needed.",
        source_policy="Use legal or government sources only; otherwise mark source quality weak.",
    ),
    "education": CustomDomainPolicy(
        domain="education",
        answer_style="Education policy guidance with fairness, validity, and stakeholder focus.",
        missing_information=["students affected", "assessment purpose", "policy constraints", "appeals", "bias checks"],
        urgent_red_flags=[],
        guidance="Focus on fairness, validity, transparency, and human accountability.",
        source_policy="Use education, assessment, or policy sources only.",
    ),
    "general_decision": CustomDomainPolicy(
        domain="general_decision",
        answer_style="Concise practical decision guidance with assumptions and next steps.",
        missing_information=["goal", "constraints", "stakes", "timeline", "alternatives"],
        urgent_red_flags=[],
        guidance="Answer directly and avoid generic phrasing.",
        source_policy="Use only clearly relevant sources; otherwise mark source quality weak.",
    ),
    "unknown": CustomDomainPolicy(
        domain="unknown",
        answer_style="Ask clarifying questions and provide only low-risk general guidance.",
        missing_information=["domain", "goal", "stakes", "timeline", "constraints"],
        urgent_red_flags=[],
        guidance="Avoid over-specific advice until the domain and stakes are clear.",
        source_policy="Mark source quality weak unless sources clearly match the prompt.",
    ),
}


def policy_for_domain(domain: str) -> CustomDomainPolicy:
    return POLICIES.get(domain, POLICIES["general_decision"])
