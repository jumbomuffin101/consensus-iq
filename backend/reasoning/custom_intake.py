import re

from llm.base import BaseLLMProvider
from models.reasoning import CustomPromptIntake
from policies.custom_domains import policy_for_domain


LOW_CONFIDENCE_THRESHOLD = 0.45


def classify_custom_prompt(
    question: str, provider: BaseLLMProvider | None = None
) -> CustomPromptIntake:
    intake = deterministic_custom_intake(question)
    if provider is None or intake.confidence >= LOW_CONFIDENCE_THRESHOLD:
        return intake

    fallback = intake.dict()
    payload = provider.complete_json(
        system_prompt=(
            "Classify a custom user prompt for a decision-support app. Return "
            "only the requested JSON. Do not answer the user's question."
        ),
        user_prompt=(
            f"Prompt: {question}\n\n"
            "Return JSON with keys: domain, intent, urgency, missing_information, "
            "retrieval_queries, answer_style, confidence. Supported domains: "
            "clinical_human, pet_health, enterprise_risk, research_eval, finance, "
            "legal, education, general_decision, unknown. Supported intents: "
            "triage, compare_options, evaluate_risk, summarize, plan, "
            "diagnose_problem, other. Supported urgency: low, moderate, high, "
            "emergency_possible."
        ),
        fallback=fallback,
        agent_name="custom intake classifier",
    )
    try:
        parsed = CustomPromptIntake.parse_obj(payload)
    except Exception:
        return intake
    return _with_policy_defaults(parsed, question)


def deterministic_custom_intake(question: str) -> CustomPromptIntake:
    normalized = question.lower()
    domain, domain_confidence = _domain(normalized)
    intent, intent_confidence = _intent(normalized)
    urgency, urgency_confidence = _urgency(normalized, domain)
    confidence = round(min(0.95, max(domain_confidence, (domain_confidence + intent_confidence + urgency_confidence) / 3)), 2)
    intake = CustomPromptIntake(
        domain=domain,
        intent=intent,
        urgency=urgency,
        missing_information=[],
        retrieval_queries=_retrieval_queries(question, domain, intent),
        answer_style="",
        confidence=confidence,
    )
    return _with_policy_defaults(intake, question)


def _with_policy_defaults(
    intake: CustomPromptIntake, question: str
) -> CustomPromptIntake:
    policy = policy_for_domain(intake.domain)
    missing = list(dict.fromkeys([*intake.missing_information, *policy.missing_information]))
    retrieval_queries = intake.retrieval_queries or _retrieval_queries(
        question, intake.domain, intake.intent
    )
    return intake.copy(
        update={
            "missing_information": missing[:12],
            "retrieval_queries": retrieval_queries[:4],
            "answer_style": intake.answer_style or policy.answer_style,
        }
    )


def _domain(normalized: str) -> tuple[str, float]:
    if any(term in normalized for term in ["dog", "cat", "puppy", "kitten", "pet", "vet", "veterinarian"]):
        return "pet_health", 0.9
    if any(term in normalized for term in ["patient", "symptom", "doctor", "hospital", "medical", "clinic"]):
        return "clinical_human", 0.78
    if any(term in normalized for term in ["contract", "lawsuit", "tenant", "landlord", "legal", "court"]):
        return "legal", 0.74
    if any(term in normalized for term in ["invest", "stock", "portfolio", "loan", "debt", "savings"]):
        return "finance", 0.78
    if any(term in normalized for term in ["student", "school", "university", "classroom", "teacher", "admissions"]):
        return "education", 0.68
    if any(term in normalized for term in ["research", "study", "validity", "reliability", "evaluate"]):
        return "research_eval", 0.68
    if any(term in normalized for term in ["company", "employee", "business", "vendor", "client", "enterprise"]):
        return "enterprise_risk", 0.68
    if any(term in normalized for term in ["should", "choose", "decide", "compare"]):
        return "general_decision", 0.52
    return "unknown", 0.25


def _intent(normalized: str) -> tuple[str, float]:
    if any(term in normalized for term in ["vet", "doctor", "urgent", "emergency", "take my", "go to"]):
        return "triage", 0.82
    if any(term in normalized for term in ["compare", "versus", " vs ", "option"]):
        return "compare_options", 0.76
    if any(term in normalized for term in ["risk", "safe", "danger", "concern"]):
        return "evaluate_risk", 0.72
    if any(term in normalized for term in ["plan", "steps", "roadmap"]):
        return "plan", 0.7
    if any(term in normalized for term in ["why", "broken", "not working", "problem"]):
        return "diagnose_problem", 0.62
    if any(term in normalized for term in ["summarize", "summary"]):
        return "summarize", 0.72
    return "other", 0.35


def _urgency(normalized: str, domain: str) -> tuple[str, float]:
    emergency_terms = [
        "can't breathe",
        "cannot breathe",
        "collapse",
        "collapsed",
        "unconscious",
        "seizure",
        "poison",
        "toxin",
        "not waking",
        "pale gums",
        "blue gums",
        "emergency",
    ]
    high_terms = ["vomiting", "diarrhea", "blood", "pain", "weak", "lethargic", "not eating"]
    if any(term in normalized for term in emergency_terms):
        return "emergency_possible", 0.88
    if domain in {"pet_health", "clinical_human"} and any(term in normalized for term in high_terms):
        return "high", 0.7
    if domain in {"pet_health", "clinical_human"}:
        return "moderate", 0.62
    return "low", 0.55


def _retrieval_queries(question: str, domain: str, intent: str) -> list[str]:
    clean = re.sub(r"\s+", " ", question.strip())
    if domain == "pet_health":
        return [
            clean,
            "dog lethargy when to call veterinarian red flags",
            "pet lethargy emergency veterinary signs",
        ]
    if domain == "clinical_human":
        return [clean, "medical symptom triage urgent red flags"]
    if domain == "legal":
        return [clean, "legal issue jurisdiction deadline consult attorney"]
    return [clean, f"{domain} {intent} decision guidance"]
