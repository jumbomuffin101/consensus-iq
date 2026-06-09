from dataclasses import dataclass

from models.reasoning import ReasoningState


@dataclass(frozen=True)
class DomainProfile:
    domain: str
    ambiguity: float
    evidence_quality: float
    source_certainty: float
    primary_focus: str


DOMAIN_KEYWORDS = {
    "clinical": [
        "patient",
        "seizure",
        "mri",
        "lumbar",
        "puncture",
        "diagnosis",
        "treatment",
        "clinical",
        "medical",
    ],
    "cybersecurity": [
        "security",
        "cyber",
        "breach",
        "incident",
        "containment",
        "malware",
        "phishing",
        "confidential",
        "client documents",
    ],
    "research": [
        "research",
        "grader",
        "student",
        "concept maps",
        "evaluation",
        "validity",
        "bias",
        "rubric",
    ],
    "enterprise": [
        "company",
        "employees",
        "public ai",
        "governance",
        "stakeholders",
        "policy",
        "client",
    ],
}


def classify_domain(question: str) -> str:
    normalized = question.lower()
    scores = {
        domain: sum(1 for keyword in keywords if keyword in normalized)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "custom"
    if best_domain == "cybersecurity" and scores.get("enterprise", 0) > 0:
        return "enterprise"
    return best_domain


def build_domain_profile(state: ReasoningState) -> DomainProfile:
    domain = classify_domain(state.question)
    evidence_quality = _average(
        [source.relevance_score for source in state.retrieved_context], default=0.55
    )
    source_certainty = min(1.0, evidence_quality + (len(state.retrieved_context) * 0.04))
    ambiguity = _question_ambiguity(state.question, domain)
    return DomainProfile(
        domain=domain,
        ambiguity=ambiguity,
        evidence_quality=evidence_quality,
        source_certainty=source_certainty,
        primary_focus={
            "clinical": "diagnosis, treatment sequence, and patient safety",
            "cybersecurity": "containment, investigation, compliance, and data exposure",
            "research": "validity, bias, evaluator reliability, and measurement design",
            "enterprise": "stakeholders, governance, confidentiality, and operational risk",
            "custom": "decision criteria, uncertainty, and evidence gaps",
        }[domain],
    )


def bounded_score(value: float) -> float:
    return round(max(0.05, min(0.98, value)), 2)


def _question_ambiguity(question: str, domain: str) -> float:
    normalized = question.lower()
    ambiguity = 0.35
    ambiguity += 0.1 if len(question.split()) < 10 else 0
    ambiguity += 0.08 if any(term in normalized for term in ["should", "trust", "allow"]) else 0
    ambiguity += 0.12 if "single" in normalized or "public" in normalized else 0
    ambiguity += {"clinical": 0.05, "cybersecurity": 0.1, "research": 0.14, "enterprise": 0.12, "custom": 0.18}[domain]
    return min(0.9, ambiguity)


def _average(values: list[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)
