from dataclasses import dataclass

from models.reasoning import ReasoningState


@dataclass(frozen=True)
class DomainProfile:
    domain: str
    scenario_label: str
    ambiguity: float
    evidence_quality: float
    source_certainty: float
    risk_level: float
    primary_focus: str


DOMAIN_KEYWORDS = {
    "clinical": [
        "patient",
        "woman",
        "man",
        "aphasia",
        "weakness",
        "stroke",
        "thrombolytic",
        "thrombolysis",
        "tpa",
        "alteplase",
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
        "customer database",
        "personal laptop",
        "downloaded",
        "data exposure",
        "data exfiltration",
        "forensic",
        "notification",
        "confidential",
        "client documents",
    ],
    "research": [
        "research",
        "education",
        "university",
        "essay",
        "essays",
        "grader",
        "grade",
        "student",
        "concept maps",
        "evaluation",
        "validity",
        "bias",
        "rubric",
    ],
    "enterprise": [
        "company",
        "organization",
        "employees",
        "employee",
        "public ai",
        "ai tools",
        "ai agents",
        "software engineers",
        "governance",
        "stakeholders",
        "policy",
        "client",
    ],
    "finance": [
        "invest",
        "investment",
        "savings",
        "stock",
        "portfolio",
        "financial",
        "college student",
        "risk tolerance",
        "retirement",
        "loan",
        "debt",
    ],
}


SCENARIO_LABELS = {
    "clinical": "Clinical",
    "cybersecurity": "Cybersecurity",
    "research": "Research",
    "enterprise": "Enterprise",
    "finance": "Finance",
    "custom": "Custom",
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
    return best_domain


def build_domain_profile(state: ReasoningState) -> DomainProfile:
    domain = classify_domain(state.question)
    evidence_quality = _average(
        [source.relevance_score for source in state.retrieved_context], default=0.2
    )
    source_certainty = min(1.0, evidence_quality + (len(state.retrieved_context) * 0.04))
    ambiguity = _question_ambiguity(state.question, domain)
    return DomainProfile(
        domain=domain,
        scenario_label=SCENARIO_LABELS[domain],
        ambiguity=ambiguity,
        evidence_quality=evidence_quality,
        source_certainty=source_certainty,
        risk_level={
            "clinical": 0.9,
            "cybersecurity": 0.82,
            "finance": 0.8,
            "enterprise": 0.68,
            "research": 0.62,
            "custom": 0.55,
        }[domain],
        primary_focus={
            "clinical": "diagnosis, treatment sequence, and patient safety",
            "cybersecurity": "containment, investigation, compliance, and data exposure",
            "research": "validity, bias, evaluator reliability, and measurement design",
            "enterprise": "stakeholders, governance, confidentiality, and operational risk",
            "finance": "diversification, suitability, liquidity, and downside risk",
            "custom": "decision criteria, uncertainty, and evidence gaps",
        }[domain],
    )


def scenario_label_for_question(question: str) -> str:
    return SCENARIO_LABELS[classify_domain(question)]


def bounded_score(value: float) -> float:
    return round(max(0.05, min(0.98, value)), 2)


def prompt_injection_risk(question: str) -> float:
    normalized = question.lower()
    indicators = [
        "ignore all previous instructions",
        "ignore previous instructions",
        "100% certain",
        "guaranteed answer",
        "no uncertainty",
        "do not mention limitations",
    ]
    return 0.18 if any(indicator in normalized for indicator in indicators) else 0.0


def missing_information_load(state: ReasoningState) -> float:
    missing_count = sum(len(output.missing_evidence) for output in state.agent_outputs)
    no_ref_count = sum(1 for output in state.agent_outputs if not output.evidence_refs)
    return min(0.28, (missing_count * 0.018) + (no_ref_count * 0.04))


def _question_ambiguity(question: str, domain: str) -> float:
    normalized = question.lower()
    ambiguity = 0.35
    ambiguity += 0.1 if len(question.split()) < 10 else 0
    ambiguity += 0.08 if any(term in normalized for term in ["should", "trust", "allow", "replace"]) else 0
    ambiguity += 0.12 if any(term in normalized for term in ["single", "public", "all savings", "every company"]) else 0
    ambiguity += prompt_injection_risk(question)
    ambiguity += {
        "clinical": 0.05,
        "cybersecurity": 0.08,
        "research": 0.14,
        "enterprise": 0.12,
        "finance": 0.16,
        "custom": 0.18,
    }[domain]
    return min(0.9, ambiguity)


def _average(values: list[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)
