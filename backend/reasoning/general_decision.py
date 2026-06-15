from dataclasses import dataclass


@dataclass(frozen=True)
class GeneralDecisionFrame:
    topic: str
    objective: str
    recommendation: str
    key_risk: str
    evidence_limitation: str
    alternative_approaches: str
    missing_assumptions: list[str]


def build_general_decision_frame(question: str) -> GeneralDecisionFrame:
    """Build prompt-specific fallback reasoning for custom decisions.

    This keeps non-preset prompts useful when live LLM calls or specific
    retrieval coverage are unavailable.
    """

    normalized = question.lower()
    topic = question.strip().rstrip("?.!") or "this decision"

    if any(
        term in normalized
        for term in [
            "strain",
            "sprain",
            "injury",
            "pain",
            "quad",
            "quadriceps",
            "hamstring",
            "ankle",
            "knee",
            "soccer",
            "basketball",
            "workout",
            "gym",
            "play through",
        ]
    ):
        return GeneralDecisionFrame(
            topic=topic,
            objective=(
                "Decide whether the athlete can safely return to sport without "
                "worsening the suspected muscle or joint injury."
            ),
            recommendation=(
                "Do not play soccer until pain-free walking, stairs, jogging, "
                "cutting, sprinting, and sport-specific movements are tolerated."
            ),
            key_risk=(
                "Playing through pain can worsen the strain, increase bleeding or "
                "tissue damage, delay recovery, and make cutting or sprinting unsafe."
            ),
            evidence_limitation=(
                "No strong retrieved sports-medicine source matched this prompt, so "
                "the recommendation relies on conservative return-to-play reasoning."
            ),
            alternative_approaches=(
                "Rest from painful play, use modified activity that does not provoke "
                "symptoms, progress through rehab and sport-specific drills, and seek "
                "medical evaluation if symptoms are severe, worsening, bruised, weak, "
                "or persistent."
            ),
            missing_assumptions=[
                "Severity of pain at rest, walking, stairs, jogging, cutting, and sprinting.",
                "Bruising, swelling, weakness, range-of-motion loss, or a popping sensation.",
                "Exam findings, time since injury, and whether symptoms are improving or worsening.",
            ],
        )

    if any(term in normalized for term in ["medical school", "medical schools"]) and any(
        term in normalized for term in ["screen", "applications", "applicants"]
    ):
        return GeneralDecisionFrame(
            topic=topic,
            objective=(
                "Decide whether AI should assist medical school application screening "
                "without unfairly excluding qualified applicants."
            ),
            recommendation=(
                "Use AI only as an audited triage aid after validation; do not let it "
                "autonomously reject or rank applicants."
            ),
            key_risk=(
                "Bias, opaque criteria, disability or accommodation effects, privacy "
                "exposure, and false negatives could deny applicants fair review."
            ),
            evidence_limitation=(
                "Retrieved evidence is general governance evidence, not admissions-specific "
                "validation data for the university's applicant pool."
            ),
            alternative_approaches=(
                "Compare human committee review, rubric-based screening, AI-assisted "
                "flagging with mandatory human review, and a retrospective pilot before use."
            ),
            missing_assumptions=[
                "Validated accuracy against prior admissions decisions and outcomes.",
                "Bias testing by demographic, disability, socioeconomic, and school-background groups.",
                "Appeals process, applicant notice, privacy controls, and final human accountability.",
            ],
        )

    if "city" in normalized and any(
        term in normalized for term in ["traffic lights", "roundabouts", "intersection"]
    ):
        return GeneralDecisionFrame(
            topic=topic,
            objective=(
                "Decide whether specific signalized intersections should become roundabouts "
                "while preserving safety, access, cost discipline, and mobility."
            ),
            recommendation=(
                "Do not replace traffic lights citywide; run intersection-level traffic "
                "engineering studies and pilot only suitable locations."
            ),
            key_risk=(
                "A blanket replacement could worsen pedestrian, cyclist, disability, transit, "
                "freight, or emergency access at intersections that are not roundabout-suitable."
            ),
            evidence_limitation=(
                "Retrieved evidence is general decision-support evidence; the decision needs "
                "local crash history, traffic volumes, geometry, and cost data."
            ),
            alternative_approaches=(
                "Compare signal timing upgrades, protected turns, pedestrian priority phases, "
                "targeted roundabouts, traffic calming, and corridor redesign."
            ),
            missing_assumptions=[
                "Intersection-by-intersection crash rates, volumes, speeds, and right-of-way constraints.",
                "Pedestrian, cyclist, transit, emergency response, and accessibility impacts.",
                "Capital cost, construction disruption, public acceptance, and maintenance capacity.",
            ],
        )

    return GeneralDecisionFrame(
        topic=topic,
        objective=f"Clarify the practical goal, affected people, constraints, and decision criteria for '{topic}'.",
        recommendation=(
            f"Use a cautious decision-support approach for '{topic}': compare realistic options, "
            "check the most important risks, and choose the lowest-regret next step."
        ),
        key_risk=(
            f"The main risk is giving a confident answer about '{topic}' before the relevant facts, "
            "constraints, and downside scenarios are clear."
        ),
        evidence_limitation=(
            "Retrieved evidence is limited and generic, so it can frame the decision but "
            "cannot by itself justify a high-confidence recommendation."
        ),
        alternative_approaches=(
            "Compare the conservative option, a monitored middle path, and a higher-commitment "
            "option using the same success criteria and stop conditions."
        ),
        missing_assumptions=[
            "Decision owner, success metric, and minimum evidence threshold.",
            "Stakeholder impact, downside severity, and legal or policy constraints.",
            "Rollback plan and conditions that would stop or reverse the decision.",
        ],
    )
