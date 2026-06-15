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

    if all(term in normalized for term in ["university", "medical school"]) and any(
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
        objective=f"Clarify the goal, stakeholders, success criteria, and reversibility for '{topic}'.",
        recommendation=(
            f"Treat '{topic}' as a conditional decision: run a small, measurable, reversible "
            "trial before broad adoption."
        ),
        key_risk=(
            f"The main risk is committing to '{topic}' before benefits, harms, owners, "
            "and rollback criteria are explicit."
        ),
        evidence_limitation=(
            "Retrieved evidence is limited and generic, so it can frame the decision but "
            "cannot by itself justify a high-confidence recommendation."
        ),
        alternative_approaches=(
            "Compare doing nothing, a limited pilot, a controlled rollout with review gates, "
            "and a higher-commitment implementation."
        ),
        missing_assumptions=[
            "Decision owner, success metric, and minimum evidence threshold.",
            "Stakeholder impact, downside severity, and legal or policy constraints.",
            "Rollback plan and conditions that would stop or reverse the decision.",
        ],
    )
