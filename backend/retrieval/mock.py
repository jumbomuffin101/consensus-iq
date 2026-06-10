from models.reasoning import RetrievedContext
from reasoning.domain import classify_domain
from retrieval.base import BaseRetrievalProvider


class MockRetrievalProvider(BaseRetrievalProvider):
    """Demo corpus provider used when live Foundry IQ credentials are unavailable."""

    name = "mock-foundry-iq"

    def retrieve(self, question: str) -> list[RetrievedContext]:
        domain = classify_domain(question)
        normalized = question.lower()
        source_rows = self._source_rows(domain, normalized)
        return self.normalize(
            [
                RetrievedContext(
                    id=f"demo-{domain}-{index}",
                    citation_id=f"S{index}",
                    title=row["title"],
                    source="Foundry IQ Retrieval Layer \u2014 Demo Corpus",
                    url=f"mock://foundry-iq/{domain}/{index}",
                    snippet=f"Demo corpus source: {row['snippet']}",
                    relevance_score=row["score"],
                )
                for index, row in enumerate(source_rows, start=1)
            ]
        )

    def _source_rows(self, domain: str, question: str) -> list[dict[str, str | float]]:
        if domain == "clinical" and any(
            term in question for term in ["stroke", "thrombolytic", "aphasia", "weakness"]
        ):
            return [
                {
                    "title": "Acute ischemic stroke treatment window",
                    "snippet": (
                        "patients within the early onset window require urgent stroke-team "
                        "assessment, non-contrast brain imaging to exclude hemorrhage, and "
                        "screening for thrombolysis contraindications."
                    ),
                    "score": 0.95,
                },
                {
                    "title": "Thrombolysis contraindication checklist",
                    "snippet": (
                        "bleeding risk, anticoagulant exposure, recent surgery, severe "
                        "hypertension, glucose abnormalities, and imaging evidence of "
                        "hemorrhage materially affect eligibility."
                    ),
                    "score": 0.9,
                },
                {
                    "title": "Time-critical stroke workflow",
                    "snippet": (
                        "rapid treatment decisions should preserve neurologic benefit while "
                        "avoiding overconfidence when onset time, imaging, or consent details "
                        "are incomplete."
                    ),
                    "score": 0.84,
                },
            ]

        if domain == "clinical":
            return [
                {
                    "title": "New-onset focal seizure diagnostic priority",
                    "snippet": (
                        "new focal neurologic features increase concern for structural lesions, "
                        "hemorrhage, or mass effect, making intracranial imaging a safety gate "
                        "before lumbar puncture in many pathways."
                    ),
                    "score": 0.93,
                },
                {
                    "title": "Lumbar puncture imaging safety criteria",
                    "snippet": (
                        "papilledema, focal deficits, immunocompromise, altered mental status, "
                        "or seizure can trigger imaging before LP to reduce herniation risk."
                    ),
                    "score": 0.88,
                },
                {
                    "title": "Parallel emergency treatment considerations",
                    "snippet": (
                        "when infection is plausible, empiric treatment and stabilization may "
                        "run in parallel with imaging rather than waiting passively."
                    ),
                    "score": 0.81,
                },
            ]

        if domain == "cybersecurity":
            return [
                {
                    "title": "Customer data exfiltration response",
                    "snippet": (
                        "personal-device copies of customer data should be treated as a security "
                        "incident requiring containment, preservation of evidence, and scope "
                        "assessment before normalizing access."
                    ),
                    "score": 0.94,
                },
                {
                    "title": "Incident response evidence handling",
                    "snippet": (
                        "forensic collection, chain-of-custody controls, device isolation, and "
                        "account access review help determine whether data was accessed, shared, "
                        "or further exposed."
                    ),
                    "score": 0.89,
                },
                {
                    "title": "Breach notification and compliance review",
                    "snippet": (
                        "customer, regulator, contractual, and insurer notification duties depend "
                        "on data type, jurisdiction, exposure evidence, and timing thresholds."
                    ),
                    "score": 0.84,
                },
            ]

        if domain == "research":
            return [
                {
                    "title": "Assessment validity for automated grading",
                    "snippet": (
                        "single-rater automated grading can introduce construct-irrelevant bias "
                        "and should be validated against expert scoring before high-stakes use."
                    ),
                    "score": 0.92,
                },
                {
                    "title": "LLM grading reliability checks",
                    "snippet": (
                        "prompt sensitivity, rubric drift, inter-rater reliability, and subgroup "
                        "performance should be measured before replacing human graders."
                    ),
                    "score": 0.87,
                },
                {
                    "title": "Hybrid evaluation design",
                    "snippet": (
                        "LLMs can support draft scoring, feedback generation, or triage when "
                        "paired with calibration sets, human adjudication, and audit sampling."
                    ),
                    "score": 0.8,
                },
            ]

        if domain == "enterprise":
            return [
                {
                    "title": "Enterprise AI governance controls",
                    "snippet": (
                        "organization-wide AI adoption requires approved use cases, data "
                        "classification rules, accountable owners, and monitoring for business "
                        "impact and compliance."
                    ),
                    "score": 0.89,
                },
                {
                    "title": "Workforce and operational impact review",
                    "snippet": (
                        "automation decisions should distinguish productivity augmentation from "
                        "role replacement and evaluate quality, accountability, morale, and "
                        "service continuity."
                    ),
                    "score": 0.83,
                },
                {
                    "title": "Confidential information handling policy",
                    "snippet": (
                        "public AI tools are unsuitable for restricted information unless vendor "
                        "terms, retention controls, client permissions, and auditability are clear."
                    ),
                    "score": 0.79,
                },
            ]

        if domain == "finance":
            return [
                {
                    "title": "Single-stock concentration risk",
                    "snippet": (
                        "putting all savings into one equity exposes the investor to unsystematic "
                        "risk, volatility, and permanent capital loss that diversification is "
                        "designed to reduce."
                    ),
                    "score": 0.96,
                },
                {
                    "title": "Young investor suitability factors",
                    "snippet": (
                        "age alone does not justify maximum risk; emergency savings, tuition needs, "
                        "debt, time horizon, and risk tolerance determine suitability."
                    ),
                    "score": 0.9,
                },
                {
                    "title": "Speculative AI equity risk",
                    "snippet": (
                        "theme-driven stocks can have valuation, hype-cycle, liquidity, and "
                        "company-specific risks even when the long-term sector outlook is strong."
                    ),
                    "score": 0.85,
                },
            ]

        return [
            {
                "title": "Decision criteria and uncertainty framing",
                "snippet": (
                    "custom decision prompts benefit from explicit objectives, stakeholder "
                    "constraints, downside scenarios, and evidence thresholds before a strong "
                    "recommendation is made."
                ),
                "score": 0.76,
            },
            {
                "title": "Comparable option analysis",
                "snippet": (
                    "reversible, conservative, and high-commitment options should be compared "
                    "using the same success metrics and failure criteria."
                ),
                "score": 0.69,
            },
            {
                "title": "Evidence gap register",
                "snippet": (
                    "missing facts should reduce confidence and become explicit conditions for "
                    "the final recommendation instead of being hidden by fluent wording."
                ),
                "score": 0.63,
            },
        ]
