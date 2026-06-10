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
                    source="Foundry IQ Retrieval Layer \u2014 Curated Public Corpus",
                    url=str(row["url"]),
                    snippet=f"Curated public corpus source: {row['snippet']}",
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
                    "url": "https://www.ahajournals.org/doi/10.1161/STR.0000000000000211",
                    "snippet": (
                        "patients within the early onset window require urgent stroke-team "
                        "assessment, non-contrast brain imaging to exclude hemorrhage, and "
                        "screening for thrombolysis contraindications."
                    ),
                    "score": 0.95,
                },
                {
                    "title": "Thrombolysis contraindication checklist",
                    "url": "https://www.ninds.nih.gov/health-information/disorders/stroke",
                    "snippet": (
                        "bleeding risk, anticoagulant exposure, recent surgery, severe "
                        "hypertension, glucose abnormalities, and imaging evidence of "
                        "hemorrhage materially affect eligibility."
                    ),
                    "score": 0.9,
                },
                {
                    "title": "Time-critical stroke workflow",
                    "url": "https://www.ninds.nih.gov/health-information/disorders/stroke",
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
                    "url": "https://www.ncbi.nlm.nih.gov/books/NBK557553/",
                    "snippet": (
                        "new focal neurologic features increase concern for structural lesions, "
                        "hemorrhage, or mass effect, making intracranial imaging a safety gate "
                        "before lumbar puncture in many pathways."
                    ),
                    "score": 0.93,
                },
                {
                    "title": "Lumbar puncture imaging safety criteria",
                    "url": "https://www.ncbi.nlm.nih.gov/books/NBK557553/",
                    "snippet": (
                        "papilledema, focal deficits, immunocompromise, altered mental status, "
                        "or seizure can trigger imaging before LP to reduce herniation risk."
                    ),
                    "score": 0.88,
                },
                {
                    "title": "Parallel emergency treatment considerations",
                    "url": "https://www.ncbi.nlm.nih.gov/books/NBK557553/",
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
                    "url": "https://csrc.nist.gov/publications/detail/sp/800-61/rev-2/final",
                    "snippet": (
                        "personal-device copies of customer data should be treated as a security "
                        "incident requiring containment, preservation of evidence, and scope "
                        "assessment before normalizing access."
                    ),
                    "score": 0.94,
                },
                {
                    "title": "Incident response evidence handling",
                    "url": "https://www.nist.gov/cyberframework",
                    "snippet": (
                        "forensic collection, chain-of-custody controls, device isolation, and "
                        "account access review help determine whether data was accessed, shared, "
                        "or further exposed."
                    ),
                    "score": 0.89,
                },
                {
                    "title": "Breach notification and compliance review",
                    "url": "https://www.cisa.gov/resources-tools/resources/federal-government-cybersecurity-incident-and-vulnerability-response-playbooks",
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
                    "url": "https://arxiv.org/abs/2409.13120",
                    "snippet": (
                        "single-rater automated grading can introduce construct-irrelevant bias "
                        "and should be validated against expert scoring before high-stakes use."
                    ),
                    "score": 0.92,
                },
                {
                    "title": "LLM grading reliability checks",
                    "url": "https://arxiv.org/abs/2508.02442",
                    "snippet": (
                        "prompt sensitivity, rubric drift, inter-rater reliability, and subgroup "
                        "performance should be measured before replacing human graders."
                    ),
                    "score": 0.87,
                },
                {
                    "title": "Hybrid evaluation design",
                    "url": "https://arxiv.org/abs/2603.18765",
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
                    "url": "https://www.nist.gov/itl/ai-risk-management-framework",
                    "snippet": (
                        "organization-wide AI adoption requires approved use cases, data "
                        "classification rules, accountable owners, and monitoring for business "
                        "impact and compliance."
                    ),
                    "score": 0.89,
                },
                {
                    "title": "Workforce and operational impact review",
                    "url": "https://www.microsoft.com/en-us/ai/responsible-ai",
                    "snippet": (
                        "automation decisions should distinguish productivity augmentation from "
                        "role replacement and evaluate quality, accountability, morale, and "
                        "service continuity."
                    ),
                    "score": 0.83,
                },
                {
                    "title": "Confidential information handling policy",
                    "url": "https://learn.microsoft.com/en-us/purview/data-classification-overview",
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
                    "url": "https://www.investor.gov/introduction-investing/investing-basics/glossary/diversification",
                    "snippet": (
                        "putting all savings into one equity exposes the investor to unsystematic "
                        "risk, volatility, and permanent capital loss that diversification is "
                        "designed to reduce."
                    ),
                    "score": 0.96,
                },
                {
                    "title": "Young investor suitability factors",
                    "url": "https://www.finra.org/investors/investing/investing-basics/asset-allocation-diversification",
                    "snippet": (
                        "age alone does not justify maximum risk; emergency savings, tuition needs, "
                        "debt, time horizon, and risk tolerance determine suitability."
                    ),
                    "score": 0.9,
                },
                {
                    "title": "Speculative AI equity risk",
                    "url": "https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks",
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
                "url": "https://www.nist.gov/itl/ai-risk-management-framework",
                "snippet": (
                    "custom decision prompts benefit from explicit objectives, stakeholder "
                    "constraints, downside scenarios, and evidence thresholds before a strong "
                    "recommendation is made."
                ),
                "score": 0.76,
            },
            {
                "title": "Comparable option analysis",
                "url": "https://learn.microsoft.com/en-us/azure/architecture/guide/responsible-innovation/",
                "snippet": (
                    "reversible, conservative, and high-commitment options should be compared "
                    "using the same success metrics and failure criteria."
                ),
                "score": 0.69,
            },
            {
                "title": "Evidence gap register",
                "url": "https://www.nist.gov/cyberframework",
                "snippet": (
                    "missing facts should reduce confidence and become explicit conditions for "
                    "the final recommendation instead of being hidden by fluent wording."
                ),
                "score": 0.63,
            },
        ]
