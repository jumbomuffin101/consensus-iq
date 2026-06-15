from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    title: str
    domain: str
    source: str
    url: str
    snippet: str
    content: str
    tags: list[str]
    score: float


CURATED_PUBLIC_CORPUS: list[CorpusDocument] = [
    CorpusDocument(
        id="clinical-aha-stroke-resources",
        title="American Stroke Association acute stroke resources",
        domain="clinical",
        source="American Stroke Association",
        url="https://www.stroke.org/en/about-stroke",
        snippet=(
            "Patients within an early stroke window need urgent stroke-team assessment, "
            "brain imaging to exclude hemorrhage, and screening for thrombolysis contraindications."
        ),
        content=(
            "Acute stroke decisions require rapid symptom timing, neurologic assessment, "
            "brain imaging, hemorrhage exclusion, contraindication review, and treatment pathway coordination."
        ),
        tags=["stroke", "thrombolysis", "brain imaging", "clinical"],
        score=0.95,
    ),
    CorpusDocument(
        id="clinical-ninds-stroke-information",
        title="NIH NINDS stroke information",
        domain="clinical",
        source="NIH National Institute of Neurological Disorders and Stroke",
        url="https://www.ninds.nih.gov/health-information/disorders/stroke",
        snippet=(
            "Bleeding risk, anticoagulant exposure, recent surgery, severe hypertension, "
            "glucose abnormalities, and imaging evidence of hemorrhage affect eligibility."
        ),
        content=(
            "NINDS stroke information describes ischemic and hemorrhagic stroke, warning signs, "
            "treatment urgency, risk factors, and the importance of fast emergency evaluation."
        ),
        tags=["stroke", "NIH", "NINDS", "bleeding risk", "clinical"],
        score=0.9,
    ),
    CorpusDocument(
        id="clinical-aha-stroke-treatment",
        title="American Stroke Association treatment overview",
        domain="clinical",
        source="American Stroke Association",
        url="https://www.stroke.org/en/about-stroke",
        snippet=(
            "Rapid treatment decisions should preserve neurologic benefit while avoiding "
            "overconfidence when onset time, imaging, or consent details are incomplete."
        ),
        content=(
            "Stroke treatment resources emphasize emergency evaluation, time-sensitive intervention, "
            "imaging, contraindication review, and post-acute care planning."
        ),
        tags=["stroke treatment", "treatment window", "clinical"],
        score=0.84,
    ),
    CorpusDocument(
        id="clinical-merck-seizure-disorders",
        title="New-onset focal seizure diagnostic priority",
        domain="clinical",
        source="Merck Manual Professional Edition",
        url="https://www.merckmanuals.com/professional/neurologic-disorders/seizure-disorders/seizure-disorders",
        snippet=(
            "New focal neurologic features increase concern for structural lesions, hemorrhage, "
            "or mass effect, making intracranial imaging a safety gate before lumbar puncture in many pathways."
        ),
        content=(
            "Seizure evaluation considers structural brain disease, metabolic causes, infection, "
            "focal signs, neuroimaging, and safety before invasive diagnostic procedures."
        ),
        tags=["seizure", "MRI", "focal neurologic signs", "clinical"],
        score=0.93,
    ),
    CorpusDocument(
        id="clinical-merck-meningitis-lp",
        title="Lumbar puncture imaging safety criteria",
        domain="clinical",
        source="Merck Manual Professional Edition",
        url="https://www.merckmanuals.com/professional/neurologic-disorders/meningitis/acute-bacterial-meningitis",
        snippet=(
            "Papilledema, focal deficits, immunocompromise, altered mental status, or seizure "
            "can trigger imaging before LP to reduce herniation risk."
        ),
        content=(
            "Meningitis and lumbar puncture guidance discusses when CT or MRI should precede LP, "
            "including focal deficits, seizure, immunocompromise, and concern for mass effect."
        ),
        tags=["lumbar puncture", "meningitis", "MRI", "clinical"],
        score=0.88,
    ),
    CorpusDocument(
        id="clinical-emergency-neurology-parallel-treatment",
        title="Parallel emergency treatment considerations",
        domain="clinical",
        source="NIH National Institute of Neurological Disorders and Stroke",
        url="https://www.ninds.nih.gov/health-information/disorders/stroke",
        snippet=(
            "When infection is plausible, empiric treatment and stabilization may run in "
            "parallel with imaging rather than waiting passively."
        ),
        content=(
            "Emergency neurologic care often requires parallel stabilization, empiric therapy, "
            "neuroimaging, and diagnostic planning when delay could increase harm."
        ),
        tags=["emergency neurology", "parallel treatment", "clinical"],
        score=0.81,
    ),
    CorpusDocument(
        id="cyber-nist-sp-800-61",
        title="Customer data exfiltration response",
        domain="cybersecurity",
        source="NIST SP 800-61 Revision 2",
        url="https://csrc.nist.gov/pubs/sp/800/61/r2/final",
        snippet=(
            "Personal-device copies of customer data should be treated as a security incident "
            "requiring containment, evidence preservation, and scope assessment."
        ),
        content=(
            "NIST SP 800-61 describes incident handling phases including preparation, detection, "
            "analysis, containment, eradication, recovery, and post-incident activity."
        ),
        tags=["incident response", "containment", "data exfiltration", "cybersecurity"],
        score=0.94,
    ),
    CorpusDocument(
        id="cyber-nist-csf",
        title="Incident response evidence handling",
        domain="cybersecurity",
        source="NIST Cybersecurity Framework",
        url="https://www.nist.gov/cyberframework",
        snippet=(
            "Forensic collection, chain-of-custody controls, device isolation, and account "
            "access review help determine whether data was accessed, shared, or further exposed."
        ),
        content=(
            "The NIST Cybersecurity Framework organizes risk management outcomes across identify, "
            "protect, detect, respond, recover, govern, and continuous improvement practices."
        ),
        tags=["forensics", "chain of custody", "NIST CSF", "cybersecurity"],
        score=0.89,
    ),
    CorpusDocument(
        id="cyber-cisa-incident-playbooks",
        title="Breach notification and compliance review",
        domain="cybersecurity",
        source="CISA Incident and Vulnerability Response Playbooks",
        url="https://www.cisa.gov/resources-tools/resources/federal-government-cybersecurity-incident-and-vulnerability-response-playbooks",
        snippet=(
            "Customer, regulator, contractual, and insurer notification duties depend on data "
            "type, jurisdiction, exposure evidence, and timing thresholds."
        ),
        content=(
            "CISA incident response resources describe coordinated response actions, severity "
            "assessment, communications, remediation, and stakeholder reporting."
        ),
        tags=["CISA", "breach notification", "incident playbook", "cybersecurity"],
        score=0.84,
    ),
    CorpusDocument(
        id="research-llm-essay-graders",
        title="Are Large Language Models Good Essay Graders?",
        domain="research",
        source="arXiv",
        url="https://arxiv.org/abs/2409.13120",
        snippet=(
            "Single-rater automated grading can misalign with human scores and should be "
            "validated against expert scoring before high-stakes use."
        ),
        content=(
            "The paper evaluates LLMs for automated essay scoring and reports limitations in "
            "human alignment, score correlation, and suitability as a replacement for human grading."
        ),
        tags=["LLM grading", "automated essay scoring", "validity", "research"],
        score=0.92,
    ),
    CorpusDocument(
        id="research-llm-reliability-validity",
        title="Assessing LLM reliability and validity for essay assessment",
        domain="research",
        source="arXiv",
        url="https://arxiv.org/abs/2508.02442",
        snippet=(
            "Prompt sensitivity, rubric drift, inter-rater reliability, and subgroup performance "
            "should be measured before replacing human graders."
        ),
        content=(
            "This higher-education essay assessment study evaluates reliability and validity across "
            "multiple LLMs, prompt replications, rubric criteria, and human-LLM agreement."
        ),
        tags=["reliability", "validity", "LLM grading", "research"],
        score=0.87,
    ),
    CorpusDocument(
        id="research-testing-standards",
        title="Educational assessment validity standards",
        domain="research",
        source="Standards for Educational and Psychological Testing",
        url="https://www.testingstandards.net/open-access-files.html",
        snippet=(
            "Validity, reliability, fairness, and intended score use should be documented before "
            "automated scoring is trusted for consequential education decisions."
        ),
        content=(
            "Assessment standards emphasize validity evidence, reliability, fairness, score use, "
            "documentation, and protections for examinees in testing programs."
        ),
        tags=["educational assessment", "validity", "fairness", "research"],
        score=0.8,
    ),
    CorpusDocument(
        id="enterprise-nist-ai-rmf",
        title="Enterprise AI governance controls",
        domain="enterprise",
        source="NIST AI Risk Management Framework",
        url="https://www.nist.gov/itl/ai-risk-management-framework",
        snippet=(
            "Organization-wide AI adoption requires approved use cases, data classification rules, "
            "accountable owners, and monitoring for business impact and compliance."
        ),
        content=(
            "The NIST AI RMF describes governance, mapping, measuring, and managing AI risks "
            "across organizational contexts and responsible deployment practices."
        ),
        tags=["AI governance", "risk management", "enterprise"],
        score=0.89,
    ),
    CorpusDocument(
        id="enterprise-microsoft-responsible-ai",
        title="Workforce and operational impact review",
        domain="enterprise",
        source="Microsoft Responsible AI",
        url="https://www.microsoft.com/en-us/ai/responsible-ai",
        snippet=(
            "Automation decisions should distinguish productivity augmentation from role replacement "
            "and evaluate quality, accountability, morale, and service continuity."
        ),
        content=(
            "Microsoft Responsible AI resources describe fairness, reliability and safety, privacy, "
            "security, inclusiveness, transparency, and accountability principles."
        ),
        tags=["responsible AI", "workforce", "enterprise"],
        score=0.83,
    ),
    CorpusDocument(
        id="enterprise-microsoft-purview-classification",
        title="Confidential information handling policy",
        domain="enterprise",
        source="Microsoft Purview",
        url="https://learn.microsoft.com/en-us/purview/data-classification-overview",
        snippet=(
            "Public AI tools are unsuitable for restricted information unless vendor terms, retention "
            "controls, client permissions, and auditability are clear."
        ),
        content=(
            "Microsoft Purview data classification supports discovery and classification of sensitive "
            "information to help govern protected data across enterprise systems."
        ),
        tags=["data classification", "confidential documents", "enterprise"],
        score=0.79,
    ),
    CorpusDocument(
        id="finance-investor-gov-diversification",
        title="Single-stock concentration risk",
        domain="finance",
        source="Investor.gov",
        url="https://www.investor.gov/introduction-investing/investing-basics/glossary/diversification",
        snippet=(
            "Putting all savings into one equity exposes the investor to unsystematic risk, "
            "volatility, and permanent capital loss that diversification is designed to reduce."
        ),
        content=(
            "Investor.gov explains diversification as spreading investments across assets to reduce "
            "the effect of any single investment on overall portfolio outcomes."
        ),
        tags=["diversification", "single stock", "finance"],
        score=0.96,
    ),
    CorpusDocument(
        id="finance-finra-asset-allocation",
        title="Young investor suitability factors",
        domain="finance",
        source="FINRA",
        url="https://www.finra.org/investors/investing/investing-basics/asset-allocation-diversification",
        snippet=(
            "Age alone does not justify maximum risk; emergency savings, tuition needs, debt, "
            "time horizon, and risk tolerance determine suitability."
        ),
        content=(
            "FINRA investor education discusses asset allocation, diversification, risk tolerance, "
            "investment goals, and the role of time horizon in investment planning."
        ),
        tags=["asset allocation", "risk tolerance", "finance"],
        score=0.9,
    ),
    CorpusDocument(
        id="finance-investor-gov-stocks",
        title="Speculative AI equity risk",
        domain="finance",
        source="Investor.gov",
        url="https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks",
        snippet=(
            "Theme-driven stocks can have valuation, hype-cycle, liquidity, and company-specific "
            "risks even when the long-term sector outlook is strong."
        ),
        content=(
            "Investor.gov stock education describes ownership, risk, price volatility, company-specific "
            "exposure, and the need to understand investment products before buying."
        ),
        tags=["stocks", "speculation", "AI stock", "finance"],
        score=0.85,
    ),
    CorpusDocument(
        id="custom-nist-ai-rmf",
        title="Decision criteria and uncertainty framing",
        domain="custom",
        source="NIST AI Risk Management Framework",
        url="https://www.nist.gov/itl/ai-risk-management-framework",
        snippet=(
            "Custom decision prompts benefit from explicit objectives, stakeholder constraints, "
            "downside scenarios, and evidence thresholds before a strong recommendation is made."
        ),
        content=(
            "General AI risk management guidance can help frame uncertainty, objectives, impacts, "
            "stakeholders, controls, and monitoring when the prompt lacks a specific domain."
        ),
        tags=["decision criteria", "uncertainty", "custom"],
        score=0.76,
    ),
    CorpusDocument(
        id="custom-microsoft-responsible-ai",
        title="Responsible AI governance checkpoints",
        domain="custom",
        source="Microsoft Responsible AI",
        url="https://www.microsoft.com/en-us/ai/responsible-ai",
        snippet=(
            "AI-assisted decisions should preserve accountability, reliability, privacy, "
            "transparency, and human oversight before deployment."
        ),
        content=(
            "Microsoft Responsible AI resources describe principles and practices for "
            "fairness, reliability and safety, privacy and security, inclusiveness, "
            "transparency, and accountability."
        ),
        tags=["AI governance", "human oversight", "automation governance", "custom"],
        score=0.72,
    ),
    CorpusDocument(
        id="custom-education-assessment-validity",
        title="Education assessment validity and fairness",
        domain="custom",
        source="Standards for Educational and Psychological Testing",
        url="https://www.testingstandards.net/open-access-files.html",
        snippet=(
            "Consequential education decisions need validity, reliability, fairness, "
            "documented score use, and protections for affected applicants or students."
        ),
        content=(
            "Assessment standards emphasize validity evidence, reliability, fairness, "
            "score interpretation, score use, documentation, and examinee protections."
        ),
        tags=["education", "assessment", "validity", "fairness", "admissions", "custom"],
        score=0.7,
    ),
    CorpusDocument(
        id="custom-ftc-data-security",
        title="Privacy and data minimization for sensitive decisions",
        domain="custom",
        source="Federal Trade Commission",
        url="https://www.ftc.gov/business-guidance/privacy-security",
        snippet=(
            "Sensitive data workflows should minimize data collection, protect personal "
            "information, and define accountability before automation is introduced."
        ),
        content=(
            "FTC business guidance covers privacy, security, data minimization, consumer "
            "protection, and organizational safeguards for personal information."
        ),
        tags=["privacy", "data security", "data minimization", "automation", "custom"],
        score=0.68,
    ),
    CorpusDocument(
        id="custom-fhwa-roundabouts",
        title="Roundabout safety and intersection suitability",
        domain="custom",
        source="Federal Highway Administration",
        url="https://highways.dot.gov/safety/intersection-safety/intersection-types/roundabouts",
        snippet=(
            "Roundabouts can improve safety at suitable intersections, but selection "
            "depends on traffic volumes, geometry, users, cost, and local operating context."
        ),
        content=(
            "FHWA roundabout resources discuss intersection safety, design considerations, "
            "operations, crash reduction potential, and context-specific suitability."
        ),
        tags=["roundabouts", "traffic lights", "intersection", "transportation", "custom"],
        score=0.72,
    ),
    CorpusDocument(
        id="custom-azure-responsible-innovation",
        title="Comparable option analysis",
        domain="custom",
        source="Microsoft Azure Architecture Center",
        url="https://learn.microsoft.com/en-us/azure/architecture/guide/responsible-innovation/",
        snippet=(
            "Reversible, conservative, and high-commitment options should be compared using the "
            "same success metrics and failure criteria."
        ),
        content=(
            "Responsible innovation guidance emphasizes harms modeling, stakeholder review, impact "
            "assessment, and tradeoff analysis for technology decisions."
        ),
        tags=["responsible innovation", "option analysis", "custom"],
        score=0.69,
    ),
    CorpusDocument(
        id="custom-nist-csf-evidence-gaps",
        title="Evidence gap register",
        domain="custom",
        source="NIST Cybersecurity Framework",
        url="https://www.nist.gov/cyberframework",
        snippet=(
            "Missing facts should reduce confidence and become explicit conditions for the final "
            "recommendation instead of being hidden by fluent wording."
        ),
        content=(
            "Risk management frameworks can support gap tracking, assumptions, controls, and "
            "explicit confidence limits for decisions made with incomplete evidence."
        ),
        tags=["evidence gaps", "risk management", "custom"],
        score=0.63,
    ),
]


def documents_for_domain(domain: str, question: str = "") -> list[CorpusDocument]:
    normalized_question = question.lower()
    if domain == "clinical" and any(
        term in normalized_question
        for term in ["stroke", "thrombolytic", "thrombolysis", "aphasia", "weakness", "tpa", "alteplase"]
    ):
        selected_ids = [
            "clinical-aha-stroke-resources",
            "clinical-ninds-stroke-information",
            "clinical-aha-stroke-treatment",
        ]
        return _by_ids(selected_ids)

    if domain == "clinical" and any(
        term in normalized_question
        for term in ["seizure", "mri", "lumbar", "puncture", "lp"]
    ):
        selected_ids = [
            "clinical-merck-seizure-disorders",
            "clinical-merck-meningitis-lp",
            "clinical-emergency-neurology-parallel-treatment",
        ]
        return _by_ids(selected_ids)

    if domain == "custom":
        if all(term in normalized_question for term in ["medical school"]) and any(
            term in normalized_question
            for term in ["ai", "screen", "applications", "applicants"]
        ):
            return _by_ids(
                [
                    "custom-nist-ai-rmf",
                    "custom-education-assessment-validity",
                    "custom-microsoft-responsible-ai",
                ]
            )
        if any(term in normalized_question for term in ["traffic lights", "roundabouts", "intersection"]):
            return _by_ids(
                [
                    "custom-fhwa-roundabouts",
                    "custom-nist-ai-rmf",
                    "custom-azure-responsible-innovation",
                ]
            )
        if any(term in normalized_question for term in ["privacy", "data", "automation", "ai"]):
            return _by_ids(
                [
                    "custom-nist-ai-rmf",
                    "custom-microsoft-responsible-ai",
                    "custom-ftc-data-security",
                ]
            )

    documents = [document for document in CURATED_PUBLIC_CORPUS if document.domain == domain]
    if documents:
        return documents[:3]
    return [document for document in CURATED_PUBLIC_CORPUS if document.domain == "custom"][:3]


def _by_ids(document_ids: list[str]) -> list[CorpusDocument]:
    lookup = {document.id: document for document in CURATED_PUBLIC_CORPUS}
    return [lookup[document_id] for document_id in document_ids if document_id in lookup]
