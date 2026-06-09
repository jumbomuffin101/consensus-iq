from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import ReasoningState, ReasoningTask
from reasoning.domain import classify_domain


class PlannerNode:
    """Creates structured reasoning tasks from the user question."""

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or MockLLMProvider()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        fallback_tasks = self._fallback_plan(state.question)
        fallback = {"tasks": [task.dict() for task in fallback_tasks]}
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ planner. Decompose the user's question "
                "into concise, non-overlapping reasoning tasks for specialist agents. "
                "Use retrieved context as grounding when it is available."
            ),
            user_prompt=(
                f"Question: {state.question}\n\n"
                f"Retrieved context: {[item.dict() for item in state.retrieved_context]}\n\n"
                "Return JSON with key 'tasks'. Each task must include id, "
                "description, owner, and priority. Owners must be one of: "
                "Risk Analyst Agent, Evidence Analyst Agent, Alternative Solutions Agent."
            ),
            fallback=fallback,
        )
        try:
            tasks = [ReasoningTask.parse_obj(task) for task in payload["tasks"]]
        except Exception:
            tasks = fallback_tasks

        return state.copy(update={"reasoning_tasks": tasks})

    def _fallback_plan(self, question: str) -> list[ReasoningTask]:
        normalized = question.lower()
        domain = classify_domain(question)

        if domain == "clinical" and any(
            term in normalized for term in ["stroke", "thrombolytic", "aphasia", "weakness"]
        ):
            return [
                ReasoningTask(
                    id="task-1",
                    description="Confirm stroke time window, last-known-well, and immediate eligibility factors.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-2",
                    description="Identify thrombolysis contraindications and patient-specific bleeding risks.",
                    owner="Risk Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-3",
                    description="Review evidence for urgent imaging and treatment sequencing in acute focal deficits.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-4",
                    description="Compare thrombolysis, thrombectomy evaluation, supportive care, and transfer options.",
                    owner="Alternative Solutions Agent",
                    priority="high",
                ),
            ]

        if all(term in normalized for term in ["seizure", "mri", "lp"]):
            return [
                ReasoningTask(
                    id="task-1",
                    description="Identify diagnostic priorities for new-onset focal seizure.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-2",
                    description="Evaluate risks of lumbar puncture before intracranial imaging.",
                    owner="Risk Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-3",
                    description="Review evidence supporting MRI before LP.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-4",
                    description="Identify alternative sequencing or urgent exceptions.",
                    owner="Alternative Solutions Agent",
                    priority="medium",
                ),
            ]

        if domain == "cybersecurity":
            return [
                ReasoningTask(
                    id="task-1",
                    description="Contain the personal laptop and preserve forensic evidence.",
                    owner="Risk Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-2",
                    description="Determine what customer data was copied, accessed, or transmitted.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-3",
                    description="Evaluate legal, contractual, customer, and regulator notification obligations.",
                    owner="Risk Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-4",
                    description="Compare containment-only, disciplinary, remediation, and disclosure paths.",
                    owner="Alternative Solutions Agent",
                    priority="medium",
                ),
            ]

        if domain == "research":
            return [
                ReasoningTask(
                    id="task-1",
                    description="Define what grading validity and reliability must mean for this assessment.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-2",
                    description="Identify bias, rubric drift, prompt sensitivity, and appeal risks.",
                    owner="Risk Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-3",
                    description="Review evidence required before a single LLM can grade high-stakes work.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-4",
                    description="Compare human grading, hybrid grading, ensemble LLMs, and audit sampling.",
                    owner="Alternative Solutions Agent",
                    priority="medium",
                ),
            ]

        if domain == "finance":
            return [
                ReasoningTask(
                    id="task-1",
                    description="Assess liquidity needs, emergency savings, debt, time horizon, and risk tolerance.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-2",
                    description="Evaluate single-stock concentration, volatility, and permanent-loss risk.",
                    owner="Risk Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-3",
                    description="Review evidence for diversification and suitability for a college student.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-4",
                    description="Compare diversified funds, staged investing, cash reserve, and education-first options.",
                    owner="Alternative Solutions Agent",
                    priority="medium",
                ),
            ]

        if domain == "enterprise":
            return [
                ReasoningTask(
                    id="task-1",
                    description="Clarify business objective, stakeholders, accountability, and operational impact.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-2",
                    description="Identify governance, compliance, workforce, quality, and confidentiality risks.",
                    owner="Risk Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-3",
                    description="Review evidence for controlled adoption versus broad replacement or unrestricted use.",
                    owner="Evidence Analyst Agent",
                    priority="high",
                ),
                ReasoningTask(
                    id="task-4",
                    description="Compare augmentation, limited pilots, approved-tool policies, and no-go boundaries.",
                    owner="Alternative Solutions Agent",
                    priority="medium",
                ),
            ]

        return [
            ReasoningTask(
                id="task-1",
                description="Identify the decision objective and diagnostic priorities.",
                owner="Evidence Analyst Agent",
                priority="high",
            ),
            ReasoningTask(
                id="task-2",
                description="Evaluate risks, limitations, and failure modes.",
                owner="Risk Analyst Agent",
                priority="high",
            ),
            ReasoningTask(
                id="task-3",
                description="Review supporting evidence and rationale.",
                owner="Evidence Analyst Agent",
                priority="high",
            ),
            ReasoningTask(
                id="task-4",
                description="Identify alternative interpretations and approaches.",
                owner="Alternative Solutions Agent",
                priority="medium",
            ),
        ]
