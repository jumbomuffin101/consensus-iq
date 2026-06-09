from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import ReasoningState, ReasoningTask


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

        return [
            ReasoningTask(
                id="task-1",
                description="Identify the decision objective and diagnostic priorities.",
                owner="Planner Agent",
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
