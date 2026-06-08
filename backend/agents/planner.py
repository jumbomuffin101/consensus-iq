from models.reasoning import ReasoningState, ReasoningTask


class PlannerNode:
    """Creates structured reasoning tasks from the user question."""

    def __call__(self, state: ReasoningState) -> ReasoningState:
        return state.copy(update={"reasoning_tasks": self._plan(state.question)})

    def _plan(self, question: str) -> list[ReasoningTask]:
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
