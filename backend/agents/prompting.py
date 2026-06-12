import json
from typing import Any

from models.reasoning import ReasoningState


def state_context_payload(state: ReasoningState) -> str:
    payload: dict[str, Any] = {
        "question": state.question,
        "retrieved_context": [item.dict() for item in state.retrieved_context],
        "reasoning_tasks": [task.dict() for task in state.reasoning_tasks],
        "citation_instruction": (
            "Use retrieved_context for evidence-based claims. Cite sources by "
            "citation_id in evidence_refs, for example S1 or S2."
        ),
    }
    return json.dumps(payload, indent=2, default=str)


def agent_outputs_payload(state: ReasoningState) -> str:
    return json.dumps(
        [output.dict() for output in state.agent_outputs],
        indent=2,
        default=str,
    )


def disagreements_payload(state: ReasoningState) -> str:
    return json.dumps(
        [disagreement.dict() for disagreement in state.disagreements],
        indent=2,
        default=str,
    )
