from typing import Any, Callable, TypedDict

from agents.alternatives_agent import AlternativesAnalystNode
from agents.consensus_agent import ConsensusJudgeNode
from agents.evidence_agent import EvidenceAnalystNode
from agents.planner import PlannerNode
from agents.risk_agent import RiskAnalystNode
from models.reasoning import ReasoningState
from retrieval.foundry import RetrievalNode


class GraphEnvelope(TypedDict):
    state: dict[str, Any]


class ConsensusReasoningGraph:
    """Owns graph orchestration and state transitions.

    The preferred runtime is LangGraph. The local fallback preserves the same
    node ordering and state contract when LangGraph is unavailable, which keeps
    the mocked MVP runnable on Python distributions without native wheels.
    """

    def __init__(self) -> None:
        self.nodes = [
            ("retrieval", RetrievalNode()),
            ("planner", PlannerNode()),
            ("risk_analyst", RiskAnalystNode()),
            ("evidence_analyst", EvidenceAnalystNode()),
            ("alternatives_analyst", AlternativesAnalystNode()),
            ("consensus_judge", ConsensusJudgeNode()),
        ]
        self._compiled_graph = self._compile_langgraph()

    def invoke(self, question: str) -> ReasoningState:
        initial_state = ReasoningState(question=question.strip())

        if self._compiled_graph is None:
            return self._invoke_local_graph(initial_state)

        result = self._compiled_graph.invoke({"state": initial_state.dict()})
        return ReasoningState.parse_obj(result["state"])

    def _compile_langgraph(self) -> Any | None:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        graph = StateGraph(GraphEnvelope)
        for name, node in self.nodes:
            graph.add_node(name, self._as_langgraph_node(node))

        graph.set_entry_point("retrieval")
        graph.add_edge("retrieval", "planner")
        graph.add_edge("planner", "risk_analyst")
        graph.add_edge("risk_analyst", "evidence_analyst")
        graph.add_edge("evidence_analyst", "alternatives_analyst")
        graph.add_edge("alternatives_analyst", "consensus_judge")
        graph.add_edge("consensus_judge", END)
        return graph.compile()

    def _as_langgraph_node(
        self, node: Callable[[ReasoningState], ReasoningState]
    ) -> Callable[[GraphEnvelope], GraphEnvelope]:
        def wrapped(envelope: GraphEnvelope) -> GraphEnvelope:
            current_state = ReasoningState.parse_obj(envelope["state"])
            next_state = node(current_state)
            return {"state": next_state.dict()}

        return wrapped

    def _invoke_local_graph(self, state: ReasoningState) -> ReasoningState:
        current_state = state
        for _, node in self.nodes:
            current_state = node(current_state)
        return current_state


def analyze_question(question: str) -> ReasoningState:
    return ConsensusReasoningGraph().invoke(question)
