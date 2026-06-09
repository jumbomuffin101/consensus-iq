from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypedDict

from agents.alternatives_agent import AlternativesAnalystNode
from agents.consensus_agent import ConsensusJudgeNode
from agents.evidence_agent import EvidenceAnalystNode
from agents.planner import PlannerNode
from agents.risk_agent import RiskAnalystNode
from llm.base import BaseLLMProvider
from llm.factory import create_llm_provider
from models.reasoning import ReasoningState
from retrieval import RetrievalNode


class GraphEnvelope(TypedDict):
    state: dict[str, Any]


class ConsensusReasoningGraph:
    """Owns graph orchestration and state transitions.

    The preferred runtime is LangGraph. The local fallback preserves the same
    node ordering and state contract when LangGraph is unavailable, which keeps
    the mocked MVP runnable on Python distributions without native wheels.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or create_llm_provider()
        self.retrieval_node = RetrievalNode()
        self.planner_node = PlannerNode(self.provider)
        self.specialist_nodes = [
            RiskAnalystNode(self.provider),
            EvidenceAnalystNode(self.provider),
            AlternativesAnalystNode(self.provider),
        ]
        self.consensus_node = ConsensusJudgeNode(self.provider)
        self.nodes = [
            ("retrieval", self.retrieval_node),
            ("planner", self.planner_node),
            ("specialists_parallel", self._run_specialists_parallel),
            ("consensus_judge", self.consensus_node),
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
        graph.add_edge("planner", "specialists_parallel")
        graph.add_edge("specialists_parallel", "consensus_judge")
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

    def _run_specialists_parallel(self, state: ReasoningState) -> ReasoningState:
        outputs_by_agent: dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=len(self.specialist_nodes)) as executor:
            future_to_node = {
                executor.submit(node, state): node for node in self.specialist_nodes
            }
            for future in as_completed(future_to_node):
                try:
                    next_state = future.result()
                except Exception:
                    continue

                for output in next_state.agent_outputs:
                    outputs_by_agent[output.agent] = output

        ordered_outputs = [
            outputs_by_agent[agent_name]
            for agent_name in [
                "Risk Analyst Agent",
                "Evidence Analyst Agent",
                "Alternative Solutions Agent",
            ]
            if agent_name in outputs_by_agent
        ]
        return state.copy(update={"agent_outputs": ordered_outputs})


def analyze_question(question: str) -> ReasoningState:
    return ConsensusReasoningGraph().invoke(question)
