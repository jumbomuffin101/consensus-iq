from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from time import perf_counter
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


logger = logging.getLogger("consensus_iq.reasoning")


class ConsensusReasoningGraph:
    """Owns graph orchestration and state transitions.

    The preferred runtime is LangGraph. The local fallback preserves the same
    node ordering and state contract when LangGraph is unavailable, which keeps
    the mocked MVP runnable on Python distributions without native wheels.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or create_llm_provider()
        logger.info("Active LLM provider for analysis graph: %s", self.provider.name)
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
        start = perf_counter()
        initial_state = ReasoningState(question=question.strip())

        if self._compiled_graph is None:
            final_state = self._invoke_local_graph(initial_state)
        else:
            result = self._compiled_graph.invoke({"state": initial_state.dict()})
            final_state = ReasoningState.parse_obj(result["state"])

        execution_time_ms = self._elapsed_ms(start)
        logger.info("analysis completed in %sms", execution_time_ms)
        return final_state.with_timing("execution_time_ms", execution_time_ms)

    def _compile_langgraph(self) -> Any | None:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        graph = StateGraph(GraphEnvelope)
        for name, node in self.nodes:
            graph.add_node(name, self._as_langgraph_node(name, node))

        graph.set_entry_point("retrieval")
        graph.add_edge("retrieval", "planner")
        graph.add_edge("planner", "specialists_parallel")
        graph.add_edge("specialists_parallel", "consensus_judge")
        graph.add_edge("consensus_judge", END)
        return graph.compile()

    def _as_langgraph_node(
        self, name: str, node: Callable[[ReasoningState], ReasoningState]
    ) -> Callable[[GraphEnvelope], GraphEnvelope]:
        def wrapped(envelope: GraphEnvelope) -> GraphEnvelope:
            current_state = ReasoningState.parse_obj(envelope["state"])
            next_state = self._run_timed_node(name, node, current_state)
            return {"state": next_state.dict()}

        return wrapped

    def _invoke_local_graph(self, state: ReasoningState) -> ReasoningState:
        current_state = state
        for name, node in self.nodes:
            current_state = self._run_timed_node(name, node, current_state)
        return current_state

    def _run_specialists_parallel(self, state: ReasoningState) -> ReasoningState:
        start = perf_counter()
        outputs_by_agent: dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=len(self.specialist_nodes)) as executor:
            future_to_node = {
                executor.submit(self._run_specialist_node, node, state): node
                for node in self.specialist_nodes
            }
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    next_state = future.result()
                except Exception:
                    logger.exception("specialist agent failed; using deterministic fallback")
                    fallback_output = self._fallback_agent_output(node, state)
                    if fallback_output is not None:
                        outputs_by_agent[fallback_output.agent] = fallback_output
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
        elapsed_ms = self._elapsed_ms(start)
        logger.info("specialist agents completed in %sms", elapsed_ms)
        return state.copy(update={"agent_outputs": ordered_outputs}).with_timing(
            "agent_time_ms", elapsed_ms
        )

    def _run_specialist_node(self, node: Any, state: ReasoningState) -> ReasoningState:
        label = self._node_label(node)
        logger.info("%s started", label)
        start = perf_counter()
        next_state = node(state)
        logger.info("%s completed in %sms", label, self._elapsed_ms(start))
        return next_state

    def _fallback_agent_output(self, node: Any, state: ReasoningState) -> Any | None:
        fallback_builder = getattr(node, "_fallback_output", None)
        if not callable(fallback_builder):
            return None
        try:
            return fallback_builder(state)
        except Exception:
            logger.exception("deterministic specialist fallback failed")
            return None

    def _run_timed_node(
        self,
        name: str,
        node: Callable[[ReasoningState], ReasoningState],
        state: ReasoningState,
    ) -> ReasoningState:
        label = self._node_label(node, name)
        logger.info("%s started", label)
        start = perf_counter()
        next_state = node(state)
        elapsed_ms = self._elapsed_ms(start)
        logger.info("%s completed in %sms", label, elapsed_ms)

        timing_field = {
            "retrieval": "retrieval_time_ms",
            "consensus_judge": "consensus_time_ms",
        }.get(name)
        if timing_field:
            return next_state.with_timing(timing_field, elapsed_ms)
        return next_state

    def _elapsed_ms(self, start: float) -> int:
        return int((perf_counter() - start) * 1000)

    def _node_label(self, node: Any, fallback: str | None = None) -> str:
        if isinstance(node, PlannerNode):
            return "planner"
        if isinstance(node, RiskAnalystNode):
            return "risk analyst"
        if isinstance(node, EvidenceAnalystNode):
            return "evidence analyst"
        if isinstance(node, AlternativesAnalystNode):
            return "alternatives analyst"
        if isinstance(node, ConsensusJudgeNode):
            return "consensus judge"
        if fallback == "retrieval":
            return "retrieval"
        if fallback == "specialists_parallel":
            return "specialist agents"
        return fallback or node.__class__.__name__


def analyze_question(question: str) -> ReasoningState:
    return ConsensusReasoningGraph().invoke(question)
