# ConsensusIQ Architecture

## Mission

ConsensusIQ is a multi-agent reasoning platform for transparent, evidence-based
consensus decisions. The current implementation keeps retrieval and agents
mocked, but the architecture is structured for Microsoft Foundry IQ and Azure
OpenAI integration.

## Shared State

The reasoning pipeline passes a typed Pydantic `ReasoningState` through every
node. The state includes:

- `question`
- `retrieved_context`
- `reasoning_tasks`
- `agent_outputs`
- `disagreements`
- `consensus`
- `confidence_score`
- `agreement_score`
- `reasoning_summary`

The models live in `backend/models/reasoning.py`.

## Graph Flow

```text
User Question
  -> RetrievalNode
  -> PlannerNode
  -> RiskAnalystNode
  -> EvidenceAnalystNode
  -> AlternativesAnalystNode
  -> ConsensusJudgeNode
  -> Consensus Report
```

`backend/reasoning/graph.py` owns orchestration. It uses LangGraph when the
package is available, and a deterministic local graph runner otherwise so the
mocked MVP remains runnable in constrained local Python environments.

## Node Responsibilities

- `RetrievalNode`: returns mocked Foundry IQ `RetrievedContext` records.
- `PlannerNode`: decomposes the question into structured reasoning tasks.
- `RiskAnalystNode`: identifies risks, limitations, and failure modes.
- `EvidenceAnalystNode`: evaluates evidence and supporting rationale.
- `AlternativesAnalystNode`: proposes alternative interpretations and approaches.
- `ConsensusJudgeNode`: consumes agent outputs, invokes disagreement detection,
  and produces the final recommendation, confidence score, agreement score, and
  reasoning summary.

## Disagreement Detection

`backend/reasoning/disagreement.py` compares independent agent outputs for:

- conflicting recommendations
- differing confidence levels
- missing evidence

The detector produces structured `Disagreement` objects and calculates an
agreement score used by the consensus judge.

## Extension Points

- Replace `retrieve_evidence` with Microsoft Foundry IQ retrieval while
  preserving `RetrievedContext`.
- Replace mocked specialist node bodies with Azure OpenAI calls that return
  `AgentOutput`.
- Add more specialist nodes by registering them in `ConsensusReasoningGraph`.
- Move sequential specialist execution to parallel graph branches once real
  latency and dependency needs are known.
- Add streaming node updates after the state contract stabilizes.

## MVP Non-Goals

- No authentication.
- No database.
- No deployment-specific configuration.
- No Azure API calls yet.
