# ConsensusIQ Architecture

## Mission

ConsensusIQ is a multi-agent reasoning platform for transparent, evidence-based
consensus decisions. Retrieval is provider-based for Microsoft Foundry IQ with a
demo corpus fallback, and agents are provider-based for Azure OpenAI with mock
fallback.

## Shared State

The reasoning pipeline passes a typed Pydantic `ReasoningState` through every
node. The state includes:

- `question`
- `scenario_label`
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
  -> EvidenceAnalystNode       (parallel specialist execution)
  -> AlternativesAnalystNode
  -> ConsensusJudgeNode
  -> Consensus Report
```

`backend/reasoning/graph.py` owns orchestration. It uses LangGraph when the
package is available, and a deterministic local graph runner otherwise so the
mocked MVP remains runnable in constrained local Python environments.

## LLM Providers

Agents receive an injected provider from `backend/llm`.

- `llm/base.py`: provider contract and resilient fallback wrapper.
- `llm/azure_openai.py`: Azure OpenAI JSON completion provider with retries and
  request timeout.
- `llm/mock.py`: deterministic fallback provider.
- `llm/factory.py`: reads `AZURE_OPENAI_*` environment variables and chooses the
  provider.

Specialist agents run concurrently after planning. Each receives the question,
retrieved context, and planner tasks, and returns a validated `AgentOutput`.
The consensus judge receives all specialist outputs plus the deterministic
disagreement report.

## Foundry IQ Retrieval

Retrieval providers live in `backend/retrieval`.

- `retrieval/base.py`: provider contract, resilient fallback wrapper, and graph node.
- `retrieval/foundry.py`: Microsoft Foundry IQ HTTP provider, request payload
  builder, API key/index configuration, and response normalization.
- `retrieval/mock.py`: domain-specific demo corpus sources for reliable demos.
- `retrieval/factory.py`: reads `FOUNDRY_IQ_*` environment variables and chooses
  the provider.

Foundry IQ reduces hallucination risk by grounding every agent in retrieved
context before generation. The shared state stores citation-ready sources with
`citation_id`, `title`, `source`, `url`, `snippet`, and `relevance_score`.
Agents are instructed to cite `citation_id` values in `evidence_refs`.

Required Foundry IQ variables:

- `FOUNDRY_IQ_ENDPOINT`
- `FOUNDRY_IQ_API_KEY`
- `FOUNDRY_IQ_INDEX_NAME`
- `FOUNDRY_IQ_API_VERSION`

If any value is missing or the provider fails, `MockRetrievalProvider` returns
clearly marked demo corpus sources so `/analyze` remains reliable. The fallback
provider classifies prompts as clinical, cybersecurity, research, enterprise,
finance, or custom so local demos still exercise citation-grounded reasoning.

## Node Responsibilities

- `RetrievalNode`: retrieves citation-ready `RetrievedContext` records from
  Foundry IQ or mock fallback.
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
agreement score used by the consensus judge. The consensus judge combines
agreement, source relevance, domain risk, question ambiguity, prompt-injection
signals, and missing evidence into the final confidence score.

## Extension Points

- Adjust Foundry IQ response normalization in `retrieval/foundry.py` if the live
  project returns different field names.
- Replace mocked specialist node bodies with Azure OpenAI calls that return
  `AgentOutput`; the provider abstraction already supports this.
- Add more specialist nodes by registering them in `ConsensusReasoningGraph`.
- Move sequential specialist execution to parallel graph branches once real
  latency and dependency needs are known.
- Add streaming node updates after the state contract stabilizes.

## MVP Non-Goals

- No authentication.
- No database.
- No deployment-specific configuration.
- No authentication, database, or deployment-specific configuration.
