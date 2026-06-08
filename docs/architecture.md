# ConsensusIQ Architecture

## Mission

ConsensusIQ is a vertical slice for transparent multi-agent reasoning. The current implementation uses mocked Foundry IQ retrieval and mocked agents so the architecture can be evaluated before Azure integration.

## Flow

```text
User Question
  -> Foundry IQ Retrieval
  -> Planner Agent
  -> Risk Analyst Agent
  -> Evidence Analyst Agent
  -> Alternative Solutions Agent
  -> Consensus Judge
  -> Consensus Report
```

## Components

### Frontend

The Next.js app provides a single-page workbench with:

- question input
- submit action
- consensus answer
- confidence and agreement metrics
- agent output cards
- disagreement analysis

The API client is isolated in `frontend/src/lib/api.ts` and reads `NEXT_PUBLIC_API_BASE_URL`.

### Backend

The FastAPI app exposes:

- `GET /health`
- `POST /analyze`

The `/analyze` route coordinates the mocked reasoning pipeline:

1. `retrieval/foundry.py` returns normalized `EvidenceItem` records.
2. `agents/planner.py` creates a reasoning plan.
3. Specialist agents inspect the same evidence from different roles.
4. `agents/consensus_agent.py` calculates confidence, agreement, final answer, and disagreements.

## Extension Points

- Replace `retrieve_evidence` with Azure AI Foundry / Foundry IQ retrieval.
- Convert simple agent functions into LangGraph nodes. Optional packages for that phase are listed in `backend/requirements-agents.txt`.
- Add real Azure OpenAI calls behind each agent module. Optional packages for that phase are listed in `backend/requirements-azure.txt`.
- Add source citations and retrieval quality scoring to `EvidenceItem`.
- Add streaming updates for agent progress once the MVP API contract is stable.

## MVP Non-Goals

- No Azure API calls.
- No authentication.
- No database.
- No persistent sessions.
- No deployment-specific configuration.
