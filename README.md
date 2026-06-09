# ConsensusIQ

Evidence-based agent consensus platform for the Microsoft Agents League hackathon.

ConsensusIQ demonstrates a grounded multi-agent reasoning workflow using Foundry IQ-ready retrieval, typed shared state, graph-based agent orchestration, disagreement detection, and transparent consensus scoring. The current MVP is intentionally local-first and uses mock fallbacks when external services are unavailable.

## Stack

- Frontend: Next.js 15, TypeScript, Tailwind CSS, shadcn-style local UI primitives
- Backend: Python 3.12, FastAPI, LangGraph-ready agent modules
- Retrieval: Microsoft Foundry IQ provider abstraction with mock fallback
- Persistence/auth: none for MVP

## Local Setup

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8000
```

If your Python distribution creates a Unix-style venv on Windows, use `.venv\bin\Activate.ps1` instead.

LangGraph is listed in `backend/requirements-agents.txt`. The graph runner uses LangGraph when installed and falls back to a deterministic local graph runner when native package wheels are unavailable. OpenAI/Azure SDK integration starts from `backend/requirements-azure.txt`.

### Azure OpenAI

The backend uses a provider factory. If all Azure variables are present and the optional SDK is installed, agents use Azure OpenAI. If Azure is missing, times out, returns invalid JSON, or the SDK is unavailable, ConsensusIQ falls back to the mock provider and keeps the API response stable.

```bash
cd backend
pip install -r requirements-azure.txt
```

Set these in `backend/.env`:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21
```

### Foundry IQ Retrieval

Foundry IQ is the grounding layer. It retrieves citation-ready context before agents reason, so specialist outputs can cite `citation_id` values instead of relying only on model memory. This reduces hallucination risk by making evidence visible, auditable, and available in the API response.

Set these in `backend/.env` to enable Foundry IQ retrieval:

```bash
FOUNDRY_IQ_ENDPOINT=https://your-foundry-iq-endpoint
FOUNDRY_IQ_API_KEY=your-key
FOUNDRY_IQ_INDEX_NAME=your-index-name
FOUNDRY_IQ_API_VERSION=2024-05-01-preview
```

If any value is missing or Foundry IQ is unavailable, the backend uses mock retrieval and still returns clearly marked mock sources.

The API runs at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

The app runs at `http://localhost:3000`.

## API

`POST /analyze`

```json
{
  "question": "Should we adopt a four-day work week?"
}
```

Returns:

```json
{
  "consensus": "...",
  "confidence_score": 0.91,
  "agreement_score": 0.87,
  "reasoning_summary": "...",
  "agent_outputs": [],
  "disagreements": [],
  "sources": [
    {
      "citation_id": "S1",
      "title": "...",
      "source": "...",
      "url": "...",
      "snippet": "...",
      "relevance_score": 0.92
    }
  ]
}
```

## Demo Script

### 60-second explanation

ConsensusIQ is an evidence-grounded consensus platform. A user asks a decision question, Foundry IQ-style retrieval returns citation-ready sources, and multiple specialist agents reason independently: risk, evidence, and alternatives. ConsensusIQ then compares their conclusions, detects disagreement, and produces a final recommendation with confidence and agreement scores.

### Recommended demo prompt

```text
Should a 63-year-old patient with new-onset focal seizure receive MRI before lumbar puncture?
```

### What judges should notice

- The agents do not simply agree by default; they expose risk, evidence, alternatives, and disagreement.
- Claims are tied back to visible source citation IDs.
- The final answer includes both confidence and agreement, making uncertainty explicit.
- The demo remains reliable locally because Azure OpenAI and Foundry IQ both have mock fallback providers.

## Project Structure

```text
consensus-iq/
  frontend/
  backend/
  docs/
```

See `docs/architecture.md` for the agent flow and extension points.
