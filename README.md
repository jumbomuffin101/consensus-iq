# ConsensusIQ

Evidence-based agent consensus platform for the Microsoft Agents League hackathon.

ConsensusIQ demonstrates a grounded multi-agent reasoning workflow using mocked Foundry IQ retrieval, typed shared state, graph-based agent orchestration, disagreement detection, and transparent consensus scoring. The current MVP is intentionally local-first and extensible, with Azure integration points isolated for later work.

## Stack

- Frontend: Next.js 15, TypeScript, Tailwind CSS, shadcn-style local UI primitives
- Backend: Python 3.12, FastAPI, LangGraph-ready agent modules
- Retrieval: mocked Foundry IQ adapter
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

No Foundry IQ integration is required yet; retrieval remains mocked.

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
  "disagreements": []
}
```

## Project Structure

```text
consensus-iq/
  frontend/
  backend/
  docs/
```

See `docs/architecture.md` for the agent flow and extension points.
