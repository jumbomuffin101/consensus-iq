# ConsensusIQ

Multi-agent decision support with evidence retrieval, disagreement analysis, and consensus generation.

## Problem

Most AI apps give one answer from one model. For complex decisions, that is not enough. Users need to see risks, alternatives, uncertainty, and where the available evidence is limited.

## Solution

ConsensusIQ routes each question through specialist agents that review the decision from different angles:

- Risk Analyst
- Evidence Analyst
- Alternative Solutions Analyst
- Consensus Judge

The result is a structured recommendation with confidence, agreement, disagreement areas, and retrieved evidence when relevant sources are available.

## How It Works

```text
User Question
-> Azure AI Search / Foundry IQ Search Service
-> Specialist Agents
-> Disagreement Analysis
-> Consensus Judge
-> Final Recommendation
```

## Microsoft Technologies

- Azure AI Search / Foundry IQ Search Service is used as the evidence retrieval layer when configured.
- Microsoft Foundry resources were used for AI infrastructure exploration and experimentation.
- Foundry IQ Knowledge Base integration was investigated, but the deployed app uses direct Azure Search retrieval because of model-region constraints.
- When Azure Search is unavailable or returns weak matches, the app falls back to a curated public evidence corpus through the same retrieval interface.

## Key Features

- Multi-agent reasoning
- Evidence retrieval
- Confidence assessment
- Disagreement detection
- Custom prompt support
- Fast deterministic demo mode

## Tech Stack

Frontend:

- Next.js
- TypeScript
- Tailwind CSS

Backend:

- FastAPI
- Python
- Azure AI Search
- OpenRouter optional provider
- Render
- Vercel

## Demo

- Frontend:
- Backend health:
- API docs:

## Example Prompts

- Should a university use AI to screen medical school applications?
- Should a company allow public AI tools for confidential client documents?
- Should a single LLM grade student work?
- Should a city replace traffic lights with roundabouts?

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8000
```

If your Python environment creates a Unix-style virtual environment on Windows, use:

```bash
.\.venv\bin\Activate.ps1
```

Backend health checks:

```bash
GET http://localhost:8000/health
GET http://localhost:8000/ready
```

API docs:

```bash
http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

Open:

```bash
http://localhost:3000
```

For local development, set:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Environment Variables

Backend:

```bash
FRONTEND_ORIGIN=http://localhost:3000,http://127.0.0.1:3000

AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=
AZURE_SEARCH_API_VERSION=2024-07-01

USE_LIVE_LLM=false

OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_NAME=ConsensusIQ
```

Frontend:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Notes:

- `USE_LIVE_LLM=false` is the default demo mode. It keeps the app fast and deterministic.
- OpenRouter is optional and is only used when `USE_LIVE_LLM=true`.
- Azure Search is optional for local development. If it is not configured, the backend uses the curated public corpus fallback.

## API

`POST /analyze`

Request:

```json
{
  "question": "Should a city replace traffic lights with roundabouts?"
}
```

Response includes:

- `consensus`
- `scenario_label`
- `confidence_score`
- `agreement_score`
- `reasoning_summary`
- `agent_outputs`
- `disagreements`
- `sources`
- optional timing metadata

## Deployment

Frontend:

- Deploy `frontend/` to Vercel.
- Set `NEXT_PUBLIC_API_URL` to the deployed backend URL.

Backend:

- Deploy `backend/` to Render.
- Set Azure Search environment variables if using live retrieval.
- Keep `USE_LIVE_LLM=false` for the stable demo path.

Additional deployment notes are in `docs/deployment.md`.

## Safety Note

ConsensusIQ is a decision-support demo. It is not professional medical, legal, financial, or security advice.

## Status

Built for Agents League Hackathon 2026.

