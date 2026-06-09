# ConsensusIQ

**Evidence-grounded multi-agent consensus for high-stakes decisions.**

- **Hackathon:** Microsoft Agents League
- **Challenge track:** Reasoning Agents
- **Microsoft IQ layer:** Microsoft Foundry IQ-style citation retrieval
- **License:** MIT

## Safety Disclaimer

ConsensusIQ is a reasoning and decision-support demo. It is not a substitute for professional medical, legal, financial, or security advice.

## Problem Statement

High-stakes decisions often depend on incomplete evidence, competing interpretations, and hidden disagreement. A single AI answer can sound confident while missing risks, alternatives, or source gaps. Teams need a transparent way to compare reasoning perspectives, detect disagreement, and produce a grounded recommendation with visible confidence.

## Solution Overview

ConsensusIQ turns one user question into a structured multi-agent review. It retrieves citation-ready context, asks specialist agents to reason independently, detects disagreement, and produces a final consensus report with confidence and agreement scores. The app is reliable for demos because Azure OpenAI and Foundry IQ integrations both have mock fallback providers.

## Architecture

```text
User Question
  -> Foundry IQ Retrieval Provider
     -> citation-ready sources
  -> Planner Agent
     -> structured reasoning tasks
  -> Specialist Agents in Parallel
     -> Risk Analyst
     -> Evidence Analyst
     -> Alternatives Analyst
  -> Disagreement Detection
     -> conflicting recommendations
     -> confidence gaps
     -> missing evidence
  -> Consensus Judge
     -> final recommendation
     -> confidence score
     -> agreement score
     -> reasoning summary
```

## Key Features

- Multi-agent reasoning with planner, risk, evidence, alternatives, and consensus judge roles.
- Microsoft Foundry IQ-style retrieval abstraction with citation IDs and source snippets.
- Azure OpenAI-ready LLM provider with retries, timeouts, and mock fallback.
- Parallel specialist agent execution.
- Structured disagreement detection.
- Confidence and agreement scoring.
- Stable `POST /analyze` API contract.
- Local demo reliability without secrets or external services.

## Tech Stack

- **Frontend:** Next.js 15, TypeScript, Tailwind CSS, shadcn-style local UI primitives
- **Backend:** Python 3.12, FastAPI, Pydantic
- **Reasoning:** LangGraph-ready graph orchestration with deterministic local fallback
- **LLM:** Azure OpenAI provider abstraction with mock fallback
- **Retrieval:** Foundry IQ provider abstraction with mock fallback

## Setup Instructions

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8000
```

If your Python distribution creates a Unix-style venv on Windows, use `.venv\bin\Activate.ps1`.

Optional Azure OpenAI support:

```bash
cd backend
pip install -r requirements-azure.txt
```

Set in `backend/.env`:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21
```

Optional Foundry IQ retrieval:

```bash
FOUNDRY_IQ_ENDPOINT=https://your-foundry-iq-endpoint
FOUNDRY_IQ_API_KEY=your-key
FOUNDRY_IQ_INDEX_NAME=your-index-name
FOUNDRY_IQ_API_VERSION=2024-05-01-preview
```

If Azure OpenAI or Foundry IQ is unavailable, ConsensusIQ falls back to mock providers and still returns a complete report.

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Deployment Readiness

- Backend Dockerfile: `backend/Dockerfile`
- Local Docker Compose: `docker-compose.yml`
- Deployment guide: `docs/deployment.md`
- Backend health checks: `GET /health` and `GET /ready`
- `POST /analyze` includes optional timing metadata for demo observability.

## Demo Script

### 60-second explanation

ConsensusIQ is an evidence-grounded consensus platform. A user asks a decision question, Foundry IQ-style retrieval returns citation-ready sources, and multiple specialist agents reason independently: risk, evidence, and alternatives. ConsensusIQ compares their conclusions, detects disagreement, and produces a final recommendation with confidence and agreement scores.

### Recommended demo prompt

```text
Should a 63-year-old patient with new-onset focal seizure receive MRI before lumbar puncture?
```

### What judges should notice

- The agents do not simply agree by default; they expose risk, evidence, alternatives, and disagreement.
- Claims are tied to visible source citation IDs.
- The final answer includes confidence and agreement scores.
- Foundry IQ is represented as the grounding layer that reduces hallucination by forcing evidence-backed claims through citation-ready retrieved context.
- The demo remains reliable locally because Azure OpenAI and Foundry IQ both have fallback providers.

## Screenshots

Add final screenshots before submission:

- Main interface: `docs/screenshots/main-interface.png`
- Consensus report: `docs/screenshots/consensus-report.png`
- Agent disagreement view: `docs/screenshots/agent-disagreement-view.png`
- Sources section: `docs/screenshots/sources-section.png`

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

## Safety and Reliability

- Public disclaimer is shown in the app and README.
- No secrets are committed; credentials belong only in local `.env` files.
- Azure OpenAI failures fall back to deterministic mock reasoning.
- Foundry IQ failures fall back to clearly marked mock sources.
- Source citations are visible so judges can inspect grounding.
- No authentication or database is required for the local demo.

## Final Validation Notes

- Frontend production build passed with `npm run build`.
- Backend `/analyze` was smoke tested through FastAPI.
- Azure credentials are not committed.
- Mock fallback allows a reliable demo even without Azure OpenAI or Foundry IQ credentials.

## Future Roadmap

- Connect to a live Foundry IQ project and tune response normalization.
- Use Azure OpenAI deployments for all agent roles in production mode.
- Add streaming graph updates for live agent progress.
- Add more specialist agents for domain-specific review.
- Add exportable consensus reports for judges, reviewers, or decision teams.

## Project Structure

```text
consensus-iq/
  frontend/
  backend/
  docs/
```

See `docs/architecture.md` for deeper architecture notes.
