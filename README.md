# ConsensusIQ

Evidence-based agent consensus platform for the Microsoft Agents League hackathon.

ConsensusIQ demonstrates a grounded multi-agent reasoning workflow using mocked Foundry IQ retrieval and mocked agent collaboration. The current MVP is intentionally local-first and extensible, with Azure integration points isolated for later work.

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

LangChain/LangGraph are listed in `backend/requirements-agents.txt`, and OpenAI/Azure SDK integration starts from `backend/requirements-azure.txt`. The MVP does not need them because retrieval and agent responses are mocked.

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
  "confidence": 0.91,
  "agreement_score": 0.87,
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
