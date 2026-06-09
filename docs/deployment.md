# Deployment Guide

ConsensusIQ is split into a Next.js frontend and FastAPI backend. The frontend can deploy on Vercel, and the backend can deploy on Render, Azure Container Apps, Azure App Service, or any container platform.

## Production Environment Variables

### Backend

Required:

- `APP_ENV=production`
- `FRONTEND_ORIGIN=https://your-frontend-domain`
- `LOG_LEVEL=INFO`

Optional Azure OpenAI:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

Optional Foundry IQ:

- `FOUNDRY_IQ_ENDPOINT`
- `FOUNDRY_IQ_API_KEY`
- `FOUNDRY_IQ_INDEX_NAME`
- `FOUNDRY_IQ_API_VERSION`

If Azure OpenAI or Foundry IQ variables are missing or unavailable, ConsensusIQ uses local fallback providers so demos continue to run. The public retrieval fallback is a clearly labeled Foundry IQ Retrieval Layer demo corpus.

### Frontend

- `NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain`

## Health Checks

Backend:

- `GET /health` returns `{ "status": "ok" }`
- `GET /ready` returns `{ "status": "ready" }`

Use `/health` for container health checks and `/ready` for platform readiness checks.

## Local Docker Development

```bash
docker compose up --build
```

Frontend: `http://localhost:3000`

Backend: `http://localhost:8000`

## Vercel Frontend Deployment

1. Create a Vercel project from the repository.
2. Set the root directory to `frontend`.
3. Add environment variable:

```bash
NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain
```

4. Deploy with the default Next.js build settings.

## Render Backend Deployment

Option A: Docker deployment

1. Create a new Render Web Service.
2. Select Docker runtime.
3. Set Dockerfile path to `backend/Dockerfile`.
4. Add backend environment variables.
5. Set health check path to `/health`.

Option B: Python deployment

1. Root directory: `backend`
2. Build command:

```bash
pip install -r requirements.txt
```

3. Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

4. Set health check path to `/health`.

## Azure Deployment Notes

Recommended backend targets:

- Azure Container Apps using `backend/Dockerfile`
- Azure App Service for Containers
- Azure Web App for Python

Recommended settings:

- Configure `FRONTEND_ORIGIN` to the deployed Vercel or Azure frontend URL.
- Store `AZURE_OPENAI_API_KEY` and `FOUNDRY_IQ_API_KEY` as secrets.
- Use `/health` as the health probe path.
- Keep local fallback providers enabled for demo reliability.

## Observability

The backend logs:

- request method, path, status, and duration
- retrieval timing
- specialist agent timing
- consensus timing
- total analysis timing

`POST /analyze` also returns optional metadata:

```json
{
  "metadata": {
    "execution_time_ms": 123,
    "retrieval_time_ms": 12,
    "agent_time_ms": 80,
    "consensus_time_ms": 20
  }
}
```
