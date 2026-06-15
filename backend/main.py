import logging
import os
from time import perf_counter

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import (
    active_reasoning_order,
    azure_openai_configured,
    azure_search_configured,
    openrouter_configured,
    openrouter_model,
    parse_frontend_origins,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("consensus_iq.api")
logger.info(
    "provider startup config openrouter_configured=%s openrouter_model=%s azure_configured=%s azure_search_configured=%s",
    openrouter_configured(),
    openrouter_model(),
    azure_openai_configured(),
    azure_search_configured(),
)

app = FastAPI(
    title="ConsensusIQ API",
    description="Evidence-based multi-agent consensus analysis API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_frontend_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    elapsed_ms = int((perf_counter() - start) * 1000)
    logger.info(
        "%s %s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "ConsensusIQ API"}


@app.get("/health/providers")
async def health_providers() -> dict[str, object]:
    return {
        "azure_configured": azure_openai_configured(),
        "azure_search_configured": azure_search_configured(),
        "openrouter_configured": openrouter_configured(),
        "openrouter_model": openrouter_model(),
        "active_reasoning_order": active_reasoning_order(),
    }


@app.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}
