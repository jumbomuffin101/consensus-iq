import os

from dotenv import load_dotenv


LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def parse_frontend_origins(value: str | None = None) -> list[str]:
    load_dotenv()
    configured = os.getenv("FRONTEND_ORIGIN", "") if value is None else value
    origins: list[str] = []

    for origin in [*configured.split(","), *LOCAL_FRONTEND_ORIGINS]:
        normalized = origin.strip().rstrip("/")
        if normalized and normalized != "*" and normalized not in origins:
            origins.append(normalized)

    return origins


def openrouter_configured() -> bool:
    load_dotenv()
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def openrouter_model() -> str:
    load_dotenv()
    return os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip() or "openai/gpt-4o-mini"


def openrouter_base_url() -> str:
    load_dotenv()
    return (
        os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        or "https://openrouter.ai/api/v1"
    )


def openrouter_app_name() -> str:
    load_dotenv()
    return os.getenv("OPENROUTER_APP_NAME", "ConsensusIQ").strip() or "ConsensusIQ"


def azure_openai_configured() -> bool:
    load_dotenv()
    return all(
        os.getenv(name, "").strip()
        for name in [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION",
        ]
    )


def azure_search_configured() -> bool:
    load_dotenv()
    return all(
        os.getenv(name, "").strip()
        for name in [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
        ]
    )


def live_llm_enabled() -> bool:
    load_dotenv()
    return os.getenv("USE_LIVE_LLM", "false").strip().lower() == "true"


def prefer_azure_openai() -> bool:
    load_dotenv()
    return os.getenv("PREFER_AZURE_OPENAI", "false").strip().lower() == "true"


def active_reasoning_order() -> list[str]:
    order: list[str] = []
    if azure_openai_configured() and prefer_azure_openai():
        order.append("AzureOpenAI")
    if openrouter_configured():
        order.append("OpenRouter")
    if azure_openai_configured() and not prefer_azure_openai():
        order.append("AzureOpenAI")
    order.append("FastDeterministic")
    return order
