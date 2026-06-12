import logging
import os

from dotenv import load_dotenv

from llm.azure_openai import AzureOpenAIProvider
from llm.base import BaseLLMProvider, ResilientLLMProvider
from llm.mock import MockLLMProvider
from llm.openrouter import OpenRouterProvider


logger = logging.getLogger("consensus_iq.llm")


def create_llm_provider() -> BaseLLMProvider:
    """Create the configured provider.

    Priority:
    1. Azure OpenAI when fully configured.
    2. OpenRouter when OPENROUTER_API_KEY is configured.
    3. Deterministic mock provider.

    Live providers are wrapped with a mock fallback so transient API errors never
    crash the reasoning pipeline.
    """

    load_dotenv()

    mock_provider = MockLLMProvider()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

    if all([endpoint, api_key, deployment, api_version]):
        try:
            azure_provider = AzureOpenAIProvider(
                endpoint=endpoint,
                api_key=api_key,
                deployment=deployment,
                api_version=api_version,
            )
            logger.info("Active LLM provider: AzureOpenAI")
            return ResilientLLMProvider(azure_provider, mock_provider)
        except Exception as exc:
            logger.warning("AzureOpenAI provider initialization failed: %s", exc)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
    openrouter_base_url = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    ).strip()
    openrouter_app_name = os.getenv("OPENROUTER_APP_NAME", "ConsensusIQ").strip()

    if openrouter_api_key:
        openrouter_provider = OpenRouterProvider(
            api_key=openrouter_api_key,
            model=openrouter_model,
            base_url=openrouter_base_url,
            app_name=openrouter_app_name,
        )
        logger.info("Active LLM provider: OpenRouter")
        return ResilientLLMProvider(openrouter_provider, mock_provider)

    logger.info("Active LLM provider: Mock")
    return mock_provider
