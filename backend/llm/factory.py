import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from llm.azure_openai import AzureOpenAIProvider
from llm.base import BaseLLMProvider, ResilientLLMProvider
from llm.mock import MockLLMProvider
from llm.openrouter import OpenRouterProvider


logger = logging.getLogger("consensus_iq.llm")


@dataclass(frozen=True)
class LLMProviderSelection:
    provider_name: str
    live_llm_enabled: bool
    openrouter_api_key_present: bool
    openrouter_model: str
    openrouter_base_url: str
    azure_configured: bool
    mock_reason: str = ""


def get_llm_provider_selection() -> LLMProviderSelection:
    """Describe which provider should be active without constructing clients."""

    load_dotenv()

    live_llm_enabled = os.getenv("USE_LIVE_LLM", "false").strip().lower() == "true"
    openrouter_api_key_present = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
    openrouter_base_url = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    ).strip()

    azure_configured = all(
        os.getenv(name, "").strip()
        for name in [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION",
        ]
    )

    if not live_llm_enabled:
        return LLMProviderSelection(
            provider_name="fast-deterministic",
            live_llm_enabled=live_llm_enabled,
            openrouter_api_key_present=openrouter_api_key_present,
            openrouter_model=openrouter_model,
            openrouter_base_url=openrouter_base_url,
            azure_configured=azure_configured,
            mock_reason="USE_LIVE_LLM is not true",
        )

    if azure_configured:
        return LLMProviderSelection(
            provider_name="azure-openai",
            live_llm_enabled=live_llm_enabled,
            openrouter_api_key_present=openrouter_api_key_present,
            openrouter_model=openrouter_model,
            openrouter_base_url=openrouter_base_url,
            azure_configured=azure_configured,
        )

    if openrouter_api_key_present:
        return LLMProviderSelection(
            provider_name="openrouter",
            live_llm_enabled=live_llm_enabled,
            openrouter_api_key_present=openrouter_api_key_present,
            openrouter_model=openrouter_model,
            openrouter_base_url=openrouter_base_url,
            azure_configured=azure_configured,
        )

    return LLMProviderSelection(
        provider_name="fast-deterministic",
        live_llm_enabled=live_llm_enabled,
        openrouter_api_key_present=openrouter_api_key_present,
        openrouter_model=openrouter_model,
        openrouter_base_url=openrouter_base_url,
        azure_configured=azure_configured,
        mock_reason=(
            "USE_LIVE_LLM is true, Azure OpenAI is incomplete, and "
            "OPENROUTER_API_KEY is missing"
        ),
    )


def create_llm_provider() -> BaseLLMProvider:
    """Create the configured provider.

    Priority:
    1. Fast deterministic provider unless USE_LIVE_LLM=true.
    2. Azure OpenAI when fully configured and live LLMs are enabled.
    3. OpenRouter when configured and live LLMs are enabled.

    Live providers are wrapped with a mock fallback so transient API errors never
    crash the reasoning pipeline.
    """

    load_dotenv()

    mock_provider = MockLLMProvider()
    selection = get_llm_provider_selection()
    live_llm_enabled = selection.live_llm_enabled
    if not live_llm_enabled:
        mock_provider.selection_reason = selection.mock_reason
        logger.info(
            "Active LLM provider: FastDeterministic reason=%s",
            selection.mock_reason,
        )
        return mock_provider

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    azure_init_error = ""

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
            azure_init_error = str(exc) or exc.__class__.__name__
            logger.warning("AzureOpenAI provider initialization failed: %s", exc)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model = selection.openrouter_model
    openrouter_base_url = selection.openrouter_base_url
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

    mock_reason = selection.mock_reason
    if azure_init_error and not openrouter_api_key:
        mock_reason = (
            "AzureOpenAI provider initialization failed and "
            "OPENROUTER_API_KEY is missing"
        )
    mock_provider.selection_reason = mock_reason
    logger.info("Active LLM provider: FastDeterministic reason=%s", mock_reason)
    return mock_provider
