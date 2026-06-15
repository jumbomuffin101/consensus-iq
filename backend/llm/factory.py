import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from llm.azure_openai import AzureOpenAIProvider
from llm.base import BaseLLMProvider, LLMUsageTracker, ResilientLLMProvider
from llm.mock import MockLLMProvider
from llm.openrouter import OpenRouterProvider


logger = logging.getLogger("consensus_iq.llm")
SUPPORTED_LIVE_LLM_MODES = {"judge_only", "all_agents", "off"}


@dataclass(frozen=True)
class LLMProviderSelection:
    provider_name: str
    use_live_llm: bool
    live_llm_mode: str
    openrouter_api_key_present: bool
    openrouter_model: str
    openrouter_base_url: str
    azure_configured: bool
    mock_reason: str = ""


def get_llm_provider_selection() -> LLMProviderSelection:
    """Describe which provider should be active without constructing clients."""

    load_dotenv()

    use_live_llm = os.getenv("USE_LIVE_LLM", "false").strip().lower() == "true"
    live_llm_mode = _live_llm_mode(use_live_llm)
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

    if live_llm_mode == "off":
        mock_reason = (
            "LIVE_LLM_MODE is off"
            if use_live_llm
            else "USE_LIVE_LLM is not true and LIVE_LLM_MODE is off"
        )
        return LLMProviderSelection(
            provider_name="fast-deterministic",
            use_live_llm=use_live_llm,
            live_llm_mode=live_llm_mode,
            openrouter_api_key_present=openrouter_api_key_present,
            openrouter_model=openrouter_model,
            openrouter_base_url=openrouter_base_url,
            azure_configured=azure_configured,
            mock_reason=mock_reason,
        )

    if azure_configured:
        return LLMProviderSelection(
            provider_name="azure-openai",
            use_live_llm=use_live_llm,
            live_llm_mode=live_llm_mode,
            openrouter_api_key_present=openrouter_api_key_present,
            openrouter_model=openrouter_model,
            openrouter_base_url=openrouter_base_url,
            azure_configured=azure_configured,
        )

    if openrouter_api_key_present:
        return LLMProviderSelection(
            provider_name="openrouter",
            use_live_llm=use_live_llm,
            live_llm_mode=live_llm_mode,
            openrouter_api_key_present=openrouter_api_key_present,
            openrouter_model=openrouter_model,
            openrouter_base_url=openrouter_base_url,
            azure_configured=azure_configured,
        )

    return LLMProviderSelection(
        provider_name="fast-deterministic",
        use_live_llm=use_live_llm,
        live_llm_mode=live_llm_mode,
        openrouter_api_key_present=openrouter_api_key_present,
        openrouter_model=openrouter_model,
        openrouter_base_url=openrouter_base_url,
        azure_configured=azure_configured,
        mock_reason=(
            "USE_LIVE_LLM is true, Azure OpenAI is incomplete, and "
            "OPENROUTER_API_KEY is missing"
        ),
    )


def create_llm_provider(
    usage_tracker: LLMUsageTracker | None = None,
) -> BaseLLMProvider:
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
    if selection.live_llm_mode == "off":
        mock_provider.selection_reason = selection.mock_reason
        logger.info(
            "Active LLM provider: FastDeterministic live_llm_mode=%s reason=%s",
            selection.live_llm_mode,
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
            logger.info(
                "Active LLM provider: AzureOpenAI live_llm_mode=%s",
                selection.live_llm_mode,
            )
            return ResilientLLMProvider(
                azure_provider, mock_provider, usage_tracker=usage_tracker
            )
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
        logger.info(
            "Active LLM provider: OpenRouter live_llm_mode=%s",
            selection.live_llm_mode,
        )
        return ResilientLLMProvider(
            openrouter_provider, mock_provider, usage_tracker=usage_tracker
        )

    mock_reason = selection.mock_reason
    if azure_init_error and not openrouter_api_key:
        mock_reason = (
            "AzureOpenAI provider initialization failed and "
            "OPENROUTER_API_KEY is missing"
        )
    mock_provider.selection_reason = mock_reason
    logger.info(
        "Active LLM provider: FastDeterministic live_llm_mode=%s reason=%s",
        selection.live_llm_mode,
        mock_reason,
    )
    return mock_provider


def _live_llm_mode(use_live_llm: bool) -> str:
    configured = os.getenv("LIVE_LLM_MODE", "").strip().lower()
    if configured:
        if configured in SUPPORTED_LIVE_LLM_MODES:
            return configured
        logger.warning(
            "Unsupported LIVE_LLM_MODE=%s; defaulting to judge_only", configured
        )
        return "judge_only" if use_live_llm else "off"
    return "judge_only" if use_live_llm else "off"
