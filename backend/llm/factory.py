import os

from dotenv import load_dotenv

from llm.azure_openai import AzureOpenAIProvider
from llm.base import BaseLLMProvider, ResilientLLMProvider
from llm.mock import MockLLMProvider


def create_llm_provider() -> BaseLLMProvider:
    """Create the configured provider.

    If Azure configuration or dependencies are missing, return the mock provider.
    If Azure is configured, wrap it with a mock fallback so transient API errors
    never crash the reasoning pipeline.
    """

    load_dotenv()

    mock_provider = MockLLMProvider()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

    if not all([endpoint, api_key, deployment, api_version]):
        return mock_provider

    try:
        azure_provider = AzureOpenAIProvider(
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            api_version=api_version,
        )
    except Exception:
        return mock_provider

    return ResilientLLMProvider(azure_provider, mock_provider)
