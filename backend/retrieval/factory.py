import os

from dotenv import load_dotenv

from retrieval.base import BaseRetrievalProvider, ResilientRetrievalProvider
from retrieval.foundry import FoundryIQRetrievalProvider
from retrieval.mock import MockRetrievalProvider


def create_retrieval_provider() -> BaseRetrievalProvider:
    """Create the configured Foundry IQ provider with curated public corpus fallback."""

    load_dotenv()

    mock_provider = MockRetrievalProvider()
    endpoint = os.getenv("FOUNDRY_IQ_ENDPOINT", "").strip()
    api_key = os.getenv("FOUNDRY_IQ_API_KEY", "").strip()
    index_name = os.getenv("FOUNDRY_IQ_INDEX_NAME", "").strip()
    api_version = os.getenv("FOUNDRY_IQ_API_VERSION", "").strip()

    if not all([endpoint, api_key, index_name, api_version]):
        return mock_provider

    foundry_provider = FoundryIQRetrievalProvider(
        endpoint=endpoint,
        api_key=api_key,
        index_name=index_name,
        api_version=api_version,
    )
    return ResilientRetrievalProvider(foundry_provider, mock_provider)
