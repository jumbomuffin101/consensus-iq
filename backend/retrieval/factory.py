import os

from dotenv import load_dotenv

from retrieval.base import BaseRetrievalProvider, ResilientRetrievalProvider
from retrieval.azure_search import AzureSearchRetrievalProvider
from retrieval.foundry import FoundryIQRetrievalProvider
from retrieval.mock import MockRetrievalProvider


def create_retrieval_provider() -> BaseRetrievalProvider:
    """Create the configured Microsoft retrieval provider with local fallback.

    Priority:
    1. Direct Azure AI Search / Foundry IQ Search Service provider.
    2. Native Foundry IQ HTTP provider.
    3. Curated local demo corpus.
    """

    load_dotenv()

    mock_provider = MockRetrievalProvider()
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
    azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY", "").strip()
    azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "").strip()
    azure_search_api_version = os.getenv("AZURE_SEARCH_API_VERSION", "2024-07-01").strip()

    if all([azure_search_endpoint, azure_search_api_key, azure_search_index_name]):
        azure_search_provider = AzureSearchRetrievalProvider(
            endpoint=azure_search_endpoint,
            api_key=azure_search_api_key,
            index_name=azure_search_index_name,
            api_version=azure_search_api_version,
        )
        return ResilientRetrievalProvider(azure_search_provider, mock_provider)

    foundry_endpoint = os.getenv("FOUNDRY_IQ_ENDPOINT", "").strip()
    foundry_api_key = os.getenv("FOUNDRY_IQ_API_KEY", "").strip()
    foundry_index_name = os.getenv("FOUNDRY_IQ_INDEX_NAME", "").strip()
    foundry_api_version = os.getenv("FOUNDRY_IQ_API_VERSION", "").strip()

    if not all([foundry_endpoint, foundry_api_key, foundry_index_name, foundry_api_version]):
        return mock_provider

    foundry_provider = FoundryIQRetrievalProvider(
        endpoint=foundry_endpoint,
        api_key=foundry_api_key,
        index_name=foundry_index_name,
        api_version=foundry_api_version,
    )
    return ResilientRetrievalProvider(foundry_provider, mock_provider)
