"""Seed Azure AI Search with the ConsensusIQ curated public evidence corpus.

Run from backend/ after setting AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY,
and AZURE_SEARCH_INDEX_NAME in .env. The resulting index is compatible with
AzureSearchRetrievalProvider.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from retrieval.corpus import CURATED_PUBLIC_CORPUS  # noqa: E402


DEFAULT_API_VERSION = "2024-07-01"


def main() -> None:
    load_dotenv(BACKEND_ROOT / ".env")

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "").strip()
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "").strip()
    api_version = os.getenv("AZURE_SEARCH_API_VERSION", DEFAULT_API_VERSION).strip()

    if not all([endpoint, api_key, index_name]):
        raise SystemExit(
            "Missing AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, or AZURE_SEARCH_INDEX_NAME."
        )

    create_or_update_index(endpoint, api_key, index_name, api_version)
    upload_documents(endpoint, api_key, index_name, api_version)
    print(
        f"Seeded {len(CURATED_PUBLIC_CORPUS)} documents into Azure AI Search index '{index_name}'."
    )


def create_or_update_index(
    endpoint: str, api_key: str, index_name: str, api_version: str
) -> None:
    request = build_request(
        url=index_url(endpoint, index_name, api_version),
        api_key=api_key,
        method="PUT",
        payload={
            "name": index_name,
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": False,
                    "sortable": True,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "domain",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "source",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "url",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "snippet",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "tags",
                    "type": "Collection(Edm.String)",
                    "searchable": True,
                    "filterable": True,
                    "sortable": False,
                    "facetable": True,
                    "retrievable": True,
                },
            ],
        },
    )
    send(request, "create or update index")


def upload_documents(
    endpoint: str, api_key: str, index_name: str, api_version: str
) -> None:
    documents = [
        {
            "@search.action": "upload",
            "id": document.id,
            "title": document.title,
            "domain": document.domain,
            "source": document.source,
            "url": document.url if is_public_url(document.url) else "",
            "snippet": document.snippet,
            "content": document.content,
            "tags": document.tags,
        }
        for document in CURATED_PUBLIC_CORPUS
    ]
    request = build_request(
        url=index_documents_url(endpoint, index_name, api_version),
        api_key=api_key,
        method="POST",
        payload={"value": documents},
    )
    send(request, "upload documents")


def build_request(
    *, url: str, api_key: str, method: str, payload: dict[str, Any]
) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": api_key},
        method=method,
    )


def send(request: urllib.request.Request, operation: str) -> None:
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(
            f"Azure AI Search {operation} failed with status {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Azure AI Search {operation} failed: {exc}") from exc


def index_url(endpoint: str, index_name: str, api_version: str) -> str:
    quoted_index = urllib.parse.quote(index_name, safe="")
    query = urllib.parse.urlencode({"api-version": api_version})
    return f"{endpoint}/indexes('{quoted_index}')?{query}"


def index_documents_url(endpoint: str, index_name: str, api_version: str) -> str:
    quoted_index = urllib.parse.quote(index_name, safe="")
    query = urllib.parse.urlencode({"api-version": api_version})
    return f"{endpoint}/indexes('{quoted_index}')/docs/search.index?{query}"


def is_public_url(url: str) -> bool:
    return url.startswith(("https://", "http://"))


if __name__ == "__main__":
    main()
