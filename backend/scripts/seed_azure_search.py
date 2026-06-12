"""Seed Azure AI Search with the ConsensusIQ curated public evidence corpus.

Run from backend/ after setting AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY,
and AZURE_SEARCH_INDEX_NAME in .env. The resulting index is compatible with
AzureSearchRetrievalProvider.
"""

import json
import os
import sys
import importlib.util
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_API_VERSION = "2024-07-01"


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(BACKEND_ROOT / ".env")
    else:
        load_env_file(BACKEND_ROOT / ".env")

    endpoint, api_key, index_name, api_version = load_config()
    corpus = load_curated_corpus()

    print(f"Using Azure AI Search endpoint: {endpoint}")
    print(f"Using Azure AI Search index: {index_name}")
    print(f"Using Azure AI Search API version: {api_version}")
    create_or_update_index(endpoint, api_key, index_name, api_version)
    uploaded_count = upload_documents(endpoint, api_key, index_name, api_version, corpus)
    print(
        f"Seeded {uploaded_count} documents into Azure AI Search index '{index_name}'."
    )


def load_config() -> tuple[str, str, str, str]:
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "").strip()
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "").strip()
    api_version = os.getenv("AZURE_SEARCH_API_VERSION", DEFAULT_API_VERSION).strip()

    missing = [
        name
        for name, value in [
            ("AZURE_SEARCH_ENDPOINT", endpoint),
            ("AZURE_SEARCH_API_KEY", api_key),
            ("AZURE_SEARCH_INDEX_NAME", index_name),
        ]
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required Azure AI Search environment variable(s): "
            f"{', '.join(missing)}. Set them in backend/.env or the shell before running "
            "`python scripts/seed_azure_search.py`. The API key value is never printed."
        )
    if not endpoint.startswith(("https://", "http://")):
        raise SystemExit(
            "AZURE_SEARCH_ENDPOINT must be a full URL such as "
            "https://<service-name>.search.windows.net."
        )
    if not api_version:
        api_version = DEFAULT_API_VERSION
    return endpoint, api_key, index_name, api_version


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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
    print(f"Created or updated Azure AI Search index '{index_name}'.")


def upload_documents(
    endpoint: str, api_key: str, index_name: str, api_version: str, corpus: list
) -> int:
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
        for document in corpus
    ]
    request = build_request(
        url=index_documents_url(endpoint, index_name, api_version),
        api_key=api_key,
        method="POST",
        payload={"value": documents},
    )
    body = send(request, "upload documents")
    succeeded_count = count_succeeded_uploads(body)
    if succeeded_count != len(documents):
        raise SystemExit(
            "Azure AI Search document upload did not confirm every document. "
            f"Confirmed {succeeded_count} of {len(documents)}."
        )
    print(f"Uploaded {succeeded_count} documents.")
    return succeeded_count


def build_request(
    *, url: str, api_key: str, method: str, payload: dict[str, Any]
) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": api_key},
        method=method,
    )


def send(request: urllib.request.Request, operation: str) -> str:
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(
            f"Azure AI Search {operation} failed with status {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Azure AI Search {operation} failed: {exc}") from exc


def count_succeeded_uploads(body: str) -> int:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return 0
    values = payload.get("value") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return 0
    return sum(
        1
        for item in values
        if isinstance(item, dict) and item.get("status") is True
    )


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


def load_curated_corpus() -> list:
    module_path = BACKEND_ROOT / "retrieval" / "corpus.py"
    spec = importlib.util.spec_from_file_location("consensusiq_seed_corpus", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load curated corpus from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = getattr(module, "CURATED_PUBLIC_CORPUS", None)
    if not isinstance(corpus, list) or not corpus:
        raise SystemExit("Curated public corpus is empty or unavailable.")
    return corpus


if __name__ == "__main__":
    main()
