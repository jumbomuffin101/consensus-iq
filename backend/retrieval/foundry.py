import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from models.reasoning import RetrievedContext
from retrieval.base import BaseRetrievalProvider, RetrievalProviderError


class FoundryIQRetrievalProvider(BaseRetrievalProvider):
    """HTTP provider for Microsoft Foundry IQ retrieval.

    This is the Microsoft IQ integration boundary. Real deployments provide the
    endpoint, API key, index name, and API version through FOUNDRY_IQ_* env vars.
    The rest of the reasoning system only depends on RetrievedContext, so live
    Foundry IQ response changes are isolated to build_request_payload and
    parse_response.
    """

    name = "foundry-iq"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        index_name: str,
        api_version: str,
        timeout_seconds: float = 12.0,
        top_k: int = 5,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.index_name = index_name
        self.api_version = api_version
        self.timeout_seconds = timeout_seconds
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedContext]:
        request = self._build_http_request(self.build_request_payload(question))
        body = self._send(request)
        contexts = self.parse_response(self._decode_json(body))
        if not contexts:
            raise RetrievalProviderError("Foundry IQ returned no usable context.")
        return contexts

    def build_request_payload(self, question: str) -> dict[str, Any]:
        """Build the provider request body for a Foundry IQ search call.

        Foundry IQ projects may expose slightly different search schemas. Keep
        those project-specific fields here, not in agents or API routes.
        """
        return {
            "query": question,
            "search": question,
            "index_name": self.index_name,
            "indexName": self.index_name,
            "top": self.top_k,
            "include_citations": True,
            "includeCitations": True,
        }

    def parse_response(self, data: Any) -> list[RetrievedContext]:
        """Normalize Foundry IQ search results into citation-ready context."""
        return self._to_contexts(self._extract_items(data))

    def _build_http_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            self._request_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def _send(self, request: urllib.request.Request) -> str:
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RetrievalProviderError("Foundry IQ request failed.") from exc

    def _decode_json(self, body: str) -> Any:
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RetrievalProviderError("Foundry IQ returned invalid JSON.") from exc

    def _request_url(self) -> str:
        query = urllib.parse.urlencode({"api-version": self.api_version})
        return f"{self.endpoint}/indexes/{self.index_name}/search?{query}"

    def _extract_items(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if not isinstance(data, dict):
            return []

        for key in ("value", "results", "documents", "data", "citations", "matches"):
            items = data.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []

    def _to_contexts(self, items: list[dict[str, Any]]) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        for index, item in enumerate(items[: self.top_k], start=1):
            title = self._first_text(item, "title", "name", "document_title")
            snippet = self._first_text(
                item, "snippet", "content", "text", "chunk", "summary", "caption"
            )
            source = self._first_text(item, "source", "source_name", "dataset")
            url = self._first_text(item, "url", "uri", "source_url")
            score = self._first_number(
                item, "relevance_score", "score", "@search.score", "relevance"
            )

            if not snippet:
                continue

            contexts.append(
                RetrievedContext(
                    citation_id=f"S{index}",
                    title=title or f"Foundry IQ result {index}",
                    source=source or "Microsoft Foundry IQ",
                    snippet=snippet,
                    url=url,
                    relevance_score=max(0.0, min(1.0, score if score is not None else 0.75)),
                )
            )
        return contexts

    def _first_text(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for nested_key in ("metadata", "document", "source", "citation"):
            metadata = item.get(nested_key)
            if not isinstance(metadata, dict):
                continue
            for key in keys:
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def _first_number(self, item: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for nested_key in ("metadata", "document", "source", "citation"):
            metadata = item.get(nested_key)
            if not isinstance(metadata, dict):
                continue
            for key in keys:
                value = metadata.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return None
