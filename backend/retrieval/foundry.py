import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from models.reasoning import RetrievedContext
from retrieval.base import BaseRetrievalProvider, RetrievalProviderError


class FoundryIQRetrievalProvider(BaseRetrievalProvider):
    """HTTP provider for Microsoft Foundry IQ retrieval.

    The request/response mapping is isolated here so the rest of the reasoning
    system only depends on RetrievedContext. If Foundry IQ response fields differ
    by project, adjust _extract_items without touching agents or API routes.
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
        payload = {
            "query": question,
            "index_name": self.index_name,
            "top": self.top_k,
        }
        request = urllib.request.Request(
            self._request_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RetrievalProviderError("Foundry IQ request failed.") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RetrievalProviderError("Foundry IQ returned invalid JSON.") from exc

        return self._to_contexts(self._extract_items(data))

    def _request_url(self) -> str:
        query = urllib.parse.urlencode({"api-version": self.api_version})
        return f"{self.endpoint}/indexes/{self.index_name}/search?{query}"

    def _extract_items(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if not isinstance(data, dict):
            return []

        for key in ("value", "results", "documents", "data"):
            items = data.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []

    def _to_contexts(self, items: list[dict[str, Any]]) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        for index, item in enumerate(items[: self.top_k], start=1):
            title = self._first_text(item, "title", "name", "document_title")
            snippet = self._first_text(
                item, "snippet", "content", "text", "chunk", "summary"
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
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
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
        return None
