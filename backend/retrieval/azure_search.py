import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from models.reasoning import RetrievedContext
from reasoning.domain import classify_domain
from retrieval.base import BaseRetrievalProvider, RetrievalProviderError


class AzureSearchRetrievalProvider(BaseRetrievalProvider):
    """Direct Azure AI Search provider for Microsoft-backed retrieval.

    This provider targets the Azure AI Search service that can back Foundry IQ
    search experiences. It keeps the external REST payload and response mapping
    isolated from agents, which only consume normalized RetrievedContext records.
    """

    name = "azure-ai-search"
    source_label = "Azure AI Search / Foundry IQ Search Service"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        index_name: str,
        api_version: str = "2024-07-01",
        timeout_seconds: float = 10.0,
        top_k: int = 5,
        minimum_raw_score: float = 0.01,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.index_name = index_name
        self.api_version = api_version
        self.timeout_seconds = timeout_seconds
        self.top_k = top_k
        self.minimum_raw_score = minimum_raw_score

    def retrieve(self, question: str) -> list[RetrievedContext]:
        domain = classify_domain(question)
        domain_filter = domain if domain != "custom" else None

        contexts = self._query(question, domain_filter=domain_filter)
        if not contexts and domain_filter:
            contexts = self._query(question, domain_filter=None)

        if not contexts:
            raise RetrievalProviderError("Azure AI Search returned no strong context.")

        return contexts

    def build_request_payload(
        self, question: str, *, domain_filter: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "search": question,
            "queryType": "simple",
            "searchMode": "any",
            "searchFields": "title,snippet,content,tags",
            "select": "id,title,domain,source,url,snippet,content,tags",
            "highlight": "title,snippet,content",
            "highlightPreTag": "",
            "highlightPostTag": "",
            "top": self.top_k,
            "count": False,
        }
        if domain_filter:
            payload["filter"] = f"domain eq '{domain_filter}'"
        return payload

    def parse_response(self, data: Any) -> list[tuple[RetrievedContext, float]]:
        if not isinstance(data, dict):
            return []

        results = data.get("value")
        if not isinstance(results, list):
            return []

        parsed: list[tuple[RetrievedContext, float]] = []
        for index, item in enumerate(results[: self.top_k], start=1):
            if not isinstance(item, dict):
                continue

            title = self._first_text(item, "title")
            document_id = self._first_text(item, "id")
            url = self._first_text(item, "url")
            snippet = (
                self._highlight_text(item)
                or self._first_caption(item)
                or self._first_text(item, "snippet", "content")
            )
            raw_score = self._first_number(
                item, "@search.rerankerScore", "@search.score", "relevance_score", "score"
            )

            if not snippet:
                continue

            parsed.append(
                (
                    RetrievedContext(
                        id=document_id or f"S{index}",
                        citation_id=f"S{index}",
                        title=title or f"Azure AI Search result {index}",
                        source=self.source_label,
                        url=url if self._is_public_url(url) else "",
                        snippet=self._clean_text(snippet),
                        relevance_score=0.5,
                    ),
                    raw_score if raw_score is not None else 0.0,
                )
            )
        return parsed

    def _query(self, question: str, *, domain_filter: str | None) -> list[RetrievedContext]:
        payload = self.build_request_payload(question, domain_filter=domain_filter)
        body = self._send(self._build_http_request(payload))
        parsed = self.parse_response(self._decode_json(body))
        return self._normalize_scored_contexts(parsed)

    def _build_http_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            self._search_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
            },
            method="POST",
        )

    def _send(self, request: urllib.request.Request) -> str:
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RetrievalProviderError(
                f"Azure AI Search request failed with status {exc.code}."
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RetrievalProviderError("Azure AI Search request failed.") from exc

    def _decode_json(self, body: str) -> Any:
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RetrievalProviderError("Azure AI Search returned invalid JSON.") from exc

    def _search_url(self) -> str:
        index_name = urllib.parse.quote(self.index_name, safe="")
        query = urllib.parse.urlencode({"api-version": self.api_version})
        return f"{self.endpoint}/indexes('{index_name}')/docs/search.post.search?{query}"

    def _normalize_scored_contexts(
        self, scored_contexts: list[tuple[RetrievedContext, float]]
    ) -> list[RetrievedContext]:
        if not scored_contexts:
            return []

        max_raw_score = max(raw_score for _, raw_score in scored_contexts)
        if max_raw_score < self.minimum_raw_score:
            return []

        normalized_contexts: list[RetrievedContext] = []
        for context, raw_score in scored_contexts:
            normalized_score = max(0.05, min(1.0, raw_score / max_raw_score))
            normalized_contexts.append(
                context.copy(update={"relevance_score": round(normalized_score, 2)})
            )
        return self.normalize(normalized_contexts)

    def _highlight_text(self, item: dict[str, Any]) -> str:
        highlights = item.get("@search.highlights")
        if not isinstance(highlights, dict):
            return ""
        for key in ("snippet", "content", "title"):
            values = highlights.get(key)
            if isinstance(values, list):
                text_values = [value for value in values if isinstance(value, str)]
                if text_values:
                    return " ".join(text_values)
        return ""

    def _first_caption(self, item: dict[str, Any]) -> str:
        captions = item.get("@search.captions")
        if not isinstance(captions, list):
            return ""
        for caption in captions:
            if not isinstance(caption, dict):
                continue
            text = caption.get("text") or caption.get("highlights")
            if isinstance(text, str) and text.strip():
                return text
        return ""

    def _first_text(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _first_number(self, item: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _clean_text(self, text: str) -> str:
        without_tags = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\s+", " ", without_tags).strip()

    def _is_public_url(self, url: str) -> bool:
        return url.startswith(("https://", "http://"))
