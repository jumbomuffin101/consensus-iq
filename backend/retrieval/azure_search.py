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
    fallback_on_empty = True

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        index_name: str,
        api_version: str = "2024-07-01",
        timeout_seconds: float = 3.0,
        top_k: int = 5,
        minimum_relevance_score: float = 0.3,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.index_name = index_name
        self.api_version = api_version
        self.timeout_seconds = timeout_seconds
        self.top_k = top_k
        self.minimum_relevance_score = minimum_relevance_score

    def retrieve(self, question: str) -> list[RetrievedContext]:
        return self._query(question)

    def build_request_payload(
        self, question: str, domain_filter: str | None = None
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

    def parse_response(
        self,
        data: Any,
        question: str,
        expected_domain: str | None = None,
        allowed_domains: set[str] | None = None,
    ) -> list[tuple[RetrievedContext, float, float]]:
        if not isinstance(data, dict):
            return []

        results = data.get("value")
        if not isinstance(results, list):
            return []

        parsed: list[tuple[RetrievedContext, float, float]] = []
        for index, item in enumerate(results[: self.top_k], start=1):
            if not isinstance(item, dict):
                continue
            item_domain = self._first_text(item, "domain").lower()
            if allowed_domains is not None and item_domain not in allowed_domains:
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
            lexical_score = self._lexical_relevance(question, item, expected_domain)

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
                    lexical_score,
                )
            )
        return parsed

    def _query(self, question: str) -> list[RetrievedContext]:
        detected_domain = classify_domain(question)
        if detected_domain == "custom":
            results = self._query_once(
                question,
                domain_filter=None,
                expected_domain="general_decision",
                allowed_domains={"general_decision", "custom"},
                minimum_relevance_score=0.45,
            )
            return self._with_limited_evidence_notice(results)

        if detected_domain != "custom":
            domain_results = self._query_once(question, detected_domain)
            if domain_results:
                return domain_results
            if detected_domain == "sports_injury":
                return []
        return self._query_once(question, None)

    def _query_once(
        self,
        question: str,
        domain_filter: str | None,
        expected_domain: str | None = None,
        allowed_domains: set[str] | None = None,
        minimum_relevance_score: float | None = None,
    ) -> list[RetrievedContext]:
        payload = self.build_request_payload(question, domain_filter)
        body = self._send(self._build_http_request(payload))
        parsed = self.parse_response(
            self._decode_json(body),
            question,
            expected_domain or domain_filter,
            allowed_domains,
        )
        return self._normalize_scored_contexts(parsed, minimum_relevance_score)

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
        self,
        scored_contexts: list[tuple[RetrievedContext, float, float]],
        minimum_relevance_score: float | None = None,
    ) -> list[RetrievedContext]:
        if not scored_contexts:
            return []

        max_raw_score = max(raw_score for _, raw_score, _ in scored_contexts)

        ranked_contexts: list[tuple[RetrievedContext, float]] = []
        for context, raw_score, lexical_score in scored_contexts:
            azure_score = raw_score / max_raw_score if max_raw_score > 0 else 0.0
            combined_score = (lexical_score * 0.8) + (azure_score * 0.2)
            ranked_contexts.append((context, round(max(0.0, min(0.92, combined_score)), 2)))

        ranked_contexts.sort(key=lambda item: item[1], reverse=True)
        minimum_score = minimum_relevance_score or self.minimum_relevance_score
        if not ranked_contexts or ranked_contexts[0][1] < minimum_score:
            return []

        normalized_contexts: list[RetrievedContext] = []
        for context, normalized_score in ranked_contexts:
            if normalized_score < minimum_score:
                continue
            normalized_contexts.append(
                context.copy(update={"relevance_score": normalized_score})
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

    def _lexical_relevance(
        self, question: str, item: dict[str, Any], expected_domain: str | None
    ) -> float:
        query_tokens = self._tokens(question)
        if not query_tokens:
            return 0.0

        title_tokens = self._tokens(self._first_text(item, "title"))
        snippet_tokens = self._tokens(self._first_text(item, "snippet"))
        content_tokens = self._tokens(self._first_text(item, "content"))
        tag_tokens = self._tokens(" ".join(self._list_text(item, "tags")))

        title_overlap = self._overlap(query_tokens, title_tokens)
        snippet_overlap = self._overlap(query_tokens, snippet_tokens)
        content_overlap = self._overlap(query_tokens, content_tokens)
        tag_overlap = self._overlap(query_tokens, tag_tokens)
        domain_match = 0.0
        item_domain = self._first_text(item, "domain").lower()
        if expected_domain and item_domain == expected_domain:
            domain_match = 0.12

        return min(
            1.0,
            (title_overlap * 0.34)
            + (snippet_overlap * 0.26)
            + (content_overlap * 0.28)
            + (tag_overlap * 0.12)
            + domain_match,
        )

    def _tokens(self, text: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "or",
            "our",
            "should",
            "the",
            "to",
            "use",
            "with",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in stopwords
        }

    def _overlap(self, query_tokens: set[str], candidate_tokens: set[str]) -> float:
        if not query_tokens or not candidate_tokens:
            return 0.0
        return len(query_tokens & candidate_tokens) / len(query_tokens)

    def _list_text(self, item: dict[str, Any], key: str) -> list[str]:
        value = item.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        return []

    def _clean_text(self, text: str) -> str:
        without_tags = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\s+", " ", without_tags).strip()

    def _is_public_url(self, url: str) -> bool:
        return url.startswith(("https://", "http://"))

    def _with_limited_evidence_notice(
        self, contexts: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        if not contexts:
            return []

        average_relevance = sum(item.relevance_score for item in contexts) / len(contexts)
        if average_relevance >= 0.55:
            return contexts

        return [
            item.copy(
                update={
                    "snippet": (
                        "Limited evidence coverage for this custom prompt. "
                        f"{item.snippet}"
                    )
                }
            )
            for item in contexts
        ]
