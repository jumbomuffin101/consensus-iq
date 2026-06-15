from abc import ABC, abstractmethod
import re

from models.reasoning import ReasoningState, RetrievedContext


class RetrievalProviderError(RuntimeError):
    """Raised when a retrieval provider cannot return usable context."""


class RetrievalAdapter(ABC):
    """Strict adapter contract for Microsoft-backed and fallback retrieval.

    All retrieval implementations must return normalized RetrievedContext
    records so agents and API routes never depend on provider-specific fields.
    """

    name: str

    @abstractmethod
    def retrieve(self, question: str) -> list[RetrievedContext]:
        raise NotImplementedError

    def normalize(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        ranked_contexts = sorted(
            contexts,
            key=lambda context: context.relevance_score,
            reverse=True,
        )
        normalized: list[RetrievedContext] = []
        seen_keys: set[str] = set()
        for context in ranked_contexts:
            source_id = context.source_id or context.id or context.citation_id
            dedupe_key = self._dedupe_key(context, source_id)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            index = len(normalized) + 1
            citation_id = f"S{index}"
            normalized.append(
                context.copy(
                    update={
                        "id": context.id or source_id or citation_id,
                        "source_id": source_id or citation_id,
                        "citation_id": citation_id,
                        "snippet": self._stable_snippet(context.snippet),
                    }
                )
            )
        return normalized

    def _dedupe_key(self, context: RetrievedContext, source_id: str) -> str:
        if source_id:
            return f"id:{source_id.lower()}"
        if context.url:
            return f"url:{context.url.lower()}"
        normalized_title = re.sub(r"\W+", " ", context.title.lower()).strip()
        normalized_snippet = re.sub(r"\W+", " ", context.snippet.lower()).strip()
        return f"text:{normalized_title}:{normalized_snippet[:120]}"

    def _stable_snippet(self, snippet: str) -> str:
        cleaned = re.sub(r"\s+", " ", snippet).strip()
        if len(cleaned) <= 900:
            return cleaned
        return f"{cleaned[:897].rstrip()}..."


class BaseRetrievalProvider(RetrievalAdapter):
    """Backward-compatible provider name for existing graph wiring."""


class ResilientRetrievalProvider(BaseRetrievalProvider):
    """Wraps a primary retrieval provider with a reliable fallback provider."""

    def __init__(
        self,
        primary: BaseRetrievalProvider,
        fallback_provider: BaseRetrievalProvider,
    ) -> None:
        self.primary = primary
        self.fallback_provider = fallback_provider
        self.name = f"{primary.name}-with-{fallback_provider.name}-fallback"

    def retrieve(self, question: str) -> list[RetrievedContext]:
        try:
            results = self.primary.retrieve(question)
            if results:
                return results
            if not getattr(self.primary, "fallback_on_empty", True):
                return []
        except Exception:
            pass
        return self.fallback_provider.retrieve(question)


class RetrievalNode:
    """Graph node that retrieves citation-ready context for downstream agents."""

    def __init__(self, provider: BaseRetrievalProvider | None = None) -> None:
        if provider is None:
            from retrieval.factory import create_retrieval_provider

            provider = create_retrieval_provider()
        self.provider = provider

    def __call__(self, state: ReasoningState) -> ReasoningState:
        return state.copy(
            update={"retrieved_context": self.provider.retrieve(state.question)}
        )
