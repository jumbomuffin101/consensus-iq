from abc import ABC, abstractmethod

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
        normalized: list[RetrievedContext] = []
        for index, context in enumerate(contexts, start=1):
            citation_id = context.citation_id or f"S{index}"
            normalized.append(
                context.copy(
                    update={
                        "id": context.id or citation_id,
                        "citation_id": citation_id,
                    }
                )
            )
        return normalized


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
