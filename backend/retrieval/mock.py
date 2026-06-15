from models.reasoning import RetrievedContext
from reasoning.domain import classify_domain
from retrieval.base import BaseRetrievalProvider
from retrieval.corpus import documents_for_domain


class MockRetrievalProvider(BaseRetrievalProvider):
    """Curated public corpus provider used when live retrieval is unavailable."""

    name = "mock-foundry-iq"
    source_label = "Foundry IQ-Compatible Demo Corpus"

    def retrieve(self, question: str) -> list[RetrievedContext]:
        domain = classify_domain(question)
        source_rows = documents_for_domain(domain, question)
        limited_prefix = (
            "Limited evidence coverage for this custom prompt. "
            if domain == "custom"
            else ""
        )
        return self.normalize(
            [
                RetrievedContext(
                    id=document.id,
                    citation_id=f"S{index}",
                    title=document.title,
                    source=self.source_label,
                    url=document.url if self._is_public_url(document.url) else "",
                    snippet=(
                        f"{limited_prefix}Curated public corpus source: "
                        f"{document.snippet}"
                    ),
                    relevance_score=(
                        min(document.score, 0.48)
                        if domain == "custom"
                        else document.score
                    ),
                )
                for index, document in enumerate(source_rows, start=1)
            ]
        )

    def _is_public_url(self, url: str) -> bool:
        return url.startswith(("https://", "http://"))
