import re

from models.reasoning import RetrievedContext
from reasoning.custom_intake import deterministic_custom_intake
from reasoning.domain import classify_domain
from retrieval.base import BaseRetrievalProvider
from retrieval.corpus import CURATED_PUBLIC_CORPUS, CorpusDocument, documents_for_domain


class MockRetrievalProvider(BaseRetrievalProvider):
    """Curated public corpus provider used when live retrieval is unavailable."""

    name = "mock-foundry-iq"
    source_label = "Foundry IQ-Compatible Demo Corpus"

    def retrieve(self, question: str) -> list[RetrievedContext]:
        domain = classify_domain(question)
        source_rows = self._rank_documents(question, domain)
        limited_prefix = (
            "Limited evidence coverage for this custom prompt. "
            if domain == "custom" and self._source_quality(source_rows) == "weak"
            else ""
        )
        return self.normalize(
            [
                RetrievedContext(
                    id=document.id,
                    source_id=document.id,
                    citation_id=f"S{index}",
                    title=document.title,
                    source=self.source_label,
                    url=document.url if self._is_public_url(document.url) else "",
                    snippet=self._snippet(document, limited_prefix),
                    relevance_score=(
                        min(document.score, 0.52)
                        if domain == "custom"
                        else document.score
                    ),
                )
                for index, document in enumerate(source_rows, start=1)
            ]
        )

    def _rank_documents(self, question: str, domain: str) -> list[CorpusDocument]:
        if domain == "custom":
            intake = deterministic_custom_intake(question)
            if intake.domain in {"pet_health", "clinical_human", "legal", "unknown"}:
                return []
        candidates = documents_for_domain(domain, question)
        if not candidates:
            return []

        scored = [
            (document, self._combined_score(question, document, domain))
            for document in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            self._with_score(document, score)
            for document, score in scored[:5]
            if score >= (0.18 if domain == "custom" else 0.26)
        ]

    def _combined_score(
        self, question: str, document: CorpusDocument, domain: str
    ) -> float:
        query_tokens = self._tokens(question)
        if not query_tokens:
            return document.score
        title_overlap = self._overlap(query_tokens, self._tokens(document.title))
        snippet_overlap = self._overlap(query_tokens, self._tokens(document.snippet))
        content_overlap = self._overlap(query_tokens, self._tokens(document.content))
        tag_overlap = self._overlap(query_tokens, self._tokens(" ".join(document.tags)))
        lexical = (
            (title_overlap * 0.28)
            + (snippet_overlap * 0.28)
            + (content_overlap * 0.24)
            + (tag_overlap * 0.20)
        )
        domain_bonus = 0.12 if domain != "custom" and document.domain == domain else 0.0
        return round(
            max(0.0, min(0.96, (document.score * 0.35) + (lexical * 0.55) + domain_bonus)),
            2,
        )

    def _with_score(self, document: CorpusDocument, score: float) -> CorpusDocument:
        return CorpusDocument(
            id=document.id,
            title=document.title,
            domain=document.domain,
            source=document.source,
            url=document.url,
            snippet=document.snippet,
            content=document.content,
            tags=document.tags,
            score=score,
        )

    def _snippet(self, document: CorpusDocument, prefix: str) -> str:
        content = document.content.strip()
        detail = f" {content}" if content and content not in document.snippet else ""
        return f"{prefix}Curated public corpus source: {document.snippet}{detail}"

    def _source_quality(self, documents: list[CorpusDocument]) -> str:
        if not documents:
            return "weak"
        if documents[0].score >= 0.7 and len(documents) >= 2:
            return "strong"
        if documents[0].score >= 0.45:
            return "partial"
        return "weak"

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
            "would",
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

    def _is_public_url(self, url: str) -> bool:
        return url.startswith(("https://", "http://"))
