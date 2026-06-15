import logging
import re

from models.reasoning import CitationValidity, ReasoningState, RetrievedContext


logger = logging.getLogger("consensus_iq.grounding")

CITATION_PATTERN = re.compile(r"(?<![A-Za-z0-9])S\d+(?![A-Za-z0-9])")


def extract_citation_ids(text: str) -> set[str]:
    """Extract source IDs such as S1 or S12 from model-visible text."""

    if not text:
        return set()
    return set(CITATION_PATTERN.findall(text))


def available_citation_ids(sources: list[RetrievedContext]) -> set[str]:
    return {source.citation_id for source in sources if source.citation_id}


def validate_citations(text: str, sources: list[RetrievedContext]) -> CitationValidity:
    available_sources = sorted(available_citation_ids(sources), key=_citation_sort_key)
    cited_sources = extract_citation_ids(text)
    invalid = sorted(cited_sources - set(available_sources), key=_citation_sort_key)

    if invalid:
        logger.warning(
            "citation validation failed invalid_citations=%s available_sources=%s",
            invalid,
            available_sources,
        )

    return CitationValidity(
        valid=not invalid,
        invalid_citations=invalid,
        available_sources=available_sources,
    )


def strip_invalid_citations(text: str, invalid_citations: list[str]) -> str:
    cleaned = text
    for citation_id in invalid_citations:
        cleaned = re.sub(
            rf"\s*[\[\(]?{re.escape(citation_id)}[\]\)]?",
            "",
            cleaned,
        )
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def validate_state_citations(state: ReasoningState) -> ReasoningState:
    """Validate and sanitize source IDs before returning the API response."""

    available = available_citation_ids(state.retrieved_context)
    sanitized_outputs = []
    for output in state.agent_outputs:
        valid_refs = [ref for ref in output.evidence_refs if ref in available]
        invalid_refs = [ref for ref in output.evidence_refs if ref not in available]
        if invalid_refs:
            logger.warning(
                "agent citation validation failed agent=%s invalid_citations=%s",
                output.agent,
                invalid_refs,
            )
        sanitized_outputs.append(output.copy(update={"evidence_refs": valid_refs}))

    combined_text = " ".join(
        [state.consensus, state.reasoning_summary]
        + [output.conclusion for output in sanitized_outputs]
        + [output.recommendation for output in sanitized_outputs]
    )
    validity = validate_citations(combined_text, state.retrieved_context)
    consensus = state.consensus
    reasoning_summary = state.reasoning_summary
    if not validity.valid:
        consensus = strip_invalid_citations(consensus, validity.invalid_citations)
        reasoning_summary = strip_invalid_citations(
            reasoning_summary, validity.invalid_citations
        )

    return state.copy(
        update={
            "agent_outputs": sanitized_outputs,
            "consensus": consensus,
            "reasoning_summary": reasoning_summary,
            "citation_validity": validity,
        }
    )


def _citation_sort_key(value: str) -> tuple[str, int]:
    match = re.match(r"([A-Za-z]+)(\d+)$", value)
    if not match:
        return (value, 0)
    return (match.group(1), int(match.group(2)))
