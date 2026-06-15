import json
import logging
import os
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from llm.base import LLMProviderError
from models.reasoning import RetrievedContext


logger = logging.getLogger("consensus_iq.llm")


class OpenRouterGroundedClient:
    """Async OpenRouter client for optional citation-grounded reasoning.

    The deterministic graph remains the local default. This client is only used
    when OPENROUTER_API_KEY is configured, and callers are expected to keep a
    deterministic fallback for provider failures.
    """

    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "ConsensusIQ",
        frontend_origin: str = "",
        timeout_seconds: float = 12.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.frontend_origin = frontend_origin
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "OpenRouterGroundedClient | None":
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return None

        return cls(
            api_key=api_key,
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
            or "openai/gpt-4o-mini",
            base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).strip()
            or "https://openrouter.ai/api/v1",
            app_name=os.getenv("OPENROUTER_APP_NAME", "ConsensusIQ").strip()
            or "ConsensusIQ",
            frontend_origin=_first_origin(os.getenv("FRONTEND_ORIGIN", "")),
        )

    async def complete_grounded_json(
        self,
        *,
        question: str,
        retrieved_context: list[RetrievedContext],
        deterministic_answer: dict[str, Any],
        agent_name: str = "grounded consensus",
        strict: bool = False,
    ) -> dict[str, Any]:
        start = perf_counter()
        logger.info(
            "OpenRouter request start: agent=%s model=%s", agent_name, self.model
        )

        payload = self._build_payload(
            question=question,
            retrieved_context=retrieved_context,
            deterministic_answer=deterministic_answer,
            strict=strict,
        )

        try:
            data = await self._post(payload)
            content = self._extract_message_content(data)
            parsed = self._parse_json(content)
        except Exception as exc:
            logger.warning(
                "OpenRouter request failed: agent=%s reason=%s", agent_name, exc
            )
            raise LLMProviderError("OpenRouter grounded completion failed.") from exc

        latency_ms = int((perf_counter() - start) * 1000)
        logger.info(
            "OpenRouter request success: agent=%s latency_ms=%s",
            agent_name,
            latency_ms,
        )
        return parsed

    def _build_payload(
        self,
        *,
        question: str,
        retrieved_context: list[RetrievedContext],
        deterministic_answer: dict[str, Any],
        strict: bool,
    ) -> dict[str, Any]:
        source_rows = [
            {
                "source_id": source.citation_id,
                "title": source.title,
                "url": source.url,
                "source": source.source,
                "snippet": source.snippet,
                "relevance_score": source.relevance_score,
            }
            for source in retrieved_context
        ]
        strict_note = (
            "Previous output cited unavailable source IDs. Regenerate once using "
            "only the exact source_id values in retrieved_context."
            if strict
            else ""
        )

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are ConsensusIQ's grounded consensus refiner. "
                        "Answer only from retrieved_context. Cite only source_id "
                        "values that appear in retrieved_context, such as S1. "
                        "If the retrieved sources do not support an answer, say "
                        "the answer could not be verified from the available "
                        "sources. Do not invent titles, URLs, authors, filenames, "
                        "or source IDs. Return valid JSON only with keys: "
                        "consensus, reasoning_summary. Keep the existing decision "
                        "structure concise and preserve uncertainty. "
                        f"{strict_note}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "retrieved_context": source_rows,
                            "deterministic_answer": deterministic_answer,
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise LLMProviderError(
                "httpx is required for OpenRouter calls. Install requirements.txt."
            ) from exc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.frontend_origin:
            headers["HTTP-Referer"] = self.frontend_origin

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise LLMProviderError("OpenRouter returned a non-object response.")
            return data

    def _extract_message_content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError("OpenRouter response did not include choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMProviderError("OpenRouter choice was not an object.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LLMProviderError("OpenRouter choice did not include a message.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("OpenRouter returned empty content.")
        return content

    def _parse_json(self, content: str) -> dict[str, Any]:
        for candidate in _json_candidates(content):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise LLMProviderError("OpenRouter response did not contain valid JSON.")


def _json_candidates(content: str) -> list[str]:
    stripped = content.strip()
    candidates = [stripped]

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])

    return [candidate for candidate in candidates if candidate]


def _first_origin(value: str) -> str:
    return next((item.strip() for item in value.split(",") if item.strip()), "")
