from abc import ABC, abstractmethod
import logging
from threading import Lock
from typing import Any


logger = logging.getLogger("consensus_iq.llm")


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce a valid structured response."""


class BaseLLMProvider(ABC):
    """Provider contract used by all reasoning agents.

    Agents ask for JSON-shaped outputs and validate them with Pydantic models.
    This keeps Azure OpenAI, local mocks, and future model providers swappable.
    """

    name: str

    @abstractmethod
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        agent_name: str = "unknown",
    ) -> dict[str, Any]:
        raise NotImplementedError


class LLMUsageTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.openrouter_call_count = 0
        self.openrouter_agents: list[str] = []
        self.fallback_reason = ""

    def record_openrouter_call(self, agent_name: str) -> None:
        with self._lock:
            self.openrouter_call_count += 1
            self.openrouter_agents.append(agent_name)

    def record_fallback(self, reason: str) -> None:
        with self._lock:
            self.fallback_reason = reason


class ResilientLLMProvider(BaseLLMProvider):
    """Wraps a primary provider and falls back to mock output on any failure."""

    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback_provider: BaseLLMProvider,
        usage_tracker: LLMUsageTracker | None = None,
    ) -> None:
        self.primary = primary
        self.fallback_provider = fallback_provider
        self.usage_tracker = usage_tracker
        self.name = f"{primary.name}-with-{fallback_provider.name}-fallback"

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        agent_name: str = "unknown",
    ) -> dict[str, Any]:
        try:
            if (
                self.usage_tracker is not None
                and "openrouter" in self.primary.name.lower()
            ):
                self.usage_tracker.record_openrouter_call(agent_name)
            return self.primary.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=fallback,
                agent_name=agent_name,
            )
        except Exception as exc:
            safe_reason = _safe_error_message(exc)
            logger.warning(
                "LLM provider %s failed for agent=%s; provider selected=%s reason=%s",
                self.primary.name,
                agent_name,
                self.fallback_provider.name,
                safe_reason,
            )
            if self.usage_tracker is not None:
                self.usage_tracker.record_fallback(safe_reason)
            return self.fallback_provider.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=fallback,
                agent_name=agent_name,
            )


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__
