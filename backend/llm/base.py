from abc import ABC, abstractmethod
import logging
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


class ResilientLLMProvider(BaseLLMProvider):
    """Wraps a primary provider and falls back to mock output on any failure."""

    def __init__(
        self, primary: BaseLLMProvider, fallback_provider: BaseLLMProvider
    ) -> None:
        self.primary = primary
        self.fallback_provider = fallback_provider
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
            return self.primary.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=fallback,
                agent_name=agent_name,
            )
        except Exception as exc:
            logger.warning(
                "LLM provider %s failed for agent=%s; falling back to %s: %s",
                self.primary.name,
                agent_name,
                self.fallback_provider.name,
                exc,
            )
            return self.fallback_provider.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=fallback,
                agent_name=agent_name,
            )
