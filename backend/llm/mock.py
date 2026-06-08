from typing import Any

from llm.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Deterministic provider used for local development and Azure fallback."""

    name = "mock"

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        return fallback
