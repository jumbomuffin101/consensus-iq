from typing import Any

from llm.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Deterministic provider used for local development and Azure fallback."""

    name = "fast-deterministic"

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        agent_name: str = "unknown",
    ) -> dict[str, Any]:
        return fallback
