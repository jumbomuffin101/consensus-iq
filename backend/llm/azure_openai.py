import json
import time
from typing import Any

from llm.base import BaseLLMProvider, LLMProviderError


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider for structured JSON reasoning calls."""

    name = "azure-openai"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        self.deployment = deployment
        self.max_retries = max_retries

        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise LLMProviderError(
                "openai package is not installed. Install backend/requirements-azure.txt."
            ) from exc

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout_seconds,
        )

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        agent_name: str = "unknown",
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"{system_prompt}\n\nReturn only valid JSON with "
                                "the requested keys. Do not include markdown."
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                content = response.choices[0].message.content
                if not content:
                    raise LLMProviderError("Azure OpenAI returned empty content.")
                return self._parse_json(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))

        raise LLMProviderError("Azure OpenAI failed after retries.") from last_error

    def _parse_json(self, content: str) -> dict[str, Any]:
        normalized = content.strip()
        if normalized.startswith("```"):
            normalized = normalized.strip("`")
            normalized = normalized.removeprefix("json").strip()

        parsed = json.loads(normalized)
        if not isinstance(parsed, dict):
            raise LLMProviderError("Azure OpenAI response was not a JSON object.")
        return parsed
