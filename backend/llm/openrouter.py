import json
import time
import urllib.error
import urllib.request
from typing import Any

from llm.base import BaseLLMProvider, LLMProviderError


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter Chat Completions provider for structured agent reasoning."""

    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "ConsensusIQ",
        timeout_seconds: float = 25.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                payload = self._build_payload(system_prompt, user_prompt)
                body = self._send(payload)
                content = self._extract_message_content(self._decode_json(body))
                return self._parse_json(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))

        raise LLMProviderError("OpenRouter failed after retries.") from last_error

    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        "Return valid JSON only. Do not include markdown fences, "
                        "commentary, prose, or keys that were not requested. If "
                        "retrieved evidence is weak or absent, state that limitation "
                        "inside the requested JSON fields and do not fabricate citations."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

    def _send(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://consensusiq.local",
                "X-Title": self.app_name,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise LLMProviderError(
                f"OpenRouter request failed with status {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMProviderError("OpenRouter request failed.") from exc

    def _decode_json(self, body: str) -> Any:
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("OpenRouter returned invalid JSON envelope.") from exc

    def _extract_message_content(self, data: Any) -> str:
        if not isinstance(data, dict):
            raise LLMProviderError("OpenRouter response envelope was not an object.")
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
        candidates = [content.strip()]
        fenced = self._extract_fenced_json(content)
        if fenced:
            candidates.append(fenced)
        object_text = self._extract_json_object(content)
        if object_text:
            candidates.append(object_text)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        raise LLMProviderError("OpenRouter response did not contain a JSON object.")

    def _extract_fenced_json(self, content: str) -> str:
        stripped = content.strip()
        if not stripped.startswith("```"):
            return ""
        lines = stripped.splitlines()
        if len(lines) < 3:
            return ""
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _extract_json_object(self, content: str) -> str:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            return ""
        return content[start : end + 1].strip()
