import asyncio
import os
import unittest
from unittest.mock import patch

from config import active_reasoning_order, parse_frontend_origins
from main import health, health_providers


class HealthAndConfigTests(unittest.TestCase):
    def test_health_returns_ok(self) -> None:
        result = asyncio.run(health())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["service"], "ConsensusIQ API")

    def test_health_providers_does_not_expose_secrets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "secret-openrouter-key",
                "OPENROUTER_MODEL": "openai/gpt-oss-120b:free",
            },
            clear=False,
        ):
            result = asyncio.run(health_providers())

        serialized = str(result)
        self.assertNotIn("secret-openrouter-key", serialized)
        self.assertTrue(result["openrouter_configured"])
        self.assertEqual(result["openrouter_model"], "openai/gpt-oss-120b:free")

    def test_frontend_origin_comma_separated_parsing(self) -> None:
        origins = parse_frontend_origins(
            "https://preview.example.vercel.app, https://app.example.com/ , *"
        )

        self.assertIn("https://preview.example.vercel.app", origins)
        self.assertIn("https://app.example.com", origins)
        self.assertIn("http://localhost:3000", origins)
        self.assertIn("http://127.0.0.1:3000", origins)
        self.assertNotIn("*", origins)

    def test_openrouter_selected_when_api_key_present(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "",
                "AZURE_OPENAI_API_KEY": "",
                "AZURE_OPENAI_DEPLOYMENT": "",
                "AZURE_OPENAI_API_VERSION": "",
                "PREFER_AZURE_OPENAI": "false",
            },
            clear=False,
        ):
            order = active_reasoning_order()

        self.assertEqual(order[0], "OpenRouter")
        self.assertIn("FastDeterministic", order)


if __name__ == "__main__":
    unittest.main()
