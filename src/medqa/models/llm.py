"""LLM client abstraction with retry logic and rate limiting."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from groq import Groq

from medqa.config import Settings, get_settings
from medqa.log import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Thin wrapper around the Groq SDK with retries and JSON parsing."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        cfg = self._settings.llm
        self._client = Groq(api_key=self._settings.groq_api_key.get_secret_value())
        self._max_retries = cfg.max_retries
        self._retry_delay = cfg.retry_delay
        self._rate_limit_delay = cfg.rate_limit_delay

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> str:
        """Send a single-turn completion request and return the text response."""
        cfg = self._settings.llm
        model = model or cfg.generation_model
        temperature = temperature if temperature is not None else cfg.temperature

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if isinstance(content, list):
                    content = "".join(part.text for part in content)
                time.sleep(self._rate_limit_delay)
                return content
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)

        raise RuntimeError(f"LLM request failed after {self._max_retries} attempts: {last_err}")

    def complete_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Send a completion and parse the result as JSON."""
        raw = self.complete(prompt, model=model, temperature=temperature)
        return parse_json_response(raw)


def parse_json_response(raw: str) -> dict[str, Any]:
    """Robustly extract a JSON object from LLM output."""
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}
