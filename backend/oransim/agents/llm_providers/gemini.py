"""Google Gemini ``generateContent`` native provider.

Differences from OpenAI chat/completions:

- URL includes the model name: ``{base}/v1beta/models/{model}:generateContent``.
- System prompt is a top-level ``systemInstruction`` object, not a role.
- Messages become ``contents[]`` with ``role`` ∈ {``user``, ``model``} and
  text inside ``parts[].text``.
- Temperature / max_tokens live under ``generationConfig``.
- Auth is via ``?key=`` query param (or ``x-goog-api-key`` header).
- Response is ``candidates[0].content.parts[].text`` and
  ``usageMetadata.{promptTokenCount,candidatesTokenCount}``.

API reference: https://ai.google.dev/api/generate-content
"""

from __future__ import annotations

import os
import time
import urllib.parse
from dataclasses import dataclass

from ...runtime.http_client import post_json
from .base import GenerateResult

DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


@dataclass
class GeminiProvider:
    """POST to ``{base_url}/v1beta/models/{model}:generateContent``."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    name: str = "gemini"

    def _headers(self) -> dict[str, str]:
        # Using header auth keeps the URL free of credentials in logs.
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def build_body(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        body: dict = {
            "contents": [
                {"role": "user", "parts": [{"text": user}]},
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def generate(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stream: bool = False,
    ) -> GenerateResult:
        # Streaming (:streamGenerateContent) omitted — buffered parity first.
        safe_model = urllib.parse.quote(model, safe="")
        url = f"{self.base_url.rstrip('/')}/v1beta/models/{safe_model}:generateContent"
        body = self.build_body(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        t0 = time.time()
        resp = post_json(url, self._headers(), body, timeout=self.timeout)
        candidates = resp.get("candidates") or []
        content = ""
        if candidates:
            parts = ((candidates[0].get("content") or {}).get("parts")) or []
            content = "".join(p.get("text", "") for p in parts)
        usage = resp.get("usageMetadata") or {}
        return GenerateResult(
            content=content,
            usage={
                "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
                "completion_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            },
            latency_ms=int((time.time() - t0) * 1000),
        )


__all__ = ["GeminiProvider"]
