"""Anthropic ``/v1/messages`` native provider.

Differences from OpenAI chat/completions:

- System prompt is a **top-level** ``system`` field, not a ``role: system``
  message. Claude expects the ``messages`` array to only contain
  ``user`` / ``assistant`` turns.
- Auth header is ``x-api-key``, plus a required ``anthropic-version`` header.
- Response shape is ``content[].text`` (list of content blocks), not
  ``choices[0].message.content``.
- Usage is ``input_tokens`` / ``output_tokens``.

API reference: https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ...runtime.http_client import post_json
from .base import GenerateResult

DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")


@dataclass
class AnthropicProvider:
    """POST to ``{base_url}/v1/messages``."""

    base_url: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT
    anthropic_version: str = ANTHROPIC_VERSION
    name: str = "anthropic"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }

    def build_body(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        return {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

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
        # Anthropic streaming uses a different event vocabulary than OpenAI
        # SSE; for now buffered mode is the safe cross-provider baseline.
        url = f"{self.base_url.rstrip('/')}/v1/messages"
        body = self.build_body(
            system, user,
            model=model, temperature=temperature, max_tokens=max_tokens,
        )
        t0 = time.time()
        resp = post_json(url, self._headers(), body, timeout=self.timeout)
        blocks = resp.get("content") or []
        content = "".join(
            b.get("text", "") for b in blocks if b.get("type") == "text"
        )
        usage = resp.get("usage") or {}
        return GenerateResult(
            content=content,
            usage={
                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            },
            latency_ms=int((time.time() - t0) * 1000),
        )


__all__ = ["AnthropicProvider"]
