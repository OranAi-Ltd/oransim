"""OpenAI-compatible chat/completions provider.

Default adapter — works with OpenAI, DeepSeek, any vLLM / TGI / OpenRouter
that speaks the OpenAI chat/completions shape. Extracted verbatim from the
pre-R-phase ``soul_llm`` module so behavior stays bit-compatible when this
is the selected provider.

Streaming is handled inline (SSE) rather than via ``post_json`` because the
retry helper only handles buffered responses.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from ...runtime.http_client import post_json
from .base import GenerateResult

DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))


@dataclass
class OpenAICompatProvider:
    """POST to ``{base_url}/chat/completions`` with Bearer auth."""

    base_url: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT
    name: str = "openai_compat"

    def _headers(self, streaming: bool) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if streaming:
            h["Accept"] = "text/event-stream"
        return h

    def build_body(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            body["stream"] = True
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
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        body = self.build_body(
            system, user,
            model=model, temperature=temperature,
            max_tokens=max_tokens, stream=stream,
        )
        t0 = time.time()
        if stream:
            content, usage = self._stream(url, self._headers(True), body)
        else:
            resp = post_json(url, self._headers(False), body, timeout=self.timeout)
            content = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {}) or {}
        return GenerateResult(
            content=content,
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
            latency_ms=int((time.time() - t0) * 1000),
        )

    def _stream(self, url: str, headers: dict, body: dict) -> tuple[str, dict]:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers=headers, method="POST",
        )
        collected: list[str] = []
        usage: dict = {}
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    collected.append(delta)
        return "".join(collected), usage


__all__ = ["OpenAICompatProvider"]
