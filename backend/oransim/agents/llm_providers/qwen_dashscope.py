"""Alibaba Qwen DashScope native provider.

DashScope ships two surfaces: an OpenAI-compat gateway (which the existing
``OpenAICompatProvider`` already handles) and a **native** endpoint at
``/services/aigc/text-generation/generation`` which exposes Qwen-specific
knobs (``result_format``, ``incremental_output``, tool schemas). This
adapter speaks the native format directly so users of DashScope don't have
to route through the compat gateway.

Differences from OpenAI chat/completions:

- Payload wraps messages under ``input.messages`` and knobs under
  ``parameters.*``.
- ``result_format: "message"`` makes the response shape similar to OpenAI
  (``output.choices[0].message.content``); without it, Qwen returns a flat
  ``output.text``.
- Auth is ``Authorization: Bearer <key>``.
- Usage is ``usage.{input_tokens,output_tokens}``.

API reference: https://help.aliyun.com/zh/model-studio/developer-reference/
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ...runtime.http_client import post_json
from .base import GenerateResult

DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"


@dataclass
class QwenDashScopeProvider:
    """POST to ``{base_url}/services/aigc/text-generation/generation``."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    name: str = "qwen_dashscope"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
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
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return {
            "model": model,
            "input": {"messages": messages},
            "parameters": {
                "result_format": "message",
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
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
        url = f"{self.base_url.rstrip('/')}/services/aigc/text-generation/generation"
        body = self.build_body(
            system,
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        t0 = time.time()
        resp = post_json(url, self._headers(), body, timeout=self.timeout)
        output = resp.get("output") or {}
        content = ""
        choices = output.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
        if not content:
            # fallback when result_format="text"
            content = output.get("text") or ""
        usage = resp.get("usage") or {}
        return GenerateResult(
            content=content,
            usage={
                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            },
            latency_ms=int((time.time() - t0) * 1000),
        )


__all__ = ["QwenDashScopeProvider"]
