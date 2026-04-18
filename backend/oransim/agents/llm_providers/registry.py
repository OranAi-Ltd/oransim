"""Provider registry — env-var routing between native LLM adapters.

``LLM_PROVIDER`` (default ``openai``) picks the active adapter:

- ``openai`` — OpenAI-compatible ``/chat/completions`` (legacy default).
- ``anthropic`` — Anthropic ``/v1/messages``.
- ``gemini`` — Google ``generateContent``.
- ``qwen`` (alias ``qwen_dashscope``) — Qwen native ``/generation``.

Base URL and API key default to provider-appropriate values but can be
overridden per-provider via env (``ANTHROPIC_BASE_URL`` etc.) or via the
shared ``LLM_BASE_URL`` / ``LLM_API_KEY`` pair used by the OpenAI-compat
path. Resolution order per provider:

1. Provider-specific env (``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, …).
2. Shared ``LLM_API_KEY`` / ``LLM_BASE_URL``.
3. Adapter's documented default (e.g., Gemini's public endpoint).

This keeps single-provider deployments simple (just set ``LLM_*``) while
still supporting multi-provider test rigs where each adapter points at its
own gateway.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .anthropic import AnthropicProvider
from .base import LLMProvider
from .gemini import GeminiProvider
from .openai_compat import OpenAICompatProvider
from .qwen_dashscope import QwenDashScopeProvider

_PROVIDER_ALIASES = {
    "openai": "openai",
    "openai_compat": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "qwen": "qwen_dashscope",
    "qwen_dashscope": "qwen_dashscope",
    "dashscope": "qwen_dashscope",
}

ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com"


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _build(name: str) -> LLMProvider:
    if name == "openai":
        return OpenAICompatProvider(
            base_url=_env("LLM_BASE_URL", default="https://api.openai.com/v1"),
            api_key=_env("LLM_API_KEY", "OPENAI_API_KEY"),
        )
    if name == "anthropic":
        return AnthropicProvider(
            base_url=_env("ANTHROPIC_BASE_URL", "LLM_BASE_URL", default=ANTHROPIC_DEFAULT_BASE),
            api_key=_env("ANTHROPIC_API_KEY", "LLM_API_KEY"),
        )
    if name == "gemini":
        kwargs = {"api_key": _env("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY")}
        base = _env("GEMINI_BASE_URL", "LLM_BASE_URL")
        if base:
            kwargs["base_url"] = base
        return GeminiProvider(**kwargs)
    if name == "qwen_dashscope":
        kwargs = {"api_key": _env("DASHSCOPE_API_KEY", "QWEN_API_KEY", "LLM_API_KEY")}
        base = _env("DASHSCOPE_BASE_URL", "LLM_BASE_URL")
        if base:
            kwargs["base_url"] = base
        return QwenDashScopeProvider(**kwargs)
    raise ValueError(f"unknown LLM provider: {name}")


def resolve_provider_name(raw: str | None = None) -> str:
    raw = raw if raw is not None else os.environ.get("LLM_PROVIDER", "openai")
    key = (raw or "openai").strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(
            f"unknown LLM_PROVIDER={raw!r}; " f"expected one of {sorted(set(_PROVIDER_ALIASES))}"
        )
    return _PROVIDER_ALIASES[key]


@lru_cache(maxsize=8)
def _cached_provider(name: str) -> LLMProvider:
    return _build(name)


def get_provider(name: str | None = None) -> LLMProvider:
    """Return the active provider, constructed on first use and cached."""
    return _cached_provider(resolve_provider_name(name))


def reset_provider_cache() -> None:
    """Drop the cached provider — useful in tests that mutate env vars."""
    _cached_provider.cache_clear()


__all__ = [
    "get_provider",
    "resolve_provider_name",
    "reset_provider_cache",
]
