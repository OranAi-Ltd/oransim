"""Provider registry — env-var routing between native LLM adapters.

``LLM_PROVIDER`` (default ``openai``) picks the active adapter:

- ``openai`` — OpenAI-compatible ``/chat/completions`` (legacy default).
- ``anthropic`` — Anthropic ``/v1/messages``.
- ``gemini`` — Google ``generateContent``.
- ``qwen`` (alias ``qwen_dashscope``) — Qwen native ``/generation``.
- ``atlascloud`` — Atlas Cloud OpenAI-compatible ``/chat/completions``.

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
    "atlascloud": "atlascloud",
    "atlas_cloud": "atlascloud",
    "atlas-cloud": "atlascloud",
    "atlas": "atlascloud",
}

OPENAI_COMPAT_DEFAULT_BASE = "https://api.openai.com/v1"
OPENAI_COMPAT_DEFAULT_MODEL = "gpt-5.4"
ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com"
ATLAS_CLOUD_DEFAULT_BASE = "https://api.atlascloud.ai/v1"
ATLAS_CLOUD_DEFAULT_MODEL = "qwen/qwen3.5-flash"


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _env_unless_default(name: str, default_value: str) -> str:
    value = os.environ.get(name, "").strip()
    if value and value.rstrip("/") != default_value.rstrip("/"):
        return value
    return ""


def resolve_atlascloud_base_url() -> str:
    """Return Atlas Cloud's endpoint, allowing explicit per-provider overrides."""
    base = _env(
        "ATLASCLOUD_BASE_URL",
        "ATLASCLOUD_API_BASE",
        "ATLAS_CLOUD_BASE_URL",
        "ATLAS_CLOUD_API_BASE",
    )
    if base:
        return base.rstrip("/")
    # A copied .env.example sets LLM_BASE_URL to the OpenAI default. Treat that
    # as "not customized" so LLM_PROVIDER=atlascloud works as a preset.
    shared = _env_unless_default("LLM_BASE_URL", OPENAI_COMPAT_DEFAULT_BASE)
    return (shared or ATLAS_CLOUD_DEFAULT_BASE).rstrip("/")


def resolve_model_name(raw: str | None = None, provider: str | None = None) -> str:
    """Resolve the runtime model, including provider-specific defaults."""
    name = resolve_provider_name(provider)
    if name == "atlascloud":
        model = _env("ATLASCLOUD_MODEL", "ATLAS_CLOUD_MODEL")
        if model:
            return model
        candidate = raw if raw is not None else os.environ.get("LLM_MODEL", "")
        if candidate and candidate.strip() != OPENAI_COMPAT_DEFAULT_MODEL:
            return candidate.strip()
        return ATLAS_CLOUD_DEFAULT_MODEL
    candidate = raw if raw is not None else os.environ.get("LLM_MODEL", "")
    return candidate.strip() or OPENAI_COMPAT_DEFAULT_MODEL


def _build(name: str) -> LLMProvider:
    if name == "openai":
        return OpenAICompatProvider(
            base_url=_env("LLM_BASE_URL", default=OPENAI_COMPAT_DEFAULT_BASE),
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
    if name == "atlascloud":
        return OpenAICompatProvider(
            base_url=resolve_atlascloud_base_url(),
            api_key=_env("ATLASCLOUD_API_KEY", "ATLAS_CLOUD_API_KEY", "LLM_API_KEY"),
            name="atlascloud",
        )
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
    "ATLAS_CLOUD_DEFAULT_BASE",
    "ATLAS_CLOUD_DEFAULT_MODEL",
    "get_provider",
    "resolve_atlascloud_base_url",
    "resolve_model_name",
    "resolve_provider_name",
    "reset_provider_cache",
]
