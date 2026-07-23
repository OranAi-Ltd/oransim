"""Native-format LLM providers.

Oransim originally only supported OpenAI-compatible ``/chat/completions``;
this package adds first-class adapters for providers whose native APIs
differ materially (Anthropic Messages, Google Gemini, Qwen DashScope),
plus a registry that routes via ``LLM_PROVIDER``.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import GenerateResult, LLMProvider
from .gemini import GeminiProvider
from .openai_compat import OpenAICompatProvider
from .qwen_dashscope import QwenDashScopeProvider
from .registry import (
    ATLAS_CLOUD_DEFAULT_BASE,
    ATLAS_CLOUD_DEFAULT_MODEL,
    get_provider,
    reset_provider_cache,
    resolve_atlascloud_base_url,
    resolve_model_name,
    resolve_provider_name,
)

__all__ = [
    "AnthropicProvider",
    "ATLAS_CLOUD_DEFAULT_BASE",
    "ATLAS_CLOUD_DEFAULT_MODEL",
    "GeminiProvider",
    "GenerateResult",
    "LLMProvider",
    "OpenAICompatProvider",
    "QwenDashScopeProvider",
    "get_provider",
    "reset_provider_cache",
    "resolve_atlascloud_base_url",
    "resolve_model_name",
    "resolve_provider_name",
]
