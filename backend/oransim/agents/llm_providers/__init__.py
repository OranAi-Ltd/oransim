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
    get_provider,
    reset_provider_cache,
    resolve_provider_name,
)

__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "GenerateResult",
    "LLMProvider",
    "OpenAICompatProvider",
    "QwenDashScopeProvider",
    "get_provider",
    "reset_provider_cache",
    "resolve_provider_name",
]
