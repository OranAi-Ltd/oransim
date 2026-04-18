"""LLM provider protocol — one shape regardless of native API format.

Oransim originally shipped only an OpenAI-compatible client (chat/completions
with ``messages``, flat ``choices[0].message.content``). That shape is the
lingua-franca of LLM gateways but is *not* the native format for several
major providers:

- **Anthropic**: ``/v1/messages`` with system as a top-level parameter and
  usage reported as ``input_tokens`` / ``output_tokens``.
- **Google Gemini**: ``/v1beta/models/{model}:generateContent`` with
  ``contents[].parts`` role-mapped and ``usageMetadata``.
- **AWS Bedrock Converse**: SigV4-signed ``converse`` endpoint with a
  provider-neutral messages shape and ``usage`` broken into input / output
  token counts.
- **Alibaba Qwen DashScope**: ``/services/aigc/text-generation/generation``
  with ``input.messages`` + ``parameters.*`` and ``usage.input_tokens`` /
  ``output_tokens``.

Forcing everyone through a proxy that speaks OpenAI-compat works but adds an
extra hop, a different cost surface, and masks provider-native features
(e.g., Anthropic tool use, Gemini JSON mode, Bedrock guardrails). This
protocol lets callers talk to each provider in its native shape without
caring which one is active.

Adapters implement ``generate(system, user, ...)`` and return a
``GenerateResult`` with a normalized ``usage`` dict using
``prompt_tokens`` / ``completion_tokens`` keys — the same keys the rest of
the codebase already consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GenerateResult:
    """Normalized result of a single completion call."""

    content: str
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    raw_preview: str = ""

    def __post_init__(self) -> None:
        if not self.raw_preview:
            self.raw_preview = self.content[:120]


class LLMProvider(Protocol):
    """Uniform LLM interface across native request formats."""

    name: str

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
        """Run a single completion.

        ``usage`` on the result must contain ``prompt_tokens`` and
        ``completion_tokens`` integer keys so callers written against the
        OpenAI-compat shape keep working unchanged.
        """
        ...


__all__ = ["GenerateResult", "LLMProvider"]
