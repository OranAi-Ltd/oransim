"""Shape-only tests for native LLM provider adapters (no network calls).

Each provider's job is to translate a ``(system, user, model, temp, max_tok)``
call into its native request body + header shape. These tests pin those
shapes so a future refactor (or a careless env-var read) can't silently
break provider parity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# --------------------------------------------------------------- registry


def test_resolve_provider_name_aliases():
    from oransim.agents.llm_providers import resolve_provider_name

    assert resolve_provider_name("openai") == "openai"
    assert resolve_provider_name("openai_compat") == "openai"
    assert resolve_provider_name("anthropic") == "anthropic"
    assert resolve_provider_name("claude") == "anthropic"
    assert resolve_provider_name("gemini") == "gemini"
    assert resolve_provider_name("google") == "gemini"
    assert resolve_provider_name("qwen") == "qwen_dashscope"
    assert resolve_provider_name("dashscope") == "qwen_dashscope"
    assert resolve_provider_name("atlascloud") == "atlascloud"
    assert resolve_provider_name("atlas-cloud") == "atlascloud"
    assert resolve_provider_name("atlas") == "atlascloud"


def test_resolve_provider_name_case_insensitive_and_defaults(monkeypatch):
    from oransim.agents.llm_providers import resolve_provider_name

    assert resolve_provider_name("Anthropic") == "anthropic"
    assert resolve_provider_name("  OPENAI  ") == "openai"
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider_name() == "openai"


def test_resolve_provider_name_rejects_unknown():
    from oransim.agents.llm_providers import resolve_provider_name

    with pytest.raises(ValueError):
        resolve_provider_name("mystery_llm")


def test_get_provider_routes_by_env(monkeypatch):
    from oransim.agents.llm_providers import (
        AnthropicProvider,
        GeminiProvider,
        OpenAICompatProvider,
        QwenDashScopeProvider,
        get_provider,
        reset_provider_cache,
    )

    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    for env_val, expected_type in [
        ("openai", OpenAICompatProvider),
        ("anthropic", AnthropicProvider),
        ("gemini", GeminiProvider),
        ("qwen", QwenDashScopeProvider),
        ("atlascloud", OpenAICompatProvider),
    ]:
        reset_provider_cache()
        monkeypatch.setenv("LLM_PROVIDER", env_val)
        provider = get_provider()
        assert isinstance(provider, expected_type), (
            f"LLM_PROVIDER={env_val} should build {expected_type.__name__}, "
            f"got {type(provider).__name__}"
        )


def test_atlascloud_provider_uses_openai_compat_preset(monkeypatch):
    from oransim.agents.llm_providers import get_provider, reset_provider_cache

    monkeypatch.setenv("LLM_PROVIDER", "atlas")
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "ak-atlas")
    # Copied .env.example values should not override the Atlas preset defaults.
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")

    reset_provider_cache()
    provider = get_provider()

    assert provider.name == "atlascloud"
    assert provider.base_url == "https://api.atlascloud.ai/v1"
    assert provider.api_key == "ak-atlas"


def test_atlascloud_allows_specific_overrides(monkeypatch):
    from oransim.agents.llm_providers import (
        get_provider,
        reset_provider_cache,
        resolve_model_name,
    )

    monkeypatch.setenv("LLM_PROVIDER", "atlascloud")
    monkeypatch.setenv("ATLAS_CLOUD_API_KEY", "ak-alias")
    monkeypatch.setenv("ATLASCLOUD_BASE_URL", "https://atlas.example/v1")
    monkeypatch.setenv("ATLASCLOUD_MODEL", "deepseek-ai/deepseek-v4-pro")

    reset_provider_cache()
    provider = get_provider()

    assert provider.base_url == "https://atlas.example/v1"
    assert provider.api_key == "ak-alias"
    assert resolve_model_name() == "deepseek-ai/deepseek-v4-pro"


# --------------------------------------------------------- OpenAI-compat


def test_openai_compat_body_shape():
    from oransim.agents.llm_providers import OpenAICompatProvider

    p = OpenAICompatProvider(base_url="https://gw/v1", api_key="sk")
    body = p.build_body(
        "SYS",
        "USER",
        model="gpt-5.4",
        temperature=0.3,
        max_tokens=100,
        stream=False,
    )
    assert body["model"] == "gpt-5.4"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "USER"}
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 100
    assert "stream" not in body  # only present when True

    body_s = p.build_body(
        "SYS",
        "USER",
        model="gpt-5.4",
        temperature=0.3,
        max_tokens=100,
        stream=True,
    )
    assert body_s["stream"] is True


# ------------------------------------------------------------- Anthropic


def test_anthropic_body_shape_and_headers():
    from oransim.agents.llm_providers import AnthropicProvider

    p = AnthropicProvider(base_url="https://api.anthropic.com", api_key="sk-ant-xxx")
    body = p.build_body(
        "SYS",
        "USER",
        model="claude-sonnet-4-6",
        temperature=0.4,
        max_tokens=128,
    )
    # System is NOT in messages — it's top-level
    assert body["system"] == "SYS"
    assert body["messages"] == [{"role": "user", "content": "USER"}]
    assert all(m["role"] != "system" for m in body["messages"])
    assert body["model"] == "claude-sonnet-4-6"
    assert body["max_tokens"] == 128

    headers = p._headers()
    assert headers["x-api-key"] == "sk-ant-xxx"
    assert "anthropic-version" in headers
    assert headers["Content-Type"] == "application/json"
    # Must NOT use Bearer auth
    assert "Authorization" not in headers


# --------------------------------------------------------------- Gemini


def test_gemini_body_shape_and_url():
    from oransim.agents.llm_providers import GeminiProvider

    p = GeminiProvider(api_key="AIza-xxx")
    body = p.build_body("SYS", "USER", temperature=0.5, max_tokens=200)
    # systemInstruction at top level (not a role message)
    assert body["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    # contents uses parts[].text, role user
    assert body["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
    # temp + max live under generationConfig
    assert body["generationConfig"]["temperature"] == 0.5
    assert body["generationConfig"]["maxOutputTokens"] == 200
    # Header auth, no key in URL
    headers = p._headers()
    assert headers["x-goog-api-key"] == "AIza-xxx"


def test_gemini_body_omits_system_when_empty():
    from oransim.agents.llm_providers import GeminiProvider

    p = GeminiProvider(api_key="AIza-xxx")
    body = p.build_body("", "USER", temperature=0.5, max_tokens=200)
    assert "systemInstruction" not in body


# ---------------------------------------------------------- Qwen DashScope


def test_qwen_dashscope_body_shape():
    from oransim.agents.llm_providers import QwenDashScopeProvider

    p = QwenDashScopeProvider(api_key="sk-qwen")
    body = p.build_body(
        "SYS",
        "USER",
        model="qwen-plus",
        temperature=0.6,
        max_tokens=150,
    )
    # native format: input.messages + parameters.*
    assert body["model"] == "qwen-plus"
    assert body["input"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert body["parameters"]["result_format"] == "message"
    assert body["parameters"]["temperature"] == 0.6
    assert body["parameters"]["max_tokens"] == 150
    # No flat "messages" or "temperature" at top level
    assert "messages" not in body
    assert "temperature" not in body

    headers = p._headers()
    assert headers["Authorization"] == "Bearer sk-qwen"


# ----------------------------------------------------- soul_llm integration


def test_llm_info_reports_active_provider(monkeypatch):
    from oransim.agents import soul_llm
    from oransim.agents.llm_providers import reset_provider_cache

    reset_provider_cache()
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setenv("LLM_MODE", "api")

    info = soul_llm.llm_info()
    assert info["provider"] == "anthropic"
    assert info["api_key_set"] is True
    assert soul_llm.llm_available() is True


def test_llm_info_reports_atlascloud_provider(monkeypatch):
    from oransim.agents import soul_llm
    from oransim.agents.llm_providers import reset_provider_cache

    reset_provider_cache()
    monkeypatch.setenv("LLM_MODE", "api")
    monkeypatch.setenv("LLM_PROVIDER", "atlas-cloud")
    monkeypatch.setenv("ATLAS_CLOUD_API_KEY", "ak-atlas")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")

    info = soul_llm.llm_info()

    assert info["provider"] == "atlascloud"
    assert info["base_url"] == "https://api.atlascloud.ai/v1"
    assert info["model"] == "qwen/qwen3.5-flash"
    assert info["api_key_set"] is True
    assert soul_llm.llm_available() is True


def test_atlascloud_legacy_json_helper_uses_atlas_endpoint(monkeypatch):
    from oransim.agents import soul_llm
    from oransim.agents.llm_providers import reset_provider_cache

    captured = {}

    def fake_http_post(url, headers, body, timeout):
        captured.update(url=url, headers=headers, body=body, timeout=timeout)
        return {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    reset_provider_cache()
    monkeypatch.setenv("LLM_MODE", "api")
    monkeypatch.setenv("LLM_PROVIDER", "atlascloud")
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "ak-atlas")
    monkeypatch.setattr(soul_llm, "_http_post", fake_http_post)

    parsed, usage = soul_llm.call_llm_json_with_retry(
        {"messages": [{"role": "user", "content": "{}"}], "model": "qwen/qwen3.5-flash"},
        use_stream=False,
    )

    assert parsed == {"ok": True}
    assert usage == {"prompt_tokens": 3, "completion_tokens": 2}
    assert captured["url"] == "https://api.atlascloud.ai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer ak-atlas"


def test_llm_info_mock_mode_has_no_key(monkeypatch):
    from oransim.agents import soul_llm
    from oransim.agents.llm_providers import reset_provider_cache

    reset_provider_cache()
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # llm_available gates on MODE too, so mock mode must be False even
    # if a key is set
    assert soul_llm.llm_available() is False
