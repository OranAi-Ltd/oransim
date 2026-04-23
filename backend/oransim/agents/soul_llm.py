"""Real LLM soul agents — multi-provider router.

Routes through :mod:`oransim.agents.llm_providers` so the same call site
works against any supported provider (OpenAI-compat, Anthropic Messages,
Google Gemini, Qwen DashScope). Select at runtime with env:

  LLM_MODE=mock|api                  (default: mock)
  LLM_PROVIDER=openai|anthropic|gemini|qwen   (default: openai)
  LLM_BASE_URL=...                   (OpenAI-compat only; per-provider
                                       overrides like ANTHROPIC_BASE_URL,
                                       GEMINI_BASE_URL, DASHSCOPE_BASE_URL)
  LLM_API_KEY=...                    (or provider-specific:
                                       OPENAI_API_KEY, ANTHROPIC_API_KEY,
                                       GEMINI_API_KEY, DASHSCOPE_API_KEY)
  LLM_MODEL=gpt-5.4 | claude-sonnet-4-6 | gemini-2.5-pro | qwen-plus | ...

When ``LLM_PROVIDER=openai`` (the default), behavior is bit-compatible with
the pre-R-phase client including SSE streaming.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .llm_providers import get_provider, resolve_provider_name
from .soul import Persona

MODE = os.environ.get("LLM_MODE", "mock")
# Base URL / API key are resolved by the provider registry; kept here for
# the legacy :func:`llm_info` report and the cost estimator.
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "gpt-5.4")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))


# ---------------------------------------------------------------------------
# Legacy low-level HTTP helpers (OpenAI-compat only)
# ---------------------------------------------------------------------------
#
# A handful of modules (``competitor_roi``, ``final_report``, ``group_chat``,
# ``discourse``, ``verdict``, ``data.world_events``) build OpenAI-compat
# request bodies directly and call these helpers. Phase R introduces the
# provider registry for the main soul-agent path but leaves these helpers in
# place so callers aren't forced to migrate in one go. They are explicitly
# OpenAI-compat and will be retired when those modules move to the registry.


def _http_post(url: str, headers: dict, body: dict, timeout: float = TIMEOUT) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_stream_post(url: str, headers: dict, body: dict, timeout: float = TIMEOUT):
    """Stream SSE chunks from an OpenAI-compatible ``/chat/completions?stream=true``.

    Returns ``(collected_content, usage_dict)``.
    """
    body = dict(body)
    body["stream"] = True
    headers = dict(headers)
    headers["Accept"] = "text/event-stream"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    collected: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
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


SYSTEM = """你是社媒用户决策的 persona-driven 模拟器。
你扮演给定的虚拟用户，站在他/她的立场，看到一条广告，决定点不点、为什么、会不会评论。
只输出严格 JSON，不要任何额外解释。"""

PROMPT_TEMPLATE = """<persona>
{persona_card}
最近兴趣倾向：{interests}
</persona>

<impression>
平台：{platform}
达人/创作者：{kol_name}（{kol_niche} 赛道，粉丝 {kol_fans}）
位置：feed 第 3 屏
素材文案：{caption}
投放品类：{kol_niche}（只按此赛道判断，别把"探店/vlog"等通用词当成旅行或其它赛道）
视觉：{visual}，BGM：{music}，时长 {duration}s
</impression>

严格输出 JSON：
{{"will_click": true/false,
  "reason": "不超过25字的中文理由",
  "comment": "如果会评论就写评论内容（10-20字），否则空字符串",
  "feel": "厌恶|无感|好奇|心动|购买冲动 里选一个",
  "purchase_intent_7d": 0-1之间的小数}}"""


def _provider_key_present() -> bool:
    """Check whichever env var the *active* provider reads for its key."""
    name = resolve_provider_name()
    if name == "openai":
        return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if name == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY"))
    if name == "gemini":
        return bool(
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
    if name == "qwen_dashscope":
        return bool(
            os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
    return False


def llm_available() -> bool:
    # Read env live so callers that flip LLM_MODE after import (tests,
    # long-lived workers that reload config) see the change.
    return os.environ.get("LLM_MODE", MODE) == "api" and _provider_key_present()


def llm_info() -> dict:
    return {
        "mode": os.environ.get("LLM_MODE", MODE),
        "provider": resolve_provider_name(),
        "base_url": os.environ.get("LLM_BASE_URL", BASE_URL),
        "model": os.environ.get("LLM_MODEL", MODEL),
        "api_key_set": _provider_key_present(),
    }


def soul_infer_llm(
    persona: Persona,
    caption: str,
    platform: str,
    kol_name: str = "无",
    kol_niche: str = "通用",
    kol_fans: int = 0,
    visual: str = "bright",
    music: str = "upbeat",
    duration: float = 15.0,
) -> dict:
    """Call real LLM for one soul-agent decision."""
    prompt = PROMPT_TEMPLATE.format(
        persona_card=persona.full_card(),
        interests=", ".join(persona.interests),
        platform=platform,
        kol_name=kol_name,
        kol_niche=kol_niche,
        kol_fans=f"{kol_fans/10000:.1f}万" if kol_fans else "无",
        caption=caption,
        visual=visual,
        music=music,
        duration=duration,
    )
    use_stream = os.environ.get("LLM_STREAM", "1") not in ("0", "false", "False")
    provider = get_provider()
    # Streaming is an OpenAI-compat optimization; other providers run
    # buffered until their native streaming surface is wired up.
    stream_ok = use_stream and resolve_provider_name() == "openai"
    t0 = time.time()
    try:
        result = provider.generate(
            system=SYSTEM,
            user=prompt,
            model=MODEL,
            temperature=0.7,
            max_tokens=250,
            stream=stream_ok,
        )
        parsed = _extract_json(result.content)
        parsed["_latency_ms"] = result.latency_ms or int((time.time() - t0) * 1000)
        parsed["_tokens_in"] = int(result.usage.get("prompt_tokens", 0) or 0)
        parsed["_tokens_out"] = int(result.usage.get("completion_tokens", 0) or 0)
        parsed["_raw_preview"] = result.raw_preview
        return parsed
    except Exception as e:
        # Surface enough detail for the caller to triage without leaking
        # raw response bodies (which may include auth echo on some gateways).
        return {"_error": f"{type(e).__name__}: {str(e)[:200]}"}


def _extract_json(s: str) -> dict:
    """Pull the first JSON object out of an LLM response, tolerating code fences / prose.

    Legacy callers (soul_infer_llm, discourse, group_chat) still rely on the
    best-effort soul-shaped dict on parse failure. New callers should use
    :func:`_extract_json_strict` instead.
    """
    try:
        return _extract_json_strict(s)
    except ValueError:
        return {
            "will_click": False,
            "reason": "(parse error) " + (s or "")[:40],
            "comment": "",
            "feel": "无感",
            "purchase_intent_7d": 0.1,
        }


def _extract_json_strict(s: str) -> dict:
    """Parse first JSON object; raise ValueError on any issue (no fallback)."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError(f"no JSON object found in LLM response: {s[:80]!r}")
    try:
        return json.loads(s[i : j + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}; content={s[i:j+1][:120]!r}")


# ---------------------------------------------------------------------------
# Retry wrapper for OpenAI-compat JSON callers
# ---------------------------------------------------------------------------
#
# All new LLM callers should use ``call_llm_json_with_retry``. It handles:
#   (1) transient network errors (timeouts / 429 / 5xx) with exponential backoff
#   (2) malformed JSON responses — next attempt appends a "strict JSON"
#       reminder to the user message so the model is less likely to wrap
#       the answer in prose or code fences on the retry
#   (3) gives up after ``max_retries`` and re-raises the last exception, so
#       the caller can fall back to a mock / template path
def call_llm_json_with_retry(
    body: dict,
    *,
    max_retries: int = 2,
    use_stream: bool | None = None,
    timeout: float = TIMEOUT,
    url: str | None = None,
) -> tuple[dict, dict]:
    """Call ``/chat/completions`` + parse JSON with automatic retry.

    Returns ``(parsed_dict, usage_dict)``. Raises the last exception on
    final failure. Requires ``llm_available()`` to be true.
    """
    if not llm_available():
        raise RuntimeError("LLM mode is not 'api' or API key missing")
    if use_stream is None:
        use_stream = os.environ.get("LLM_STREAM", "1") not in ("0", "false", "False")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    target = url or f"{BASE_URL}/chat/completions"
    # clone body so we don't mutate the caller's dict while injecting hints
    body = json.loads(json.dumps(body))
    last_err: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            if use_stream:
                content, usage = _http_stream_post(target, headers, body, timeout=timeout)
            else:
                resp = _http_post(target, headers, body, timeout=timeout)
                content = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage", {})
            parsed = _extract_json_strict(content)
            return parsed, usage
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(0.5 * (2 ** attempt))  # 0.5s → 1s → 2s
            # On JSON parse failure, nudge the model toward a stricter format
            if isinstance(e, (ValueError, json.JSONDecodeError)):
                for msg in reversed(body.get("messages", [])):
                    if msg.get("role") == "user":
                        if "严格输出纯 JSON" not in msg["content"]:
                            msg["content"] = (
                                msg["content"]
                                + "\n\n【重要】严格输出纯 JSON 对象，不要任何代码块围栏、前后文字、解释。"
                            )
                        break
            elif isinstance(e, urllib.error.HTTPError) and e.code == 429:
                time.sleep(1.5)  # extra slack on explicit rate-limit
    raise last_err  # type: ignore[misc]


# Cost estimation table (CNY per 1M tokens, rough as of 2025-2026)
COST_TABLE_CNY = {
    # model substring → (input_per_1m, output_per_1m)
    "deepseek-chat": (1.0, 2.0),  # ¥/M tokens (cache miss)
    "deepseek-v3": (1.0, 2.0),
    "qwen-turbo": (0.3, 0.6),
    "qwen-plus": (2.0, 6.0),
    "qwen3": (0.6, 1.2),
    "gpt-4o-mini": (1.1, 4.4),  # ~USD0.15/0.60 → CNY
    "gpt-4o": (18.0, 72.0),
    "gpt-5": (0.7, 5.0),  # OpenAI-compat 实测价 (¥0.7/M in · ¥5/M out)
    "claude": (22.0, 110.0),
    "gemini-2.5-flash": (2.2, 8.8),  # approx USD0.30/1.20 → CNY
    "gemini-2.5-pro": (9.0, 36.0),
}


def estimate_cost_cny(tokens_in: int, tokens_out: int, model: str = MODEL) -> float:
    model_l = model.lower()
    for k, (pin, pout) in COST_TABLE_CNY.items():
        if k in model_l:
            return (tokens_in / 1_000_000) * pin + (tokens_out / 1_000_000) * pout
    return 0.0
