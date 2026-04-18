"""Real LLM soul agents — OpenAI-compatible client (DeepSeek / Qwen / local vLLM / GPT).

Zero extra deps (uses urllib). Switch via env:
  LLM_MODE=mock|api         (default: mock)
  LLM_BASE_URL=...          (default: https://api.deepseek.com/v1)
  LLM_API_KEY=...
  LLM_MODEL=deepseek-chat   (or qwen-turbo, gpt-4o-mini, Qwen/Qwen3-4B-Instruct, ...)

Drop-in replacement for SoulAgentPool.infer_one when LLM_MODE=api.
"""
from __future__ import annotations
import os, json, time, urllib.request, urllib.error
from typing import Dict, Optional
from .soul import Persona

MODE = os.environ.get("LLM_MODE", "mock")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))


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
视觉：{visual}，BGM：{music}，时长 {duration}s
</impression>

严格输出 JSON：
{{"will_click": true/false,
  "reason": "不超过25字的中文理由",
  "comment": "如果会评论就写评论内容（10-20字），否则空字符串",
  "feel": "厌恶|无感|好奇|心动|购买冲动 里选一个",
  "purchase_intent_7d": 0-1之间的小数}}"""


def _http_post(url: str, headers: Dict, body: Dict, timeout=TIMEOUT) -> Dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_stream_post(url: str, headers: Dict, body: Dict, timeout=TIMEOUT):
    """Stream SSE chunks from an OpenAI-compatible /chat/completions?stream=true.
    Yields per-chunk dict; returns total collected content + usage at end via
    final 'end' event.
    """
    body = dict(body); body["stream"] = True
    headers = dict(headers); headers["Accept"] = "text/event-stream"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=headers, method="POST"
    )
    collected = []
    usage = {}
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
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                collected.append(delta)
    return "".join(collected), usage


def llm_available() -> bool:
    return MODE == "api" and bool(API_KEY)


def llm_info() -> Dict:
    return {
        "mode": MODE, "base_url": BASE_URL, "model": MODEL,
        "api_key_set": bool(API_KEY),
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
) -> Dict:
    """Call real LLM for one soul-agent decision."""
    prompt = PROMPT_TEMPLATE.format(
        persona_card=persona.full_card(),
        interests=", ".join(persona.interests),
        platform=platform, kol_name=kol_name, kol_niche=kol_niche,
        kol_fans=f"{kol_fans/10000:.1f}万" if kol_fans else "无",
        caption=caption, visual=visual, music=music, duration=duration,
    )
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 250,
    }
    # Some upstream (OpenAI-compat GPT-5 relay) don't support response_format; omit it.
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json"}
    use_stream = os.environ.get("LLM_STREAM", "1") not in ("0", "false", "False")
    t0 = time.time()
    try:
        if use_stream:
            content, usage = _http_stream_post(
                f"{BASE_URL}/chat/completions", headers, body)
        else:
            resp = _http_post(f"{BASE_URL}/chat/completions", headers, body)
            content = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
        # extract JSON object from content (robust to code-fence wrapping)
        parsed = _extract_json(content)
        parsed["_latency_ms"] = int((time.time() - t0) * 1000)
        parsed["_tokens_in"] = usage.get("prompt_tokens", 0)
        parsed["_tokens_out"] = usage.get("completion_tokens", 0)
        parsed["_raw_preview"] = content[:120]
        return parsed
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"}
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def _extract_json(s: str) -> Dict:
    """Pull the first JSON object out of an LLM response, tolerating code fences / prose."""
    s = s.strip()
    # strip ``` fences
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"): s = s[4:]
    # find first { and last }
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s[i:j+1])
        except Exception:
            pass
    # fallback: best-effort synthesis
    return {"will_click": False, "reason": "(parse error) " + s[:40],
            "comment": "", "feel": "无感", "purchase_intent_7d": 0.1}


# Cost estimation table (CNY per 1M tokens, rough as of 2025-2026)
COST_TABLE_CNY = {
    # model substring → (input_per_1m, output_per_1m)
    "deepseek-chat":    (1.0,  2.0),      # ¥/M tokens (cache miss)
    "deepseek-v3":      (1.0,  2.0),
    "qwen-turbo":       (0.3,  0.6),
    "qwen-plus":        (2.0,  6.0),
    "qwen3":            (0.6,  1.2),
    "gpt-4o-mini":      (1.1,  4.4),       # ~USD0.15/0.60 → CNY
    "gpt-4o":           (18.0, 72.0),
    "gpt-5":            (0.7,  5.0),        # OpenAI-compat 实测价 (¥0.7/M in · ¥5/M out)
    "claude":           (22.0, 110.0),
}


def estimate_cost_cny(tokens_in: int, tokens_out: int, model: str = MODEL) -> float:
    model_l = model.lower()
    for k, (pin, pout) in COST_TABLE_CNY.items():
        if k in model_l:
            return (tokens_in/1_000_000) * pin + (tokens_out/1_000_000) * pout
    return 0.0
