"""Broad-spectrum world events ingestion.

Pulls today's top events across ALL categories (politics, sports, entertainment,
disasters, viral memes, tech launches, economy, celebrity news, ...) — not
limited to marketing-specific events.

Uses LLM (already configured) as the curator: asks GPT to enumerate plausible
top events with consumer-impact scoring. Cached to disk; refreshable on demand.

For production: swap LLM curation for real news APIs (NewsAPI / GDELT /
微博热搜 / Reddit hot / Google Trends).
"""
from __future__ import annotations
import os, json, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

CACHE = Path("/tmp/oransim_world.json")
CACHE_TTL = 6 * 3600  # 6 hours


def _read_cache(max_age_sec: int = CACHE_TTL) -> Optional[Dict]:
    if not CACHE.exists():
        return None
    age = time.time() - CACHE.stat().st_mtime
    if age > max_age_sec:
        return None
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return None


def _write_cache(state: Dict) -> None:
    try:
        CACHE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception:
        pass


WORLD_PROMPT = """今天是 {today}（请基于你的常识猜测这一天前后全球的真实热度事件，不要捏造太离谱的）。

列出 10 件目前/近期最热的事件——不限于营销/消费类，**任何类型都可以**：
政治外交、体育赛事、电影电视剧、热门梗/迷因、自然灾害、新品发布、明星动态、
经济新闻、社会议题、科技突破、游戏发行、网红事件 等等。

每件事给：
- title: 一句话
- category: politics/sports/entertainment/disaster/meme/tech/economy/celebrity/social/gaming/...
- consumer_impact: 对整体消费意愿/社媒注意力的影响系数 (-0.3 到 +0.3, 大多在 -0.05 到 +0.1)
- attention_share: 这件事在大众注意力里的占比 (0..1)
- affected_categories: 受影响的消费品类列表 (如 ["奶茶", "外卖", "电影周边", "羽绒服"])

严格 JSON：
{{"events": [{{...}}, ...]}}"""


def refresh_world_state(force: bool = False) -> Dict:
    """Refresh today's world snapshot. Returns the new state dict."""
    if not force:
        cached = _read_cache()
        if cached:
            return cached

    today = datetime.now().date().isoformat()
    state = {
        "today": today,
        "fetched_at": datetime.now().isoformat(),
        "source": "fallback-mock",
        "events": [],
        "sentiment": "neutral",
        "avg_consumer_impact": 0.0,
        "total_attention_share": 0.0,
    }

    # Try LLM curation
    try:
        from ..agents.soul_llm import (
            llm_available, _http_stream_post, _http_post, _extract_json,
            MODEL, BASE_URL, API_KEY,
        )
        if llm_available():
            body = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "你是消费洞察分析师。基于常识列举真实近期热门事件。只输出 JSON。"},
                    {"role": "user", "content": WORLD_PROMPT.format(today=today)},
                ],
                "temperature": 0.4,
                "max_tokens": 2000,
            }
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            content, _ = _http_stream_post(f"{BASE_URL}/chat/completions", headers, body)
            data = _extract_json(content)
            events = data.get("events", [])
            if events:
                state["source"] = f"llm/{MODEL}"
                state["events"] = events
                state["avg_consumer_impact"] = round(
                    sum(e.get("consumer_impact", 0) for e in events) / len(events), 3)
                state["total_attention_share"] = round(
                    sum(e.get("attention_share", 0) for e in events), 3)
                ai = state["avg_consumer_impact"]
                state["sentiment"] = (
                    "viral" if ai > 0.12 else
                    "positive" if ai > 0.04 else
                    "crisis" if ai < -0.12 else
                    "negative" if ai < -0.04 else "neutral"
                )
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    _write_cache(state)
    return state


def get_world_state() -> Dict:
    cached = _read_cache()
    if cached:
        return cached
    return refresh_world_state()


def category_lift(world: Dict, category: str) -> float:
    """How much today's events lift/dampen this consumer category."""
    if not world or not world.get("events"):
        return 1.0
    lift = 1.0
    for e in world["events"]:
        affected = e.get("affected_categories", []) or []
        if any(category.lower() in c.lower() or c.lower() in category.lower() for c in affected):
            lift *= 1.0 + e.get("consumer_impact", 0) * (e.get("attention_share", 0.1) * 5)
    return float(max(0.5, min(1.8, lift)))
