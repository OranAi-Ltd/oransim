"""Final LLM verdict — 把所有预测数据交给 GPT 写成客户能看懂的中文结论。

输入：world model 预测（含 P10/P50/P90）+ 场景信息
输出：2-4 句客户视角的中文结论 + 推荐动作
"""
from __future__ import annotations
import json, time
from typing import Dict, Optional


VERDICT_SYSTEM = """你是营销效果分析专家。根据给定的预测数据，用 2-4 句中文给客户一个实用结论。

结构：
1. 一句话总结预期效果（好/中/差 + 核心数字）
2. 置信区间说明（最坏情况 / 最好情况）
3. 推荐动作（加大/保持/调整/放弃）

要求：
- 口语化，避免术语
- 数字精确但不啰嗦
- 如果 CI 特别宽，明确警告"风险大"
- 如果 Like 率过低（<0.3%），建议换素材
- 如果 Like 率偏高（>3%），强调加大预算"""

VERDICT_PROMPT = """场景：{scenario}

预测:
- 曝光 P50: {exp_p50:,.0f}
- Like 率 P50: {like_rate:.2f}% (80% CI: {like_p10:.2f}% - {like_p90:.2f}%)
- Read 率 P50: {read_rate:.2f}% (80% CI: {read_p10:.2f}% - {read_p90:.2f}%)

用中文 2-4 句给客户结论。"""


def generate_verdict(prediction: Dict, scenario_desc: str = "") -> Dict:
    """Return dict with verdict text + latency + cost."""
    from .soul_llm import (llm_available, _http_stream_post,
                            MODEL, BASE_URL, API_KEY, estimate_cost_cny)
    if not llm_available():
        return _mock_verdict(prediction, scenario_desc)

    prompt = VERDICT_PROMPT.format(
        scenario=scenario_desc or "广告投放",
        exp_p50=prediction.get("exp_p50", 0),
        like_rate=prediction.get("_like_rate_p50", 0),
        like_p10=prediction.get("_like_rate_p10", 0),
        like_p90=prediction.get("_like_rate_p90", 0),
        read_rate=prediction.get("_read_rate_p50", 0),
        read_p10=prediction.get("_read_rate_p10", 0),
        read_p90=prediction.get("_read_rate_p90", 0),
    )
    body = {
        "model": MODEL, "temperature": 0.4, "max_tokens": 300,
        "messages": [
            {"role": "system", "content": VERDICT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json"}
    t0 = time.time()
    try:
        content, usage = _http_stream_post(
            f"{BASE_URL}/chat/completions", headers, body)
        return {
            "verdict": content.strip(),
            "latency_ms": int((time.time() - t0) * 1000),
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "cost_cny": round(estimate_cost_cny(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0)), 4),
            "source": "llm",
        }
    except Exception as e:
        return {"verdict": f"(LLM 失败, fallback) {_mock_verdict(prediction, scenario_desc)['verdict']}",
                "source": "fallback", "error": str(e)[:100]}


def _mock_verdict(prediction: Dict, scenario_desc: str) -> Dict:
    """规则模板 fallback."""
    like_rate = prediction.get("_like_rate_p50", 0)
    like_p10 = prediction.get("_like_rate_p10", 0)
    like_p90 = prediction.get("_like_rate_p90", 0)
    exp_p50 = prediction.get("exp_p50", 0)

    if like_rate > 3:
        verdict_tone = f"预期 Like 率 {like_rate:.1f}%，属爆款潜力区间（小红书平均 1-2%），"\
                       f"建议加大预算。最坏情况 {like_p10:.1f}% 也仍高于平均，风险低。"
    elif like_rate > 1:
        verdict_tone = f"预期 Like 率 {like_rate:.1f}%，达到平均水平。"\
                       f"80% 概率在 {like_p10:.1f}%-{like_p90:.1f}% 之间，可按计划投放。"
    elif like_rate > 0.3:
        verdict_tone = f"预期 Like 率 {like_rate:.1f}%，低于小红书平均（1-2%），"\
                       f"建议优化素材后再投。"
    else:
        verdict_tone = f"预期 Like 率仅 {like_rate:.1f}%，严重低于平均，"\
                       f"不建议投放当前素材，建议重新制作。"

    return {
        "verdict": f"{scenario_desc}: 预期曝光 {exp_p50:,.0f} 次。{verdict_tone}",
        "source": "mock",
        "cost_cny": 0,
    }
