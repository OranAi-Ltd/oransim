"""T1-A3 competitor_audience_roi — LLM-based competitor analysis.

Given a brand name + list of competitor brand names + own budget, asks GPT-5.4
(via OpenAI-compat proxy) to estimate:
  - 粉丝重叠率 (overlap_ratio)
  - TGI top tags
  - estimated抢夺 ROI
  - estimated conversion count

Uses the same LLM client as soul_llm (urllib synchronous) with a strict JSON
schema. One LLM call per competitor.
"""

from __future__ import annotations

import time
import uuid

from .soul_llm import (
    MODEL,
    TIMEOUT,
    call_llm_json_with_retry,
    estimate_cost_cny,
    llm_available,
)

COMPETITOR_SYSTEM = """你是中国数字营销行业的竞品分析专家，熟悉小红书/抖音/微博的品牌生态。
给定本品牌和一个竞品，你基于对两个品牌的已知认知（粉丝人群/调性/价格段/主流营销玩法），
估算该竞品对本品牌的受众重叠和抢夺潜力。严格只输出 JSON。"""

COMPETITOR_PROMPT = """<我方品牌>
名称: {own_brand}
品类: {category}
主要平台: {platforms}
</我方品牌>

<竞品>
名称: {competitor}
</竞品>

<投放预算（元）>
{budget}
</投放预算>

严格输出 JSON（字段必填）：
{{
  "competitor_brand_id": "brand_xxx",  // 给竞品编一个 ID 字符串
  "own_total_fans_estimate": 数字,    // 本品牌在上述平台的估计粉丝量
  "competitor_total_fans_estimate": 数字,
  "overlap_ratio": 0.0~0.5,            // 受众重叠比例
  "overlap_fans_count": 数字,          // overlap_ratio × own_total_fans
  "tgi_top_tags": {{"标签名": 数字TGI>120, ...}},  // 3-6 个高 TGI 标签
  "gender_dist": {{"M": 0~1, "F": 0~1}},
  "age_dist": {{"18-24":0~1, "25-34":0~1, "35-44":0~1, "45+":0~1}},
  "city_tier_dist": {{"T1":0~1, "T2":0~1, "T3":0~1, "T4+":0~1}},
  "estimated_roi": 0.5~5.0,             // 本品牌从该竞品抢夺人群的 ROI 估算
  "estimated_conversion": 数字,        // 预估可抢夺转化人数（基于预算 + overlap + TGI）
  "estimated_cost": 数字,               // 预估所需费用（小于等于预算）
  "historical_deviation": 0.0~0.4,     // 与同行案例的偏差（越低越可靠）
  "reasoning": "不超过80字的中文理由"
}}"""


def _call_llm(
    own_brand: str, category: str, platforms: list[str], competitor: str, budget: float
) -> dict:
    prompt = COMPETITOR_PROMPT.format(
        own_brand=own_brand,
        category=category,
        platforms=",".join(platforms),
        competitor=competitor,
        budget=int(budget),
    )
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": COMPETITOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    t0 = time.time()
    try:
        parsed, usage = call_llm_json_with_retry(
            body, max_retries=2, timeout=TIMEOUT * 2, use_stream=False
        )
        parsed["_latency_ms"] = int((time.time() - t0) * 1000)
        parsed["_tokens_in"] = usage.get("prompt_tokens", 0)
        parsed["_tokens_out"] = usage.get("completion_tokens", 0)
        return parsed
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def estimate_competitor_roi(
    own_brand: str, category: str, platforms: list[str], competitors: list[str], total_budget: float
) -> dict:
    """Batch estimate ROI against a list of competitors. Returns schema-aligned rows."""
    if not competitors:
        return {"rows": [], "total_cost_cny": 0.0, "llm_available": llm_available()}
    # Split budget per competitor
    per_budget = total_budget / len(competitors)
    rows = []
    total_in = total_out = 0
    for comp in competitors:
        if not llm_available():
            # Mock fallback
            row = {
                "competitor_brand_id": f"brand_{comp[:6]}",
                "competitor_name": comp,
                "own_total_fans_estimate": 300_000,
                "competitor_total_fans_estimate": 500_000,
                "overlap_ratio": 0.18,
                "overlap_fans_count": 54_000,
                "tgi_top_tags": {"年轻": 150, "性价比": 135, "颜值": 128},
                "estimated_roi": 2.1,
                "estimated_conversion": 4500,
                "estimated_cost": per_budget * 0.9,
                "historical_deviation": 0.2,
                "reasoning": "(mock) 基于行业平均估算",
                "source": "mock",
            }
        else:
            r = _call_llm(own_brand, category, platforms, comp, per_budget)
            if "_error" in r:
                row = {"competitor_name": comp, "_error": r["_error"], "source": "llm_error"}
            else:
                r["competitor_name"] = comp
                r["source"] = "llm"
                total_in += r.get("_tokens_in", 0)
                total_out += r.get("_tokens_out", 0)
                row = r
        row["estimation_id"] = f"roi_{uuid.uuid4().hex[:6]}"
        row["strategy_name"] = f"从 {comp} 抢夺受众"
        row["run_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rows.append(row)
    cost = estimate_cost_cny(total_in, total_out)
    return {
        "rows": rows,
        "llm_available": llm_available(),
        "tokens_in": total_in,
        "tokens_out": total_out,
        "total_cost_cny": round(cost, 4),
    }
