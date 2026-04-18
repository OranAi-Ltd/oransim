"""Final aggregated prediction report — T(x)-report_strategy_case.

Ingests the full /api/predict response and produces a polished Markdown report
that covers ALL schema outputs into one narrative. Two modes:

  1. LLM mode (GPT-5.4): GPT reads all schema outputs + predictions and writes
     a coherent ~800-word CMO-facing report with actionable recommendations.
  2. Template mode: deterministic Jinja-style MD assembly (fallback when LLM off).

Called from /api/predict as part of schema_outputs.report_strategy_case.
"""

from __future__ import annotations

import json
import time
import uuid

from .soul_llm import (
    API_KEY,
    BASE_URL,
    MODEL,
    TIMEOUT,
    _http_post,
    estimate_cost_cny,
    llm_available,
)

REPORT_SYSTEM = """你是资深营销效果分析顾问（4A 公司水平），为品牌方 CMO 写投放前预测报告。
你基于一批算法输出生成**一份专业、务实、带可执行建议的中文 Markdown 报告**。
要求：
1. 开头写"TL;DR：一句话结论 + 推荐动作"
2. 分 5-6 个小节：投放方案 / 核心预测 / 人群与舆情 / KOL 策略 / 风险点 / 行动建议
3. 数字用千分位，百分比保留 1 位
4. 风险点必须具体（哪个参数敏感、哪个人群不买账）
5. 结尾给 3 条可立即执行的下一步
6. 控制在 700-1000 字
7. 输出纯 Markdown，不要 ``` 包裹"""

REPORT_PROMPT_TEMPLATE = """# 输入数据

## 投放方案
{scenario_json}

## KPI 核心预测
{kpis_json}

## 五阶漏斗预测 (T1-A2)
{funnel_json}

## 本广告 AI 投票情感 (T1-A5)
{sentiment_json}

## 竞品 ROI 估算 (T1-A3)
{competitor_json}

## KOL 组合优化 (T2-A1)
{kol_json}

## KOL 复投排序 Top5 (T2-A5)
{reinvest_top5_json}

## 扩散曲线 (T2-A4)
{diffusion_json}

## 敏感性 Top3 风险参数 (T3-A5)
{sensitivity_top3_json}

## 搜索弹性 (T3-A6)
{elasticity_json}

---

基于上述数据，生成完整中文 Markdown 报告（700-1000 字）。"""


def _truncate_json(obj, limit: int = 1200) -> str:
    """Compact JSON with hard char cap to keep prompt tokens predictable."""
    s = json.dumps(obj, ensure_ascii=False, indent=None, separators=(",", ":"))
    if len(s) > limit:
        s = s[:limit] + "...[截断]"
    return s


def _template_report(scenario: dict, kpis: dict, ps: dict | None, schema: dict) -> str:
    """Deterministic fallback when LLM unavailable."""
    caption = scenario.get("caption", "?")
    budget = scenario.get("total_budget", 0)
    alloc = scenario.get("platform_alloc", {})

    funnel = schema.get("T1_A2_mc_funnel_prediction") or {}
    comp = schema.get("T1_A3_competitor_audience_roi") or {}
    kol = schema.get("T2_A1_kol_mix_optimization") or {}
    reinvest = schema.get("T2_A5_kol_reinvest_ranking") or []
    diff = schema.get("T2_A4_ugc_diffusion_simulation") or {}
    sens = schema.get("T3_A5_sensitivity_analysis") or {}
    elast = schema.get("T3_A6_search_elasticity") or {}

    ctr_pct = (kpis.get("ctr", 0) or 0) * 100
    cvr_pct = (kpis.get("cvr", 0) or 0) * 100
    roi_v = kpis.get("roi", 0) or 0
    net = (ps or {}).get("net_sentiment_score", 0)
    pos_pct = ((ps or {}).get("sentiment_distribution", {}).get("positive", 0) or 0) * 100

    recommend = (
        "✅ 建议投放"
        if roi_v > 1.5 and net > 0.2
        else ("⚠️ 建议先小范围测试" if roi_v > 0.8 else "❌ 建议换素材/调方案")
    )

    top_sens = (sens.get("parameters") or [])[:3]
    top_reinvest = reinvest[:3]

    md = f"""# 投放预测报告 · {caption[:40]}

**TL;DR**：{recommend} · 整体投放 ROI **{roi_v:.2f}x**（revenue/cost，见 §2 · 与 §5 KOL 层 ROI 不是同一口径） · 情感净值 **{net:.2f}**（正面占比 {pos_pct:.1f}%）

## 1. 投放方案
- 素材：{caption}
- 总预算：¥{budget:,.0f}
- 平台分配：{', '.join(f'{k}={v*100:.0f}%' for k,v in alloc.items())}

## 2. 核心预测（蒙特卡洛 10 次均值）
| 指标 | 值 |
|---|---|
| 预估曝光 | {kpis.get('impressions',0):,.0f} |
| 预估点击 | {kpis.get('clicks',0):,.0f} (CTR {ctr_pct:.2f}%) |
| 预估转化 | {kpis.get('conversions',0):,.0f} (CVR {cvr_pct:.2f}%) |
| 整体投放 ROI | **{roi_v:.2f}x** &nbsp;<sub>(revenue/cost)</sub> |
| 预估 GMV | ¥{kpis.get('revenue',0):,.0f} |

### 五阶漏斗 (P25 悲观 / P50 / P75 乐观)
"""
    for stage, label in [
        ("A1_awareness", "曝光"),
        ("A2_interest", "兴趣"),
        ("A3_engagement", "互动"),
        ("A4_conversion", "转化"),
        ("A5_loyalty", "复购"),
    ]:
        b = funnel.get(stage)
        if b:
            md += f"- **{label}** ({stage}): P25={b.get('p25',0):,.0f} · P50=**{b.get('p50',0):,.0f}** · P75={b.get('p75',0):,.0f}\n"

    md += f"""
## 3. 人群与舆情 (基于 {(ps or {}).get('agent_count',0)} AI 用户投票)
- 情感分布：正面 {pos_pct:.1f}% / 中性 {((ps or {}).get('sentiment_distribution',{}).get('neutral',0) or 0)*100:.1f}% / 负面 {((ps or {}).get('sentiment_distribution',{}).get('negative',0) or 0)*100:.1f}%
- 净情感分数：{net:.2f}
- 高购意用户占比：{((ps or {}).get('high_intent_pct',0) or 0)*100:.1f}%
- 高频关注主题：{', '.join(t.get('theme','') for t in (ps or {}).get('key_opinion_themes',[])[:4])}

"""

    if comp and comp.get("rows"):
        md += "## 4. 竞品对比\n\n| 竞品 | 重合率 | 预估 ROI | 可抢夺转化 |\n|---|---|---|---|\n"
        for r in comp["rows"]:
            md += f"| {r.get('competitor_name','?')} | {(r.get('overlap_ratio',0) or 0)*100:.1f}% | {r.get('estimated_roi','?')} | {r.get('estimated_conversion',0):,} |\n"

    if kol:
        kol_roi = kol.get("estimated_roi", 0) or 0
        kol_roi_fmt = (
            f"**{kol_roi:.2f}x**" if kol_roi else "— <sub>(未跑 KOL 组合优化或无有效解)</sub>"
        )
        md += f"""
## 5. KOL 组合策略
- 入选达人：**{kol.get('total_selected',0)}** 位（KOL:KOC = {kol.get('kol_koc_ratio','?')}）
- 预估总触达：{kol.get('estimated_total_reach',0):,}
- **KOL 池加权 ROI（ILP 估算）**：{kol_roi_fmt} &nbsp;<sub>ILP optimizer 基于预算 × 单 KOL 历史 ROI 的加权平均 · **与 §2 整体投放 ROI 不是同一口径**</sub>
- 预算利用率：{(kol.get('budget_utilization',0) or 0)*100:.1f}%

### 优先复投 Top3
"""
        for r in top_reinvest:
            md += f"- **{r.get('name','?')}**（score {r.get('reinvest_score','?')}, {r.get('recommendation','?')}）\n"

    md += f"""
## 6. 扩散预期 (14 天)
- 峰值日：Day **{diff.get('peak_day','?')}**
- 半衰期：**{diff.get('half_life_days','?')}** 天
- 总 UGC 预估：{diff.get('total_ugc_predicted',0):,}
- 搜索弹性 ε：**{elast.get('elasticity_coeff','?')}**（R²={elast.get('r_squared','?')}）—— {elast.get('interpretation','')}

## 7. 风险点（敏感性 Top 3）
"""
    for r in top_sens:
        elas = r.get("elasticity", 0)
        direction = "正向" if elas > 0 else "负向"
        md += f"- **{r.get('parameter_name','?')}** ±20% → GMV 波动 ¥{r.get('gmv_change_amplitude',0):,.0f}（{direction}弹性 {elas:+.2f}）\n"

    md += f"""
## 8. 行动建议
1. {'加大投放预算 20-30%' if roi_v > 2 else '先 30% 预算做 3 天测试，实际 CTR 达 ' + f'{ctr_pct*0.7:.2f}%' + ' 再加投'}
2. {'优先头部 KOL' if kol.get('kol_count',0) > 2 else '重心放在 KOC 矩阵起量'}
3. {'重点监控负面主题，提前准备公关话术' if ((ps or {}).get('sentiment_distribution',{}).get('negative',0) or 0) > 0.15 else '保持素材节奏，14 天内二次触达'}

---

_报告生成：{time.strftime('%Y-%m-%d %H:%M:%S')} · 基于 {(ps or {}).get('agent_count',0)} AI 用户 + Oransim 六层因果仿真_
"""
    return md


def _llm_report(scenario: dict, kpis: dict, ps: dict | None, schema: dict) -> str | None:
    """Ask GPT-5.4 to write a polished report. Returns None on failure."""
    if not llm_available():
        return None

    funnel = schema.get("T1_A2_mc_funnel_prediction") or {}
    comp_rows = (schema.get("T1_A3_competitor_audience_roi") or {}).get("rows") or []
    kol = schema.get("T2_A1_kol_mix_optimization") or {}
    reinvest_top5 = (schema.get("T2_A5_kol_reinvest_ranking") or [])[:5]
    diff = schema.get("T2_A4_ugc_diffusion_simulation") or {}
    sens_params = (schema.get("T3_A5_sensitivity_analysis") or {}).get("parameters", [])[:3]
    elast = schema.get("T3_A6_search_elasticity") or {}

    # Strip heavy/noisy fields
    comp_compact = [
        {
            "name": r.get("competitor_name"),
            "overlap_ratio": r.get("overlap_ratio"),
            "estimated_roi": r.get("estimated_roi"),
            "estimated_conversion": r.get("estimated_conversion"),
            "tgi_top_tags": r.get("tgi_top_tags"),
        }
        for r in comp_rows[:5]
    ]

    kol_compact = (
        {
            "selected_count": kol.get("total_selected"),
            "kol_koc_ratio": kol.get("kol_koc_ratio"),
            "estimated_total_reach": kol.get("estimated_total_reach"),
            "estimated_roi": kol.get("estimated_roi"),
            "budget_utilization": kol.get("budget_utilization"),
        }
        if kol
        else {}
    )

    prompt = REPORT_PROMPT_TEMPLATE.format(
        scenario_json=_truncate_json(scenario, 400),
        kpis_json=_truncate_json(kpis, 400),
        funnel_json=_truncate_json(funnel, 600),
        sentiment_json=_truncate_json(ps or {}, 400),
        competitor_json=_truncate_json(comp_compact, 600),
        kol_json=_truncate_json(kol_compact, 400),
        reinvest_top5_json=_truncate_json(reinvest_top5, 500),
        diffusion_json=_truncate_json(diff, 400),
        sensitivity_top3_json=_truncate_json(sens_params, 400),
        elasticity_json=_truncate_json(elast, 300),
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 2000,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        resp = _http_post(f"{BASE_URL}/chat/completions", headers, body, timeout=TIMEOUT * 4)
        content = resp["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("markdown") or content.startswith("md"):
                content = content.split("\n", 1)[-1]
        usage = resp.get("usage", {})
        return content, usage
    except Exception:
        return None


def build_final_report(
    scenario: dict,
    kpis: dict,
    predicted_sentiment: dict | None,
    schema_outputs: dict,
    use_llm: bool = True,
) -> dict:
    """Main entry. Returns schema report_strategy_case payload."""
    t0 = time.time()
    llm_result = None
    if use_llm:
        llm_result = _llm_report(scenario, kpis, predicted_sentiment, schema_outputs)

    if llm_result:
        content, usage = llm_result
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        cost = estimate_cost_cny(tokens_in, tokens_out)
        source = "gpt-5.4"
    else:
        content = _template_report(scenario, kpis, predicted_sentiment, schema_outputs)
        tokens_in = tokens_out = 0
        cost = 0.0
        source = "template"

    return {
        "report_id": f"rpt_strategy_{uuid.uuid4().hex[:8]}",
        "report_type": "STRATEGY_CASE",
        "report_content": content,
        "output_formats": ["MD"],
        "source": source,
        "generation_ms": int((time.time() - t0) * 1000),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_cny": round(cost, 4),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
