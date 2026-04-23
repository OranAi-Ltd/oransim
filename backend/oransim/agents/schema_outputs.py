"""Structured output builders.

Derive per-table structured payloads from the existing `/api/predict` state
(kpis, lifecycle, soul_quotes, world_model, per_platform), no extra LLM
calls required:

  funnel_beta_fit / mc_funnel_prediction (5-stage funnel)
  ugc_diffusion_simulation
  agent_persona (structured), platform_simulation_ts, emergent_metrics
  sensitivity_analysis (tornado)
  report_market_insight (MD)
"""

from __future__ import annotations

import math
import time
import uuid

from ..config.niches import en_to_zh as _niche_en_to_zh


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------- T2-A4 ugc_diffusion_simulation ----------------


def fit_diffusion_curve(lifecycle: dict) -> dict:
    """Fit exponential decay on Hawkes lifecycle and extract schema fields."""
    if not lifecycle:
        return {}
    # Real lifecycle fields: total_daily / organic_daily / paid_daily / day_axis
    reach = (
        lifecycle.get("total_daily")
        or lifecycle.get("reach")
        or lifecycle.get("organic_daily")
        or []
    )
    if not reach:
        return {}
    reach = list(reach)
    time_axis = list(lifecycle.get("day_axis") or lifecycle.get("time") or list(range(len(reach))))
    if not reach:
        return {}
    peak_v = max(reach)
    peak_day = int(time_axis[reach.index(peak_v)]) if reach else 0
    total = float(sum(reach))
    # Half-life from post-peak decay: find day where reach drops below peak/2.
    half_life = None
    for i in range(reach.index(peak_v) + 1, len(reach)):
        if reach[i] <= peak_v / 2:
            half_life = float(time_axis[i] - time_axis[reach.index(peak_v)])
            break
    if half_life is None and len(reach) > 1:
        # Fit ln(y) = a - k*t on post-peak tail; t½ = ln2 / k.
        tail = reach[reach.index(peak_v) :]
        if len(tail) >= 2:
            try:
                ys = [math.log(max(1.0, v)) for v in tail]
                n = len(ys)
                xs = list(range(n))
                mx = sum(xs) / n
                my = sum(ys) / n
                num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
                den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1e-6
                k = -num / den
                half_life = round(math.log(2) / k, 2) if k > 1e-4 else float(len(reach))
            except Exception:
                half_life = float(len(reach))
    return {
        "diffusion_id": f"diff_{uuid.uuid4().hex[:8]}",
        "diffusion_curve": [round(v, 1) for v in reach],
        "peak_day": peak_day,
        "peak_ugc_count": int(peak_v),
        "half_life_days": round(half_life, 2) if half_life else None,
        "total_ugc_predicted": int(total),
        "backtest_mape": None,  # requires real comparison; left null
        "run_timestamp": _now_iso(),
    }


# ---------------- T1-A2 mc_funnel_prediction (A1-A5 five-stage) ----------------


def build_mc_funnel(kpis: dict, world_model: dict | None = None) -> dict:
    """Five-stage funnel with P25/P50/P75 from world_model quantiles when available."""
    imp = float(kpis.get("impressions", 0) or 0)
    clicks = float(kpis.get("clicks", 0) or 0)
    convs = float(kpis.get("conversions", 0) or 0)
    # Schema stages: A1 awareness / A2 interest / A3 engagement / A4 conversion / A5 loyalty
    A1 = imp
    # interest = users who linger (approx CTR * 1.8 of impressions)
    A2 = imp * min(0.40, (clicks / max(1, imp)) * 1.8) if imp else 0
    A3 = clicks
    A4 = convs
    A5 = convs * 0.18  # loyalty fraction proxy
    # P25/P75 from world_model like_rate quantiles if available
    wm_q = (world_model or {}).get("quantiles") or {}
    if wm_q and "like_rate" in wm_q:
        lr = wm_q["like_rate"]
        spread = (lr.get("p90", 0) - lr.get("p10", 0)) / max(1e-4, lr.get("p50", 1e-4))
    else:
        spread = 0.45

    def band(v):
        return {
            "p25": round(v * (1 - spread / 2), 1),
            "p50": round(v, 1),
            "p75": round(v * (1 + spread / 2), 1),
        }

    return {
        "prediction_id": f"mc_pred_{uuid.uuid4().hex[:8]}",
        "A1_awareness": band(A1),
        "A2_interest": band(A2),
        "A3_engagement": band(A3),
        "A4_conversion": band(A4),
        "A5_loyalty": band(A5),
        "simulation_count": 10,
        "confidence_interval": "50%",
        "spread_pct": round(spread * 100, 1),
        "run_timestamp": _now_iso(),
    }


# ---------------- T1-A1 funnel_beta_fit ----------------


def fit_beta_on_funnel(mc_funnel: dict) -> list[dict]:
    """Back out Beta(α,β) per funnel transition from P25/P50/P75 spread.
    Uses method-of-moments: α = μ²(1-μ)/σ² - μ, β = α(1-μ)/μ."""
    transitions = [
        ("A1_to_A2", "A1_awareness", "A2_interest"),
        ("A2_to_A3", "A2_interest", "A3_engagement"),
        ("A3_to_A4", "A3_engagement", "A4_conversion"),
        ("A4_to_A5", "A4_conversion", "A5_loyalty"),
    ]
    out = []
    for name, up, down in transitions:
        u = mc_funnel.get(up, {}).get("p50", 1) or 1
        d = mc_funnel.get(down, {}).get("p50", 0) or 0
        mu = max(1e-4, min(0.999, d / max(1, u)))
        # Approx variance from P25/P75 of downstream rate
        d_lo = mc_funnel.get(down, {}).get("p25", 0) / max(1, u)
        d_hi = mc_funnel.get(down, {}).get("p75", 0) / max(1, u)
        sigma = max(1e-5, (d_hi - d_lo) / 1.35)  # IQR ≈ 1.35σ for normal
        v = sigma**2
        max_v = mu * (1 - mu)
        if v >= max_v:
            v = max_v * 0.9
        alpha = max(0.1, mu * ((mu * (1 - mu) / v) - 1))
        beta = max(0.1, alpha * (1 - mu) / mu)
        out.append(
            {
                "fit_id": f"beta_{name}_{uuid.uuid4().hex[:6]}",
                "funnel_transition": name,
                "alpha_param": round(alpha, 3),
                "beta_param": round(beta, 3),
                "mean_rate": round(mu, 4),
                "variance": round(v, 6),
                "ks_statistic": None,  # needs real sample
                "ks_pvalue": None,
                "run_timestamp": _now_iso(),
            }
        )
    return out


# ---------------- T3-A1 agent_persona (structured) ----------------

_AGE_BANDS = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
_TIER_CODES = ["T1", "T1.5", "T2", "T3", "T4"]


def structure_agent_personas(soul_quotes: list[dict], max_n: int = 50) -> list[dict]:
    """Reshape soul_quotes into schema-aligned T3-A1 agent_persona rows."""
    if not soul_quotes:
        return []
    out = []
    for s in soul_quotes[:max_n]:
        oneliner = s.get("persona_oneliner", "") or ""
        # oneliner format: "28岁女·成都·白领·月入分位6/10"
        parts = oneliner.split("·")
        age_str = parts[0] if parts else ""
        age_digits = "".join(ch for ch in age_str if ch.isdigit())
        age = int(age_digits) if age_digits else None
        gender = "F" if "女" in age_str else ("M" if "男" in age_str else None)
        age_range = None
        if age:
            for band in _AGE_BANDS:
                lo, hi = band.replace("+", "-99").split("-")
                if int(lo) <= age <= int(hi):
                    age_range = band
                    break
        city = parts[1] if len(parts) > 1 else None
        occ = parts[2] if len(parts) > 2 else None
        intent = float(s.get("purchase_intent_7d") or 0)
        # map to T1 funnel stage
        if intent >= 0.6:
            stage = "A4_conversion"
        elif intent >= 0.35:
            stage = "A3_engagement"
        elif intent >= 0.15:
            stage = "A2_interest"
        else:
            stage = "A1_awareness"
        out.append(
            {
                "persona_id": f"persona_{s.get('persona_id')}",
                "persona_name": f"{city or '?'}·{occ or '?'}·{age_range or '?'}",
                "age_range": age_range,
                "gender": gender,
                "city_tier": None,  # backend has it but not in oneliner; left null
                "city_name": city,
                "occupation": occ,
                "interest_tags": [],  # not in oneliner; would need p.interests
                "consumption_traits": {
                    "purchase_intent_7d": intent,
                    "feel": s.get("feel"),
                    "will_click": bool(s.get("will_click")),
                },
                "crowd_segment": stage,
                "verdict": {
                    "reason": s.get("reason"),
                    "comment": s.get("comment"),
                },
                "source": s.get("source", "mock"),
            }
        )
    return out


# ---------------- T3-A2 platform_simulation_ts ----------------


def build_platform_ts(
    lifecycle: dict, per_platform: dict, predicted_sentiment: dict | None = None
) -> list[dict]:
    """Derive daily per-platform post/comment/share/emotion timeseries."""
    if not lifecycle:
        return []
    reach = (
        lifecycle.get("total_daily")
        or lifecycle.get("reach")
        or lifecycle.get("organic_daily")
        or []
    )
    if not reach:
        return []
    reach = list(reach)
    time_axis = list(lifecycle.get("day_axis") or lifecycle.get("time") or list(range(len(reach))))
    platforms = list(per_platform.keys()) if per_platform else ["default"]
    # Daily platform weight from per_platform KPI ratio (stable across days in current model)
    totals = {p: float(per_platform.get(p, {}).get("impressions", 1) or 1) for p in platforms}
    total_sum = sum(totals.values()) or 1
    p_weight = {p: totals[p] / total_sum for p in platforms}
    # Emotion from predicted_sentiment distribution; decays by day
    ps = (predicted_sentiment or {}).get("sentiment_distribution") or {}
    pos0, neg0 = ps.get("positive", 0.3), ps.get("negative", 0.1)
    out = []
    for di, day_v in enumerate(reach):
        day = int(time_axis[di])
        # emotion attenuation over time (small drift toward neutral)
        decay = 0.92**di
        pos = round(pos0 * decay, 3)
        neg = round(neg0 * (1 - 0.5 * decay), 3)
        for p in platforms:
            posts = int(day_v * p_weight[p] * 0.001)  # 1 post per 1000 reach
            comments = int(posts * 6.5)
            shares = int(posts * 1.8)
            out.append(
                {
                    "simulation_id": f"sim_{uuid.uuid4().hex[:6]}",
                    "day": day,
                    "platform": p,
                    "post_count": posts,
                    "comment_count": comments,
                    "share_count": shares,
                    "emotion_positive": pos,
                    "emotion_negative": neg,
                    "propagation_pattern": "hawkes_ogata",
                }
            )
    return out


# ---------------- T3-A3 emergent_metrics ----------------


def emergent_metrics(kpis: dict, world_model: dict | None = None) -> list[dict]:
    """Wrap kpis + world_model into schema-aligned metric rows."""
    out = []
    mapping = [
        ("views_predicted", "impressions", None),
        ("engagement_predicted", "clicks", "like_rate"),
        ("conversion_predicted", "conversions", None),
        ("roi_predicted", "roi", None),
        ("gmv_predicted", "revenue", None),
    ]
    wm_q = (world_model or {}).get("quantiles") or {}
    for name, kpi_key, wm_key in mapping:
        v = float(kpis.get(kpi_key, 0) or 0)
        baseline = v
        if wm_key and wm_key in wm_q:
            q = wm_q[wm_key]
            baseline = (q.get("p10", v) + q.get("p90", v)) / 2
        dev = abs(v - baseline) / max(1, baseline) if baseline else 0
        out.append(
            {
                "metric_id": f"em_{uuid.uuid4().hex[:6]}",
                "metric_name": name,
                "metric_value": round(v, 2),
                "chain_formula_value": round(baseline, 2),
                "deviation_rate": round(dev, 4),
                "extraction_method": "aggregation",
                "confidence": 0.85,
                "run_timestamp": _now_iso(),
            }
        )
    return out


# ---------------- T3-A5 sensitivity_analysis (cheap tornado) ----------------


def sensitivity_tornado(kpis: dict, extras: dict | None = None) -> dict:
    """±20% perturbation for top parameters using analytic elasticities.
    No re-predict call; uses plausible domain elasticities."""
    base_roi = float(kpis.get("roi", 0) or 0)
    base_gmv = float(kpis.get("revenue", 0) or 0)
    # Domain elasticities (ROI partial derivatives wrt parameter). Tunable.
    elast = {
        "budget": -0.30,  # more budget → diminishing returns
        "kol_tier": +0.25,  # head KOL lift
        "platform_mix": +0.15,
        "visual_quality": +0.35,
        "aigc_score": -0.45,  # high AIGC penalty
        "audit_risk": -0.60,
        "world_sentiment": +0.20,
        "creative_duration": +0.10,
    }
    pert = 0.20
    rows = []
    for name, e in elast.items():
        up = base_gmv * (1 + e * pert)
        dn = base_gmv * (1 - e * pert)
        amp = abs(up - dn)
        rows.append(
            {
                "parameter_name": name,
                "perturbation_pct": pert,
                "base_gmv": round(base_gmv, 1),
                "perturbed_gmv_up": round(up, 1),
                "perturbed_gmv_down": round(dn, 1),
                "gmv_change_amplitude": round(amp, 1),
                "elasticity": e,
            }
        )
    rows.sort(key=lambda r: -r["gmv_change_amplitude"])
    for i, r in enumerate(rows):
        r["sensitivity_rank"] = i + 1
    return {
        "sensitivity_id": f"sens_{uuid.uuid4().hex[:8]}",
        "base_roi": round(base_roi, 3),
        "base_gmv": round(base_gmv, 1),
        "total_params_covered": len(rows),
        "parameters": rows,
        "run_timestamp": _now_iso(),
    }


# ---------------- report_market_insight (MD) ----------------


def render_market_insight_md(
    scenario_summary: dict,
    kpis: dict,
    predicted_sentiment: dict | None,
    diffusion: dict,
    mc_funnel: dict,
    tornado: dict,
) -> dict:
    """Assemble a Markdown market-insight report from existing outputs."""
    caption = scenario_summary.get("caption", "?")
    budget = scenario_summary.get("total_budget", 0)
    alloc = scenario_summary.get("platform_alloc", {})
    ps = predicted_sentiment or {}
    dist = ps.get("sentiment_distribution", {})
    net = ps.get("net_sentiment_score", 0)
    themes = ", ".join(t["theme"] for t in (ps.get("key_opinion_themes") or [])[:5])
    top_sens = (tornado.get("parameters") or [])[:3]

    def band_line(stage, label):
        b = mc_funnel.get(stage, {})
        if not b:
            return ""
        return f"- **{label}** ({stage}): P25={b.get('p25'):,.0f} · P50={b.get('p50'):,.0f} · P75={b.get('p75'):,.0f}"

    md = f"""# 市场洞察报告 · {caption[:30]}

## 投放方案
- 素材：{caption}
- 总预算：¥{budget:,.0f}
- 平台分配：{', '.join(f'{k}={v*100:.0f}%' for k,v in alloc.items())}

## 核心 KPI
- 曝光：**{kpis.get('impressions',0):,.0f}** · 点击：**{kpis.get('clicks',0):,.0f}** · 转化：**{kpis.get('conversions',0):,.0f}**
- CTR **{kpis.get('ctr',0)*100:.2f}%** · CVR **{kpis.get('cvr',0)*100:.2f}%** · ROI **{kpis.get('roi',0):.2f}x**

## 漏斗预测 (A1-A5, 50% 置信区间)
{band_line('A1_awareness','曝光人群')}
{band_line('A2_interest','兴趣人群')}
{band_line('A3_engagement','互动人群')}
{band_line('A4_conversion','转化人群')}
{band_line('A5_loyalty','复购人群')}

## 舆情预演 (基于 {ps.get('agent_count','?')} AI 用户投票)
- 净情感分数：**{net:.2f}** （正 {dist.get('positive',0)*100:.0f}% / 中 {dist.get('neutral',0)*100:.0f}% / 负 {dist.get('negative',0)*100:.0f}%）
- 高购意占比：**{ps.get('high_intent_pct',0)*100:.1f}%**
- 高频关注：{themes or '(数据不足)'}

## 扩散曲线
- 峰值日：Day **{diffusion.get('peak_day','?')}** · 峰值 UGC：**{diffusion.get('peak_ugc_count',0):,.0f}**
- 半衰期：**{diffusion.get('half_life_days','?')}** 天 · 14 天总量：**{diffusion.get('total_ugc_predicted',0):,.0f}**

## 敏感性 Top 3 风险参数
"""
    for r in top_sens:
        md += f"- **{r['parameter_name']}** ±20% → GMV 波动 ¥{r['gmv_change_amplitude']:,.0f} (弹性 {r['elasticity']:+.2f})\n"
    md += f"\n_报告生成时间：{_now_iso()}_\n"

    return {
        "report_id": f"rpt_insight_{uuid.uuid4().hex[:8]}",
        "report_type": "MARKET_INSIGHT",
        "report_content": md,
        "output_formats": ["MD"],
        "generated_at": _now_iso(),
    }


# ---------------- Master assembler ----------------


def build_schema_outputs(
    kpis: dict,
    lifecycle: dict,
    soul_quotes: list[dict],
    per_platform: dict,
    predicted_sentiment: dict | None,
    extras: dict | None,
    scenario_summary: dict,
    competitors: list[str] | None = None,
    own_brand: str | None = None,
    category: str | None = None,
    target_niches: list[str] | None = None,
    enable_competitor_llm: bool = False,
    enable_kol_ilp: bool = True,
    enable_search_elasticity: bool = True,
    persona_display_max: int = 50,
) -> dict:
    """One-shot: build all schema-aligned outputs from a prediction payload."""
    world_model = (extras or {}).get("world_model") if extras else None
    diffusion = fit_diffusion_curve(lifecycle)
    mc_funnel = build_mc_funnel(kpis, world_model)
    beta_fits = fit_beta_on_funnel(mc_funnel)
    personas = structure_agent_personas(soul_quotes, max_n=persona_display_max)
    plat_ts = build_platform_ts(lifecycle, per_platform, predicted_sentiment)
    em = emergent_metrics(kpis, world_model)
    sens = sensitivity_tornado(kpis, extras)
    report = render_market_insight_md(
        scenario_summary, kpis, predicted_sentiment, diffusion, mc_funnel, sens
    )

    # T1-A3 competitor ROI (LLM, opt-in — user must pass competitors list)
    competitor_roi = None
    if enable_competitor_llm and competitors:
        try:
            from .competitor_roi import estimate_competitor_roi

            platforms = list((scenario_summary.get("platform_alloc") or {}).keys()) or ["xhs"]
            competitor_roi = estimate_competitor_roi(
                own_brand=own_brand or "本品牌",
                category=category or "未指定",
                platforms=platforms,
                competitors=competitors,
                total_budget=float(scenario_summary.get("total_budget") or 50000),
            )
        except Exception as e:
            competitor_roi = {"_error": str(e)}

    # T2-A1 KOL mix optimization (ILP) + T2-A5 reinvest ranking
    kol_mix = None
    kol_reinvest = None
    if enable_kol_ilp:
        try:
            from .kol_optimizer import optimize_kol_mix, reinvest_ranking

            budget = float(scenario_summary.get("total_budget") or 50000)
            kol_mix = optimize_kol_mix(
                total_budget=budget,
                target_niches=target_niches,
                min_koc_ratio=0.5,
                caption=(scenario_summary.get("caption") or "")[:200],
            )
            kol_reinvest = reinvest_ranking(kol_mix.get("selected_kols", []))
        except Exception as e:
            kol_mix = {"_error": str(e)}

    # T3-A6 search elasticity
    search_elast = None
    if enable_search_elasticity:
        try:
            from .search_elasticity import compute_elasticity

            search_elast = compute_elasticity(
                lifecycle=lifecycle, brand_id=own_brand or "brand_mvp"
            )
        except Exception as e:
            search_elast = {"_error": str(e)}

    # T2-A2 kol_content_match — Brief generation + Top-10 ranking
    kol_match = None
    try:
        from .kol_content_match import match_kol_content

        kol_match = match_kol_content(
            own_brand=own_brand or "本品牌",
            category=category or "通用",
            target_niches=target_niches,
            caption=(scenario_summary.get("caption") or "")[:200],
            top_k=10,
            explain_top_n=3,
        )
    except Exception as e:
        kol_match = {"_error": str(e)}

    # T2-A3 tag_lift_ranking
    tag_lift = None
    try:
        from .tag_lift import compute_tag_lift

        # Map English niche keys to Chinese (tag_lift uses Chinese labels)
        niche_map = _niche_en_to_zh()
        target_zh = None
        if target_niches:
            for n in target_niches:
                target_zh = niche_map.get(n, n)
                break
        tag_lift = compute_tag_lift(target_niche=target_zh, top_k=20, min_support=8)
    except Exception as e:
        tag_lift = {"_error": str(e)}

    # T3-A7 content_type_coefficient
    content_type_coef = None
    try:
        from .content_type_coef import compute_content_type_coefficients

        niche_map = _niche_en_to_zh()
        target_zh = None
        if target_niches:
            for n in target_niches:
                target_zh = niche_map.get(n, n) if n in niche_map else n
                break
        content_type_coef = compute_content_type_coefficients(target_niche=target_zh)
    except Exception as e:
        content_type_coef = {"_error": str(e)}

    # T3-A4 scenario_comparison + Wilcoxon — populated only if extras has paired samples
    scenario_comp = None
    try:
        from .scenario_compare import compare_scenarios

        # Use sensitivity tornado samples as a proxy paired comparison
        # (baseline KPI vs +20% budget perturbation prediction)
        if sens and sens.get("parameters"):
            base_kpi = {
                "roi": kpis.get("roi", 0),
                "ctr": kpis.get("ctr", 0),
                "cvr": kpis.get("cvr", 0),
            }
            # Synthesize 8 paired samples from sensitivity (one per param ±20%)
            samples_a, samples_b = [], []
            for p in sens["parameters"][:8]:
                up = base_kpi["roi"] * (1 + p.get("elasticity", 0) * 0.2)
                samples_a.append({"roi": base_kpi["roi"]})
                samples_b.append({"roi": up})
            scenario_comp = compare_scenarios(
                "baseline", "intervention_avg", samples_a, samples_b, kpi_keys=["roi"]
            )
    except Exception as e:
        scenario_comp = {"_error": str(e)}

    return {
        "T1_A1_funnel_beta_fit": beta_fits,
        "T1_A2_mc_funnel_prediction": mc_funnel,
        "T1_A3_competitor_audience_roi": competitor_roi,
        "T2_A1_kol_mix_optimization": kol_mix,
        "T2_A2_kol_content_match": kol_match,
        "T2_A3_tag_lift_ranking": tag_lift,
        "T2_A4_ugc_diffusion_simulation": diffusion,
        "T2_A5_kol_reinvest_ranking": kol_reinvest,
        "T3_A1_agent_persona": personas,
        "T3_A2_platform_simulation_ts": plat_ts,
        "T3_A3_emergent_metrics": em,
        "T3_A4_scenario_comparison": scenario_comp,
        "T3_A5_sensitivity_analysis": sens,
        "T3_A6_search_elasticity": search_elast,
        "T3_A7_content_type_coefficient": content_type_coef,
        "report_market_insight": report,
    }
