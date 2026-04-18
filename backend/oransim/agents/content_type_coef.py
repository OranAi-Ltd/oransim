"""T3-A7 content_type_coefficient — per-format engagement coefficients.

For each content format (教程 / 测评 / 种草 / 开箱 / vlog / 攻略), estimate a
multiplicative coefficient on baseline engagement from the synthetic notes
corpus. Coefficient > 1 means that format over-performs the niche average.

Methodology: bucket notes into format by keyword heuristics on text_zh, then
compute mean engagement_rate per format and divide by the niche's overall mean.
"""

from __future__ import annotations

import time
import uuid

from .tag_lift import _load_notes, _resolve_niche  # reuse cached corpus

_FORMAT_KEYWORDS: dict[str, list[str]] = {
    "教程": ["教程", "教你", "步骤", "新手", "入门", "指南"],
    "测评": ["测评", "对比", "实测", "评测", "横评"],
    "种草": ["种草", "安利", "推荐", "必入", "真香"],
    "开箱": ["开箱", "到手", "拆箱", "购入"],
    "vlog": ["vlog", "日常", "记录", "生活"],
    "攻略": ["攻略", "避坑", "清单", "合集"],
}


def _classify(text: str) -> str:
    if not text:
        return "其他"
    t = text.lower()
    for fmt, kws in _FORMAT_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in t:
                return fmt
    return "其他"


def compute_content_type_coefficients(target_niche: str | None = None) -> dict:
    """Schema T3-A7 content_type_coefficient payload."""
    notes = _load_notes()
    if not notes:
        return {"_error": "synthetic notes corpus missing", "rows": []}
    if target_niche:
        resolved = _resolve_niche(target_niche)
        notes = [n for n in notes if n.get("niche") == resolved]
    if not notes:
        return {
            "_error": f"niche '{target_niche}' has no notes",
            "rows": [],
        }

    by_format: dict[str, list[float]] = {}
    for n in notes:
        rate = float(n.get("metrics", {}).get("engagement_rate") or 0)
        fmt = _classify(n.get("text_zh") or n.get("text_en") or "")
        by_format.setdefault(fmt, []).append(rate)

    all_rates = [r for rates in by_format.values() for r in rates]
    baseline = sum(all_rates) / max(1, len(all_rates))

    rows = []
    for fmt, rates in by_format.items():
        if not rates:
            continue
        mean_rate = sum(rates) / len(rates)
        coef = mean_rate / baseline if baseline > 0 else 1.0
        rows.append(
            {
                "content_type": fmt,
                "coefficient": round(coef, 3),
                "mean_engagement_rate": round(mean_rate, 4),
                "baseline_rate": round(baseline, 4),
                "n_samples": len(rates),
                "recommendation": (
                    "优先投放" if coef >= 1.2 else "可选" if coef >= 0.9 else "不推荐"
                ),
            }
        )
    rows.sort(key=lambda r: -r["coefficient"])

    return {
        "run_id": f"ctc_{uuid.uuid4().hex[:8]}",
        "target_niche": target_niche or "all",
        "baseline_engagement_rate": round(baseline, 4),
        "n_notes_total": len(notes),
        "n_formats": len(rows),
        "rows": rows,
        "data_source": f"synthetic notes ({len(notes)} notes)",
        "note": "Synthetic corpus — coefficients calibrated against generative priors.",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
