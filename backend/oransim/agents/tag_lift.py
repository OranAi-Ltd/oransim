"""T2-A3 tag_lift_ranking — tag/topic prevalence Lift per niche.

Uses the synthetic notes corpus shipped in ``data/synthetic/notes_v3.json``:
each note has a ``niche`` (Chinese label) and ``text_zh`` we tokenize.

Lift(tag, niche) = P(tag | niche) / P(tag | all).  Lift > 1 means the tag is
over-represented in this niche vs the full corpus; high Lift often correlates
with higher engagement.

When real-panel data is swapped in, the same algorithm runs unchanged — just
point ``_NOTES_PATH`` at the new jsonl. Enterprise Edition ships with a
continuously-updated real-panel index.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter

_NOTES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "synthetic", "notes_v3.json"
)

_NOTES_CACHE: list[dict] | None = None
_TAG_INDEX_CACHE: dict | None = None

_STOP = set(
    "的了是和与及或而但虽然然后所以因为如果就要这那一个我你他她它我们你们他们"
    "她们什么怎么哪个为什么很非常超级特别好看真的还行可以应该不会有没有"
    "月关于系列学到"
)

# Chinese niche label → English key used in synthetic notes corpus.
_ZH_TO_EN = {
    "美妆": "beauty",
    "穿搭": "fashion",
    "美食": "food",
    "数码": "electronics",
    "旅行": "travel",
    "母婴": "parenting",
    "健身": "fitness",
    "理财": "finance",
    "宠物": "pet",
    "家居": "home",
    "饮品": "beverage",
}


def _resolve_niche(n: str | None) -> str | None:
    """Accept either English key or Chinese label — normalize to the key used
    in notes_v3.json (English)."""
    if not n:
        return None
    return _ZH_TO_EN.get(n, n)


def _ngrams(s: str, n: int) -> list[str]:
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _all_tags(text: str) -> list[str]:
    """Extract candidate tags: explicit #tag# plus 2-3 char Chinese n-grams."""
    if not text:
        return []
    tags: list[str] = []
    for m in re.finditer(r"#([^\s#]{2,20})#", text):
        tags.append(m.group(1))
    text_cn = re.sub(r"[^\u4e00-\u9fff]+", "", text)
    for n in (2, 3):
        for g in _ngrams(text_cn, n):
            if g in _STOP:
                continue
            if n == 2 and any(c in _STOP for c in g):
                continue
            tags.append(g)
    return tags


def _load_notes() -> list[dict]:
    global _NOTES_CACHE
    if _NOTES_CACHE is None:
        try:
            with open(os.path.abspath(_NOTES_PATH), encoding="utf-8") as f:
                _NOTES_CACHE = json.load(f)
        except Exception:
            _NOTES_CACHE = []
    return _NOTES_CACHE


def _build_tag_index() -> dict:
    """Pure function of the notes corpus; cache so tokenization runs once."""
    global _TAG_INDEX_CACHE
    if _TAG_INDEX_CACHE is not None:
        return _TAG_INDEX_CACHE
    notes = _load_notes()
    all_tags_global: Counter = Counter()
    tags_by_niche: dict[str, Counter] = {}
    notes_by_niche: dict[str, list[dict]] = {}
    n_notes_by_niche: dict[str, int] = {}
    for note in notes:
        niche = note.get("niche") or "other"
        notes_by_niche.setdefault(niche, []).append(note)
        n_notes_by_niche[niche] = n_notes_by_niche.get(niche, 0) + 1
        text = note.get("text_zh") or note.get("text_en") or ""
        for t in set(_all_tags(text)):
            all_tags_global[t] += 1
            tags_by_niche.setdefault(niche, Counter())[t] += 1
    _TAG_INDEX_CACHE = {
        "notes_by_niche": notes_by_niche,
        "all_tags_global": all_tags_global,
        "tags_by_niche": tags_by_niche,
        "n_notes_global": len(notes),
        "n_notes_by_niche": n_notes_by_niche,
    }
    return _TAG_INDEX_CACHE


def compute_tag_lift(
    target_niche: str | None = None,
    min_support: int = 8,
    top_k: int = 20,
) -> dict:
    """Schema T2-A3 tag_lift_ranking payload."""
    idx = _build_tag_index()
    notes_by_niche = idx["notes_by_niche"]
    if not notes_by_niche:
        return {
            "_error": "synthetic notes corpus missing",
            "rows": [],
            "platform": "xhs",
        }
    niches = list(notes_by_niche.keys())
    target_niche = _resolve_niche(target_niche) or niches[0]
    if target_niche not in idx["tags_by_niche"]:
        return {
            "_error": f"niche '{target_niche}' not in pool",
            "rows": [],
            "available_niches": niches,
        }

    all_tags = idx["all_tags_global"]
    niche_tags = idx["tags_by_niche"][target_niche]
    n_global = idx["n_notes_global"]
    n_niche = idx["n_notes_by_niche"][target_niche]

    rows = []
    for tag, count_in_niche in niche_tags.most_common(top_k * 4):
        if count_in_niche < min_support:
            continue
        p_base = all_tags[tag] / max(1, n_global)
        p_niche = count_in_niche / max(1, n_niche)
        if p_base <= 0:
            continue
        rows.append(
            {
                "tag": tag,
                "lift": round(p_niche / p_base, 3),
                "count_in_niche": count_in_niche,
                "count_global": all_tags[tag],
                "p_niche": round(p_niche, 4),
                "p_global": round(p_base, 4),
                "support": count_in_niche,
                "confidence": round(p_niche, 4),
            }
        )
    rows.sort(key=lambda r: -r["lift"])
    rows = rows[:top_k]

    return {
        "run_id": f"tl_{uuid.uuid4().hex[:8]}",
        "target_niche": target_niche,
        "platform": "xhs",
        "min_support": min_support,
        "top_k": top_k,
        "n_notes_global": n_global,
        "n_notes_in_niche": n_niche,
        "n_unique_tags": len(niche_tags),
        "rows": rows,
        "data_source": f"synthetic notes_v3 ({n_global} notes, {len(niches)} niches)",
        "note": "Synthetic corpus — Enterprise Edition swaps in a real-panel index.",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
