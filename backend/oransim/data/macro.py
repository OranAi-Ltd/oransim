"""Macro environment factors: holidays, season, dayparting, weather.

All return small multipliers (typically 0.7 - 1.4) to nudge user_state and
impression dynamics. Realistic-but-stylized.
"""
from __future__ import annotations
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Dict, Optional


# --- Holiday calendar (CN + Global) — date → (label, lift) ---
# lift = consumption boost multiplier (1.0 = neutral)
_HOLIDAYS = {
    # China
    (1, 1):  ("元旦",        1.10),
    (2, 14): ("情人节",      1.20),
    (3, 8):  ("妇女节",      1.15),
    (4, 5):  ("清明",        0.95),
    (5, 1):  ("劳动节",      1.10),
    (5, 20): ("520",         1.30),
    (6, 1):  ("儿童节",      1.10),
    (6, 18): ("618 大促",     1.55),
    (8, 7):  ("七夕",        1.25),
    (9, 10): ("教师节",      1.05),
    (10, 1): ("国庆",        1.20),
    (11, 11):("双十一",       1.80),  # peak
    (12, 12):("双十二",       1.40),
    (12, 25):("圣诞",        1.20),
    # Global
    (11, 26):("黑五",        1.50),
    (5, 12): ("母亲节(近似)", 1.20),  # 2nd Sun of May (approx)
    (6, 16): ("父亲节(近似)", 1.10),
}

# Pre-/post-holiday halo: 3 days before, 1 day after
def holiday_factor(d: date) -> Dict:
    best = ("平日", 1.0, 0)   # label, lift, days_to
    for offset in range(-3, 2):
        d2 = d + timedelta(days=offset)
        key = (d2.month, d2.day)
        if key in _HOLIDAYS:
            label, lift = _HOLIDAYS[key]
            # decay halo
            decay = 1.0 - 0.15 * abs(offset)
            effective = 1.0 + (lift - 1.0) * max(decay, 0.3)
            if effective > best[1]:
                best = (label, effective, offset)
    return {"label": best[0], "lift": round(best[1], 3), "days_to": best[2]}


# --- Season (Northern Hemisphere) ---
def season_factor(d: date, category_hint: str = "general") -> float:
    m = d.month
    if category_hint in ("beverage", "drink", "ice"):
        return [0.7, 0.7, 0.85, 1.0, 1.2, 1.4, 1.5, 1.45, 1.2, 0.95, 0.8, 0.75][m-1]
    if category_hint in ("apparel_warm", "down"):
        return [1.4, 1.3, 1.0, 0.7, 0.5, 0.4, 0.4, 0.45, 0.7, 1.0, 1.3, 1.5][m-1]
    if category_hint in ("travel"):
        return [0.85, 1.2, 1.0, 1.1, 1.2, 1.05, 1.4, 1.35, 1.1, 1.3, 0.9, 1.15][m-1]
    return 1.0


# --- Day-of-week ---
_DOW_LIFT = [1.0, 1.0, 1.05, 1.05, 1.15, 1.30, 1.20]   # Mon..Sun
def dow_factor(d: date) -> float:
    return _DOW_LIFT[d.weekday()]


# --- Day-parting (which feed window the campaign is dripped in) ---
# Each window: (label, ctr_mult, cvr_mult, share_of_day)
DAYPART = {
    "morning":   ("06-10 早通勤",   0.90, 0.85, 0.10),
    "noon":      ("11-13 午休",     1.10, 1.00, 0.18),
    "afternoon": ("14-17 摸鱼",     1.05, 1.05, 0.15),
    "evening":   ("18-22 黄金档",   1.30, 1.20, 0.40),
    "late":      ("22-02 睡前",     1.20, 1.10, 0.17),
}

def daypart_factor(window: str) -> Dict:
    if window == "auto":   # weighted average across day
        ctr = sum(v[1] * v[3] for v in DAYPART.values())
        cvr = sum(v[2] * v[3] for v in DAYPART.values())
        return {"label": "全天", "ctr_mult": round(ctr, 3), "cvr_mult": round(cvr, 3)}
    if window not in DAYPART:
        return {"label": "全天", "ctr_mult": 1.0, "cvr_mult": 1.0}
    label, c, v, _ = DAYPART[window]
    return {"label": label, "ctr_mult": c, "cvr_mult": v}


# --- Weather (very rough) ---
def weather_factor(temp_c: float, rainy: bool = False, category: str = "general") -> float:
    if category in ("beverage", "drink", "ice"):
        return 1.0 + 0.02 * max(0, temp_c - 22)
    if category == "delivery":
        return 1.2 if rainy else 1.0
    return 1.0


# --- Public sentiment / outbreak risk (mock — could be wired to news) ---
def sentiment_factor(label: str = "neutral") -> float:
    return {"crisis": 0.7, "negative": 0.85, "neutral": 1.0,
            "positive": 1.1, "viral": 1.25}.get(label, 1.0)


@dataclass
class MacroContext:
    today: date
    category: str = "general"
    daypart: str = "auto"          # "morning"/"noon"/"afternoon"/"evening"/"late"/"auto"
    weather_temp_c: float = 20.0
    rainy: bool = False
    sentiment: str = "neutral"

    def summary(self) -> Dict:
        h = holiday_factor(self.today)
        s = season_factor(self.today, self.category)
        dow = dow_factor(self.today)
        dp = daypart_factor(self.daypart)
        w = weather_factor(self.weather_temp_c, self.rainy, self.category)
        sent = sentiment_factor(self.sentiment)
        # combined macro multipliers
        macro_lift = h["lift"] * s * dow * w * sent
        return {
            "today": self.today.isoformat(),
            "holiday": h, "season_mult": round(s, 3),
            "dow_mult": round(dow, 3),
            "daypart": dp,
            "weather_mult": round(w, 3),
            "sentiment_mult": round(sent, 3),
            "ctr_macro_lift": round(macro_lift * dp["ctr_mult"], 3),
            "cvr_macro_lift": round(macro_lift * dp["cvr_mult"], 3),
        }
