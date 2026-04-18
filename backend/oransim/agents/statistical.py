"""L2a statistical agents — vectorized decision model.

Given an ImpressionResult from the world model, compute:
  click_prob, engage_prob, convert_prob  per shown agent.

All numpy — 100k agents < 50ms per call.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..data.population import Population
from ..data.creatives import Creative
from ..data.kols import KOL
from ..world_model.model import ImpressionResult


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class OutcomeBatch:
    agent_idx: np.ndarray
    click: np.ndarray          # (K,) binary
    engage: np.ndarray         # (K,) binary
    convert: np.ndarray        # (K,) binary
    click_prob: np.ndarray     # (K,) float
    engage_prob: np.ndarray
    convert_prob: np.ndarray
    # Latent noise used (abduction-ready)
    u_noise: np.ndarray        # (K,) standard normal noise driving decisions


class StatisticalAgents:
    """Deterministic vectorized decision function, fully differentiable-compatible."""

    # learned weights (hand-initialized; would be fit on real data)
    W_CLICK = np.array([1.8, 1.2, 0.9, 0.4, -0.7, 0.25, 0.3], dtype=np.float32)
    # order: content_score, platform_activity, audience_filter, kol_boost,
    #        fatigue_proxy, bigfive_openness, celeb_flag
    W_ENGAGE = np.array([1.4, 0.9, 0.6, 0.5, -0.3, 0.4], dtype=np.float32)
    W_CONVERT = np.array([1.1, 0.6, 0.7, 0.8, 0.35], dtype=np.float32)
    # order: click_prob, purchase_intent, kol_trust, audience_match, price_sensitivity

    def __init__(self, population: Population):
        self.pop = population

    def simulate(
        self,
        impression: ImpressionResult,
        creative: Creative,
        kol: Optional[KOL] = None,
        fixed_noise: Optional[np.ndarray] = None,
        rng_seed: int = 1,
        macro_ctr_lift: float = 1.0,
        macro_cvr_lift: float = 1.0,
    ) -> OutcomeBatch:
        idx = impression.agent_idx
        K = len(idx)
        if K == 0:
            z = np.zeros(0, dtype=np.float32)
            return OutcomeBatch(idx, z.astype(bool), z.astype(bool), z.astype(bool),
                                z, z, z, z)

        rng = np.random.default_rng(rng_seed)

        content = impression.score_breakdown["content"]
        plat    = impression.score_breakdown["platform_activity"]
        aud     = impression.score_breakdown["audience_filter"]
        kol_b   = impression.score_breakdown["kol_boost"]

        bigfive = self.pop.bigfive[idx]       # (K, 5)
        openness = bigfive[:, 0]              # 0..1
        neurotic = bigfive[:, 4]

        # fatigue proxy: how similar content is to user's current state drift
        state = self.pop.state[idx]           # (K, 16)
        fatigue = np.clip(np.abs(state[:, 0]), 0, 1)

        celeb = np.full(K, 1.0 if creative.has_celeb else 0.0, dtype=np.float32)

        click_feat = np.stack([
            content, plat, aud, kol_b, fatigue, openness, celeb
        ], axis=1)
        click_logit = click_feat @ self.W_CLICK - 1.2

        # Compliance / AIGC penalty: audit_risk throttles distribution; aigc shows turn off some users
        click_logit -= 1.5 * float(creative.audit_risk)
        # AIGC hits older + lower-tier users harder
        aigc_pen = float(creative.aigc_score) * (
            0.4 + 0.5 * (self.pop.age_idx[idx] / 5.0).astype(np.float32)
        )
        click_logit -= aigc_pen

        # Macro multiplier (holiday/season/dayparting/weather etc.)
        click_logit += np.log(max(macro_ctr_lift, 1e-3))

        # noise: u_i controls this agent's personal response (abduction handle)
        if fixed_noise is not None and len(fixed_noise) == K:
            u = fixed_noise
        else:
            u = rng.normal(0, 1, K).astype(np.float32)
        click_logit = click_logit + 0.7 * u

        click_prob = _sigmoid(click_logit)

        # engage = like/save/comment conditional on click
        engage_feat = np.stack([
            content, plat, aud, kol_b, neurotic, openness
        ], axis=1)
        engage_logit = engage_feat @ self.W_ENGAGE - 1.5
        engage_prob = click_prob * _sigmoid(engage_logit + 0.5 * u)

        # convert: purchase intent
        purchase_intent = 0.5 * (1 - neurotic) + 0.3 * openness
        income = self.pop.income[idx] / 9.0  # 0..1
        price_sens = 1.0 - income
        kol_trust = kol_b  # treat as proxy
        audience_match = aud
        convert_feat = np.stack([
            click_prob, purchase_intent, kol_trust, audience_match, price_sens * 0.3
        ], axis=1)
        convert_logit = convert_feat @ self.W_CONVERT - 4.2 + np.log(max(macro_cvr_lift, 1e-3))
        convert_prob = click_prob * _sigmoid(convert_logit + 0.3 * u)

        # sample outcomes
        r = rng.uniform(0, 1, (3, K)).astype(np.float32)
        click = (r[0] < click_prob)
        engage = click & (r[1] < _sigmoid(engage_logit))
        convert = click & (r[2] < _sigmoid(convert_logit))

        return OutcomeBatch(
            agent_idx=idx,
            click=click, engage=engage, convert=convert,
            click_prob=click_prob, engage_prob=engage_prob, convert_prob=convert_prob,
            u_noise=u,
        )

    @staticmethod
    def aggregate_kpis(outcome: OutcomeBatch, impression: ImpressionResult,
                       budget: float, conv_value_cny: float = 45.0) -> Dict[str, float]:
        """Roll per-agent outcomes into campaign KPIs."""
        if len(outcome.agent_idx) == 0:
            return {"impressions": 0, "clicks": 0, "conversions": 0,
                    "ctr": 0.0, "cvr": 0.0, "roi": 0.0, "cost": budget}
        imps = impression.total_impressions
        # "realized" clicks: sum of click_prob weighted by impression weight
        clicks = float(np.sum(outcome.click_prob * impression.weight))
        conversions = float(np.sum(outcome.convert_prob * impression.weight))
        ctr = clicks / max(imps, 1)
        cvr = conversions / max(clicks, 1)
        revenue = conversions * conv_value_cny
        roi = (revenue - budget) / max(budget, 1)
        return {
            "impressions": float(imps),
            "clicks": clicks,
            "conversions": conversions,
            "ctr": ctr, "cvr": cvr, "roi": roi,
            "revenue": revenue, "cost": budget,
        }
