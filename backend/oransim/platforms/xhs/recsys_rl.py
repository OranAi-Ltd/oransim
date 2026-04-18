"""C. Platform RecSys RL loop.

Normal impression run: one-shot score → top-K.
RL loop: multi-round iterative — platform "learns" from early engagement signals.

  Round 1: initial score weights → serve to 10% of budget → observe CTR per
           score dimension → update weights for over-performing dims
  Round 2: use updated weights → serve next 20% → observe → update
  ...
  Round N: converged weights reflect "what this content actually attracts"

This simulates the real-world ByteDance/小红书 冷启 → 破圈 dynamic and predicts
peak day + break-out likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data.creatives import Creative
from ...data.kols import KOL
from ...data.platforms import budget_to_impressions
from .world_model_legacy import ImpressionResult, PlatformWorldModel


@dataclass
class RLRoundResult:
    round_id: int
    weights: np.ndarray  # 5 dims: content, platform, audience, kol, noise
    impression_idx: np.ndarray
    click_rate: float
    engage_rate: float
    breakout: bool  # did it trigger next-pool release


@dataclass
class RecSysRLReport:
    rounds: list[RLRoundResult]
    cumulative_impression: ImpressionResult
    peak_round: int
    break_out: bool
    final_weights: np.ndarray


class RecSysRLSimulator:
    """Multi-round impression with weight adaptation."""

    N_DIMS = 5  # content, platform_activity, audience, kol, noise/exploration

    def __init__(self, world_model: PlatformWorldModel):
        self.wm = world_model
        self.pop = world_model.pop

    def _score(
        self,
        creative: Creative,
        platform: str,
        weights: np.ndarray,
        aud_filter=None,
        kol: KOL | None = None,
        rng_seed: int = 0,
    ) -> np.ndarray:
        rng = np.random.default_rng(rng_seed)
        plat_i = self.wm.platform_idx.get(platform, 0)
        content = (self.pop.interest @ creative.content_emb + 1) / 2
        plat = self.pop.platform_activity[:, plat_i]
        aud = self.wm._audience_score(aud_filter)
        if kol is not None:
            kol_s = 1.0 + 0.5 * np.clip(self.pop.interest @ kol.emb, 0, 1)
        else:
            kol_s = np.ones_like(content)
        noise = rng.uniform(0.6, 1.4, self.pop.N).astype(np.float32)
        feats = np.stack([content, plat, aud, kol_s, noise], axis=1).astype(np.float32)
        return feats @ weights

    def simulate(
        self,
        creative: Creative,
        platform: str,
        total_budget: float,
        audience_filter=None,
        kol: KOL | None = None,
        n_rounds: int = 5,
        lr: float = 0.25,
        breakout_threshold: float = 0.035,  # click-rate needed to "break out"
        seed: int = 0,
    ) -> RecSysRLReport:
        rng = np.random.default_rng(seed)
        # initial weights: platform default
        w = np.array([1.0, 0.9, 1.0, 0.5, 0.3], dtype=np.float32)
        round_fractions = np.array([0.1, 0.15, 0.2, 0.25, 0.30], dtype=np.float32)
        round_fractions = round_fractions[:n_rounds] / round_fractions[:n_rounds].sum()

        total_imps = budget_to_impressions(total_budget, platform)
        served_mask = np.zeros(self.pop.N, dtype=bool)
        all_idx: list[int] = []
        all_w: list[float] = []
        rounds: list[RLRoundResult] = []
        broke = False

        for r in range(n_rounds):
            round_imps = int(total_imps * round_fractions[r])
            # score everyone
            score = self._score(creative, platform, w, audience_filter, kol, rng_seed=seed + r * 17)
            # mask already-served (prevent infinite loops)
            score = np.where(served_mask, -1e9, score)
            if round_imps <= 0 or (~served_mask).sum() < round_imps:
                round_imps = min(round_imps, int((~served_mask).sum()))
                if round_imps <= 0:
                    break
            idx = np.argpartition(-score, round_imps - 1)[:round_imps]
            served_mask[idx] = True
            all_idx.extend(idx.tolist())

            # observe clicks: use content-affinity proxy (would be stat agent in prod)
            content = (self.pop.interest[idx] @ creative.content_emb + 1) / 2
            plat_i = self.wm.platform_idx.get(platform, 0)
            plat_act = self.pop.platform_activity[idx, plat_i]
            # proxy engagement
            eng_prob = content * plat_act * 0.4
            click_rate = float(eng_prob.mean())
            engage_rate = click_rate * 0.7

            rounds.append(
                RLRoundResult(
                    round_id=r,
                    weights=w.copy(),
                    impression_idx=idx.astype(np.int64),
                    click_rate=click_rate,
                    engage_rate=engage_rate,
                    breakout=(click_rate > breakout_threshold),
                )
            )
            all_w.extend([float(eng_prob[i]) for i in range(len(idx))])

            if click_rate > breakout_threshold:
                broke = True

            # UPDATE weights:  dim that correlates with engagement gets boosted
            feats_idx = np.stack(
                [
                    (self.pop.interest[idx] @ creative.content_emb + 1) / 2,
                    self.pop.platform_activity[idx, plat_i],
                    np.ones(len(idx), dtype=np.float32),
                    (
                        np.ones(len(idx), dtype=np.float32)
                        if kol is None
                        else 1.0 + 0.5 * np.clip(self.pop.interest[idx] @ kol.emb, 0, 1)
                    ),
                    rng.uniform(0.6, 1.4, len(idx)).astype(np.float32),
                ],
                axis=1,
            )
            # correlation of each dim with eng_prob → gradient
            eng_c = eng_prob - eng_prob.mean()
            feat_c = feats_idx - feats_idx.mean(axis=0, keepdims=True)
            grad = (feat_c * eng_c[:, None]).mean(axis=0)
            w = w + lr * grad * 5  # scale
            w = np.clip(w, 0.05, 3.0)

        # peak round = max click_rate
        peak_round = max(range(len(rounds)), key=lambda i: rounds[i].click_rate) if rounds else 0

        # cumulative impression
        if all_idx:
            idx_arr = np.array(all_idx, dtype=np.int64)
            w_arr = np.array(all_w, dtype=np.float32)
            w_arr = w_arr / (w_arr.max() + 1e-8)
            cum_imp = ImpressionResult(
                agent_idx=idx_arr,
                weight=w_arr,
                total_impressions=float(total_imps),
                platform=platform,
                score_breakdown={
                    "content": (self.pop.interest[idx_arr] @ creative.content_emb + 1) / 2,
                    "platform_activity": self.pop.platform_activity[
                        idx_arr, self.wm.platform_idx.get(platform, 0)
                    ],
                    "audience_filter": np.ones(len(idx_arr), dtype=np.float32),
                    "kol_boost": (
                        np.ones(len(idx_arr), dtype=np.float32)
                        if kol is None
                        else 1.0 + 0.5 * np.clip(self.pop.interest[idx_arr] @ kol.emb, 0, 1)
                    ),
                },
            )
        else:
            cum_imp = ImpressionResult(
                agent_idx=np.array([], dtype=np.int64),
                weight=np.array([], dtype=np.float32),
                total_impressions=0.0,
                platform=platform,
                score_breakdown={
                    "content": np.array([]),
                    "platform_activity": np.array([]),
                    "audience_filter": np.array([]),
                    "kol_boost": np.array([]),
                },
            )

        return RecSysRLReport(
            rounds=rounds,
            cumulative_impression=cum_imp,
            peak_round=peak_round,
            break_out=broke,
            final_weights=w,
        )


def rl_report_to_dict(rep: RecSysRLReport) -> dict:
    return {
        "n_rounds": len(rep.rounds),
        "break_out": rep.break_out,
        "peak_round": rep.peak_round,
        "final_weights": {
            k: round(float(v), 3)
            for k, v in zip(
                ["content", "platform_activity", "audience", "kol", "exploration"],
                rep.final_weights,
                strict=False,
            )
        },
        "per_round": [
            {
                "round": r.round_id,
                "click_rate": round(r.click_rate, 4),
                "engage_rate": round(r.engage_rate, 4),
                "broke_out": r.breakout,
                "reach": int(len(r.impression_idx)),
                "weights": [round(float(x), 3) for x in r.weights.tolist()],
            }
            for r in rep.rounds
        ],
    }
