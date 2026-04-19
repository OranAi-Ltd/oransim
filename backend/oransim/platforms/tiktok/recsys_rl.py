"""TikTok FYP cold-start → breakout simulator.

Mirror of :mod:`oransim.platforms.xhs.recsys_rl` calibrated to FYP dynamics.

Key calibration differences from XHS
-----------------------------------

  - **Exponential escalation of reach pool.** TikTok's FYP classically
    tests content on a small seed cohort (~200–1000 impressions), gates
    the next tier on CTR/completion, and exponentially widens the pool
    on each breakout. We model this with 6 rounds whose budget fractions
    grow geometrically (1%, 3%, 8%, 18%, 30%, 40%).
  - **Lower breakout threshold.** FYP uses retention + CTR blend; we
    approximate "breakout" as CTR > 2.8% (XHS uses 3.5%) so cold-start
    exits faster.
  - **Wider exploration noise.** The per-agent exploration draw uses
    [0.5, 1.5] instead of [0.6, 1.4], reflecting FYP's higher variance.
  - **Retention-weighted reward.** The per-round engagement proxy
    multiplies content × platform_activity × duration_retention so that
    short-enough videos benefit even in cold-start.

The output shape matches ``RecSysRLReport`` from the XHS module so
API-level consumers (``rl_report_to_dict``) can stay platform-agnostic.
"""

from __future__ import annotations

import numpy as np

from ...data.creatives import Creative
from ...data.kols import KOL
from ...data.platforms import budget_to_impressions
from ..xhs.recsys_rl import RecSysRLReport, RecSysRLSimulator, RLRoundResult
from .world_model_legacy import ImpressionResult, TikTokWorldModel


class TikTokRecSysRLSimulator(RecSysRLSimulator):
    """FYP-calibrated breakout simulator."""

    # 6 geometric fractions; sum = 1.0 after normalization inside simulate().
    _fyp_fractions = np.array([0.01, 0.03, 0.08, 0.18, 0.30, 0.40], dtype=np.float32)

    def __init__(self, world_model: TikTokWorldModel):
        super().__init__(world_model)
        # self.wm is now TikTokWorldModel (inherits PlatformWorldModel).

    def simulate(
        self,
        creative: Creative,
        platform: str = "tiktok",
        total_budget: float = 0.0,
        audience_filter=None,
        kol: KOL | None = None,
        n_rounds: int = 6,
        lr: float = 0.3,
        breakout_threshold: float = 0.028,
        seed: int = 0,
    ) -> RecSysRLReport:
        rng = np.random.default_rng(seed)
        # Initial FYP weights — content and exploration noise weighted
        # higher than explicit audience targeting.
        w = np.array([1.0, 0.85, 0.55, 0.45, 0.6], dtype=np.float32)
        fractions = self._fyp_fractions[:n_rounds]
        fractions = fractions / fractions.sum()

        # Route "tiktok" impressions through douyin's CPM (same-stack).
        total_imps = budget_to_impressions(total_budget, "tiktok")
        served_mask = np.zeros(self.pop.N, dtype=bool)
        all_idx: list[int] = []
        all_w: list[float] = []
        rounds: list[RLRoundResult] = []
        broke = False

        duration_retention = self.wm._duration_retention(creative)  # type: ignore[attr-defined]

        for r in range(n_rounds):
            round_imps = int(total_imps * fractions[r])
            # Score with current weights. Reuse parent's _score and then
            # apply duration retention on top.
            score = self._score(
                creative,
                "douyin",  # underlying activity column
                w,
                audience_filter,
                kol,
                rng_seed=seed + r * 17,
            )
            score = score * duration_retention
            # Extra FYP exploration noise draw per round.
            score = score * rng.uniform(0.5, 1.5, self.pop.N).astype(np.float32)
            score = np.where(served_mask, -1e9, score)
            if round_imps <= 0 or (~served_mask).sum() < round_imps:
                round_imps = min(round_imps, int((~served_mask).sum()))
                if round_imps <= 0:
                    break
            idx = np.argpartition(-score, round_imps - 1)[:round_imps]
            served_mask[idx] = True
            all_idx.extend(idx.tolist())

            plat_i = self.wm.platform_idx.get("douyin", 0)
            content = (self.pop.interest[idx] @ creative.content_emb + 1) / 2
            plat_act = self.pop.platform_activity[idx, plat_i]
            # Retention-weighted engagement proxy — captures FYP's
            # completion-rate signal rather than click alone.
            eng_prob = content * plat_act * duration_retention * 0.45
            click_rate = float(eng_prob.mean())
            engage_rate = click_rate * 0.75

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

            # Weight update: correlation of each score dim with engagement.
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
                    rng.uniform(0.5, 1.5, len(idx)).astype(np.float32),
                ],
                axis=1,
            )
            eng_c = eng_prob - eng_prob.mean()
            feat_c = feats_idx - feats_idx.mean(axis=0, keepdims=True)
            grad = (feat_c * eng_c[:, None]).mean(axis=0)
            w = w + lr * grad * 5
            w = np.clip(w, 0.05, 3.0)

        peak_round = max(range(len(rounds)), key=lambda i: rounds[i].click_rate) if rounds else 0

        if all_idx:
            idx_arr = np.array(all_idx, dtype=np.int64)
            w_arr = np.array(all_w, dtype=np.float32)
            w_arr = w_arr / (w_arr.max() + 1e-8)
            plat_i = self.wm.platform_idx.get("douyin", 0)
            cum_imp = ImpressionResult(
                agent_idx=idx_arr,
                weight=w_arr,
                total_impressions=float(total_imps),
                platform="tiktok",
                score_breakdown={
                    "content": (self.pop.interest[idx_arr] @ creative.content_emb + 1) / 2,
                    "platform_activity": self.pop.platform_activity[idx_arr, plat_i],
                    "audience_filter": np.ones(len(idx_arr), dtype=np.float32),
                    "kol_boost": (
                        np.ones(len(idx_arr), dtype=np.float32)
                        if kol is None
                        else 1.0 + 0.5 * np.clip(self.pop.interest[idx_arr] @ kol.emb, 0, 1)
                    ),
                    "duration_retention": np.full(
                        len(idx_arr), duration_retention, dtype=np.float32
                    ),
                },
            )
        else:
            cum_imp = ImpressionResult(
                agent_idx=np.array([], dtype=np.int64),
                weight=np.array([], dtype=np.float32),
                total_impressions=0.0,
                platform="tiktok",
                score_breakdown={
                    "content": np.array([]),
                    "platform_activity": np.array([]),
                    "audience_filter": np.array([]),
                    "kol_boost": np.array([]),
                    "duration_retention": np.array([]),
                },
            )

        return RecSysRLReport(
            rounds=rounds,
            cumulative_impression=cum_imp,
            peak_round=peak_round,
            break_out=broke,
            final_weights=w,
        )
