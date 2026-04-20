"""Pluggable Agent Provider interface.

Abstracts over different agent backends so API can switch modes:
  - "voronoi"   : our default — 100k stat + 100 LLM + Voronoi sampling calibration
  - "oasis"     : pure multi-agent LLM (OASIS-compatible), each agent = 1 LLM call
  - "hybrid"    : stat backbone + OASIS discourse overlay

Why: lets us benchmark Voronoi vs OASIS head-to-head, give enterprise clients
a "high-fidelity mode" without rewriting pipelines.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..data.creatives import Creative
from ..data.kols import KOL
from ..platforms.xhs.world_model_legacy import ImpressionResult


@dataclass
class ProviderResult:
    """Unified output format from any agent provider."""

    provider: str  # "voronoi" / "oasis" / ...
    n_agents_total: int  # theoretical scale
    n_agents_active: int  # actually computed this run
    n_llm_calls: int  # real LLM calls made
    cost_cny: float
    latency_s: float
    click_prob_mean: float
    click_prob_std: float
    ctr: float
    cvr: float
    per_agent_samples: list[dict] = field(default_factory=list)  # small sample for viz
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "n_agents_total": self.n_agents_total,
            "n_agents_active": self.n_agents_active,
            "n_llm_calls": self.n_llm_calls,
            "cost_cny": round(self.cost_cny, 4),
            "latency_s": round(self.latency_s, 2),
            "click_prob_mean": round(self.click_prob_mean, 4),
            "click_prob_std": round(self.click_prob_std, 4),
            "ctr": round(self.ctr, 4),
            "cvr": round(self.cvr, 4),
            "per_agent_samples": self.per_agent_samples[:20],
            "notes": self.notes,
        }


class AgentProvider(ABC):
    """Abstract agent provider — compute CTR/CVR for a scenario."""

    name: str = "abstract"

    @abstractmethod
    def simulate(
        self,
        impression: ImpressionResult,
        creative: Creative,
        kol: KOL | None,
        platform: str,
        **kwargs,
    ) -> ProviderResult: ...


# ---------------- OASIS-style provider (pure multi-agent LLM) ----------------


class OASISProvider(AgentProvider):
    """Pure multi-agent LLM simulation — each activated agent calls LLM
    independently (NO Voronoi representative sampling).

    This mimics OASIS / CAMEL-AI's scaling strategy:
      - N total agents
      - Each step activates p% (activation_probability)
      - Each active agent → 1 LLM call

    Tries to import real OASIS if available; otherwise runs the same algorithm
    with our LLM gateway (honest approximation).
    """

    name = "oasis"

    def __init__(self, soul_pool, population, n_total: int = 10_000, activation_prob: float = 0.05):
        self.souls = soul_pool
        self.pop = population
        self.n_total = n_total
        self.activation_prob = activation_prob
        self._oasis_available = self._check_oasis()

    def _check_oasis(self) -> bool:
        try:
            import oasis  # noqa

            return True
        except ImportError:
            return False

    def simulate(self, impression, creative, kol, platform, **kwargs) -> ProviderResult:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from .soul_llm import estimate_cost_cny, llm_available, soul_infer_llm

        t0 = time.time()

        # activation_probability × N_total = how many LLM calls we should make
        n_active = max(5, int(self.n_total * self.activation_prob))
        # Cap at 200 for demo reality — production would relax
        n_active_real = min(n_active, 200)

        if not llm_available():
            return self._mock_result(n_active, n_active_real, t0, notes="LLM unavailable — stub")

        # Sample n_active_real personas WITHOUT Voronoi-aware weighting
        # (pure random sample — this is the "OASIS way")
        import random

        rng = random.Random(int(time.time()))
        persona_ids = rng.sample(
            list(self.souls.personas.keys()), min(n_active_real, len(self.souls.personas))
        )

        def call_one(pid):
            p = self.souls.personas[pid]
            r = soul_infer_llm(
                persona=p,
                caption=creative.caption,
                platform=platform,
                kol_name=kol.name if kol else "无",
                kol_niche=kol.niche if kol else "通用",
                kol_fans=kol.fan_count if kol else 0,
                visual=creative.visual_style,
                music=creative.music_mood,
                duration=creative.duration_sec,
            )
            return pid, r

        import os

        workers = int(os.environ.get("LLM_CONCURRENCY", "15"))
        llm_calls = 0
        tok_in = tok_out = 0
        verdicts = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(call_one, pid) for pid in persona_ids]
            for f in as_completed(futs):
                pid, r = f.result()
                llm_calls += 1
                if "_error" not in r:
                    tok_in += r.get("_tokens_in", 0)
                    tok_out += r.get("_tokens_out", 0)
                    verdicts.append(
                        {
                            "pid": pid,
                            "clicked": bool(r.get("will_click", False)),
                            "intent": float(r.get("purchase_intent_7d", 0.1)),
                            "reason": r.get("reason", ""),
                            "persona": self.souls.personas[pid].one_liner(),
                        }
                    )

        if not verdicts:
            return self._mock_result(n_active, n_active_real, t0, notes="all calls failed")

        click_arr = np.array([1.0 if v["clicked"] else 0.0 for v in verdicts], dtype=np.float32)
        ctr = float(click_arr.mean())
        cvr_arr = np.array([v["intent"] for v in verdicts], dtype=np.float32)
        cvr = float((cvr_arr > 0.3).mean())

        cost = estimate_cost_cny(tok_in, tok_out)
        # Scale cost to theoretical full activation (honest accounting)
        theoretical_cost = cost * (n_active / max(len(verdicts), 1))

        return ProviderResult(
            provider="oasis" + (":real" if self._oasis_available else ":shim"),
            n_agents_total=self.n_total,
            n_agents_active=n_active,
            n_llm_calls=llm_calls,
            cost_cny=cost,
            latency_s=time.time() - t0,
            click_prob_mean=ctr,
            click_prob_std=float(click_arr.std()),
            ctr=ctr,
            cvr=cvr,
            per_agent_samples=verdicts,
            notes=(
                f"nominal {n_active} agents active (of {self.n_total}), "
                f"actually computed {len(verdicts)} LLM calls. "
                f"Theoretical full-scale cost ≈ ¥{theoretical_cost:.2f}. "
                f"OASIS library {'available' if self._oasis_available else 'not installed — shim mode'}."
            ),
        )

    def _mock_result(self, n_active, n_real, t0, notes):
        return ProviderResult(
            provider="oasis:stub",
            n_agents_total=self.n_total,
            n_agents_active=n_active,
            n_llm_calls=0,
            cost_cny=0.0,
            latency_s=time.time() - t0,
            click_prob_mean=0.02,
            click_prob_std=0.01,
            ctr=0.02,
            cvr=0.1,
            notes=notes,
        )


# ---------------- Voronoi provider (our default) ----------------


class SLOCProvider(AgentProvider):
    """Sparse LLM Oracle Calibration (SLOC) — Oransim's core IP.

    100 LLM oracles anchor 100k-1M statistical agents via KNN-stratified
    importance weighting. Theoretical backbone: Kish 1965 survey sampling
    + a 1/√N heuristic decay curve (central-limit shape; not a derived
    PAC-Bayes bound). See ``sloc.py`` for the curve target.
    """

    name = "sloc"

    def __init__(self, statistical_agents, soul_pool, partition, persona_to_slot):
        self.stat = statistical_agents
        self.souls = soul_pool
        self.partition = partition
        self.persona_to_slot = persona_to_slot

    def simulate(
        self,
        impression,
        creative,
        kol,
        platform,
        use_llm: bool = True,
        n_souls: int = 30,
        macro_ctr_lift: float = 1.0,
        macro_cvr_lift: float = 1.0,
        **kwargs,
    ) -> ProviderResult:
        from .calibration import calibrate_per_territory
        from .soul_llm import llm_available

        t0 = time.time()

        outcome = self.stat.simulate(
            impression,
            creative,
            kol,
            macro_ctr_lift=macro_ctr_lift,
            macro_cvr_lift=macro_cvr_lift,
        )
        click_prob_by_agent = {
            int(a): float(p) for a, p in zip(outcome.agent_idx, outcome.click_prob, strict=False)
        }

        llm_calls = 0
        cost = 0.0
        cal_info = {}
        if use_llm and llm_available():
            souls = self.souls.infer_batch(
                creative,
                click_prob_by_agent,
                kol=kol,
                platform=platform,
                n_sample=n_souls,
                use_llm=True,
            )
            llm_calls = sum(1 for s in souls if s.get("source") == "llm")
            cost = souls[0].get("_batch_cost_cny", 0.0) if souls else 0.0
            # Voronoi calibration
            llm_souls = [s for s in souls if s.get("source") == "llm"]
            if len(llm_souls) >= 5:
                # Need stat probs at soul agent indices
                soul_pids = [
                    int(s.get("persona_id")) for s in llm_souls if s.get("persona_id") is not None
                ]
                # Quick approximation: get existing probs where overlap
                stat_probs_at_souls = {
                    pid: click_prob_by_agent.get(pid, float(np.mean(outcome.click_prob)))
                    for pid in soul_pids
                }
                cal = calibrate_per_territory(
                    llm_souls,
                    self.partition,
                    stat_probs_at_souls,
                    persona_id_to_slot=self.persona_to_slot,
                )
                cal_info = cal

        # KPI — use calibrated if available
        impressions = float(impression.total_impressions)
        clicks = float(np.sum(outcome.click_prob * impression.weight))
        convs = float(np.sum(outcome.convert_prob * impression.weight))
        if cal_info.get("global_factor"):
            f = cal_info["global_factor"]
            clicks *= f
            convs *= f
        ctr = clicks / max(impressions, 1)
        cvr = convs / max(clicks, 1)

        samples = []
        if use_llm:
            # Top 20 LLM souls for viz
            import random

            rng = random.Random(0)
            sample_ids = rng.sample(
                list(self.souls.personas.keys()), min(20, len(self.souls.personas))
            )
            for pid in sample_ids:
                samples.append(
                    {
                        "pid": pid,
                        "persona": self.souls.personas[pid].one_liner(),
                        "stat_click_prob": round(click_prob_by_agent.get(pid, 0.05), 3),
                    }
                )

        return ProviderResult(
            provider="sloc",
            n_agents_total=self.stat.pop.N,
            n_agents_active=self.stat.pop.N,  # all 100k evaluated (vectorized)
            n_llm_calls=llm_calls,
            cost_cny=cost,
            latency_s=time.time() - t0,
            click_prob_mean=float(outcome.click_prob.mean()),
            click_prob_std=float(outcome.click_prob.std()),
            ctr=ctr,
            cvr=cvr,
            per_agent_samples=samples,
            notes=(
                f"SLOC: 100k statistical fabric + {llm_calls} LLM oracles "
                f"+ stratified coreset weighting "
                f"({'with' if cal_info else 'without'} calibration factor "
                f"{cal_info.get('global_factor', 'N/A')})."
            ),
        )


# Backward-compat alias — old code imports VoronoiProvider
VoronoiProvider = SLOCProvider


def compare_providers(providers: list[ProviderResult]) -> dict:
    """Side-by-side comparison stats."""
    if not providers:
        return {}
    base = providers[0]
    rows = []
    for p in providers:
        rows.append(
            {
                "provider": p.provider,
                "n_agents_total": p.n_agents_total,
                "n_agents_active": p.n_agents_active,
                "n_llm_calls": p.n_llm_calls,
                "latency_s": round(p.latency_s, 2),
                "cost_cny": round(p.cost_cny, 4),
                "ctr": round(p.ctr, 4),
                "cvr": round(p.cvr, 4),
                "ctr_delta_vs_first": round(p.ctr - base.ctr, 4),
                "cost_ratio_vs_first": round(p.cost_cny / max(base.cost_cny, 0.0001), 1),
            }
        )
    return {"rows": rows}
