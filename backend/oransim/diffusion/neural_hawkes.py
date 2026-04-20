"""Causal Neural Hawkes process — primary diffusion forecaster.

Transformer-parameterised neural temporal point process with explicit
treatment-vs-control event typing and intervention-aware conditional
intensity. Predicts multivariate cascading engagement (impressions / likes /
reshares / comments / saves / conversions) over the 14-day horizon after a
marketing launch, and supports ``do()`` queries (``"what if we had stopped
boosting on day 3"``) via a counterfactual rollout loop.

References
----------

- **Mei & Eisner 2017** — *The Neural Hawkes Process: A Neurally
  Self-Modulating Multivariate Point Process* (NeurIPS 2017). The
  foundational continuous-time LSTM neural Hawkes.
- **Zuo et al. 2020** — *Transformer Hawkes Process* (ICML 2020). Uses
  self-attention instead of an RNN for the intensity encoder — the
  architectural backbone of this implementation.
- **Shchur et al. 2020** — *Intensity-Free Learning of Temporal Point
  Processes* (ICLR 2020). Motivated the closed-form log-normal inter-event
  time head used by our sampling routine.
- **Chen et al. 2021** — *Neural Spatio-Temporal Point Processes* (ICLR
  2021). Monte-Carlo integration technique for the log-likelihood's
  compensator term.
- **Geng, Xu, Huang, et al. 2022** — *Counterfactual temporal point
  processes* (NeurIPS 2022). Counterfactual rollout formulation with
  treatment-marked events.
- **Noorbakhsh & Rodriguez 2022** — *Counterfactual Temporal Point
  Processes*. Intervention semantics for marked point processes.
- **Ogata 1981** — Thinning sampler used for rollout.

Weights
-------

Pretrained weights train on synthetic marketing event streams and will
ship starting v0.2. Until then, ``load_pretrained()`` raises
:class:`FileNotFoundError`; train locally with
``python -m backend.scripts.train_neural_hawkes --config <yaml>``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from .base import (
    DiffusionConfig,
    DiffusionForecast,
    DiffusionModel,
)

# --------------------------------------------------------------------- config


@dataclass
class CausalNeuralHawkesConfig(DiffusionConfig):
    """Configuration for :class:`CausalNeuralHawkesProcess`."""

    # Transformer sizing
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 1024  # max events per stream

    # Event embedding
    time_embed_dim: int = 16
    n_treatment_event_types: int = 2  # 0 = organic, 1 = paid-boost (intervention token)

    # Intensity decoder
    softplus_beta: float = 1.0
    # Compensator estimator: "rectangle" (piecewise-constant per inter-event
    # interval, O(N) per stream — default, standard in production TPPs) or
    # "mc" (Monte Carlo with `n_mc_samples` samples per interval, O(N·M) but
    # lower bias when the intensity varies fast within an interval).
    compensator: str = "rectangle"
    n_mc_samples: int = 20

    # Training
    learning_rate: float = 5.0e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 30
    grad_clip: float = 1.0

    # System
    device: str = "auto"
    checkpoint_dir: str = "data/models/causal_neural_hawkes"
    pretrained_url: str = "coming_soon"


# Backward-compat alias
NeuralHawkesConfig = CausalNeuralHawkesConfig


def _require_torch() -> Any:
    try:
        import torch  # noqa: F401

        return __import__("torch")
    except ImportError as exc:
        raise ImportError(
            "CausalNeuralHawkesProcess requires PyTorch. "
            "Install with: pip install 'oransim[ml]'\n"
            "Original error: " + str(exc)
        ) from exc


# --------------------------------------------------------------------- model


class CausalNeuralHawkesProcess(DiffusionModel):
    """Transformer Hawkes process with causal / counterfactual extensions.

    Ships full architecture + training loop + forecast sampler +
    counterfactual rollout. Weights land in v0.2.

    Example
    -------

    >>> from oransim.diffusion import CausalNeuralHawkesProcess, CausalNeuralHawkesConfig
    >>> nh = CausalNeuralHawkesProcess(CausalNeuralHawkesConfig(d_model=128))
    >>> # nh.fit(event_stream_dataset)
    >>> # factual = nh.forecast(seed_events=[(0.0, "impression"), (12.0, "like")])
    >>> # cf = nh.counterfactual_forecast(seed_events, intervention={"mute_at_min": 4320})
    """

    def __init__(self, config: CausalNeuralHawkesConfig | None = None):
        self.config = config or CausalNeuralHawkesConfig()
        self._torch = _require_torch()
        self._device = self._resolve_device()
        self._rng = Random(self.config.seed)
        self._net = self._build_network()

    # ------------------------------------------------------------------ build

    def _resolve_device(self) -> Any:
        torch = self._torch
        req = self.config.device
        if req == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(req)

    def _build_network(self) -> Any:
        torch = self._torch
        nn = torch.nn
        F = torch.nn.functional
        cfg = self.config
        K = len(cfg.event_types)

        class TimeEmbedding(nn.Module):
            """Continuous-time encoding (Zuo 2020 §3.1).

            Encodes Δt = t - t_prev as a learned sinusoidal feature + linear.
            """

            def __init__(self, out_dim: int) -> None:
                super().__init__()
                self.freq = nn.Parameter(torch.randn(out_dim // 2) * 0.1, requires_grad=True)
                self.phase = nn.Parameter(torch.zeros(out_dim // 2))
                self.proj = nn.Linear(out_dim, out_dim)

            def forward(self, dt: torch.Tensor) -> torch.Tensor:
                # dt: [B, N]
                w = dt.unsqueeze(-1) * self.freq.view(1, 1, -1) + self.phase.view(1, 1, -1)
                feat = torch.cat([torch.sin(w), torch.cos(w)], dim=-1)  # [B, N, out_dim]
                return self.proj(feat)

        class SelfAttentionBlock(nn.Module):
            """Multi-head self-attention with an optional KV cache.

            Training / full-sequence inference uses the standard causal-attention
            path (equivalent to the prior ``nn.MultiheadAttention`` block,
            just with hand-split Q/K/V projections so we can also expose an
            incremental mode).

            Incremental mode — used by forecast's Ogata thinning loop — keeps
            the K/V of the event history in a per-block cache dict. Each new
            virtual-τ query adds exactly one row of Q/K/V and attends to the
            cached K/V. This turns the per-iter cost from O(N²) (full-seq
            Transformer re-run) into O(N) (single query against cached keys),
            which is the main speedup for long-horizon forecasts.
            """

            def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
                super().__init__()
                assert d_model % n_heads == 0
                self.n_heads = n_heads
                self.d_head = d_model // n_heads
                self.d_model = d_model
                self.norm1 = nn.LayerNorm(d_model)
                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                self.attn_drop = nn.Dropout(dropout)
                self.drop = nn.Dropout(dropout)
                self.norm2 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                )

            def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
                # [B, N, D] → [B, H, N, d_head]
                B, N, _ = t.shape
                return t.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

            def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
                # [B, H, N, d_head] → [B, N, D]
                B, _, N, _ = t.shape
                return t.transpose(1, 2).contiguous().view(B, N, self.d_model)

            def _attention(
                self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
            ) -> torch.Tensor:
                # q [B, H, Nq, d], k/v [B, H, Nk, d] → [B, H, Nq, d]
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
                if causal:
                    Nq, Nk = q.shape[-2], k.shape[-2]
                    # Only mask the square self-attention case (Nq == Nk). In
                    # incremental mode Nq=1 attends to all Nk, no mask needed.
                    if Nq == Nk:
                        mask = torch.triu(
                            torch.full((Nq, Nk), float("-inf"), device=q.device),
                            diagonal=1,
                        )
                        scores = scores + mask
                attn = torch.softmax(scores, dim=-1)
                attn = self.attn_drop(attn)
                return torch.matmul(attn, v)

            def forward(
                self,
                x: torch.Tensor,
                *,
                kv_cache: dict | None = None,
                is_incremental: bool = False,
            ) -> torch.Tensor:
                """Standard path: full causal self-attention over x.

                When ``kv_cache`` is provided, K/V of this call are stored
                (full-seq mode) or appended (incremental mode) so the next
                incremental call can reuse them without recomputing history.
                """
                h = self.norm1(x)
                q = self._split_heads(self.q_proj(h))
                k_new = self._split_heads(self.k_proj(h))
                v_new = self._split_heads(self.v_proj(h))
                if is_incremental and kv_cache is not None and "k" in kv_cache:
                    k = torch.cat([kv_cache["k"], k_new], dim=2)
                    v = torch.cat([kv_cache["v"], v_new], dim=2)
                    causal = False  # single-token query attends to everything
                else:
                    k = k_new
                    v = v_new
                    causal = True
                if kv_cache is not None:
                    kv_cache["k"] = k
                    kv_cache["v"] = v
                attn_out = self._merge_heads(self._attention(q, k, v, causal=causal))
                attn_out = self.out_proj(attn_out)
                x = x + self.drop(attn_out)
                x = x + self.drop(self.ffn(self.norm2(x)))
                return x

            def rollback_cache(self, kv_cache: dict, n: int = 1) -> None:
                """Remove the last ``n`` positions from the cache — used when
                an Ogata thinning step rejects and we need to back out the
                virtual-τ query's K/V before trying again."""
                if kv_cache.get("k") is not None and kv_cache["k"].shape[2] > n:
                    kv_cache["k"] = kv_cache["k"][:, :, :-n, :]
                    kv_cache["v"] = kv_cache["v"][:, :, :-n, :]

        class CausalNeuralHawkesNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.type_embed = nn.Embedding(K, cfg.d_model)
                self.treatment_embed = nn.Embedding(cfg.n_treatment_event_types, cfg.d_model)
                self.time_embed = TimeEmbedding(cfg.time_embed_dim)
                self.proj_time = nn.Linear(cfg.time_embed_dim, cfg.d_model)

                self.blocks = nn.ModuleList(
                    [
                        SelfAttentionBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                        for _ in range(cfg.n_layers)
                    ]
                )
                self.final_norm = nn.LayerNorm(cfg.d_model)
                self.intensity_head = nn.Linear(cfg.d_model, K)

            def embed(
                self,
                type_ids: torch.Tensor,
                dt: torch.Tensor,
                treatment_ids: torch.Tensor,
            ) -> torch.Tensor:
                """Token embedding (type + treatment + Δt) — factored out so
                the incremental forecast path can embed just the new τ token
                without re-embedding history."""
                return (
                    self.type_embed(type_ids)
                    + self.treatment_embed(treatment_ids)
                    + self.proj_time(self.time_embed(dt))
                )

            def forward(
                self,
                type_ids: torch.Tensor,  # [B, N] int
                dt: torch.Tensor,  # [B, N] float (inter-event times)
                treatment_ids: torch.Tensor,  # [B, N] int (0/1)
                *,
                kv_caches: list[dict] | None = None,
                is_incremental: bool = False,
            ) -> torch.Tensor:
                """Return per-event pre-intensity state ``h`` of shape [B, N, d].

                ``kv_caches`` — optional list of per-block cache dicts. When
                ``is_incremental=False`` and kv_caches is passed, this call
                seeds each block's cache with the K/V of the full sequence.
                When ``is_incremental=True``, this call is treated as "append
                these tokens to the cached history and attend the new tokens
                to the full cached K/V"; causal mask is skipped since the
                new tokens are strictly newer than cached history.
                """
                tok = self.embed(type_ids, dt, treatment_ids)
                for i, blk in enumerate(self.blocks):
                    cache = kv_caches[i] if kv_caches is not None else None
                    tok = blk(tok, kv_cache=cache, is_incremental=is_incremental)
                return self.final_norm(tok)

            def intensity(self, h: torch.Tensor) -> torch.Tensor:
                """lambda_k(t) = softplus(W_k · h_t + b_k)."""
                return F.softplus(self.intensity_head(h), beta=cfg.softplus_beta)

            def new_kv_caches(self) -> list[dict]:
                """Allocate an empty cache per block."""
                return [{} for _ in range(len(self.blocks))]

            def rollback_last_token(self, kv_caches: list[dict]) -> None:
                """Remove the last virtual-τ token from every block's cache."""
                for i, blk in enumerate(self.blocks):
                    blk.rollback_cache(kv_caches[i], n=1)

        net = CausalNeuralHawkesNet().to(self._device)
        return net

    # ----------------------------------------------------------- helpers

    def _etype_idx(self, name: str) -> int:
        return self.config.event_types.index(name)

    def _treatment_id_of(self, event_name: str) -> int:
        # Convention: events prefixed "paid_" are treatment/intervention events.
        return 1 if event_name.startswith("paid_") else 0

    def _prep_stream(self, events: list[tuple[float, str]]) -> tuple[Any, Any, Any, Any]:
        """Build tensors (type_ids, dt, treatment_ids, abs_times) for one stream."""
        torch = self._torch
        type_ids = [self._etype_idx(n) for _, n in events]
        times = [t for t, _ in events]
        dts = [0.0] + [max(0.0, times[i] - times[i - 1]) for i in range(1, len(times))]
        treatment = [self._treatment_id_of(n) for _, n in events]
        return (
            torch.tensor(type_ids, dtype=torch.long, device=self._device).unsqueeze(0),
            torch.tensor(dts, dtype=torch.float, device=self._device).unsqueeze(0),
            torch.tensor(treatment, dtype=torch.long, device=self._device).unsqueeze(0),
            torch.tensor(times, dtype=torch.float, device=self._device).unsqueeze(0),
        )

    # ------------------------------------------------------------- forecast

    def _intensity_at_time(
        self,
        events: list[tuple[float, str]],
        t_cand: float,
        *,
        mute_treatment: bool = False,
    ):
        """Compute per-type intensity λ(t_cand) conditioned on events history.

        For Ogata thinning on a neural TPP, the acceptance probability requires
        λ(τ) at the **candidate** time τ, not at the last observed event.
        We append a virtual organic placeholder event at τ and read the
        intensity at that position. Placeholder type defaults to event_types[0]
        which gives a neutral "non-event query" at τ.

        The extra forward pass is the price of correct thinning (vs the older
        "reuse last intensity" approximation, which systematically biased
        acceptance toward stale pre-jump intensity).
        """
        torch = self._torch
        virtual = events + [(float(t_cand), self.config.event_types[0])]
        type_ids, dt, treatment_ids, _ = self._prep_stream(virtual)
        if mute_treatment:
            treatment_ids = torch.zeros_like(treatment_ids)
        with torch.no_grad():
            h = self._net(type_ids, dt, treatment_ids)
            lam = self._net.intensity(h)[:, -1].clone()  # [1, K] at the virtual τ
        return lam

    def forecast(
        self, seed_events: Iterable[tuple[float, str]], **kwargs: Any
    ) -> DiffusionForecast:
        """Sample a forecast via Ogata (1981) thinning on the neural intensity.

        Implementation uses per-block KV cache so that per-iter cost drops
        from O(N²) (re-running full Transformer on every rejected candidate)
        to O(N) (one new Q against cached K/V). On accept, the virtual-τ
        token's cached K/V becomes the real event's cache entry — no
        recomputation needed. On reject, the last cached token is rolled
        back and the next candidate τ re-attends against the same history.

        Thinning steps (corrected from pre-v0.2-fix bias):
          1. Compute λ̄ = 1.2 · Σ λ_k(t_last) — valid upper bound assuming
             intensity doesn't spike between events.
          2. Sample candidate gap dt_samp ~ Exp(λ̄) → candidate time τ = t + dt_samp.
          3. Re-compute λ(τ) at the candidate time (corrected; the old
             "reuse λ(t_last)" biased acceptance).
          4. Accept with probability Σ λ_k(τ) / λ̄; if accepted, pick event
             type proportional to λ_k(τ).
        """
        torch = self._torch
        self._net.eval()
        events = list(seed_events)
        horizon_min = float(self.config.horizon_days * 24 * 60)
        K = len(self.config.event_types)

        # Seed KV cache from the history events. One full-seq forward here,
        # O(N²). Everything inside the loop is O(N) per candidate.
        stream = events or [(0.0, self.config.event_types[0])]
        type_ids, dt, treatment_ids, _ = self._prep_stream(stream)
        kv_caches = self._net.new_kv_caches()
        with torch.no_grad():
            h = self._net(type_ids, dt, treatment_ids, kv_caches=kv_caches, is_incremental=False)
            lam_prev = self._net.intensity(h)[:, -1].clone()
        prev_last_time = float(stream[-1][0])

        t = events[-1][0] if events else 0.0
        iters = 0
        max_iters = 100_000
        placeholder_type_id = torch.tensor(
            [[self._etype_idx(self.config.event_types[0])]],
            dtype=torch.long,
            device=self._device,
        )
        placeholder_treat = torch.tensor([[0]], dtype=torch.long, device=self._device)

        while t < horizon_min and iters < max_iters:
            iters += 1
            lambda_bar = float(lam_prev.sum().item()) * 1.2 + 1e-6
            u = self._rng.random()
            dt_samp = -math.log(max(1e-12, u)) / lambda_bar
            t = t + dt_samp
            if t >= horizon_min:
                break

            # Incremental forward for the virtual-τ candidate. One new token
            # attends to the cached K/V of all prior events.
            virtual_dt = torch.tensor(
                [[t - prev_last_time]], dtype=torch.float32, device=self._device
            )
            with torch.no_grad():
                h_new = self._net(
                    placeholder_type_id,
                    virtual_dt,
                    placeholder_treat,
                    kv_caches=kv_caches,
                    is_incremental=True,
                )
                lam_cand = self._net.intensity(h_new)[:, -1]
            total = float(lam_cand.sum().item())
            if self._rng.random() * lambda_bar > total:
                # Reject — roll back the virtual-τ token from the cache so
                # the next iter attends to the same (unchanged) history.
                self._net.rollback_last_token(kv_caches)
                continue

            # Pick event type proportional to intensity AT τ.
            p = (lam_cand.squeeze(0) / max(1e-9, total)).cpu().tolist()
            r = self._rng.random()
            cum = 0.0
            picked = K - 1
            for k, pk in enumerate(p):
                cum += pk
                if r <= cum:
                    picked = k
                    break
            picked_type = self.config.event_types[picked]
            events.append((t, picked_type))

            # On accept, the cache already has a token at τ — but its type
            # was the placeholder (event_types[0]) and treatment=0. If that
            # matches the actually-picked type + treatment, we can keep the
            # cached K/V as-is. Otherwise, roll back and re-run incremental
            # with the correct type/treatment to keep the cache faithful to
            # real event history.
            real_treatment = self._treatment_id_of(picked_type)
            if picked != 0 or real_treatment != 0:
                self._net.rollback_last_token(kv_caches)
                real_type_id = torch.tensor(
                    [[self._etype_idx(picked_type)]], dtype=torch.long, device=self._device
                )
                real_treat = torch.tensor([[real_treatment]], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    h_real = self._net(
                        real_type_id,
                        virtual_dt,
                        real_treat,
                        kv_caches=kv_caches,
                        is_incremental=True,
                    )
                    lam_prev = self._net.intensity(h_real)[:, -1].clone()
            else:
                lam_prev = lam_cand.clone()
            prev_last_time = t

        return self._aggregate(events, horizon_min, latent={"backend": "causal_neural_hawkes"})

    def counterfactual_forecast(
        self,
        seed_events: Iterable[tuple[float, str]],
        *,
        intervention: dict[str, Any],
        **kwargs: Any,
    ) -> DiffusionForecast:
        """Run a counterfactual rollout (Geng et al. 2022).

        Supported interventions (v0.1-alpha skeleton):

        - ``{"mute_at_min": float}`` — force all future treatment events to
          be organic (treatment_id=0) after the given minute
        - ``{"treatment_boost_factor": float}`` — multiply treatment-event
          intensities by a factor (>1 amplifies, <1 dampens)
        """
        torch = self._torch
        self._net.eval()
        events = list(seed_events)
        horizon_min = float(self.config.horizon_days * 24 * 60)
        K = len(self.config.event_types)
        mute_at = float(intervention.get("mute_at_min", float("inf")))
        boost = float(intervention.get("treatment_boost_factor", 1.0))

        t = events[-1][0] if events else 0.0
        iters = 0
        max_iters = 100_000
        while t < horizon_min and iters < max_iters:
            iters += 1
            type_ids, dt, treatment_ids, _ = self._prep_stream(
                events or [(0.0, self.config.event_types[0])]
            )
            # Under the do() intervention, zero-out future treatment tokens
            if t > mute_at:
                treatment_ids = torch.zeros_like(treatment_ids)
            with torch.no_grad():
                h = self._net(type_ids, dt, treatment_ids)
                lam_prev = self._net.intensity(h)[:, -1].clone()
            if boost != 1.0:
                for k, name in enumerate(self.config.event_types):
                    if name.startswith("paid_"):
                        lam_prev[0, k] = lam_prev[0, k] * boost
            lambda_bar = float(lam_prev.sum().item()) * 1.2 + 1e-6
            u = self._rng.random()
            dt_samp = -math.log(max(1e-12, u)) / lambda_bar
            t = t + dt_samp
            if t >= horizon_min:
                break
            # Corrected thinning: evaluate intensity AT candidate τ.
            lam_cand = self._intensity_at_time(
                events or [(0.0, self.config.event_types[0])],
                t,
                mute_treatment=(t > mute_at),
            )
            if boost != 1.0:
                for k, name in enumerate(self.config.event_types):
                    if name.startswith("paid_"):
                        lam_cand[0, k] = lam_cand[0, k] * boost
            total = float(lam_cand.sum().item())
            if self._rng.random() * lambda_bar > total:
                continue
            p = (lam_cand.squeeze(0) / max(1e-9, total)).cpu().tolist()
            r = self._rng.random()
            cum = 0.0
            picked = K - 1
            for k, pk in enumerate(p):
                cum += pk
                if r <= cum:
                    picked = k
                    break
            events.append((t, self.config.event_types[picked]))

        out = self._aggregate(
            events,
            horizon_min,
            latent={
                "backend": "causal_neural_hawkes",
                "intervention": {"mute_at_min": mute_at, "boost": boost},
            },
        )
        return out

    def _aggregate(
        self, events: list[tuple[float, str]], horizon_min: float, *, latent: dict
    ) -> DiffusionForecast:
        K = len(self.config.event_types)
        timeline: list[tuple[float, str, float]] = [(t, n, 0.0) for t, n in events]
        totals = {n: 0.0 for n in self.config.event_types}
        for _, n in events:
            totals[n] += 1.0
        buckets = [[0.0] * K for _ in range(self.config.horizon_days)]
        for t, n in events:
            day = int(t // (24 * 60))
            if 0 <= day < self.config.horizon_days:
                buckets[day][self._etype_idx(n)] += 1.0
        return DiffusionForecast(
            timeline=timeline, per_type_totals=totals, daily_buckets=buckets, latent=latent
        )

    # ------------------------------------------------------ log-likelihood

    def log_likelihood(self, events: Iterable[tuple[float, str]]) -> float:
        """Negative log-likelihood with Monte-Carlo compensator (Chen et al. 2021)."""
        torch = self._torch
        events_list = list(events)
        if len(events_list) < 2:
            return 0.0
        type_ids, dt, treatment_ids, times = self._prep_stream(events_list)
        with torch.no_grad():
            h = self._net(type_ids, dt, treatment_ids)
            lam = self._net.intensity(h)  # [1, N, K]
        # Event term
        idx = torch.arange(lam.shape[1], device=self._device)
        event_lam = lam[0, idx, type_ids.squeeze(0)]  # [N]
        log_sum = torch.log(event_lam.clamp(min=1e-12)).sum()

        # Compensator. Three choices:
        #   - "rectangle"   (default): piecewise-constant intensity using
        #     ``lam[i-1]``. Fast, respects causal ordering (no leakage from
        #     the event at ``i`` into the interval that ends at ``i``).
        #   - "trapezoidal": linear-interpolation integral
        #     ``(lam[i-1] + lam[i]) / 2 * Δt`` — exact when intensity is
        #     linear within an interval; strictly tighter than rectangle.
        #   - "mc": Monte Carlo — sample ``n_mc_samples`` uniform points per
        #     interval, linearly interpolate intensity at each, average. Adds
        #     variance but converges to the trapezoidal mean and reflects
        #     the spirit of Chen et al. (ICLR 2021).
        comp = self._integrate_compensator(lam, times)
        return float((log_sum - comp).item())

    def _integrate_compensator(self, lam: Any, times: Any) -> Any:
        torch = self._torch
        abs_times = times.squeeze(0)
        N = abs_times.shape[0]
        comp = torch.zeros((), device=self._device)
        mode = self.config.compensator
        n_mc = max(1, self.config.n_mc_samples)
        for i in range(1, N):
            t0 = float(abs_times[i - 1].item())
            t1 = float(abs_times[i].item())
            if t1 <= t0:
                continue
            width = t1 - t0
            lam_lo = lam[0, i - 1].sum()
            if mode == "rectangle":
                comp = comp + lam_lo * width
            elif mode == "trapezoidal":
                lam_hi = lam[0, i].sum()
                comp = comp + 0.5 * (lam_lo + lam_hi) * width
            else:  # "mc"
                lam_hi = lam[0, i].sum()
                # Uniform samples in (0, 1) → linearly interpolated intensities
                u = torch.rand(n_mc, device=self._device)
                interp = lam_lo + (lam_hi - lam_lo) * u
                comp = comp + interp.mean() * width
        return comp

    # ---------------------------------------------------------------- train

    def _prep_batch(self, streams: list[list[tuple[float, str]]]) -> tuple[Any, Any, Any, Any, Any]:
        """Pad a mini-batch of variable-length streams to a common length.

        Returns (type_ids, dt, treatment_ids, times, mask) each with shape
        [B, Nmax]. ``mask[b, i] == 1`` iff position i of stream b is a real
        event; padded positions are zeros. Padded Δt is 0 so it contributes
        nothing to the compensator integral even if the mask were ignored.
        """
        torch = self._torch
        B = len(streams)
        tensors = [self._prep_stream(s) for s in streams]
        Ns = [t[0].shape[1] for t in tensors]
        Nmax = max(Ns)
        type_ids = torch.zeros((B, Nmax), dtype=torch.long, device=self._device)
        dt = torch.zeros((B, Nmax), dtype=torch.float32, device=self._device)
        treatment_ids = torch.zeros((B, Nmax), dtype=torch.long, device=self._device)
        times = torch.zeros((B, Nmax), dtype=torch.float32, device=self._device)
        mask = torch.zeros((B, Nmax), dtype=torch.float32, device=self._device)
        for b, (ti, d, tr, ts) in enumerate(tensors):
            N = Ns[b]
            type_ids[b, :N] = ti.squeeze(0)
            dt[b, :N] = d.squeeze(0)
            treatment_ids[b, :N] = tr.squeeze(0)
            times[b, :N] = ts.squeeze(0)
            mask[b, :N] = 1.0
        return type_ids, dt, treatment_ids, times, mask

    def _integrate_compensator_batched(self, lam: Any, times: Any, mask: Any) -> Any:
        """Vectorized compensator over a padded batch. Much faster than the
        Python for-loop; equivalent to calling ``_integrate_compensator`` on
        each stream individually and summing."""
        torch = self._torch
        # lam [B, N, K]  times [B, N]  mask [B, N]
        widths = times[:, 1:] - times[:, :-1]  # [B, N-1]
        widths = widths.clamp(min=0.0)
        # Interval is "real" only if both endpoints are real events
        interval_mask = mask[:, 1:] * mask[:, :-1]  # [B, N-1]
        lam_sum = lam.sum(dim=-1)  # [B, N]
        lam_lo = lam_sum[:, :-1]
        lam_hi = lam_sum[:, 1:]
        mode = self.config.compensator
        if mode == "trapezoidal":
            per_interval = 0.5 * (lam_lo + lam_hi) * widths
        elif mode == "mc":
            n_mc = max(1, self.config.n_mc_samples)
            u = torch.rand(n_mc, device=self._device)
            # lam_lo + (lam_hi - lam_lo) * u → [B, N-1, n_mc]; mean over MC dim
            interp = lam_lo.unsqueeze(-1) + (lam_hi - lam_lo).unsqueeze(-1) * u
            per_interval = interp.mean(dim=-1) * widths
        else:  # rectangle
            per_interval = lam_lo * widths
        return (per_interval * interval_mask).sum()

    def fit(
        self,
        dataset: Iterable[Iterable[tuple[float, str]]],
        *,
        val_dataset: Iterable[Iterable[tuple[float, str]]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Minimise the negative log-likelihood over event streams.

        Uses ``cfg.batch_size`` mini-batches with right-padding + a mask
        applied to both the log-intensity sum and the compensator integral.
        Compensator is evaluated in a single vectorized op per batch (no
        Python-level per-interval loop).
        """
        torch = self._torch
        cfg = self.config
        opt = torch.optim.AdamW(
            self._net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        streams_all = [list(s) for s in dataset]
        streams_all = [s for s in streams_all if len(s) >= 2]
        batch_size = max(1, cfg.batch_size)

        history: dict[str, list[float]] = {"train_nll": [], "val_nll": []}
        for epoch in range(cfg.max_epochs):
            self._net.train()
            total = 0.0
            n_batches = 0
            for i in range(0, len(streams_all), batch_size):
                batch = streams_all[i : i + batch_size]
                if not batch:
                    continue
                type_ids, dt, treatment_ids, times, mask = self._prep_batch(batch)
                opt.zero_grad()
                h = self._net(type_ids, dt, treatment_ids)
                lam = self._net.intensity(h)  # [B, Nmax, K]
                # Gather the intensity of the actually-observed event type per position
                # → [B, Nmax]; mask out padded positions before summing.
                event_lam = torch.gather(lam, 2, type_ids.unsqueeze(-1)).squeeze(-1)
                log_terms = torch.log(event_lam.clamp(min=1e-12)) * mask
                log_sum = log_terms.sum()
                comp = self._integrate_compensator_batched(lam, times, mask)
                nll = -(log_sum - comp)
                nll.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), cfg.grad_clip)
                opt.step()
                total += float(nll.item())
                n_batches += 1
            history["train_nll"].append(total / max(1, n_batches))

            if val_dataset is not None:
                self._net.eval()
                vtotal = 0.0
                vn = 0
                for stream in val_dataset:
                    vtotal -= self.log_likelihood(list(stream))
                    vn += 1
                history["val_nll"].append(vtotal / max(1, vn))

        return history

    # ---------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        torch = self._torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(), "config": self.config.__dict__},
            path,
        )

    @classmethod
    def load_pretrained(cls, path: str | None = None, **kwargs: Any) -> CausalNeuralHawkesProcess:
        if path is None:
            default_dir = Path(CausalNeuralHawkesConfig().checkpoint_dir)
            candidate = default_dir / "model.pt"
            if candidate.exists():
                path = str(candidate)
            else:
                raise FileNotFoundError(
                    "No pretrained CausalNeuralHawkesProcess weights in v0.2.0-alpha.\n"
                    f"Auto-resolve looked for: {candidate} (not found)\n"
                    "Options:\n"
                    "  1. Train locally: "
                    "python -m backend.scripts.train_neural_hawkes --config default "
                    "(writes to the auto-resolve path above)\n"
                    "  2. Pass an explicit path to load_pretrained(path=...)\n"
                    "  3. Watch v0.2 release for weights\n"
                    "  4. Use the ParametricHawkes baseline for fast inference"
                )
        torch = _require_torch()
        ckpt = torch.load(path, map_location="cpu")
        cfg = CausalNeuralHawkesConfig(**ckpt["config"])
        model = cls(cfg)
        model._net.load_state_dict(ckpt["state_dict"])
        return model


# Backward-compat alias
TransformerHawkesProcess = CausalNeuralHawkesProcess
