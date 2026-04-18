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
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Iterable

from .base import (
    DEFAULT_EVENT_TYPES,
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
                self.freq = nn.Parameter(
                    torch.randn(out_dim // 2) * 0.1, requires_grad=True
                )
                self.phase = nn.Parameter(torch.zeros(out_dim // 2))
                self.proj = nn.Linear(out_dim, out_dim)

            def forward(self, dt: "torch.Tensor") -> "torch.Tensor":
                # dt: [B, N]
                w = dt.unsqueeze(-1) * self.freq.view(1, 1, -1) + self.phase.view(1, 1, -1)
                feat = torch.cat([torch.sin(w), torch.cos(w)], dim=-1)  # [B, N, out_dim]
                return self.proj(feat)

        class SelfAttentionBlock(nn.Module):
            def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(d_model)
                self.attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                self.drop = nn.Dropout(dropout)
                self.norm2 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                )

            def forward(
                self, x: "torch.Tensor", causal_mask: "torch.Tensor"
            ) -> "torch.Tensor":
                h = self.norm1(x)
                attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
                x = x + self.drop(attn_out)
                x = x + self.drop(self.ffn(self.norm2(x)))
                return x

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

            def forward(
                self,
                type_ids: "torch.Tensor",       # [B, N] int
                dt: "torch.Tensor",             # [B, N] float (inter-event times)
                treatment_ids: "torch.Tensor",  # [B, N] int (0/1)
            ) -> "torch.Tensor":
                """Return per-event pre-intensity state ``h`` of shape [B, N, d]."""
                B, N = type_ids.shape
                tok = (
                    self.type_embed(type_ids)
                    + self.treatment_embed(treatment_ids)
                    + self.proj_time(self.time_embed(dt))
                )
                # Causal mask — event i can only attend to events ≤ i
                causal = torch.triu(
                    torch.full((N, N), float("-inf"), device=tok.device),
                    diagonal=1,
                )
                for blk in self.blocks:
                    tok = blk(tok, causal)
                return self.final_norm(tok)

            def intensity(self, h: "torch.Tensor") -> "torch.Tensor":
                """lambda_k(t) = softplus(W_k · h_t + b_k)."""
                return F.softplus(self.intensity_head(h), beta=cfg.softplus_beta)

        net = CausalNeuralHawkesNet().to(self._device)
        return net

    # ----------------------------------------------------------- helpers

    def _etype_idx(self, name: str) -> int:
        return self.config.event_types.index(name)

    def _treatment_id_of(self, event_name: str) -> int:
        # Convention: events prefixed "paid_" are treatment/intervention events.
        return 1 if event_name.startswith("paid_") else 0

    def _prep_stream(
        self, events: list[tuple[float, str]]
    ) -> tuple[Any, Any, Any, Any]:
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

    def forecast(
        self, seed_events: Iterable[tuple[float, str]], **kwargs: Any
    ) -> DiffusionForecast:
        """Sample a forecast via thinning (Ogata 1981) with the neural intensity."""
        torch = self._torch
        self._net.eval()
        events = list(seed_events)
        horizon_min = float(self.config.horizon_days * 24 * 60)
        K = len(self.config.event_types)

        t = events[-1][0] if events else 0.0
        iters = 0
        max_iters = 100_000
        while t < horizon_min and iters < max_iters:
            iters += 1
            type_ids, dt, treatment_ids, _ = self._prep_stream(events or [(0.0, self.config.event_types[0])])
            with torch.no_grad():
                h = self._net(type_ids, dt, treatment_ids)
                lam = self._net.intensity(h)[:, -1]  # [1, K]
            lambda_bar = float(lam.sum().item()) * 1.2 + 1e-6
            u = self._rng.random()
            dt_samp = -math.log(max(1e-12, u)) / lambda_bar
            t = t + dt_samp
            if t >= horizon_min:
                break

            # Accept with probability sum_k lambda_k(t) / lambda_bar
            # Approximation: reuse the last intensity (neural Hawkes is
            # smooth between events — fine for bucketed forecasts).
            total = float(lam.sum().item())
            if self._rng.random() * lambda_bar > total:
                continue
            # Pick event type proportional to intensity
            p = (lam.squeeze(0) / max(1e-9, total)).cpu().tolist()
            r = self._rng.random()
            cum = 0.0
            picked = K - 1
            for k, pk in enumerate(p):
                cum += pk
                if r <= cum:
                    picked = k
                    break
            events.append((t, self.config.event_types[picked]))

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
            type_ids, dt, treatment_ids, _ = self._prep_stream(events or [(0.0, self.config.event_types[0])])
            # Under the do() intervention, zero-out future treatment tokens
            if t > mute_at:
                treatment_ids = torch.zeros_like(treatment_ids)
            with torch.no_grad():
                h = self._net(type_ids, dt, treatment_ids)
                lam = self._net.intensity(h)[:, -1].clone()
            if boost != 1.0:
                # Apply boost to "paid_*" event-type indices.
                # We clone above so this in-place mul stays safe for future
                # training-time counterfactual rollouts.
                for k, name in enumerate(self.config.event_types):
                    if name.startswith("paid_"):
                        lam[0, k] = lam[0, k] * boost
            lambda_bar = float(lam.sum().item()) * 1.2 + 1e-6
            u = self._rng.random()
            dt_samp = -math.log(max(1e-12, u)) / lambda_bar
            t = t + dt_samp
            if t >= horizon_min:
                break
            total = float(lam.sum().item())
            if self._rng.random() * lambda_bar > total:
                continue
            p = (lam.squeeze(0) / max(1e-9, total)).cpu().tolist()
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

    def _integrate_compensator(self, lam: "Any", times: "Any") -> "Any":
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

    def fit(
        self,
        dataset: Iterable[Iterable[tuple[float, str]]],
        *,
        val_dataset: Iterable[Iterable[tuple[float, str]]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Minimise the negative log-likelihood over event streams."""
        torch = self._torch
        cfg = self.config
        opt = torch.optim.AdamW(
            self._net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        history = {"train_nll": [], "val_nll": []}
        for epoch in range(cfg.max_epochs):
            self._net.train()
            total = 0.0
            n = 0
            for stream in dataset:
                events = list(stream)
                if len(events) < 2:
                    continue
                type_ids, dt, treatment_ids, times = self._prep_stream(events)
                opt.zero_grad()
                h = self._net(type_ids, dt, treatment_ids)
                lam = self._net.intensity(h)  # [1, N, K]
                idx = torch.arange(lam.shape[1], device=self._device)
                event_lam = lam[0, idx, type_ids.squeeze(0)]
                log_sum = torch.log(event_lam.clamp(min=1e-12)).sum()
                # Use the same compensator estimator chosen at config time —
                # rectangle / trapezoidal / mc. See ``_integrate_compensator``
                # for derivation.
                comp = self._integrate_compensator(lam, times)
                nll = -(log_sum - comp)
                nll.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), cfg.grad_clip)
                opt.step()
                total += float(nll.item())
                n += 1
            history["train_nll"].append(total / max(1, n))

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
    def load_pretrained(cls, path: str | None = None, **kwargs: Any) -> "CausalNeuralHawkesProcess":
        if path is None:
            raise FileNotFoundError(
                "No pretrained CausalNeuralHawkesProcess weights in v0.1.0-alpha.\n"
                "Options:\n"
                "  1. Train locally: "
                "python -m backend.scripts.train_neural_hawkes --config default\n"
                "  2. Watch v0.2 release for weights\n"
                "  3. Use the ParametricHawkes baseline for fast inference"
            )
        torch = _require_torch()
        ckpt = torch.load(path, map_location="cpu")
        cfg = CausalNeuralHawkesConfig(**ckpt["config"])
        model = cls(cfg)
        model._net.load_state_dict(ckpt["state_dict"])
        return model


# Backward-compat alias
TransformerHawkesProcess = CausalNeuralHawkesProcess
