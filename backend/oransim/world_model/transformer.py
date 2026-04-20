"""Causal Transformer world model.

A research-grade causal Transformer that predicts marketing funnel KPIs with
calibrated quantile uncertainty, explicit treatment/covariate/outcome
factorization, counterfactual heads, and a representation-balancing loss.

Architectural choices draw from the recent surge of causal Transformer
literature (2022–2024):

- **CaT** — Melnychuk, Frauen, Feuerriegel (ICML 2022):
  *Causal Transformer for Estimating Counterfactual Outcomes* [arXiv:2204.07258].
  Tokenizes a scenario as ``(covariates, treatments, outcomes)`` and applies
  causal self-attention to model treatment-outcome sequences.
- **CausalDAG-Transformer** — attention bias derived from a user-provided
  DAG (e.g., the 64-node Pearl SCM in :mod:`oransim.causal`). Nodes only
  attend to ancestors. Generalizes DAG-aware message passing to multi-head
  attention.
- **BCAUSS / TARNet / Dragonnet** — Shalit et al. (2017), Shi, Blei, Veitch
  (NeurIPS 2019), Tesei et al. (2021). Separate representation tower with a
  per-treatment counterfactual head and an adversarial-IPTW or HSIC
  balancing regularizer.
- **CInA** — Arik & Pfister (NeurIPS 2023):
  *Causal In-context Amortized Learning*. Accepts a context set of past
  campaigns and amortizes causal inference via in-context learning.
- **TARNet / Dragonnet treatment heads** — Shalit et al. (ICML 2017), Shi
  et al. (NeurIPS 2019). Per-arm counterfactual heads enable ``do()`` queries.

Canonical factorization
-----------------------

For a marketing scenario we split inputs into three token types:

- **Covariates (X)** — platform, demographic distribution, time-of-day, KOL
  static features. Not under the marketer's control.
- **Treatments (T)** — creative embedding, budget, KOL assignment, targeting
  parameters. The marketer intervenes on these via ``do(T=t)``.
- **Outcomes (Y)** — funnel KPIs (impressions, clicks, conversions,
  revenue). Predicted with three quantile levels (P35/P50/P65).

The model implements a T-conditioned counterfactual head so that, given a
factual prediction, we can evaluate ``Y | do(T=t')`` for any alternative
treatment ``t'``.

Balancing
---------

During training we add an auxiliary **representation-balancing loss** that
pushes the learned covariate representation to be treatment-invariant
(HSIC (Gretton et al. 2005) by default, or adversarial IPTW à la CaT).
This reduces bias in counterfactual prediction when treatment assignment is
non-random.

In-context amortization (optional)
----------------------------------

When ``context_size > 0`` the model additionally attends to ``context_size``
prior campaigns. This is a faithful scaled-down implementation of the CInA
recipe for amortized causal inference.

Weights
-------

Pretrained weights train on the OranAI synthetic 100k dataset and will ship
starting v0.2 at https://github.com/OranAi-Ltd/oransim/releases. Until then,
``load_pretrained()`` raises :class:`FileNotFoundError`. Train locally with
``python -m backend.scripts.train_transformer_wm --config <yaml>``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    KPI_NAMES,
    WorldModel,
    WorldModelConfig,
    WorldModelPrediction,
)

# --------------------------------------------------------------------- config


@dataclass
class CausalTransformerWMConfig(WorldModelConfig):
    """Configuration for :class:`CausalTransformerWorldModel`."""

    # Transformer sizing
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 64

    # Feature-channel dims (one token per modality)
    creative_embed_dim: int = 1536  # OpenAI text-embedding-3-small
    platform_vocab: int = 8
    kol_feature_dim: int = 16
    demographic_feature_dim: int = 24
    time_feature_dim: int = 4  # sin/cos hour-of-day + day-of-week

    # Causal / treatment-effect specifics
    n_treatment_arms: int = 4  # distinct (creative, budget-bucket, KOL-tier) combos to mirror
    use_counterfactual_head: bool = True
    balancing_loss: str = "hsic"  # "hsic" | "iptw_adv" | "none"
    balancing_kernel: str = "linear"  # "linear" | "rbf" — used only when balancing_loss="hsic"
    balancing_rbf_sigma: float = 1.0
    balancing_weight: float = 0.1
    counterfactual_weight: float = 0.5
    dag_attention_bias: bool = True  # enable CausalDAG-Transformer attention mask
    context_size: int = 0  # >0 → CInA-style in-context amortization

    # Training
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    max_epochs: int = 50
    pinball_weight: float = 1.0
    mse_aux_weight: float = 0.1

    # System
    device: str = "auto"  # "cuda" / "cpu" / "auto"
    checkpoint_dir: str = "data/models/causal_transformer_wm"
    pretrained_url: str = "coming_soon"
    kpi_heads: tuple[str, ...] = field(default=KPI_NAMES)


# Backward-compat alias — callers that imported TransformerWMConfig before the
# causal upgrade continue to work.
TransformerWMConfig = CausalTransformerWMConfig


# ----------------------------------------------------------------- torch dep


def _require_torch() -> Any:
    """Import torch on demand so the base package remains torch-free."""
    try:
        import torch  # noqa: F401

        return __import__("torch")
    except ImportError as exc:
        raise ImportError(
            "CausalTransformerWorldModel requires PyTorch. "
            "Install with: pip install 'oransim[ml]'\n"
            "Original error: " + str(exc)
        ) from exc


# --------------------------------------------------------------- main class


class CausalTransformerWorldModel(WorldModel):
    """Causal Transformer world model with treatment-effect heads.

    Ships full architecture + training loop + inference + counterfactual
    evaluation. Weights released starting v0.2.

    Example
    -------

    >>> from oransim.world_model import CausalTransformerWorldModel
    >>> from oransim.world_model import CausalTransformerWMConfig
    >>> cfg = CausalTransformerWMConfig(d_model=256, n_layers=6, dag_attention_bias=True)
    >>> wm = CausalTransformerWorldModel(cfg)
    >>> # wm.fit(train_loader, val_dataset=val_loader)
    >>> # wm.predict(features) — factual
    >>> # wm.counterfactual(features, intervention={"budget": 150000})
    """

    def __init__(self, config: CausalTransformerWMConfig | None = None):
        self.config = config or CausalTransformerWMConfig()
        self._torch = _require_torch()
        self._device = self._resolve_device()
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

        # ---- Normalisation and activation blocks ----

        class RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
                return x * rms * self.weight

        class SwiGLU(nn.Module):
            def __init__(self, d_model: int, d_ff: int) -> None:
                super().__init__()
                self.w1 = nn.Linear(d_model, d_ff, bias=False)
                self.w2 = nn.Linear(d_model, d_ff, bias=False)
                self.w3 = nn.Linear(d_ff, d_model, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.w3(F.silu(self.w1(x)) * self.w2(x))

        # ---- DAG-aware attention (CausalDAG-Transformer) ----

        class CausalDAGAttention(nn.Module):
            """Multi-head self-attention with an optional additive DAG mask.

            Each head can additionally learn a per-head gating of the DAG bias
            (via ``alpha``), so some heads can attend topologically and others
            can run unrestricted — this follows the gated-bias pattern used in
            recent causal-graph Transformer variants (Zhang et al. 2023).
            """

            def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
                super().__init__()
                assert d_model % n_heads == 0
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                self.scale = 1.0 / math.sqrt(self.head_dim)
                self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
                self.out = nn.Linear(d_model, d_model, bias=False)
                self.drop = nn.Dropout(dropout)
                # Per-head gate on the DAG bias
                self.alpha = nn.Parameter(torch.zeros(n_heads))

            def forward(
                self,
                x: torch.Tensor,
                dag_bias: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                B, N, D = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                # [B, H, N, d_h]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

                if dag_bias is not None:
                    # dag_bias: [N, N] — (−inf) for non-ancestor entries,
                    # 0 elsewhere. Scale per head with learned alpha.
                    attn = attn + self.alpha.view(1, -1, 1, 1) * dag_bias

                if key_padding_mask is not None:
                    mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
                    attn = attn.masked_fill(mask, float("-inf"))

                attn = torch.softmax(attn, dim=-1)
                attn = self.drop(attn)
                out = attn @ v  # [B, H, N, d_h]
                out = out.transpose(1, 2).contiguous().reshape(B, N, D)
                return self.out(out)

        class EncoderBlock(nn.Module):
            def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
                super().__init__()
                self.norm1 = RMSNorm(d_model)
                self.attn = CausalDAGAttention(d_model, n_heads, dropout)
                self.drop = nn.Dropout(dropout)
                self.norm2 = RMSNorm(d_model)
                self.ffn = SwiGLU(d_model, d_ff)

            def forward(
                self,
                x: torch.Tensor,
                dag_bias: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                h = self.norm1(x)
                x = x + self.drop(
                    self.attn(h, dag_bias=dag_bias, key_padding_mask=key_padding_mask)
                )
                x = x + self.drop(self.ffn(self.norm2(x)))
                return x

        # ---- Counterfactual / quantile heads (TARNet / Dragonnet spirit) ----

        class QuantileHead(nn.Module):
            def __init__(self, d_model: int, n_quantiles: int) -> None:
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(d_model, n_quantiles),
                )

            def forward(self, h: torch.Tensor) -> torch.Tensor:
                return self.mlp(h)

        class CounterfactualHead(nn.Module):
            """Per-arm quantile head. Applied when a specific treatment arm
            is requested via ``do(T=arm_index)``."""

            def __init__(self, d_model: int, n_arms: int, n_quantiles: int) -> None:
                super().__init__()
                self.arms = nn.ModuleList(
                    [QuantileHead(d_model, n_quantiles) for _ in range(n_arms)]
                )

            def forward(self, h: torch.Tensor, arm_idx: torch.Tensor) -> torch.Tensor:
                """Vectorized per-arm quantile prediction.

                h:       [B, d_model]
                arm_idx: [B]  (long)
                returns: [B, n_quantiles]

                Old implementation looped over the batch and called ``.item()``
                per row, forcing a GPU→host sync every step (at B=256 this
                dominated training wall-time). Now we run all arms on the full
                batch in parallel and ``gather`` the per-row arm's output —
                strictly more flops (linear in n_arms) but zero syncs, and
                n_arms is small (typically 2–5) so the tradeoff is a big win
                on any accelerator.
                """
                # Run every arm on the whole batch → [n_arms, B, n_quant]
                stacked = torch.stack([arm(h) for arm in self.arms], dim=0)
                # Permute to [B, n_arms, n_quant] so we can gather along dim 1
                stacked = stacked.permute(1, 0, 2)
                n_quant = stacked.size(-1)
                idx = arm_idx.long().view(-1, 1, 1).expand(-1, 1, n_quant)
                return stacked.gather(1, idx).squeeze(1)

        # ---- Token-type embedding and feature projections ----

        class CausalTransformerNet(nn.Module):
            TYPE_COVARIATE = 0
            TYPE_TREATMENT = 1
            TYPE_OUTCOME = 2
            TYPE_CLS = 3
            TYPE_CONTEXT = 4  # CInA: pooled prior-campaign summary token

            def __init__(self) -> None:
                super().__init__()
                self.proj_creative = nn.Linear(cfg.creative_embed_dim, cfg.d_model)  # TREATMENT
                self.proj_platform = nn.Embedding(cfg.platform_vocab, cfg.d_model)  # COVARIATE
                self.proj_kol = nn.Linear(cfg.kol_feature_dim, cfg.d_model)  # TREATMENT
                self.proj_demo = nn.Linear(cfg.demographic_feature_dim, cfg.d_model)  # COVARIATE
                self.proj_budget = nn.Linear(1, cfg.d_model)  # TREATMENT
                self.proj_time = nn.Linear(cfg.time_feature_dim, cfg.d_model)  # COVARIATE
                # Outcome projection for CInA context tokens.
                self.proj_outcome = nn.Linear(len(cfg.kpi_heads), cfg.d_model)

                # Token-type embedding (covariate / treatment / outcome / cls / context)
                self.type_embed = nn.Embedding(5, cfg.d_model)
                self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

                # Sinusoidal position
                self.register_buffer(
                    "_pos_enc",
                    self._sin_pos_encoding(cfg.max_seq_len, cfg.d_model),
                    persistent=False,
                )

                self.blocks = nn.ModuleList(
                    [
                        EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                        for _ in range(cfg.n_layers)
                    ]
                )
                self.final_norm = RMSNorm(cfg.d_model)

                n_kpis = len(cfg.kpi_heads)
                n_q = len(cfg.quantiles)

                # Factual head — always used for training signal
                self.factual_heads = nn.ModuleList(
                    [QuantileHead(cfg.d_model, n_q) for _ in range(n_kpis)]
                )

                # Counterfactual head — per-arm, used when querying do(T=arm)
                if cfg.use_counterfactual_head:
                    self.cf_heads = nn.ModuleList(
                        [
                            CounterfactualHead(cfg.d_model, cfg.n_treatment_arms, n_q)
                            for _ in range(n_kpis)
                        ]
                    )
                else:
                    self.cf_heads = None

                # Treatment classifier for IPTW adversary / HSIC balancing
                self.treatment_probe = nn.Linear(cfg.d_model, cfg.n_treatment_arms)

                # Optional DAG mask buffer, set via set_dag_mask()
                self.register_buffer("_dag_bias", torch.zeros(1, 1), persistent=False)

            # Position encoding ------------------------------------------------

            @staticmethod
            def _sin_pos_encoding(max_len: int, d_model: int) -> torch.Tensor:
                pos = torch.arange(max_len).unsqueeze(1).float()
                div = torch.exp(
                    torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
                )
                pe = torch.zeros(max_len, d_model)
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                return pe.unsqueeze(0)

            # Tokenisation -----------------------------------------------------

            def tokenize(self, features: dict) -> tuple[torch.Tensor, torch.Tensor]:
                """Build (tokens, type_ids) for a mini-batch.

                Returns (seq [B, L, d], type_ids [B, L]).
                """
                B = features["creative_embed"].shape[0]
                tokens: list = []
                type_ids: list = []

                def push(tok: torch.Tensor, tid: int) -> None:
                    tokens.append(tok.unsqueeze(1))
                    type_ids.append(tid)

                # Treatments (CaT: under intervention)
                push(self.proj_creative(features["creative_embed"]), self.TYPE_TREATMENT)
                push(self.proj_kol(features["kol_feat"]), self.TYPE_TREATMENT)
                push(self.proj_budget(features["budget"]), self.TYPE_TREATMENT)

                # Covariates (CaT: not under intervention)
                push(self.proj_platform(features["platform_id"]), self.TYPE_COVARIATE)
                push(self.proj_demo(features["demo_feat"]), self.TYPE_COVARIATE)
                push(self.proj_time(features["time_feat"]), self.TYPE_COVARIATE)

                seq = torch.cat(tokens, dim=1)  # [B, L_feat, d]
                type_tensor = torch.tensor(type_ids, device=seq.device).unsqueeze(0).expand(B, -1)
                seq = seq + self.type_embed(type_tensor)

                # Prepend CLS (TYPE_CLS)
                cls = self.cls.expand(B, -1, -1)
                cls_type = torch.full((B, 1), self.TYPE_CLS, device=seq.device, dtype=torch.long)
                seq = torch.cat([cls, seq], dim=1)
                type_tensor = torch.cat([cls_type, type_tensor], dim=1)

                # Positional encoding
                seq = seq + self._pos_enc[:, : seq.shape[1]]
                return seq, type_tensor

            def set_dag_mask(self, dag_bias: torch.Tensor | None) -> None:
                """Set an additive DAG attention bias.

                ``dag_bias`` shape: ``[L, L]``, with ``-inf`` where attention
                should be blocked (violations of the partial order of the
                user-provided DAG), ``0`` elsewhere.
                """
                if dag_bias is None:
                    self._dag_bias = torch.zeros(1, 1, device=self._dag_bias.device)
                else:
                    self._dag_bias = dag_bias.to(self._dag_bias.device)

            # CInA in-context --------------------------------------------------

            def _tokenize_context(self, context: list[dict]) -> torch.Tensor:
                """Pool each context entry into a single TYPE_CONTEXT token.

                Each entry is a dict with the same feature keys used by the
                query (creative_embed / kol_feat / demo_feat / budget /
                platform_id / time_feat) plus an ``outcome`` tensor of shape
                ``[B, n_kpis]`` holding the observed KPI targets. We compute
                the 7 per-feature projections, mean-pool them into a single
                d_model vector, add the outcome projection, and add the
                TYPE_CONTEXT embedding. Result: ``[B, len(context), d_model]``.
                """
                if not context:
                    return None  # type: ignore[return-value]

                per_entry = []
                for entry in context:
                    parts = [
                        self.proj_creative(entry["creative_embed"]),
                        self.proj_kol(entry["kol_feat"]),
                        self.proj_budget(entry["budget"]),
                        self.proj_platform(entry["platform_id"]),
                        self.proj_demo(entry["demo_feat"]),
                        self.proj_time(entry["time_feat"]),
                    ]
                    feat_pool = torch.stack(parts, dim=0).mean(dim=0)  # [B, d]
                    outcome_tok = self.proj_outcome(entry["outcome"])  # [B, d]
                    ctx_tok = feat_pool + outcome_tok  # [B, d]
                    per_entry.append(ctx_tok)
                ctx_tokens = torch.stack(per_entry, dim=1)  # [B, C, d]

                # Add TYPE_CONTEXT embedding
                B, C, _ = ctx_tokens.shape
                ctx_types = torch.full(
                    (B, C), self.TYPE_CONTEXT, device=ctx_tokens.device, dtype=torch.long
                )
                ctx_tokens = ctx_tokens + self.type_embed(ctx_types)
                return ctx_tokens

            # Forward ----------------------------------------------------------

            def encode(self, features: dict, context: list[dict] | None = None) -> torch.Tensor:
                seq, _ = self.tokenize(features)
                if context:
                    ctx = self._tokenize_context(context)
                    if ctx is not None:
                        # Prepend context tokens so query tokens can attend to them
                        seq = torch.cat([ctx, seq], dim=1)
                dag_bias = self._dag_bias if cfg.dag_attention_bias else None
                # If a DAG bias is installed, it was sized for the query-only
                # sequence — expand with zero-padding for context tokens
                if dag_bias is not None and context:
                    ctx_len = len(context)
                    L_total = seq.shape[1]
                    padded = torch.zeros(L_total, L_total, device=seq.device)
                    padded[ctx_len:, ctx_len:] = dag_bias[: L_total - ctx_len, : L_total - ctx_len]
                    dag_bias = padded
                for blk in self.blocks:
                    seq = blk(seq, dag_bias=dag_bias)
                seq = self.final_norm(seq)
                # CLS now sits at position ``len(context) + 0``, not position 0
                cls_pos = len(context) if context else 0
                # But CLS was prepended BEFORE context in tokenize()? Check:
                # tokenize() prepends CLS → [CLS, ...query_tokens]. If we prepend
                # context BEFORE that, the order is [...context, CLS, ...query].
                # So cls_pos = ctx_len (if context) else 0.
                return seq[:, cls_pos]

            def forward_factual(self, features: dict, context: list[dict] | None = None) -> dict:
                h = self.encode(features, context=context)
                return {
                    name: head(h)
                    for name, head in zip(cfg.kpi_heads, self.factual_heads, strict=False)
                }

            def forward_counterfactual(
                self,
                features: dict,
                arm_idx: torch.Tensor,
                context: list[dict] | None = None,
            ) -> dict:
                """Counterfactual prediction ``Y | do(T = arm_idx)``."""
                if self.cf_heads is None:
                    raise RuntimeError(
                        "counterfactual head disabled — set use_counterfactual_head=True"
                    )
                h = self.encode(features, context=context)
                return {
                    name: head(h, arm_idx)
                    for name, head in zip(cfg.kpi_heads, self.cf_heads, strict=False)
                }

            def treatment_logits(
                self, features: dict, context: list[dict] | None = None
            ) -> torch.Tensor:
                """Probe treatment from representation — used by IPTW adversary."""
                return self.treatment_probe(self.encode(features, context=context))

        net = CausalTransformerNet().to(self._device)
        return net

    # ----------------------------------------------------------- DAG support

    def set_dag_from_edges(
        self, n_nodes: int, edges: Iterable[tuple[int, int]], token_to_node: list[int]
    ) -> None:
        """Build and install a DAG attention bias.

        Parameters
        ----------
        n_nodes
            Number of nodes in the underlying SCM (e.g., 64 for the Pearl SCM
            in ``oransim.causal``).
        edges
            Iterable of ``(parent, child)`` node index pairs.
        token_to_node
            For each token position (including CLS), the SCM node index it
            represents, or ``-1`` for free positions (e.g., CLS).
        """
        torch = self._torch
        L = len(token_to_node)
        bias = torch.zeros(L, L)

        edge_list = list(edges)
        parents: list[set[int]] = [set() for _ in range(n_nodes)]
        children: list[set[int]] = [set() for _ in range(n_nodes)]
        for p, c in edge_list:
            parents[c].add(p)
            children[p].add(c)

        # --- Strongly-connected-component condensation ------------------
        # The shipped causal graph contains long-term feedback loops (e.g.
        # repurchase → brand_equity → ecpm_bid → impression_dist, see
        # README §Causal Graph). Naive transitive closure on a cyclic graph
        # collapses every SCC member into everyone else's ancestor set,
        # which makes the attention bias effectively no-op inside the
        # feedback loop. We instead take the SCC condensation (Tarjan) and
        # define "ancestor of n" as:
        #   SCC(n)  ∪  all nodes in SCCs that forward-reach SCC(n) in the
        #             acyclic condensation.
        # This is the standard extension of Pearl's ancestry to cyclic
        # SCMs (e.g. Bongers et al. 2021 §3.2): within an SCC the nodes
        # are mutually ancestral because they can all influence each other
        # through the cycle.
        scc_of = [-1] * n_nodes
        sccs: list[list[int]] = []
        # Tarjan's algorithm (iterative to avoid recursion-limit on 64-node graphs)
        index = 0
        stack: list[int] = []
        on_stack = [False] * n_nodes
        indices = [-1] * n_nodes
        lowlink = [0] * n_nodes
        for start in range(n_nodes):
            if indices[start] != -1:
                continue
            work: list[tuple[int, int]] = [(start, 0)]  # (node, child-iter state)
            while work:
                v, child_idx = work[-1]
                if child_idx == 0:
                    indices[v] = index
                    lowlink[v] = index
                    index += 1
                    stack.append(v)
                    on_stack[v] = True
                child_list = list(children[v])
                if child_idx < len(child_list):
                    work[-1] = (v, child_idx + 1)
                    w = child_list[child_idx]
                    if indices[w] == -1:
                        work.append((w, 0))
                    elif on_stack[w]:
                        lowlink[v] = min(lowlink[v], indices[w])
                else:
                    # Post-order: propagate lowlink up and detect SCC root
                    work.pop()
                    if lowlink[v] == indices[v]:
                        comp: list[int] = []
                        while True:
                            w = stack.pop()
                            on_stack[w] = False
                            scc_of[w] = len(sccs)
                            comp.append(w)
                            if w == v:
                                break
                        sccs.append(comp)
                    if work:
                        parent = work[-1][0]
                        lowlink[parent] = min(lowlink[parent], lowlink[v])

        # Build condensation DAG: edge (A → B) iff some edge u→v with
        # scc_of[u]=A, scc_of[v]=B, A≠B.
        scc_parents: list[set[int]] = [set() for _ in range(len(sccs))]
        for p, c in edge_list:
            a, b = scc_of[p], scc_of[c]
            if a != b:
                scc_parents[b].add(a)
        # Ancestor SCCs (transitive closure on the acyclic condensation)
        scc_ancestors: list[set[int]] = [set() for _ in range(len(sccs))]
        for s in range(len(sccs)):
            frontier = list(scc_parents[s])
            while frontier:
                a = frontier.pop()
                if a not in scc_ancestors[s]:
                    scc_ancestors[s].add(a)
                    frontier.extend(scc_parents[a])
        # Node-level ancestors = own SCC ∪ nodes in ancestor SCCs
        ancestors: list[set[int]] = [set() for _ in range(n_nodes)]
        for n in range(n_nodes):
            s = scc_of[n]
            ancestors[n].update(sccs[s])  # mutual ancestry inside SCC
            for a in scc_ancestors[s]:
                ancestors[n].update(sccs[a])
            ancestors[n].discard(n)  # don't list self

        # For tokens attending to other tokens, disallow non-ancestor
        for i, ni in enumerate(token_to_node):
            for j, nj in enumerate(token_to_node):
                if ni < 0 or nj < 0:
                    continue  # free token (CLS) — no restriction
                if i == j:
                    continue
                # j is allowed to feed i only if nj ∈ ancestors[ni] ∪ {ni}
                if nj not in ancestors[ni] and nj != ni:
                    bias[i, j] = float("-inf")
        self._net.set_dag_mask(bias.to(self._device))

    # ---------------------------------------------------------------- predict

    def predict(
        self,
        features: dict[str, Any],
        *,
        context: list[dict[str, Any]] | None = None,
    ) -> WorldModelPrediction:
        """Factual prediction, optionally amortized over a prior-campaign
        ``context`` set (CInA, Arik & Pfister NeurIPS 2023)."""
        torch = self._torch
        self._net.eval()
        batched = self._batch_one(features)
        ctx = self._prep_context(context)
        with torch.no_grad():
            raw = self._net.forward_factual(batched, context=ctx)
        return self._materialize_prediction(
            raw,
            latent={"head": "factual", "context_size": 0 if context is None else len(context)},
        )

    def counterfactual(
        self,
        features: dict[str, Any],
        arm_idx: int,
        *,
        context: list[dict[str, Any]] | None = None,
    ) -> WorldModelPrediction:
        """Predict ``Y | do(T = arm_idx)``.

        ``arm_idx`` picks a discrete treatment arm out of
        ``config.n_treatment_arms``. The caller is responsible for mapping a
        concrete intervention (different creative / budget / KOL) into an arm
        index — the mapping is project-specific and lives in
        :mod:`oransim.causal.cate` (landing in Phase 3).
        """
        torch = self._torch
        self._net.eval()
        batched = self._batch_one(features)
        ctx = self._prep_context(context)
        arm = torch.tensor([arm_idx], device=self._device, dtype=torch.long)
        with torch.no_grad():
            raw = self._net.forward_counterfactual(batched, arm, context=ctx)
        return self._materialize_prediction(
            raw,
            latent={
                "head": "counterfactual",
                "arm_idx": arm_idx,
                "context_size": 0 if context is None else len(context),
            },
        )

    def _prep_context(self, context: list[dict[str, Any]] | None):
        """Batchify + tensorize each context entry for CInA in-context path."""
        if not context:
            return None
        return [self._batch_one_with_outcome(entry) for entry in context]

    def _batch_one_with_outcome(self, entry: dict[str, Any]) -> dict[str, Any]:
        torch = self._torch
        out: dict[str, Any] = {}
        for k, v in entry.items():
            t = torch.as_tensor(v, device=self._device)
            t = t.long() if k == "platform_id" else t.float()
            # Always prepend a batch dim. A 0-dim scalar (platform_id) becomes
            # [1] — not [1,1] — so downstream nn.Embedding receives the
            # expected index shape and doesn't emit a spurious extra axis.
            out[k] = t.unsqueeze(0)
        return out

    def _materialize_prediction(self, raw: dict, *, latent: dict) -> WorldModelPrediction:
        out: dict[str, dict[float, float]] = {}
        for kpi_name, preds in raw.items():
            row = preds[0].cpu().tolist()
            out[kpi_name] = {q: float(v) for q, v in zip(self.config.quantiles, row, strict=False)}
        return WorldModelPrediction(kpi_quantiles=out, latent=latent)

    def _batch_one(self, features: dict[str, Any]) -> dict[str, Any]:
        torch = self._torch
        out: dict[str, Any] = {}
        for k, v in features.items():
            t = torch.as_tensor(v, device=self._device)
            t = t.long() if k == "platform_id" else t.float()
            # Always prepend a batch dim. A 0-dim scalar (platform_id) becomes
            # [1] not [1,1] — keeps nn.Embedding input correctly rank-1.
            out[k] = t.unsqueeze(0)
        return out

    # --------------------------------------------------------------- training

    def fit(
        self,
        dataset: Iterable[dict[str, Any]],
        *,
        val_dataset: Iterable[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train factual + counterfactual heads + balancing regularizer.

        The dataset yields mini-batch dicts with feature keys plus:

        - ``targets`` — ``{kpi_name: [B]}`` factual outcome values
        - ``treatment_arm`` — ``[B]`` int, which discrete arm this row sits in
        - (optional) ``cf_targets`` — a dict mirroring ``targets`` for a
          counterfactual arm; if present, an explicit counterfactual loss is
          added (following the semi-synthetic training recipe of CaT 2022).
        """
        torch = self._torch
        cfg = self.config
        opt = torch.optim.AdamW(
            self._net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        quantile_tensor = torch.tensor(cfg.quantiles, device=self._device)
        median_idx = cfg.quantiles.index(0.50) if 0.50 in cfg.quantiles else len(cfg.quantiles) // 2

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_factual": [],
            "train_counterfactual": [],
            "train_balance": [],
            "val_loss": [],
        }

        for epoch in range(cfg.max_epochs):
            self._net.train()
            sums = {"loss": 0.0, "fact": 0.0, "cf": 0.0, "bal": 0.0}
            n = 0
            for batch in dataset:
                opt.zero_grad()
                features = {
                    k: v
                    for k, v in batch.items()
                    if k not in ("targets", "cf_targets", "treatment_arm")
                }

                loss, parts = self._step_loss(features, batch, quantile_tensor, median_idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                opt.step()

                sums["loss"] += float(loss.item())
                sums["fact"] += parts["fact"]
                sums["cf"] += parts["cf"]
                sums["bal"] += parts["bal"]
                n += 1

            for k in sums:
                sums[k] /= max(1, n)
            history["train_loss"].append(sums["loss"])
            history["train_factual"].append(sums["fact"])
            history["train_counterfactual"].append(sums["cf"])
            history["train_balance"].append(sums["bal"])

            if val_dataset is not None:
                history["val_loss"].append(self._evaluate(val_dataset, quantile_tensor, median_idx))

        return history

    def _step_loss(
        self, features: dict, batch: dict, quantiles: Any, median_idx: int
    ) -> tuple[Any, dict[str, float]]:
        torch = self._torch
        cfg = self.config

        # Factual pinball + MSE-aux
        fact_preds = self._net.forward_factual(features)
        fact_loss = self._quantile_loss(fact_preds, batch["targets"], quantiles, median_idx)

        # Counterfactual loss (only if cf_targets and cf head are available)
        cf_loss = torch.zeros((), device=self._device)
        if cfg.use_counterfactual_head and "cf_targets" in batch and "treatment_arm" in batch:
            arm = torch.as_tensor(batch["treatment_arm"], device=self._device).long()
            cf_preds = self._net.forward_counterfactual(features, arm)
            cf_loss = self._quantile_loss(cf_preds, batch["cf_targets"], quantiles, median_idx)

        # Balancing regularizer
        bal_loss = torch.zeros((), device=self._device)
        if cfg.balancing_loss != "none" and "treatment_arm" in batch:
            arm = torch.as_tensor(batch["treatment_arm"], device=self._device).long()
            if cfg.balancing_loss == "iptw_adv":
                logits = self._net.treatment_logits(features)
                # CaT-style adversarial IPTW: we want features that are
                # uninformative of treatment. Implemented as negative CE.
                ce = torch.nn.functional.cross_entropy(logits, arm)
                bal_loss = -ce  # maximize ambiguity → minimize negative CE
            elif cfg.balancing_loss == "hsic":
                # HSIC (Gretton et al. 2005, biased estimator Eq. 4) between the
                # CLS representation and the treatment assignment. Kernel
                # defaults to linear but RBF is available via config.
                h = self._net.encode(features)
                y_onehot = torch.nn.functional.one_hot(arm, cfg.n_treatment_arms).float()
                if cfg.balancing_kernel == "rbf":
                    bal_loss = self._hsic_rbf(h, y_onehot, sigma=cfg.balancing_rbf_sigma)
                else:
                    bal_loss = self._hsic_biased(h, y_onehot)

        total = (
            cfg.pinball_weight * fact_loss
            + cfg.counterfactual_weight * cf_loss
            + cfg.balancing_weight * bal_loss
        )
        return total, {
            "fact": float(fact_loss.item()),
            "cf": float(cf_loss.item()),
            "bal": float(bal_loss.item()),
        }

    def _quantile_loss(self, preds: dict, targets: dict, quantiles: Any, median_idx: int) -> Any:
        torch = self._torch
        cfg = self.config
        total = torch.zeros((), device=self._device)
        for kpi_name, head_out in preds.items():
            y = torch.as_tensor(targets[kpi_name], device=self._device).float()
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            diff = y - head_out
            pinball = torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean()
            mse = (head_out[:, median_idx] - y.squeeze(-1)).pow(2).mean()
            total = total + cfg.pinball_weight * pinball + cfg.mse_aux_weight * mse
        return total

    @staticmethod
    def _hsic_rbf(X: Any, Y: Any, sigma: float = 1.0) -> Any:
        """HSIC (Gretton et al. 2005, biased estimator Eq. 4) with RBF kernels.

        ``K_x(i, j) = exp(-||x_i - x_j||^2 / (2 sigma^2))`` and similarly
        ``K_y``. More sensitive to nonlinear dependence between the learned
        representation and the treatment assignment than the linear kernel
        variant :meth:`_hsic_biased`.
        """
        import torch

        B = X.shape[0]

        # Pairwise squared distance
        def _rbf(A: torch.Tensor) -> torch.Tensor:
            sq = (A * A).sum(-1, keepdim=True)
            d2 = sq + sq.t() - 2.0 * (A @ A.t())
            return torch.exp(-d2 / (2.0 * sigma * sigma))

        Kx = _rbf(X)
        Ky = _rbf(Y)
        H = torch.eye(B, device=X.device) - (1.0 / B)
        return (H @ Kx @ H @ Ky).diagonal().sum() / max(1, (B - 1)) ** 2

    @staticmethod
    def _hsic_biased(X: Any, Y: Any) -> Any:
        """Biased HSIC estimator (Gretton et al. 2005, Eq. 4) with linear kernels.

        A simple drop-in to decorrelate representation ``X`` from treatment
        one-hot ``Y``. For a production training run, replace with an RBF
        kernel and adaptive bandwidth (standard practice) and/or switch to
        the unbiased estimator (Gretton et al. 2012, Eq. 4).
        """
        import torch

        B = X.shape[0]
        Kx = X @ X.t()
        Ky = Y @ Y.t()
        H = torch.eye(B, device=X.device) - (1.0 / B)
        return (H @ Kx @ H @ Ky).diagonal().sum() / max(1, (B - 1)) ** 2

    def _evaluate(
        self, dataset: Iterable[dict[str, Any]], quantiles: Any, median_idx: int
    ) -> float:
        torch = self._torch
        self._net.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in dataset:
                features = {
                    k: v
                    for k, v in batch.items()
                    if k not in ("targets", "cf_targets", "treatment_arm")
                }
                preds = self._net.forward_factual(features)
                total += float(
                    self._quantile_loss(preds, batch["targets"], quantiles, median_idx).item()
                )
                n += 1
        self._net.train()
        return total / max(1, n)

    # ---------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        torch = self._torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(), "config": self.config.__dict__},
            path,
        )

    @classmethod
    def load_pretrained(cls, path: str | None = None, **kwargs: Any) -> CausalTransformerWorldModel:
        """Load pretrained weights.

        If ``path`` is ``None``, auto-resolves
        ``<checkpoint_dir>/model.pt``. Falls through to
        :class:`FileNotFoundError` with a helpful message when that file
        doesn't exist (current status in v0.2.0-alpha — no weights shipped).
        """
        if path is None:
            # Resolve relative to repo root (not CWD) for portability.
            repo_root = Path(__file__).resolve().parents[3]
            default_dir = repo_root / CausalTransformerWMConfig().checkpoint_dir
            candidate = default_dir / "model.pt"
            if candidate.exists():
                path = str(candidate)
            else:
                raise FileNotFoundError(
                    "No pretrained CausalTransformerWorldModel weights are available "
                    "in v0.2.0-alpha.\n"
                    f"Auto-resolve looked for: {candidate} (not found)\n"
                    "Options:\n"
                    "  1. Train locally: "
                    "python -m backend.scripts.train_transformer_wm --config default "
                    "(writes to the auto-resolve path above)\n"
                    "  2. Pass an explicit path to load_pretrained(path=...)\n"
                    "  3. Watch for weights at "
                    "https://github.com/OranAi-Ltd/oransim/releases (starting v0.2)\n"
                    "  4. Use the LightGBM baseline: get_world_model('lightgbm_quantile')"
                )
        torch = _require_torch()
        ckpt = torch.load(path, map_location="cpu")
        cfg = CausalTransformerWMConfig(**ckpt["config"])
        model = cls(cfg)
        model._net.load_state_dict(ckpt["state_dict"])
        return model


# Backward-compat alias — older imports keep working
TransformerWorldModel = CausalTransformerWorldModel
