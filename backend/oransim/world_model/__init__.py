"""World model layer — predicts funnel KPIs (impressions, clicks, conversions, revenue).

Two concrete world models ship:

- :class:`CausalTransformerWorldModel` (primary) — research-grade causal
  Transformer with explicit treatment / covariate / outcome factorisation,
  DAG-aware self-attention, per-arm counterfactual heads, and a
  representation-balancing loss. Draws from the recent causal-Transformer
  literature (CaT, CausalDAG-Transformer, BCAUSS, CInA). See the module
  docstring of :mod:`oransim.world_model.transformer` for citations.

- :class:`LightGBMQuantileWorldModel` — fast baseline. Three quantile
  regressors (P35/P50/P65) per KPI. Zero-dependency fallback when PyTorch is
  unavailable or sub-millisecond inference is required.

Pick via the registry::

    from oransim.world_model import get_world_model

    wm = get_world_model("causal_transformer")        # primary
    wm = get_world_model("lightgbm_quantile")          # baseline

Pretrained weights will ship at
https://github.com/deterministically/oransim/releases (starting v0.2).
"""

from .base import WorldModel, WorldModelConfig, WorldModelPrediction
from .registry import REGISTRY, get_world_model, list_world_models
from .transformer import (
    CausalTransformerWMConfig,
    CausalTransformerWorldModel,
    TransformerWMConfig,  # backward-compat alias
    TransformerWorldModel,  # backward-compat alias
)
from .lightgbm_quantile import LightGBMQuantileWorldModel, LightGBMWMConfig

__all__ = [
    "WorldModel",
    "WorldModelConfig",
    "WorldModelPrediction",
    "CausalTransformerWorldModel",
    "CausalTransformerWMConfig",
    "TransformerWorldModel",
    "TransformerWMConfig",
    "LightGBMQuantileWorldModel",
    "LightGBMWMConfig",
    "REGISTRY",
    "get_world_model",
    "list_world_models",
]
