"""Diffusion forecasting — predicts 14-day cascading engagement after launch.

Two concrete models ship:

- :class:`CausalNeuralHawkesProcess` (primary) — a Transformer-parameterised
  neural point process with explicit treatment / control event types and
  intervention-aware intensity (``lambda(t | H, do(T=t_star))``). Combines
  ideas from Mei & Eisner (2017, *Neural Hawkes*), Zuo et al. (2020,
  *Transformer Hawkes Process*), and recent causal temporal-point-process
  work (Geng et al. 2022, Noorbakhsh & Rodriguez 2022).

- :class:`ParametricHawkes` — classical exponential-kernel Hawkes baseline
  with closed-form intensity. Kept for zero-dependency fallback and for
  ablation comparisons in OrancBench.

Pick via the registry::

    from oransim.diffusion import get_diffusion_model

    diff = get_diffusion_model("causal_neural_hawkes")   # primary
    diff = get_diffusion_model("parametric_hawkes")      # baseline

Pretrained weights train on OranAI synthetic event streams and ship at
https://github.com/ORAN-cgsj/oransim/releases starting v0.2.
"""

from .base import DiffusionModel, DiffusionConfig, DiffusionForecast
from .registry import REGISTRY, get_diffusion_model, list_diffusion_models
from .neural_hawkes import (
    CausalNeuralHawkesProcess,
    CausalNeuralHawkesConfig,
    TransformerHawkesProcess,  # backward-compat alias
    NeuralHawkesConfig,  # backward-compat alias
)
from .hawkes import ParametricHawkes, ParametricHawkesConfig

__all__ = [
    "DiffusionModel",
    "DiffusionConfig",
    "DiffusionForecast",
    "CausalNeuralHawkesProcess",
    "CausalNeuralHawkesConfig",
    "TransformerHawkesProcess",
    "NeuralHawkesConfig",
    "ParametricHawkes",
    "ParametricHawkesConfig",
    "REGISTRY",
    "get_diffusion_model",
    "list_diffusion_models",
]
