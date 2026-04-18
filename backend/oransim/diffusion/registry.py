"""Registry for diffusion-model variants."""

from __future__ import annotations

from typing import Any, Callable

from .base import DiffusionModel


def _load_causal_neural_hawkes() -> type[DiffusionModel]:
    from .neural_hawkes import CausalNeuralHawkesProcess

    return CausalNeuralHawkesProcess


def _load_parametric() -> type[DiffusionModel]:
    from .hawkes import ParametricHawkes

    return ParametricHawkes


REGISTRY: dict[str, Callable[[], type[DiffusionModel]]] = {
    "causal_neural_hawkes": _load_causal_neural_hawkes,
    "parametric_hawkes": _load_parametric,
    # Aliases
    "neural_hawkes": _load_causal_neural_hawkes,
    "transformer_hawkes": _load_causal_neural_hawkes,
    "thp": _load_causal_neural_hawkes,
    "hawkes": _load_parametric,
}


def get_diffusion_model(name: str, **kwargs: Any) -> DiffusionModel:
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown diffusion model '{name}'. Available: {sorted(REGISTRY)}"
        ) from None
    return factory()(**kwargs)


def list_diffusion_models() -> list[str]:
    return ["causal_neural_hawkes", "parametric_hawkes"]
