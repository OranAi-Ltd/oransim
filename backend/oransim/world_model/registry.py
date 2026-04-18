"""Registry for world-model variants.

Use :func:`get_world_model` to instantiate a model by name without importing
its implementation module. Lazy import keeps cold-start fast even when
PyTorch is not installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import WorldModel


def _load_causal_transformer() -> type[WorldModel]:
    from .transformer import CausalTransformerWorldModel

    return CausalTransformerWorldModel


def _load_lightgbm() -> type[WorldModel]:
    from .lightgbm_quantile import LightGBMQuantileWorldModel

    return LightGBMQuantileWorldModel


REGISTRY: dict[str, Callable[[], type[WorldModel]]] = {
    # Canonical names
    "causal_transformer": _load_causal_transformer,
    "lightgbm_quantile": _load_lightgbm,
    # Aliases
    "transformer": _load_causal_transformer,  # back-compat
    "transformer_wm": _load_causal_transformer,
    "ct_wm": _load_causal_transformer,
    "lgbm": _load_lightgbm,
    "lightgbm": _load_lightgbm,
}


def get_world_model(name: str, **kwargs: Any) -> WorldModel:
    """Instantiate a world model by name.

    Parameters
    ----------
    name
        One of ``"causal_transformer"``, ``"lightgbm_quantile"``, or a
        registered alias. ``"transformer"`` remains a supported alias and
        resolves to the causal variant.
    kwargs
        Forwarded to the model's constructor (typically the config dataclass).
    """
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown world model '{name}'. Available: {sorted(REGISTRY)}") from None
    cls = factory()
    return cls(**kwargs)


def list_world_models() -> list[str]:
    """Return canonical names (drops aliases)."""
    return ["causal_transformer", "lightgbm_quantile"]
