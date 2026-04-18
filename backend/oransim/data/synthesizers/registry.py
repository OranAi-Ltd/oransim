"""Registry for population synthesizers.

Bayesian-network, TabDDPM, and causal-DAG-guided variants are reserved
through the registry but raise :class:`NotImplementedError` until they land
with the corresponding roadmap milestone.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import PopulationSynthesizer


def _load_ipf() -> type[PopulationSynthesizer]:
    from .ipf import IPFSynthesizer
    return IPFSynthesizer


def _load_bayes_net() -> type[PopulationSynthesizer]:
    from .bayes_net import BayesianNetworkSynthesizer
    return BayesianNetworkSynthesizer


def _not_yet(name: str, milestone: str) -> Callable[[], type[PopulationSynthesizer]]:
    class _PendingSynthesizer(PopulationSynthesizer):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise NotImplementedError(
                f"{name} synthesizer is on the roadmap for {milestone}. "
                "See https://github.com/OranAi-Ltd/oransim/blob/main/ROADMAP.md. "
                "Fall back: get_synthesizer('ipf')."
            )

        def generate(self, N: int, *, seed: int | None = None, **kwargs: Any) -> Any:
            raise NotImplementedError

    def _factory() -> type[PopulationSynthesizer]:
        return _PendingSynthesizer

    return _factory


REGISTRY: dict[str, Callable[[], type[PopulationSynthesizer]]] = {
    "ipf": _load_ipf,
    "bayes_net":          _load_bayes_net,
    "tabddpm":            _not_yet("TabDDPM (tabular diffusion)",            "v0.5"),
    "causal_dag_tabddpm": _not_yet("Causal-DAG-guided TabDDPM",              "v1.0 (research)"),
    "ctgan":              _not_yet("CTGAN",                                  "v0.5"),
}


def get_synthesizer(name: str, **kwargs: Any) -> PopulationSynthesizer:
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown synthesizer '{name}'. Available: {sorted(REGISTRY)}"
        ) from None
    return factory()(**kwargs)


def list_synthesizers() -> list[str]:
    return ["ipf", "bayes_net", "tabddpm", "causal_dag_tabddpm", "ctgan"]
