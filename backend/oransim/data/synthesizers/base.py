"""Abstract base for population synthesizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SynthesizerConfig:
    """Shared configuration fields across synthesizers."""

    target_marginals: dict[str, list[float]] = field(default_factory=dict)
    seed: int = 42
    region: str = "CN"


@dataclass
class SynthesizedPopulation:
    """Minimal return type so a synthesizer can be swapped without callers
    needing to know whether the backing implementation is IPF, Bayes net,
    TabDDPM, or something else.
    """

    N: int
    attributes: dict[str, Any] = field(default_factory=dict)
    latent: dict[str, Any] = field(default_factory=dict)


class PopulationSynthesizer(ABC):
    """Abstract population synthesizer.

    Implementations should be deterministic given the same ``seed``.
    """

    config: SynthesizerConfig

    @abstractmethod
    def generate(self, N: int, *, seed: int | None = None, **kwargs: Any) -> SynthesizedPopulation:
        """Draw ``N`` virtual consumers matching the configured targets."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.__class__.__name__, "config": self.config.__dict__}
