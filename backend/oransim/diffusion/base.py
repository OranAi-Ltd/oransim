"""Abstract base for diffusion forecasting models.

A :class:`DiffusionModel` predicts the 14-day time-series of cascading
engagement events (impressions / reshares / conversions) triggered by a
marketing launch. Every implementation must support:

- :meth:`forecast` — forward simulation of the expected event stream
- :meth:`log_likelihood` — negative log-likelihood on a real event stream
- :meth:`counterfactual_forecast` — same as :meth:`forecast` but under a
  ``do()`` intervention (e.g., "what if we had stopped boosting on day 3")

Implementations:

- :class:`~oransim.diffusion.neural_hawkes.CausalNeuralHawkesProcess`
- :class:`~oransim.diffusion.hawkes.ParametricHawkes`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


DEFAULT_EVENT_TYPES = ("impression", "like", "comment", "share", "save", "conversion")


@dataclass
class DiffusionConfig:
    """Shared config across diffusion models."""

    horizon_days: int = 14
    event_types: tuple[str, ...] = DEFAULT_EVENT_TYPES
    resolution_minutes: int = 60  # bucket granularity for forecasts
    platform_id: str = "xhs"
    seed: int = 42


@dataclass
class DiffusionForecast:
    """Structured result of :meth:`DiffusionModel.forecast`.

    Attributes
    ----------
    timeline
        List of ``(timestamp_minutes, event_type, intensity)`` triples over
        the configured horizon.
    per_type_totals
        Aggregate event count by type over the forecast window.
    daily_buckets
        ``[horizon_days, n_event_types]`` — event counts per day per type.
    latent
        Model-specific debug payload (attention weights, confidence bands).
    """

    timeline: list[tuple[float, str, float]]
    per_type_totals: dict[str, float]
    daily_buckets: list[list[float]]
    latent: dict[str, Any] = field(default_factory=dict)


class DiffusionModel(ABC):
    """Abstract diffusion forecaster interface."""

    config: DiffusionConfig

    @abstractmethod
    def forecast(
        self, seed_events: Iterable[tuple[float, str]], **kwargs: Any
    ) -> DiffusionForecast:
        """Simulate the expected cascade given the seed event stream."""

    @abstractmethod
    def counterfactual_forecast(
        self,
        seed_events: Iterable[tuple[float, str]],
        *,
        intervention: dict[str, Any],
        **kwargs: Any,
    ) -> DiffusionForecast:
        """Simulate the cascade under a ``do()`` intervention.

        ``intervention`` keys are model-specific — e.g., ``{"mute_at_min": 4320}``
        to stop boosting 3 days in, or ``{"treatment_arm": 2}`` to switch the
        treatment schedule.
        """

    @abstractmethod
    def log_likelihood(self, events: Iterable[tuple[float, str]]) -> float:
        """Log-likelihood of a realised event stream under the model."""

    @abstractmethod
    def fit(
        self,
        dataset: Iterable[Iterable[tuple[float, str]]],
        *,
        val_dataset: Iterable[Iterable[tuple[float, str]]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model on event streams."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load_pretrained(cls, path: str | None = None, **kwargs: Any) -> "DiffusionModel":
        """Load pretrained weights or raise :class:`FileNotFoundError`."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "horizon_days": self.config.horizon_days,
            "event_types": list(self.config.event_types),
            "resolution_minutes": self.config.resolution_minutes,
        }
