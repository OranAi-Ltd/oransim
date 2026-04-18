"""Abstract base for world models.

A :class:`WorldModel` maps a bundle of campaign features (creative, platform,
KOLs, budget, targeting) to a distribution over funnel KPIs. The interface
intentionally returns three quantile estimates (P35, P50, P65) rather than a
point estimate — calibrated uncertainty is first-class in Oransim.

Implementations:

- :class:`~oransim.world_model.transformer.TransformerWorldModel`
- :class:`~oransim.world_model.lightgbm_quantile.LightGBMQuantileWorldModel`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

KPI_NAMES = ("impressions", "clicks", "conversions", "revenue")
DEFAULT_QUANTILES = (0.35, 0.50, 0.65)


@dataclass
class WorldModelConfig:
    """Shared config fields across world models."""

    kpis: tuple[str, ...] = KPI_NAMES
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    platform_id: str = "xhs"
    feature_version: str = "v1.1"
    seed: int = 42


@dataclass
class WorldModelPrediction:
    """Structured return value of :meth:`WorldModel.predict`.

    Attributes
    ----------
    kpi_quantiles
        Nested mapping ``kpi_name -> quantile_level -> predicted_value``.
        Example: ``{"impressions": {0.35: 82000.0, 0.50: 105000.0, 0.65: 128000.0}}``
    latent
        Optional model-specific debug payload (attention weights, feature
        importances, calibration stats). Safe to drop in production.
    """

    kpi_quantiles: dict[str, dict[float, float]]
    latent: dict[str, Any] = field(default_factory=dict)

    def point_estimate(self, kpi: str, quantile: float = 0.50) -> float:
        return self.kpi_quantiles[kpi][quantile]

    def interval(self, kpi: str, lo: float = 0.35, hi: float = 0.65) -> tuple[float, float]:
        return self.kpi_quantiles[kpi][lo], self.kpi_quantiles[kpi][hi]


class WorldModel(ABC):
    """Abstract world model interface.

    Implementations must be picklable or provide their own persistence via
    :meth:`save` / :meth:`load_pretrained`.
    """

    config: WorldModelConfig

    @abstractmethod
    def predict(self, features: dict[str, Any]) -> WorldModelPrediction:
        """Forward pass — produce a prediction for one campaign scenario.

        Parameters
        ----------
        features
            Dictionary of canonical features. Schema defined in
            :mod:`oransim.data.schema` (lands with v0.2).
        """

    @abstractmethod
    def fit(
        self,
        dataset: Iterable[dict[str, Any]],
        *,
        val_dataset: Iterable[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train or fit the model on a dataset of scenario records.

        Returns a metrics dict (loss, val_loss, R² per KPI, calibration error).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to ``path``."""

    @classmethod
    @abstractmethod
    def load_pretrained(cls, path: str | None = None, **kwargs: Any) -> WorldModel:
        """Load a pretrained checkpoint.

        If ``path`` is ``None``, implementations may try the default checkpoint
        bundled with a release. If the bundled checkpoint is unavailable
        (weights not yet released), raise :class:`FileNotFoundError` with a
        clear message pointing the user at the release page or
        ``scripts/train_*.py`` to train locally.
        """

    def describe(self) -> dict[str, Any]:
        """Human-readable summary — name, version, dimensions, status."""
        return {
            "name": self.__class__.__name__,
            "config": self.config.__dict__,
            "kpis": list(self.config.kpis),
            "quantiles": list(self.config.quantiles),
        }
