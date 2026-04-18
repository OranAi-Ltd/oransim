"""LightGBM quantile regression world model — fast baseline.

Kept as the default zero-dependency fallback when PyTorch is unavailable or
sub-millisecond inference is required. Trains three gradient-boosted quantile
regressors per KPI (P35/P50/P65).

References
----------

- Ke et al. 2017 — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*
- Koenker 2005 — *Quantile Regression*

Status
------

Reference implementation shipping in v0.2 (code migrates from the internal
prototype that produced the R² numbers reported in the README benchmarks
table). The API below is the public-facing contract; implementation details
will land alongside the synthetic data generator in Phase 3.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    WorldModel,
    WorldModelConfig,
    WorldModelPrediction,
)


@dataclass
class LightGBMWMConfig(WorldModelConfig):
    """Configuration for :class:`LightGBMQuantileWorldModel`."""

    n_estimators: int = 300
    max_depth: int = 7
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    pca_n_components: int = 32  # 0 disables PCA
    feature_subsample: float = 0.9
    row_subsample: float = 0.9
    reg_lambda: float = 0.1
    checkpoint_dir: str = "data/models/lightgbm_quantile"
    pretrained_url: str = "coming_soon"


def _require_lightgbm() -> Any:
    try:
        import lightgbm  # noqa: F401

        return __import__("lightgbm")
    except ImportError as exc:
        raise ImportError(
            "LightGBMQuantileWorldModel requires lightgbm. "
            "Install with: pip install lightgbm\n"
            "Original error: " + str(exc)
        ) from exc


class LightGBMQuantileWorldModel(WorldModel):
    """LightGBM quantile regression baseline.

    Trains ``len(kpis) × len(quantiles)`` independent boosters. Inference is
    a dict-lookup + N independent tree evaluations (sub-millisecond on CPU).

    This is the production default until the Transformer world model has
    trained weights released in v0.2+.
    """

    def __init__(self, config: LightGBMWMConfig | None = None):
        self.config = config or LightGBMWMConfig()
        self._lgb = _require_lightgbm()
        # nested map: kpi -> quantile -> Booster
        self._boosters: dict[str, dict[float, Any]] = {}
        self._pca: Any | None = None

    # ---------------------------------------------------------------- predict

    def predict(self, features: dict[str, Any]) -> WorldModelPrediction:
        if not self._boosters:
            raise RuntimeError(
                "LightGBMQuantileWorldModel has no trained boosters. "
                "Either call fit() or load_pretrained(path)."
            )

        x = self._featurize(features)
        out: dict[str, dict[float, float]] = {}
        for kpi_name in self.config.kpis:
            out[kpi_name] = {}
            for q in self.config.quantiles:
                booster = self._boosters[kpi_name][q]
                pred = booster.predict(x.reshape(1, -1))[0]
                out[kpi_name][q] = float(pred)
        return WorldModelPrediction(kpi_quantiles=out, latent={"backend": "lightgbm"})

    def _featurize(self, features: dict[str, Any]) -> Any:
        """Flatten canonical features into a feature vector.

        Phase 3 ships the full feature extraction; for now this is a
        placeholder that concatenates numerical fields.
        """
        import numpy as np

        parts = []
        for key in ("creative_embed", "kol_feat", "demo_feat", "time_feat"):
            if key in features:
                parts.append(np.asarray(features[key]).ravel())
        if "platform_id" in features:
            parts.append(np.asarray([float(features["platform_id"])]))
        if "budget" in features:
            parts.append(np.asarray([float(features["budget"])]).ravel())
        x = np.concatenate(parts).astype(np.float32)
        if self._pca is not None:
            x = self._pca.transform(x.reshape(1, -1)).ravel()
        return x

    # --------------------------------------------------------------- training

    def fit(
        self,
        dataset: Iterable[dict[str, Any]],
        *,
        val_dataset: Iterable[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train one quantile booster per (kpi, quantile) pair.

        For the full training harness (PCA fit, feature pipeline, early
        stopping, calibration eval), see
        ``scripts/train_lightgbm_quantile.py`` landing with v0.2.
        """
        lgb = self._lgb
        import numpy as np

        X_rows = []
        y_rows: dict[str, list[float]] = {k: [] for k in self.config.kpis}
        for sample in dataset:
            feats = {k: v for k, v in sample.items() if k != "targets"}
            X_rows.append(self._featurize(feats))
            for k in self.config.kpis:
                y_rows[k].append(float(sample["targets"][k]))
        X = np.stack(X_rows)

        if self.config.pca_n_components > 0 and X.shape[1] > self.config.pca_n_components:
            try:
                from sklearn.decomposition import PCA

                self._pca = PCA(n_components=self.config.pca_n_components)
                X = self._pca.fit_transform(X)
            except ImportError:
                self._pca = None

        history = {"train_loss": {k: [] for k in self.config.kpis}}
        for kpi_name in self.config.kpis:
            y = np.asarray(y_rows[kpi_name])
            self._boosters[kpi_name] = {}
            for q in self.config.quantiles:
                params = {
                    "objective": "quantile",
                    "alpha": float(q),
                    "num_leaves": self.config.num_leaves,
                    "learning_rate": self.config.learning_rate,
                    "feature_fraction": self.config.feature_subsample,
                    "bagging_fraction": self.config.row_subsample,
                    "bagging_freq": 5,
                    "min_child_samples": self.config.min_child_samples,
                    "lambda_l2": self.config.reg_lambda,
                    "max_depth": self.config.max_depth,
                    "verbose": -1,
                }
                dtrain = lgb.Dataset(X, y)
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.config.n_estimators,
                )
                self._boosters[kpi_name][q] = booster
                history["train_loss"][kpi_name].append(
                    {"quantile": q, "best_iter": booster.best_iteration or self.config.n_estimators}
                )
        return history

    # ---------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Serialize booster strings instead of pickling raw objects for portability
        dumped = {
            "config": self.config.__dict__,
            "boosters": {
                kpi: {str(q): bst.model_to_string() for q, bst in per_q.items()}
                for kpi, per_q in self._boosters.items()
            },
            "pca": pickle.dumps(self._pca) if self._pca is not None else None,
        }
        with open(path, "wb") as f:
            pickle.dump(dumped, f)

    @classmethod
    def load_pretrained(cls, path: str | None = None, **kwargs: Any) -> LightGBMQuantileWorldModel:
        if path is None:
            raise FileNotFoundError(
                "No bundled LightGBMQuantileWorldModel weights in v0.2.0-alpha.\n"
                "Options:\n"
                "  1. Train locally: python -m backend.scripts.train_lightgbm_quantile --config default\n"
                "  2. Watch v0.2 release for a pretrained pkl"
            )
        with open(path, "rb") as f:
            blob = pickle.load(f)
        cfg = LightGBMWMConfig(**blob["config"])
        model = cls(cfg)
        lgb = model._lgb
        for kpi, per_q in blob["boosters"].items():
            model._boosters[kpi] = {float(q): lgb.Booster(model_str=s) for q, s in per_q.items()}
        if blob.get("pca") is not None:
            model._pca = pickle.loads(blob["pca"])
        return model
