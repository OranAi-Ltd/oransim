"""Platform Recommender Surrogate (PRS) stub for Instagram Reels.

Mirrors :class:`oransim.platforms.tiktok.prs.TikTokPRS`. No trained pkl
ships in v0.2 OSS; callers should check ``is_ready()`` and fall through
to :class:`InstagramWorldModel` for structural predictions.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

V25_QUANTILE_PATH = Path("data/models/instagram_world_model_v25_quantile.pkl")
V2_PCA_PATH = Path("data/models/instagram_world_model_v2_pca.pkl")


class InstagramPRS:
    """Instagram PRS stub — same interface as TikTokPRS / XHSPRS."""

    def __init__(self) -> None:
        self.model = None
        self.models = None
        self.quantile_models = None
        self.target_names: list[str] = []
        self.niche_list: list[str] = []
        self.top_topics: list[str] = []
        self.feature_dim = 0
        self.n_training = 0
        self.eval_cv: dict = {}
        self.version = "not-loaded"
        self.pca = None
        self.pca_dim = 0
        self._load()

    def _load(self) -> None:
        for path in (V25_QUANTILE_PATH, V2_PCA_PATH):
            if path.exists():
                with open(path, "rb") as f:
                    blob = pickle.load(f)
                slot = "quantile_models" if path is V25_QUANTILE_PATH else "models"
                setattr(self, slot, blob["models"])
                self.version = blob.get("version", path.stem)
                self.target_names = blob["target_names"]
                self.niche_list = blob["niche_list"]
                self.top_topics = blob.get("top_topics", [])
                self.feature_dim = blob["feature_dim"]
                self.n_training = blob.get("n_training", 0)
                self.eval_cv = blob.get("eval_cv", {})
                self.pca = blob.get("pca")
                self.pca_dim = blob.get("pca_dim", 0)
                return

    def is_ready(self) -> bool:
        return self.quantile_models is not None or self.models is not None

    def info(self) -> dict:
        if not self.is_ready():
            return {
                "loaded": False,
                "reason": (
                    "Instagram PRS pkl not shipped in OSS build. "
                    "Fall through to InstagramWorldModel for structural predictions."
                ),
            }
        return {
            "loaded": True,
            "version": self.version,
            "n_training": self.n_training,
            "targets": self.target_names,
            "niches": self.niche_list,
            "feature_dim": self.feature_dim,
            "n_topics": len(self.top_topics),
            "eval_cv_r2": {
                k: round(v.get("r2", v.get("r2_p50", 0)), 3) for k, v in self.eval_cv.items()
            },
        }

    def predict(
        self,
        caption_emb: np.ndarray,
        author_fans: int,
        niche: str,
        duration_sec: float = 22.0,
        post_time: datetime | None = None,
        desc_emb: np.ndarray | None = None,
        topics: list | None = None,
        has_img: bool = False,
        img_count: int = 0,
        post_type: str = "reels",
    ) -> dict:
        if not self.is_ready():
            return {}
        raise NotImplementedError(
            "InstagramPRS pkl loaded but predict() not wired. "
            "Port XHSPRS.predict feature-build here when shipping."
        )


PRS = InstagramPRS()
