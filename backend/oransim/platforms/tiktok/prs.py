"""Platform Recommender Surrogate (PRS) stub for TikTok.

Unlike the XHS PRS — which ships a trained LightGBM pkl because the OSS
repo includes a synthetic-but-calibrated v0.2 training pipeline for
Xiaohongshu — the TikTok PRS currently has **no trained pkl** in the
public release. Reasons:

  1. The FYP signal (completion rate, watch-time percentile, re-watch
     probability) is materially different from XHS's like/collect/comment
     observable set, so a direct pkl port would misrepresent TikTok.
  2. The OSS repo does not ship a TikTok training-data ingestion script
     (upstream vendor's data contract is out of scope for the public
     release). Without real watch-time histograms the quantile targets
     we'd train against are synthetic-on-synthetic.

Until a TikTok-specific training pipeline lands, this module exposes
an XHSPRS-shaped stub so downstream code can branch uniformly on
``is_ready()``. When ``is_ready()`` returns False, callers should fall
through to :class:`TikTokWorldModel` for structural predictions.

When a real pkl arrives, drop it at ``data/models/tiktok_world_model_*.pkl``
and enrich :meth:`TikTokPRS._load` to mirror
:class:`oransim.platforms.xhs.prs.XHSPRS`.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

V25_QUANTILE_PATH = Path("data/models/tiktok_world_model_v25_quantile.pkl")
V2_PCA_PATH = Path("data/models/tiktok_world_model_v2_pca.pkl")


class TikTokPRS:
    """TikTok PRS stub. Loads a pkl if present; otherwise stays not-ready."""

    def __init__(self):
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

    def _load(self):
        if V25_QUANTILE_PATH.exists():
            with open(V25_QUANTILE_PATH, "rb") as f:
                blob = pickle.load(f)
            self.quantile_models = blob["models"]
            self.version = blob.get("version", "v2.5-quantile")
            self.target_names = blob["target_names"]
            self.niche_list = blob["niche_list"]
            self.top_topics = blob.get("top_topics", [])
            self.feature_dim = blob["feature_dim"]
            self.n_training = blob.get("n_training", 0)
            self.eval_cv = blob.get("eval_cv", {})
            self.pca = blob.get("pca")
            self.pca_dim = blob.get("pca_dim", 0)
            return
        if V2_PCA_PATH.exists():
            with open(V2_PCA_PATH, "rb") as f:
                blob = pickle.load(f)
            self.models = blob["models"]
            self.version = blob.get("version", "v2-pca")
            self.target_names = blob["target_names"]
            self.niche_list = blob["niche_list"]
            self.top_topics = blob.get("top_topics", [])
            self.feature_dim = blob["feature_dim"]
            self.n_training = blob.get("n_training", 0)
            self.eval_cv = blob.get("eval_cv", {})
            self.pca = blob.get("pca")
            self.pca_dim = blob.get("pca_dim", 0)

    def is_ready(self) -> bool:
        return self.quantile_models is not None or self.models is not None

    def info(self) -> dict:
        if not self.is_ready():
            return {
                "loaded": False,
                "reason": (
                    "TikTok PRS pkl not shipped in OSS build. "
                    "Fall through to TikTokWorldModel for structural predictions."
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
        duration_sec: float = 20.0,
        post_time: datetime | None = None,
        desc_emb: np.ndarray | None = None,
        topics: list | None = None,
        has_img: bool = False,
        img_count: int = 0,
        post_type: str = "video",
    ) -> dict:
        # Until a TikTok pkl ships, return empty so callers fall back.
        if not self.is_ready():
            return {}
        # When a pkl arrives, mirror XHSPRS.predict. Left as a stub here
        # deliberately — we want the bug to surface loudly rather than
        # silently returning miscalibrated numbers.
        raise NotImplementedError(
            "TikTokPRS pkl loaded but predict() not wired. "
            "Port XHSPRS.predict feature-build here when shipping."
        )


PRS = TikTokPRS()
