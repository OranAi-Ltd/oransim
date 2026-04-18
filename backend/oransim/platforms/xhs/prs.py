"""Platform Recommender Surrogate (PRS) for XHS.

Loads the LightGBM quantile world model trained on 10k synthetic XHS notes (v0.2+).
Given a creative + author fans + niche, predicts (exp, read, like, coll, comm).

This replaces the hand-coded structural formula in PlatformWorldModel with
empirically-calibrated predictions from real data.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np


V25_QUANTILE_PATH = Path("data/models/xhs_world_model_v25_quantile.pkl")
V2_PCA_PATH = Path("data/models/xhs_world_model_v2_pca.pkl")
V2_PATH = Path("data/models/xhs_world_model_v2.pkl")
V1_PATH = Path("data/models/xhs_world_model.pkl")


class XHSPRS:
    def __init__(self):
        self.model = None            # v1 single model or None
        self.models = None           # v2 dict of per-target models
        self.target_names = []
        self.niche_list = []
        self.top_topics = []
        self.feature_dim = 0
        self.n_training = 0
        self.eval_cv = {}
        self.version = "unknown"
        self._load()

    def _load(self):
        # Priority: v2.5-quantile > v2-pca > v2 > v1
        if V25_QUANTILE_PATH.exists():
            with open(V25_QUANTILE_PATH, "rb") as f:
                blob = pickle.load(f)
            self.quantile_models = blob["models"]   # {target: {p10, p50, p90}}
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
            return
        if V2_PATH.exists():
            with open(V2_PATH, "rb") as f:
                blob = pickle.load(f)
            self.models = blob["models"]
            self.version = blob.get("version", "v2")
            self.target_names = blob["target_names"]
            self.niche_list = blob["niche_list"]
            self.top_topics = blob.get("top_topics", [])
            self.feature_dim = blob["feature_dim"]
            self.n_training = blob.get("n_training", 0)
            self.eval_cv = blob.get("eval_cv", {})
            return
        if V1_PATH.exists():
            with open(V1_PATH, "rb") as f:
                blob = pickle.load(f)
            self.model = blob["model"]
            self.version = "v1"
            self.target_names = blob["target_names"]
            self.niche_list = blob["niche_list"]
            self.feature_dim = blob["feature_dim"]
            self.n_training = blob.get("n_training", 0)
            self.eval_cv = blob.get("eval_cv", {})

    quantile_models = None   # default unset

    def is_ready(self) -> bool:
        return (self.model is not None or self.models is not None
                or getattr(self, 'quantile_models', None) is not None)

    def info(self) -> Dict:
        if not self.is_ready():
            return {"loaded": False}
        return {
            "loaded": True,
            "version": self.version,
            "n_training": self.n_training,
            "targets": self.target_names,
            "niches": self.niche_list,
            "feature_dim": self.feature_dim,
            "n_topics": len(self.top_topics),
            "eval_cv_r2": {k: round(v.get("r2", v.get("r2_p50", 0)), 3)
                            for k, v in self.eval_cv.items()},
        }

    def _project_emb(self, emb, dim=768):
        if len(emb) == dim: return emb
        return (emb[:dim] if len(emb) > dim
                else np.concatenate([emb, np.zeros(dim - len(emb), np.float32)]))

    def predict(self, caption_emb: np.ndarray, author_fans: int,
                niche: str, duration_sec: float = 15.0,
                post_time: Optional[datetime] = None,
                desc_emb: Optional[np.ndarray] = None,
                topics: Optional[list] = None,
                has_img: bool = False,
                img_count: int = 0,
                post_type: str = "video") -> Dict:
        """Returns dict of {exp, read, like, coll, comm}."""
        if not self.is_ready():
            return {}
        if post_time is None:
            post_time = datetime.now()

        caption_emb = self._project_emb(caption_emb)

        # v2.5 quantile: returns P10/P50/P90 for each target
        if self.version == "v2.5-quantile":
            de = self._project_emb(desc_emb if desc_emb is not None else caption_emb)
            X_text = np.concatenate([caption_emb, de]).reshape(1, -1)
            X_pca = self.pca.transform(X_text)
            n_topics = len(self.top_topics)
            hand = np.zeros(3 + len(self.niche_list) + n_topics + 6, dtype=np.float32)
            col = 0
            hand[col] = np.log1p(max(author_fans, 0)); col += 1
            hand[col] = min(duration_sec, 600) / 60.0; col += 1
            hand[col] = 1.0 if duration_sec > 0 else 0.0; col += 1
            if niche in self.niche_list:
                hand[col + self.niche_list.index(niche)] = 1.0
            col += len(self.niche_list)
            topic_set = set(topics or [])
            for ti, t in enumerate(self.top_topics):
                if t in topic_set:
                    hand[col + ti] = 1.0
            col += n_topics
            h, wd = post_time.hour, post_time.weekday()
            hand[col]     = np.sin(2*np.pi*h/24); col += 1
            hand[col]     = np.cos(2*np.pi*h/24); col += 1
            hand[col]     = np.sin(2*np.pi*wd/7); col += 1
            hand[col]     = np.cos(2*np.pi*wd/7); col += 1
            hand[col]     = min(img_count, 20) / 20.0; col += 1
            hand[col]     = 0.0 if post_type == "video" else 1.0
            X = np.concatenate([X_pca, hand.reshape(1, -1)], axis=1).astype(np.float32)
            out = {}
            for name, qm in self.quantile_models.items():
                # 兼容新旧键：新用 p_low/p_high, 旧用 p10/p90
                k_lo = "p_low" if "p_low" in qm else "p10"
                k_hi = "p_high" if "p_high" in qm else "p90"
                lo = float(max(0, np.expm1(qm[k_lo].predict(X)[0])))
                p50 = float(max(0, np.expm1(qm["p50"].predict(X)[0])))
                hi = float(max(0, np.expm1(qm[k_hi].predict(X)[0])))
                out[name] = p50
                # New canonical keys
                out[f"{name}_low"] = lo
                out[f"{name}_p50"] = p50
                out[f"{name}_high"] = hi
                # Legacy aliases kept for API compat
                out[f"{name}_p10"] = lo
                out[f"{name}_p90"] = hi
            return out

        # v2-pca: project (title + desc) via PCA then concat hand features
        if self.version == "v2-pca":
            de = self._project_emb(desc_emb if desc_emb is not None else caption_emb)
            X_text = np.concatenate([caption_emb, de]).reshape(1, -1)
            X_pca = self.pca.transform(X_text)
            # Build hand features (matches train_world_model_v2_pca.py)
            n_topics = len(self.top_topics)
            hand = np.zeros(3 + len(self.niche_list) + n_topics + 6, dtype=np.float32)
            col = 0
            hand[col] = np.log1p(max(author_fans, 0)); col += 1
            hand[col] = min(duration_sec, 600) / 60.0; col += 1
            hand[col] = 1.0 if duration_sec > 0 else 0.0; col += 1
            if niche in self.niche_list:
                hand[col + self.niche_list.index(niche)] = 1.0
            col += len(self.niche_list)
            topic_set = set(topics or [])
            for ti, t in enumerate(self.top_topics):
                if t in topic_set:
                    hand[col + ti] = 1.0
            col += n_topics
            h, wd = post_time.hour, post_time.weekday()
            hand[col]     = np.sin(2*np.pi*h/24); col += 1
            hand[col]     = np.cos(2*np.pi*h/24); col += 1
            hand[col]     = np.sin(2*np.pi*wd/7); col += 1
            hand[col]     = np.cos(2*np.pi*wd/7); col += 1
            hand[col]     = min(img_count, 20) / 20.0; col += 1
            hand[col]     = 0.0 if post_type == "video" else 1.0
            X = np.concatenate([X_pca, hand.reshape(1, -1)], axis=1).astype(np.float32)
            out = {}
            for name, m in self.models.items():
                v = np.expm1(m.predict(X)[0])
                out[name] = float(max(0, v))
            return out

        if self.version == "v2":
            de = self._project_emb(desc_emb if desc_emb is not None else caption_emb)
            n_topics = len(self.top_topics)
            hand = np.zeros(3 + len(self.niche_list) + n_topics + 6, dtype=np.float32)
            col = 0
            hand[col] = np.log1p(max(author_fans, 0)); col += 1
            hand[col] = min(duration_sec, 600) / 60.0; col += 1
            hand[col] = 1.0 if duration_sec > 0 else 0.0; col += 1
            if niche in self.niche_list:
                hand[col + self.niche_list.index(niche)] = 1.0
            col += len(self.niche_list)
            topic_set = set(topics or [])
            for ti, t in enumerate(self.top_topics):
                if t in topic_set:
                    hand[col + ti] = 1.0
            col += n_topics
            h = post_time.hour; wd = post_time.weekday()
            hand[col]     = np.sin(2*np.pi*h/24); col += 1
            hand[col]     = np.cos(2*np.pi*h/24); col += 1
            hand[col]     = np.sin(2*np.pi*wd/7); col += 1
            hand[col]     = np.cos(2*np.pi*wd/7); col += 1
            hand[col]     = min(img_count, 20) / 20.0; col += 1
            hand[col]     = 0.0 if post_type == "video" else 1.0
            X = np.concatenate([caption_emb, de, hand]).reshape(1, -1).astype(np.float32)
            out = {}
            for name, m in self.models.items():
                v = np.expm1(m.predict(X)[0])
                out[name] = float(max(0, v))
            return out
        # v1 fallback
        hand = np.zeros(3 + len(self.niche_list) + 2, dtype=np.float32)
        hand[0] = np.log1p(max(author_fans, 0))
        hand[1] = min(duration_sec, 600) / 60.0
        hand[2] = 1.0 if duration_sec > 0 else 0.0
        if niche in self.niche_list:
            hand[3 + self.niche_list.index(niche)] = 1.0
        hand[3 + len(self.niche_list)] = post_time.hour / 23.0
        hand[3 + len(self.niche_list) + 1] = post_time.weekday() / 6.0
        X = np.concatenate([caption_emb, hand]).reshape(1, -1).astype(np.float32)
        log_pred = self.model.predict(X)[0]
        if np.isscalar(log_pred): log_pred = np.array([log_pred])
        vals = np.expm1(log_pred)
        return {name: float(max(0, v)) for name, v in zip(self.target_names, vals)}


# Singleton
PRS = XHSPRS()
