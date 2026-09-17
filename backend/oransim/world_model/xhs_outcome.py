"""Inference from released numeric XHS outcome parameters and prepared features."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


class XHSOutcomeModel:
    """Load native LightGBM trees and PCA arrays without training records."""

    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.models = {
            name: lgb.Booster(model_file=str(directory / f"{name}.lgb.txt"))
            for name in self.metadata["target_names"]
        }
        with np.load(directory / "pca.npz", allow_pickle=False) as pca:
            self.components = pca["components"].copy()
            self.mean = pca["mean"].copy()

    def prepare_features(self, text_embeddings, hand_features):
        """Project original-space title/description embeddings and append hand features.

        Callers supply the original numeric feature encoding. This class does not
        recreate the undistributed embedding service or learned topic vocabulary.
        """
        text = np.asarray(text_embeddings)
        hand = np.asarray(hand_features)
        if text.ndim != 2 or text.shape[1] != len(self.mean):
            raise ValueError(f"Expected a matrix with {len(self.mean)} embedding columns")
        expected = self.metadata["feature_dim"] - self.metadata["pca_dim"]
        if hand.shape != (len(text), expected):
            raise ValueError(f"Expected hand feature shape {(len(text), expected)}")
        projected = text @ self.components.T - self.mean @ self.components.T
        return np.concatenate([projected, hand], axis=1).astype(np.float32)

    def predict(self, features):
        """Predict counts/ratios from features; read_pct retains the identity scale."""
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.metadata["feature_dim"]:
            raise ValueError(f'Expected {self.metadata["feature_dim"]} feature columns')
        if not np.isfinite(x).all():
            raise ValueError("Features must be finite")
        result = {}
        for name, model in self.models.items():
            value = model.predict(x, num_threads=1)
            if self.metadata["target_transforms"][name] == "log1p":
                value = np.maximum(0, np.expm1(value))
            result[name] = value
        return result
