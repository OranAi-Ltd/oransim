"""Real text embedder via OpenAI-compatible API (reusing OpenAI / DeepSeek / Qwen / Anthropic / Gemini).

Replaces hash mock embedders with actual learned embeddings.
Uses the same LLM gateway as soul_llm.py — no new API key needed.
"""
from __future__ import annotations
import os
import json
import hashlib
import urllib.request
import urllib.error
import numpy as np
from typing import List, Optional

from .embedding_bus import Embedder, EMB_DIM


# Reuse the same env vars as soul_llm.py — one key for everything
EMB_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
EMB_API_KEY = os.environ.get("LLM_API_KEY", "")
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")  # 1536-d native
EMB_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))


# Local cache to save API $$ on repeated texts
_CACHE: dict[str, np.ndarray] = {}
_CACHE_MAX = 5000


def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()[:24]


def _embed_text_api(text: str, model: str = EMB_MODEL) -> Optional[np.ndarray]:
    """Call the embeddings endpoint. Returns None on failure."""
    if not EMB_API_KEY:
        return None
    key = _cache_key(text, model)
    if key in _CACHE:
        return _CACHE[key]

    body = {"model": model, "input": text}
    req = urllib.request.Request(
        f"{EMB_BASE_URL}/embeddings",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {EMB_API_KEY}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=EMB_TIMEOUT) as r:
            resp = json.loads(r.read().decode())
        vec = np.asarray(resp["data"][0]["embedding"], dtype=np.float32)
        # project to EMB_DIM if needed
        if len(vec) != EMB_DIM:
            if len(vec) > EMB_DIM:
                vec = vec[:EMB_DIM]
            else:
                vec = np.concatenate([vec, np.zeros(EMB_DIM - len(vec), np.float32)])
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        if len(_CACHE) < _CACHE_MAX:
            _CACHE[key] = vec
        return vec
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError):
        return None


class RealTextEmbedder(Embedder):
    """Real text embedder — OpenAI-compatible endpoint.

    Falls back to hash mock if API unavailable (graceful degradation).
    """
    name = "openai-compat-text"
    modality = "text"

    def __init__(self, model: str = EMB_MODEL, dim: int = EMB_DIM, fallback_seed: int = 0):
        self.model = model
        self.output_dim = dim
        self.fallback_seed = fallback_seed
        self._fallback_hits = 0
        self._api_hits = 0

    def embed(self, item) -> np.ndarray:
        text = str(item)
        vec = _embed_text_api(text, self.model)
        if vec is not None:
            self._api_hits += 1
            return vec
        # Fallback to deterministic hash mock
        self._fallback_hits += 1
        h = hashlib.sha256(
            (text + str(self.fallback_seed)).encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        v = rng.normal(0, 1, self.output_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def embed_batch(self, items: List) -> np.ndarray:
        """Batch via the batch endpoint (OpenAI supports arrays)."""
        if not EMB_API_KEY or not items:
            return super().embed_batch(items)
        # split into chunks of 100 (OpenAI limit varies)
        chunks = [items[i:i+64] for i in range(0, len(items), 64)]
        all_vecs: List[np.ndarray] = []
        for chunk in chunks:
            body = {"model": self.model,
                    "input": [str(x) for x in chunk]}
            req = urllib.request.Request(
                f"{EMB_BASE_URL}/embeddings",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {EMB_API_KEY}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=EMB_TIMEOUT) as r:
                    resp = json.loads(r.read().decode())
                for d in resp["data"]:
                    vec = np.asarray(d["embedding"], dtype=np.float32)
                    if len(vec) != self.output_dim:
                        vec = (vec[:self.output_dim] if len(vec) > self.output_dim
                               else np.concatenate([vec, np.zeros(self.output_dim - len(vec), np.float32)]))
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    all_vecs.append(vec)
                self._api_hits += len(chunk)
            except Exception:
                # fallback this chunk
                for it in chunk:
                    all_vecs.append(self.embed(it))
        return np.stack(all_vecs) if all_vecs else np.zeros((0, self.output_dim), np.float32)

    def info(self):
        return {**super().info(),
                "real_model": self.model,
                "api_base": EMB_BASE_URL,
                "api_hits": self._api_hits,
                "fallback_hits": self._fallback_hits}


def is_real_embedder_available() -> bool:
    return bool(EMB_API_KEY)
