"""Universal Embedding Bus (UEB).

Core idea: any new data source can be plugged in via a tiny `Embedder` class
that produces a fixed-D vector. Downstream modules (agent / world_model /
causal) consume the registry uniformly — no source-specific code changes
needed when new data arrives.

This is the architectural lever for "more data → more accurate" because:
  1. Adding a source is O(50 lines), not O(1000)
  2. All downstream automatically benefits
  3. Vectors live in shared space (alignable, comparable, fusable)
  4. Online learning hooks can re-fit any embedder as data accumulates

Production: replace hash-based mock embedders with real models
(Qwen3-Embedding for text, SigLIP for image, Whisper-v3 for audio,
TabNet for tabular, GraphSAGE for graph, etc).
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EMB_DIM = 768  # standard production dim (Qwen3-Embedding-base / SigLIP-base default)


# ---------------- Base ----------------


class Embedder(ABC):
    """Plug-in for one data source.  Implement `embed(item) -> (D,) np.float32`."""

    name: str = "abstract"
    modality: str = "unknown"  # text / image / audio / tabular / graph / event ...
    output_dim: int = EMB_DIM
    fit_required: bool = False  # True for learnable embedders

    @abstractmethod
    def embed(self, item: Any) -> np.ndarray: ...

    def embed_batch(self, items: list[Any]) -> np.ndarray:
        return np.stack([self.embed(it) for it in items])

    def fit(self, samples: list[Any]) -> None:  # noqa: B027 — optional hook, not abstract
        """Override for online/continual learning embedders."""
        pass

    def info(self) -> dict:
        return {
            "name": self.name,
            "modality": self.modality,
            "output_dim": self.output_dim,
            "fit_required": self.fit_required,
        }


# ---------------- Built-in embedders (mock for MVP, swap real later) ----------------


class HashTextEmbedder(Embedder):
    """Deterministic mock text embedder — replace with Qwen3-Embedding in prod."""

    name = "hash-text-mock"
    modality = "text"

    def __init__(self, dim: int = EMB_DIM, seed_offset: int = 0):
        self.output_dim = dim
        self.seed_offset = seed_offset

    def embed(self, item: Any) -> np.ndarray:
        s = str(item) + str(self.seed_offset)
        h = hashlib.sha256(s.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        v = rng.normal(0, 1, self.output_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


class TabularEmbedder(Embedder):
    """Tabular features → vector via random projection.  In prod: TabNet/SAINT."""

    name = "tabular-rand-proj"
    modality = "tabular"

    def __init__(self, in_dim: int, out_dim: int = EMB_DIM, seed: int = 42):
        self.in_dim = in_dim
        self.output_dim = out_dim
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)

    def embed(self, item: np.ndarray) -> np.ndarray:
        x = np.asarray(item, dtype=np.float32).ravel()
        if len(x) < self.in_dim:
            x = np.concatenate([x, np.zeros(self.in_dim - len(x), np.float32)])
        elif len(x) > self.in_dim:
            x = x[: self.in_dim]
        v = x @ self.W
        return v / (np.linalg.norm(v) + 1e-8)


class CategoricalEmbedder(Embedder):
    """Category labels → learnable lookup (mock = hash here)."""

    name = "categorical-hash"
    modality = "categorical"

    def __init__(self, dim: int = EMB_DIM):
        self.output_dim = dim

    def embed(self, item: Any) -> np.ndarray:
        h = hashlib.md5(str(item).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        v = rng.normal(0, 0.5, self.output_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


class TimeSeriesEmbedder(Embedder):
    """Variable-length numerical sequence → vector via Fourier features.
    In prod: PatchTST / TimesNet."""

    name = "ts-fourier"
    modality = "timeseries"

    def __init__(self, n_freqs: int = 32, dim: int = EMB_DIM):
        self.n_freqs = n_freqs
        self.output_dim = dim

    def embed(self, series: list[float]) -> np.ndarray:
        x = np.asarray(series, dtype=np.float32)
        if len(x) == 0:
            return np.zeros(self.output_dim, dtype=np.float32)
        # FFT of normalized series
        x = (x - x.mean()) / (x.std() + 1e-8)
        F = np.fft.rfft(x, n=self.n_freqs * 2)
        feat = np.concatenate([F.real, F.imag])[: self.n_freqs * 2]
        # pad/project to output_dim
        if len(feat) < self.output_dim:
            feat = np.concatenate([feat, np.zeros(self.output_dim - len(feat))])
        else:
            feat = feat[: self.output_dim]
        return feat.astype(np.float32) / (np.linalg.norm(feat) + 1e-8)


class GeoEmbedder(Embedder):
    """(lat, lon) → vector via positional encoding (Mercator-ish)."""

    name = "geo-pos-enc"
    modality = "geospatial"

    def __init__(self, n_freqs: int = 16, dim: int = EMB_DIM):
        self.n_freqs = n_freqs
        self.output_dim = dim

    def embed(self, latlon: tuple) -> np.ndarray:
        lat, lon = float(latlon[0]), float(latlon[1])
        feats = []
        for k in range(1, self.n_freqs + 1):
            feats.extend(
                [
                    np.sin(k * lat * np.pi / 180),
                    np.cos(k * lat * np.pi / 180),
                    np.sin(k * lon * np.pi / 180),
                    np.cos(k * lon * np.pi / 180),
                ]
            )
        feats = np.asarray(feats, dtype=np.float32)
        if len(feats) < self.output_dim:
            feats = np.concatenate(
                [feats, np.zeros(self.output_dim - len(feats), dtype=np.float32)]
            )
        else:
            feats = feats[: self.output_dim]
        return feats / (np.linalg.norm(feats) + 1e-8)


class EventEmbedder(Embedder):
    """A world event (dict with title/category/impact) → vector via concat of
    text emb + categorical emb + impact scalar projection."""

    name = "event-composite"
    modality = "event"

    def __init__(self, dim: int = EMB_DIM):
        self.output_dim = dim
        self._text = HashTextEmbedder(dim=dim)
        self._cat = CategoricalEmbedder(dim=dim)

    def embed(self, event: dict) -> np.ndarray:
        v_text = self._text.embed(event.get("title", ""))
        v_cat = self._cat.embed(event.get("category", "?"))
        impact = float(event.get("consumer_impact", 0))
        attn = float(event.get("attention_share", 0.1))
        # weighted sum: text dominates, category small bias, impact/attn modulates magnitude
        v = (v_text * 0.7 + v_cat * 0.3) * (1.0 + impact)
        v *= 1.0 + attn
        return v.astype(np.float32) / (np.linalg.norm(v) + 1e-8)


# ---------------- Multi-modal stubs (v0.5 roadmap hooks) ----------------
#
# Oransim v0.2 only ships a real *text* embedder (OpenAI-compat
# text-embedding-3-small via real_embedder.RealTextEmbedder). Image / video /
# audio embedders are stubs that raise NotImplementedError pointing at
# ROADMAP.md#v05. The ABC contract, `modality` field, and the registry are
# all ready — dropping a real CLIP / SigLIP / Whisper implementation in is a
# ~50-line Embedder subclass with no downstream changes.
#
# Why stubs now: documents the extension surface in code (not just
# docs/ROADMAP), gives callers a typed error when they try to embed a
# non-text modality, and prevents silent fallbacks to hash-mock that would
# misleadingly "just work" on image inputs.

_MULTIMODAL_ROADMAP_MSG = (
    "Multi-modal embedders (image / video / audio) are a v0.5 roadmap item. "
    "Oransim v0.2 ships only the real text embedder. "
    "See ROADMAP.md#v05 and docs/en/architecture.md for the planned "
    "CLIP / Qwen-VL / SigLIP (image), I-JEPA / TimeSformer (video), "
    "Whisper-v3 / CLAP (audio) backends."
)


class ImageEmbedderStub(Embedder):
    """Stub for image embedders. Raises NotImplementedError until v0.5 lands.

    Planned backends: CLIP (OpenAI), Qwen-VL (Alibaba), SigLIP (Google),
    ImageBind (Meta). Drop-in swap once the weights are selected.
    """

    name = "image-stub-v0.2"
    modality = "image"

    def __init__(self, dim: int = EMB_DIM):
        self.output_dim = dim

    def embed(self, item: Any) -> np.ndarray:
        raise NotImplementedError(_MULTIMODAL_ROADMAP_MSG)


class VideoEmbedderStub(Embedder):
    """Stub for video embedders. Raises NotImplementedError until v0.5 lands.

    Planned backends: I-JEPA v2 (Meta), TimeSformer, VideoMAE v2,
    or Qwen-VL video mode. Typical choice is image-backbone + temporal
    pooling for short-form video (TikTok / Reels / Shorts 15-60s).
    """

    name = "video-stub-v0.2"
    modality = "video"

    def __init__(self, dim: int = EMB_DIM):
        self.output_dim = dim

    def embed(self, item: Any) -> np.ndarray:
        raise NotImplementedError(_MULTIMODAL_ROADMAP_MSG)


class AudioEmbedderStub(Embedder):
    """Stub for audio embedders. Raises NotImplementedError until v0.5 lands.

    Planned backends: Whisper-v3 encoder (OpenAI, speech-heavy), CLAP
    (music / ambient), AudioMAE. Primary use case: BGM-mood recognition
    for short-video creatives.
    """

    name = "audio-stub-v0.2"
    modality = "audio"

    def __init__(self, dim: int = EMB_DIM):
        self.output_dim = dim

    def embed(self, item: Any) -> np.ndarray:
        raise NotImplementedError(_MULTIMODAL_ROADMAP_MSG)


# ---------------- Registry ----------------


@dataclass
class _SourceRecord:
    embedder: Embedder
    n_items_indexed: int = 0
    last_updated: float = field(default_factory=time.time)
    notes: str = ""


class EmbeddingBus:
    """Singleton registry of all data-source embedders + cached vector indexes."""

    def __init__(self):
        self._sources: dict[str, _SourceRecord] = {}
        self._lock = threading.Lock()
        self._vector_indexes: dict[str, np.ndarray] = {}  # source_name -> (N, D) vectors
        self._items_meta: dict[str, list] = {}  # source_name -> list of items

    def register(self, source_name: str, embedder: Embedder, notes: str = "") -> None:
        with self._lock:
            self._sources[source_name] = _SourceRecord(embedder=embedder, notes=notes)

    def list_sources(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "source": name,
                    **r.embedder.info(),
                    "n_items": r.n_items_indexed,
                    "notes": r.notes,
                    "last_updated": r.last_updated,
                }
                for name, r in self._sources.items()
            ]

    def index(self, source_name: str, items: list[Any]) -> np.ndarray:
        """Embed a batch from a registered source and append to its index."""
        with self._lock:
            if source_name not in self._sources:
                raise KeyError(f"unknown source: {source_name}")
            rec = self._sources[source_name]
        vectors = rec.embedder.embed_batch(items)
        with self._lock:
            existing = self._vector_indexes.get(source_name)
            if existing is None:
                self._vector_indexes[source_name] = vectors
                self._items_meta[source_name] = list(items)
            else:
                self._vector_indexes[source_name] = np.concatenate([existing, vectors], axis=0)
                self._items_meta[source_name].extend(items)
            rec.n_items_indexed = len(self._items_meta[source_name])
            rec.last_updated = time.time()
        return vectors

    def vectors(self, source_name: str) -> np.ndarray | None:
        return self._vector_indexes.get(source_name)

    def search(self, query_vec: np.ndarray, source_name: str, top_k: int = 5) -> list[dict]:
        vecs = self._vector_indexes.get(source_name)
        items = self._items_meta.get(source_name, [])
        if vecs is None or len(vecs) == 0:
            return []
        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        sims = vecs @ q
        order = np.argsort(-sims)[:top_k]
        return [{"item": items[i], "score": float(sims[i])} for i in order]

    def fuse_to_unified(
        self, item_per_source: dict[str, Any], weights: dict[str, float] | None = None
    ) -> np.ndarray:
        """Fuse representations from multiple sources for a single conceptual entity
        (e.g. a brand has text caption + image + LBS + financial timeseries)."""
        weights = weights or {}
        vecs = []
        ws = []
        for src, item in item_per_source.items():
            if src not in self._sources:
                continue
            rec = self._sources[src]
            v = rec.embedder.embed(item)
            vecs.append(v)
            ws.append(weights.get(src, 1.0))
        if not vecs:
            return np.zeros(EMB_DIM, dtype=np.float32)
        ws_arr = np.array(ws, dtype=np.float32)
        ws_arr = ws_arr / ws_arr.sum()
        # Padding/truncation for mixed-dim embedders (project to EMB_DIM)
        unified = np.zeros(EMB_DIM, dtype=np.float32)
        for v, w in zip(vecs, ws_arr, strict=False):
            if len(v) < EMB_DIM:
                v = np.concatenate([v, np.zeros(EMB_DIM - len(v), dtype=np.float32)])
            else:
                v = v[:EMB_DIM]
            unified += w * v
        return unified / (np.linalg.norm(unified) + 1e-8)

    def learning_stats(self) -> dict:
        """For online-learning tracking: how many items, how recent."""
        return {
            "n_sources": len(self._sources),
            "total_items": sum(r.n_items_indexed for r in self._sources.values()),
            "sources": self.list_sources(),
            "scaling_law_estimate": self._scaling_law(),
        }

    def _scaling_law(self) -> dict:
        """Heuristic 1/√N decay curve for the 'more data → more accurate'
        visualization. NOT a derived PAC-Bayes bound — just the canonical
        central-limit shape (error ≈ c/√N) pinned at c = 0.30 with a 0.005
        irreducible floor. Kept for the scaling-law widget only; callers
        that need a formal generalization bound must plug in their own."""
        n_total = sum(r.n_items_indexed for r in self._sources.values())
        err = max(0.005, 0.30 / np.sqrt(max(n_total, 1)))
        return {
            "n_total_items": n_total,
            # Key name is kept for frontend / external-API backward compat,
            # but the value is a heuristic curve point, not a derived bound.
            "estimated_generalization_err_upper_bound": round(float(err), 4),
            "halving_at_n": int(n_total * 4) if n_total > 0 else 0,
            "interpretation": "启发式 1/√N 衰减曲线（非严格上界）；累 4 倍数据 → 误差减半",
            "curve_family": "heuristic_inv_sqrt_n",
        }


# Global bus singleton (each backend instance has one)
BUS = EmbeddingBus()


def bootstrap_default_sources() -> None:
    """Register the standard set so backend boots with embedders ready."""
    # Real text embedders via existing LLM API (OpenAI-compatible)
    try:
        from .real_embedder import RealTextEmbedder, is_real_embedder_available

        has_real = is_real_embedder_available()
    except Exception:
        has_real = False

    if has_real:
        BUS.register(
            "creative_caption",
            RealTextEmbedder(fallback_seed=1),
            notes="素材文案；真 embedder (OpenAI-compat endpoint)",
        )
        BUS.register(
            "creative_visual",
            RealTextEmbedder(fallback_seed=2),
            notes="素材视觉描述；v2 接 SigLIP-2 / Qwen2.5-VL",
        )
        BUS.register(
            "creative_audio",
            RealTextEmbedder(fallback_seed=3),
            notes="素材音频描述；v2 接 Whisper-v3 + CLAP",
        )
    else:
        BUS.register(
            "creative_caption",
            HashTextEmbedder(seed_offset=1),
            notes="素材文案；LLM_API_KEY 未设置 — hash fallback",
        )
        BUS.register(
            "creative_visual", HashTextEmbedder(seed_offset=2), notes="素材视觉；hash fallback"
        )
        BUS.register(
            "creative_audio", HashTextEmbedder(seed_offset=3), notes="素材音频；hash fallback"
        )
    BUS.register("user_demo", TabularEmbedder(in_dim=12), notes="用户人口属性；TabNet/SAINT")
    BUS.register(
        "user_interest", TabularEmbedder(in_dim=64), notes="用户兴趣 64-d；与 creative 共空间"
    )
    BUS.register("kol_audience", TabularEmbedder(in_dim=64), notes="KOL 粉丝画像")
    BUS.register(
        "brand_text_uplifting", HashTextEmbedder(seed_offset=10), notes="品牌官网/Wiki 文本"
    )
    BUS.register("world_event", EventEmbedder(), notes="全谱事件流（GPT 拉的）")
    BUS.register("comment_text", HashTextEmbedder(seed_offset=20), notes="评论区文本")
    BUS.register("hawkes_curve", TimeSeriesEmbedder(), notes="生命周期曲线")
    BUS.register("brand_lift_curve", TimeSeriesEmbedder(), notes="90 天品牌曲线")
    BUS.register("region_geo", GeoEmbedder(), notes="地理坐标 (lat, lon)")
    BUS.register("competitor_signal", HashTextEmbedder(seed_offset=30), notes="竞品历史投放素材")
    BUS.register("macro_econ", TabularEmbedder(in_dim=10), notes="宏观经济 / CPI / 消费信心")
    BUS.register("weather", TabularEmbedder(in_dim=5), notes="天气：温度/湿度/降水/风速/能见度")
    BUS.register(
        "real_campaign_log", TabularEmbedder(in_dim=20), notes="客户真实投放历史（一方数据）"
    )
