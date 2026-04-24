"""Niche registry — single source of truth for niche keys, Chinese labels,
synonyms, and CTR priors.

Loads from ``data/niches.json`` at import time (overridable via the env var
``ORAN_NICHES_PATH``). All modules that used to hard-code small EN→ZH maps or
8-niche lists pull from helpers here instead, so adding a new niche is one
edit to the JSON registry.

Example — point at your own registry:

    ORAN_NICHES_PATH=/srv/my_niches.json python -m uvicorn oransim.api:app

The shipped JSON covers the 10 niches in ``data/synthetic/notes_v3.json`` so
the default demo runs coherent. Replace it with your own niches (keeping the
schema) when you wire in a real ``DataProvider``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "niches.json"


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    path = Path(os.environ.get("ORAN_NICHES_PATH") or _DEFAULT_PATH)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["niches"]


def reload() -> None:
    """Bust the lru cache. Tests / admin-hot-reload flows can call this."""
    _load.cache_clear()


def niches() -> list[dict]:
    """Full list of niche dicts (key / zh / en / synonyms / ctr_prior / ...)."""
    return _load()


def niche_keys() -> list[str]:
    """Ordered list of EN keys — used by kol library + model training."""
    return [n["key"] for n in _load()]


def niche_zh_list() -> list[str]:
    """Ordered list of Chinese display labels."""
    return [n["zh"] for n in _load()]


def en_to_zh() -> dict[str, str]:
    """EN key → Chinese display label (the map that used to be repeated in
    soul.py / schema_outputs.py / predict.py / kol_optimizer.py ...)."""
    return {n["key"]: n["zh"] for n in _load()}


def zh_to_en() -> dict[str, str]:
    """Chinese label → EN key (reverse map, for caption-detect fallbacks)."""
    return {n["zh"]: n["key"] for n in _load()}


def synonyms() -> dict[str, list[str]]:
    """EN key → list of keyword/brand synonyms used for caption→niche match."""
    return {n["key"]: list(n.get("synonyms", [])) for n in _load()}


def ctr_priors() -> dict[str, dict[str, float]]:
    """EN key → {mu, sigma, n} industry CTR priors (for CTR fallback / display)."""
    return {n["key"]: dict(n["ctr_prior"]) for n in _load()}


def bias_captions() -> dict[str, str]:
    """EN key → short caption used by mock KOL embedding (kols.py NICHE_BIAS_CAPTIONS)."""
    return {n["key"]: n.get("bias_caption", n["zh"]) for n in _load()}


def female_ratio() -> dict[str, int]:
    """EN key → approximate female fan ratio (for mock KOL demographics)."""
    return {n["key"]: int(n.get("female_ratio", 50)) for n in _load()}
