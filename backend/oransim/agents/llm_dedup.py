"""LLM request dedup + in-flight coalescing layer.

Two optimizations:
  1. Response cache: identical (persona, impression) → cached response
  2. In-flight coalescing: concurrent identical requests share one future

Opt-in via env DEDUP=1. Stats exposed via dedup_stats().

Design:
  - Hash key = sha1(persona_card + caption + kol_id + platform + visual + music)
  - LRU cache with maxsize 5000 (~5MB at 1KB/entry)
  - In-flight registry for coalescing
  - Thread-safe (used from ThreadPoolExecutor and asyncio threadpool)
"""
from __future__ import annotations
import os
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Dict, Callable, Any, Optional


_ENABLED = os.environ.get("DEDUP", "0") in ("1", "true", "True")
_CACHE_MAX = int(os.environ.get("DEDUP_CACHE_MAX", "5000"))


class _LRU:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.d: "OrderedDict[str, Any]" = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key]
        return None

    def put(self, key: str, val: Any):
        with self.lock:
            self.d[key] = val
            self.d.move_to_end(key)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)

    def __len__(self) -> int:
        return len(self.d)


_cache = _LRU(_CACHE_MAX)
_inflight: Dict[str, "threading.Event"] = {}
_inflight_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0, "coalesced": 0, "calls": 0}


def enabled() -> bool:
    return _ENABLED


def make_key(persona_card: str, caption: str, platform: str,
             kol_name: str = "无", visual: str = "", music: str = "",
             duration: float = 0.0, memory_hint: str = "") -> str:
    # memory_hint is inlined in the Stream Memory path and goes into the prompt —
    # forgetting it caused different-memory-state responses to collide in cache.
    raw = f"{persona_card}|{caption}|{platform}|{kol_name}|{visual}|{music}|{duration}|{memory_hint}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def async_coalesce(key: str):
    """Lock-free atomic claim for async paths: returns (is_leader, event).

    Leader runs factory, followers wait on event. This is the async-friendly
    equivalent of dedup_call's threading.Event handshake. Call .set() when leader
    finishes (via async_release below).
    """
    _stats["calls"] += 1
    cached = _cache.get(key)
    if cached is not None:
        _stats["hits"] += 1
        return False, None, _clone(cached)
    with _inflight_lock:
        ev = _inflight.get(key)
        if ev is not None:
            _stats["coalesced"] += 1
            return False, ev, None
        ev = threading.Event()
        _inflight[key] = ev
        return True, ev, None


def async_release(key: str, result: Dict):
    """Leader publishes result + notifies followers."""
    _stats["misses"] += 1
    if isinstance(result, dict) and "_error" not in result:
        _cache.put(key, _freeze(result))
    with _inflight_lock:
        ev = _inflight.pop(key, None)
    if ev is not None:
        ev.set()


def dedup_call(key: str, factory: Callable[[], Dict],
               wait_timeout: float = 60.0) -> Dict:
    """Call factory(), but coalesce concurrent duplicates and cache results.

    Returns a copy of the cached/computed dict so callers can mutate freely.
    """
    _stats["calls"] += 1

    cached = _cache.get(key)
    if cached is not None:
        _stats["hits"] += 1
        return _clone(cached)

    with _inflight_lock:
        ev = _inflight.get(key)
        if ev is not None:
            _stats["coalesced"] += 1
            wait = True
        else:
            ev = threading.Event()
            _inflight[key] = ev
            wait = False

    if wait:
        ev.wait(timeout=wait_timeout)
        cached = _cache.get(key)
        if cached is not None:
            return _clone(cached)
        _stats["misses"] += 1
        return factory()

    try:
        _stats["misses"] += 1
        result = factory()
        if isinstance(result, dict) and "_error" not in result:
            _cache.put(key, _freeze(result))
        return result
    finally:
        with _inflight_lock:
            _inflight.pop(key, None)
        ev.set()


def _freeze(d: Dict) -> Dict:
    return {k: v for k, v in d.items()
            if not k.startswith("_batch") and k not in ("persona_id", "persona_oneliner",
                                                        "persona_card", "source")}


def _clone(d: Dict) -> Dict:
    return dict(d)


def dedup_stats() -> Dict:
    total = max(1, _stats["calls"])
    return {
        "enabled": _ENABLED,
        "cache_size": len(_cache),
        "cache_max": _CACHE_MAX,
        "calls": _stats["calls"],
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "coalesced": _stats["coalesced"],
        "hit_rate": round(_stats["hits"] / total, 4),
        "coalesce_rate": round(_stats["coalesced"] / total, 4),
    }


def reset_stats():
    for k in _stats:
        _stats[k] = 0


def reset_cache():
    with _cache.lock:
        _cache.d.clear()
