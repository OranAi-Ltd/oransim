"""Stream Memory — AgentSociety-style per-agent persistent state.

Each tracked agent has four streams:
  Profile         (static)   : age / gender / city / interests
  Status          (dynamic)  : current_mood / fatigue / brand_attitude / recency
  EventFlow       (deque)    : (ts, action, content) — what happened to/by them
  PerceptionFlow  (deque)    : (ts, thought, event_ref) — what they thought

Opt-in via env STREAM_MEMORY=1. Stored to disk so cross-prediction state survives.
Only tracks the LLM-soul subset (typically 100-10k agents), not the full 1M pool.

Disk layout: ${STREAM_MEMORY_DIR:-/tmp/oransim_memory}/agent_{id}.json
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

_ENABLED = os.environ.get("STREAM_MEMORY", "0") in ("1", "true", "True")
_DIR = os.environ.get("STREAM_MEMORY_DIR", "/tmp/oransim_memory")
_MAX_EVENTS = int(os.environ.get("STREAM_MEMORY_MAX_EVENTS", "50"))
_MAX_PERCEPTIONS = int(os.environ.get("STREAM_MEMORY_MAX_PERCEPTIONS", "30"))


def enabled() -> bool:
    return _ENABLED


@dataclass
class AgentMemory:
    agent_id: int
    profile: dict
    status: dict
    events: deque[dict]
    perceptions: deque[dict]
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "profile": self.profile,
            "status": self.status,
            "events": list(self.events),
            "perceptions": list(self.perceptions),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AgentMemory:
        return cls(
            agent_id=int(d["agent_id"]),
            profile=d.get("profile", {}),
            status=d.get("status", {}),
            events=deque(d.get("events", []), maxlen=_MAX_EVENTS),
            perceptions=deque(d.get("perceptions", []), maxlen=_MAX_PERCEPTIONS),
            updated_at=float(d.get("updated_at", time.time())),
        )

    def recent_events(self, k: int = 5) -> list[dict]:
        return list(self.events)[-k:]

    def recent_perceptions(self, k: int = 3) -> list[dict]:
        return list(self.perceptions)[-k:]

    def summary_for_prompt(self) -> str:
        evs = self.recent_events(3)
        if not evs:
            return ""
        bits = []
        for e in evs:
            kind = e.get("kind", "?")
            content = e.get("content", "")[:40]
            bits.append(f"- {kind}: {content}")
        return "近期经历：\n" + "\n".join(bits)


class StreamMemoryStore:
    def __init__(self, persist_dir: str = _DIR):
        self.dir = persist_dir
        self.cache: dict[int, AgentMemory] = {}
        self.lock = threading.Lock()
        self._stats = {"reads": 0, "writes": 0, "disk_hits": 0, "events_recorded": 0}
        if _ENABLED:
            os.makedirs(self.dir, exist_ok=True)

    def _path(self, agent_id: int) -> str:
        return os.path.join(self.dir, f"agent_{agent_id}.json")

    def get(self, agent_id: int) -> AgentMemory | None:
        if not _ENABLED:
            return None
        self._stats["reads"] += 1
        with self.lock:
            if agent_id in self.cache:
                return self.cache[agent_id]
        p = self._path(agent_id)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    data = json.load(f)
                mem = AgentMemory.from_dict(data)
                with self.lock:
                    self.cache[agent_id] = mem
                self._stats["disk_hits"] += 1
                return mem
            except Exception:
                return None
        return None

    def get_or_create(self, agent_id: int, profile: dict | None = None) -> AgentMemory:
        mem = self.get(agent_id)
        if mem is not None:
            return mem
        mem = AgentMemory(
            agent_id=agent_id,
            profile=profile or {},
            status={"fatigue": 0.0, "brand_attitude": 0.0, "exposure_count": 0},
            events=deque(maxlen=_MAX_EVENTS),
            perceptions=deque(maxlen=_MAX_PERCEPTIONS),
            updated_at=time.time(),
        )
        with self.lock:
            self.cache[agent_id] = mem
        return mem

    def record_event(
        self,
        agent_id: int,
        kind: str,
        content: str,
        metadata: dict | None = None,
        profile: dict | None = None,
    ):
        if not _ENABLED:
            return
        mem = self.get_or_create(agent_id, profile=profile)
        mem.events.append(
            {
                "ts": time.time(),
                "kind": kind,
                "content": content,
                "meta": metadata or {},
            }
        )
        mem.updated_at = time.time()
        self._stats["events_recorded"] += 1

    def record_perception(self, agent_id: int, thought: str, event_ref: int | None = None):
        if not _ENABLED:
            return
        mem = self.get_or_create(agent_id)
        mem.perceptions.append(
            {
                "ts": time.time(),
                "thought": thought,
                "event_ref": event_ref,
            }
        )
        mem.updated_at = time.time()

    def update_status(self, agent_id: int, **patch):
        if not _ENABLED:
            return
        mem = self.get_or_create(agent_id)
        mem.status.update(patch)
        mem.updated_at = time.time()

    def flush(self, agent_ids: list[int] | None = None):
        if not _ENABLED:
            return
        ids = agent_ids if agent_ids is not None else list(self.cache.keys())
        for aid in ids:
            mem = self.cache.get(aid)
            if mem is None:
                continue
            try:
                with open(self._path(aid), "w") as f:
                    json.dump(mem.to_dict(), f, ensure_ascii=False)
                self._stats["writes"] += 1
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "enabled": _ENABLED,
            "dir": self.dir,
            "in_memory": len(self.cache),
            "max_events": _MAX_EVENTS,
            "max_perceptions": _MAX_PERCEPTIONS,
            **self._stats,
        }


_global_store = StreamMemoryStore()


def store() -> StreamMemoryStore:
    return _global_store


def memory_stats() -> dict:
    return _global_store.stats()
