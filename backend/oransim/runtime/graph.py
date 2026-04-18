"""Causal Computation Graph (CCG) — declarative DAG of computation steps.

Why: replace the old ad-hoc imperative pipeline (impression → outcome → kpi → ...)
with an explicit graph where:
  - each node has named inputs and a callable
  - dependencies are auto-resolved + topologically sorted
  - results cached by input hash → repeat queries are free
  - any node can be `intervene()`-d (replace value or function) → first-class do-operator
  - execution is traced with per-node timing → observability
  - independent branches run in parallel (ThreadPoolExecutor)

This is the architectural backbone for every other 2026 module
(uncertainty, multi-fidelity, multimodal, geo) to plug into uniformly.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    name: str
    fn: Callable[..., Any]
    deps: list[str] = field(default_factory=list)
    cache_key_fn: Callable[..., str] | None = None  # custom hashing
    description: str = ""
    parallel_safe: bool = True


@dataclass
class NodeTrace:
    name: str
    started_at: float
    duration_ms: float
    cache_hit: bool
    intervened: bool
    deps: list[str]
    output_summary: str


@dataclass
class GraphRun:
    results: dict[str, Any]
    traces: list[NodeTrace]
    total_ms: float
    cache_stats: dict[str, int]


def _safe_hash(obj: Any, depth: int = 0) -> str:
    """Best-effort stable hash for cache keys. Falls back to id() for unhashables."""
    if depth > 4:
        return f"<deep:{type(obj).__name__}>"
    try:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return f"{type(obj).__name__}:{obj}"
        if isinstance(obj, (list, tuple)):
            inner = ",".join(_safe_hash(x, depth + 1) for x in obj[:50])
            return f"[{inner}]"
        if isinstance(obj, dict):
            items = sorted(obj.items())[:50]
            inner = ",".join(f"{k}={_safe_hash(v, depth+1)}" for k, v in items)
            return f"{{{inner}}}"
        # numpy
        if hasattr(obj, "tobytes") and hasattr(obj, "shape"):
            h = hashlib.md5(obj.tobytes()[:1024]).hexdigest()[:12]
            return f"<np:{obj.shape}:{h}>"
        # has hash_tuple method
        if hasattr(obj, "hash_tuple"):
            return _safe_hash(obj.hash_tuple(), depth + 1)
        return f"<{type(obj).__name__}:{id(obj)}>"
    except Exception:
        return f"<err:{type(obj).__name__}>"


class CausalGraph:
    """Declarative DAG executor with caching, intervention, tracing."""

    def __init__(self, name: str = "graph"):
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._cache: dict[tuple[str, str], Any] = {}  # (node_name, key_hash) -> result
        self._cache_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def node(
        self,
        name: str,
        fn: Callable,
        deps: list[str] = None,
        description: str = "",
        parallel_safe: bool = True,
    ) -> CausalGraph:
        if name in self._nodes:
            raise ValueError(f"node already exists: {name}")
        self._nodes[name] = Node(
            name=name,
            fn=fn,
            deps=deps or [],
            description=description,
            parallel_safe=parallel_safe,
        )
        return self

    # ---- Topology ----

    def _topo_order(self, targets: list[str], known_inputs: set[str] | None = None) -> list[str]:
        """Topologically order computation. Deps not in self._nodes are treated as
        external inputs (must be provided via `inputs` dict at run time)."""
        known_inputs = known_inputs or set()
        order: list[str] = []
        visited: set[str] = set()
        temp: set[str] = set()

        def visit(n):
            if n in visited:
                return
            if n in known_inputs:  # external input — skip recursion
                visited.add(n)
                return
            if n not in self._nodes:
                # external input not provided — let runtime fail gracefully
                visited.add(n)
                return
            if n in temp:
                raise ValueError(f"cycle through {n}")
            temp.add(n)
            for d in self._nodes[n].deps:
                visit(d)
            temp.remove(n)
            visited.add(n)
            order.append(n)

        for t in targets:
            visit(t)
        return order

    # ---- Run ----

    def run(
        self,
        *,
        targets: list[str] | None = None,
        inputs: dict[str, Any] | None = None,
        interventions: dict[str, Any] | None = None,
        use_cache: bool = True,
        parallel: bool = True,
    ) -> GraphRun:
        """Execute the graph.

        inputs: pre-set values for source nodes (override fn output)
        interventions: do(X = value) — replaces node's computed value with this
            (the difference vs inputs: interventions also break the cache key,
            making downstream re-execute under the new value)
        """
        inputs = inputs or {}
        interventions = interventions or {}
        if targets is None:
            # default: run all leaf nodes (no consumers)
            consumed = {d for n in self._nodes.values() for d in n.deps}
            targets = [n for n in self._nodes if n not in consumed]
        order = self._topo_order(
            targets, known_inputs=set(inputs.keys()) | set(interventions.keys())
        )

        results: dict[str, Any] = dict(inputs)
        traces: list[NodeTrace] = []
        t_start = time.time()

        # Detect what's in dirty downstream of an intervention — invalidate cache
        intervened_set = set(interventions.keys())
        dirty: set[str] = set(intervened_set)
        for n in order:
            node = self._nodes[n]
            if any(d in dirty for d in node.deps):
                dirty.add(n)

        # Execute layer-by-layer for parallelism
        executed: set[str] = set(inputs.keys())
        while True:
            ready = [
                n
                for n in order
                if n not in executed
                and all(d in executed or d in inputs for d in self._nodes[n].deps)
            ]
            if not ready:
                break

            def run_node(name: str) -> tuple[str, Any, NodeTrace]:
                node = self._nodes[name]
                t0 = time.time()
                # Intervention: skip computation
                if name in interventions:
                    val = interventions[name]
                    tr = NodeTrace(
                        name=name,
                        started_at=t0,
                        duration_ms=0,
                        cache_hit=False,
                        intervened=True,
                        deps=node.deps,
                        output_summary=_safe_hash(val)[:80],
                    )
                    return name, val, tr
                # Pre-set input
                if name in inputs:
                    val = inputs[name]
                    return (
                        name,
                        val,
                        NodeTrace(name, t0, 0, False, False, node.deps, _safe_hash(val)[:80]),
                    )
                # Build kwargs from deps
                kwargs = {d: results[d] for d in node.deps}
                # Cache key
                if use_cache and name not in dirty:
                    key = node.cache_key_fn(**kwargs) if node.cache_key_fn else _safe_hash(kwargs)
                    with self._cache_lock:
                        cached = self._cache.get((name, key))
                    if cached is not None:
                        self.cache_hits += 1
                        return (
                            name,
                            cached,
                            NodeTrace(
                                name,
                                t0,
                                (time.time() - t0) * 1000,
                                True,
                                False,
                                node.deps,
                                _safe_hash(cached)[:80],
                            ),
                        )
                    self.cache_misses += 1
                else:
                    key = None
                # Compute
                val = node.fn(**kwargs)
                if use_cache and key is not None and name not in dirty:
                    with self._cache_lock:
                        self._cache[(name, key)] = val
                tr = NodeTrace(
                    name=name,
                    started_at=t0,
                    duration_ms=(time.time() - t0) * 1000,
                    cache_hit=False,
                    intervened=False,
                    deps=node.deps,
                    output_summary=_safe_hash(val)[:80],
                )
                return name, val, tr

            if parallel and len(ready) > 1:
                par_safe = [n for n in ready if self._nodes[n].parallel_safe]
                seq = [n for n in ready if not self._nodes[n].parallel_safe]
                if par_safe:
                    with ThreadPoolExecutor(max_workers=min(8, len(par_safe))) as ex:
                        futs = {ex.submit(run_node, n): n for n in par_safe}
                        for f in as_completed(futs):
                            name, val, tr = f.result()
                            results[name] = val
                            traces.append(tr)
                            executed.add(name)
                for n in seq:
                    name, val, tr = run_node(n)
                    results[name] = val
                    traces.append(tr)
                    executed.add(name)
            else:
                for n in ready:
                    name, val, tr = run_node(n)
                    results[name] = val
                    traces.append(tr)
                    executed.add(name)

        return GraphRun(
            results={k: v for k, v in results.items() if k in targets},
            traces=traces,
            total_ms=(time.time() - t_start) * 1000,
            cache_stats={
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "size": len(self._cache),
            },
        )

    # ---- do-operator wrapper ----

    def intervene(
        self, *, targets: list[str], do: dict[str, Any], inputs: dict[str, Any] = None
    ) -> GraphRun:
        """Pearl's do() at graph level: replaces node values, invalidates downstream cache."""
        return self.run(
            targets=targets, inputs=inputs or {}, interventions=do, use_cache=True, parallel=True
        )

    # ---- Introspection ----

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": [
                {
                    "name": n.name,
                    "deps": n.deps,
                    "description": n.description,
                    "parallel_safe": n.parallel_safe,
                }
                for n in self._nodes.values()
            ],
        }

    def trace_to_dict(self, run: GraphRun) -> dict:
        return {
            "total_ms": round(run.total_ms, 1),
            "cache_stats": run.cache_stats,
            "traces": [
                {
                    "name": t.name,
                    "duration_ms": round(t.duration_ms, 1),
                    "cache_hit": t.cache_hit,
                    "intervened": t.intervened,
                    "deps": t.deps,
                    "output": t.output_summary,
                }
                for t in run.traces
            ],
        }
