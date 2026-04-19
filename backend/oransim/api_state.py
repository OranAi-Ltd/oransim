"""Shared runtime state for the FastAPI app.

The app was historically a single 1700-line ``api.py`` god-file with
module-level globals and all endpoints inline. As we split it into
router modules, the globals have to live somewhere both the main app
and the routers can reach — that's this module.

**Access pattern:** routers should do ``from .. import api_state`` and
reference ``api_state.POP`` etc. at call time, NOT ``from ..api_state
import POP`` which would bind to the ``None`` sentinel captured at
import time (before the lifespan bootstrap has run).

``bootstrap()`` is idempotent and called from ``api._lifespan`` on app
startup. Previously it ran at import-time, which forced tests and
lightweight scripts to pay ~10s of init cost just to introspect the
module. Moving it to lifespan keeps ``import oransim.api`` cheap.
"""

from __future__ import annotations

import logging
import os
import time

from .agents.agent_provider import OASISProvider, SLOCProvider
from .agents.brand_memory import BrandMemoryState
from .agents.calibration import voronoi_partition
from .agents.soul import SoulAgentPool
from .agents.soul_llm import llm_info
from .agents.statistical import StatisticalAgents
from .causal.counterfactual import ScenarioRunner
from .data.kols import generate_kol_library
from .data.population import generate_population, marginal_fit_report
from .data.world_events import get_world_state
from .diffusion.legacy_hawkes import HawkesSimulator
from .platforms.xhs.recsys_rl import RecSysRLSimulator
from .platforms.xhs.world_model_legacy import PlatformWorldModel
from .runtime.embedding_bus import BUS, bootstrap_default_sources
from .sandbox.engine import SandboxStore

logger = logging.getLogger("oransim.api")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s · %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ---------------- Runtime state (populated by bootstrap via lifespan) ----------------
POP = None
WM = None
AG = None
SOULS = None
KOLS = None
RUNNER = None
SANDBOX = None
HAWKES = None
RECSYS_RL = None
BRAND_STORE = None
SLOC_PROVIDER = None
OASIS_PROVIDER = None
VORONOI_PROVIDER = None
PARTITION = None
PERSONA_TO_SLOT: dict[int, int] = {}


def bootstrap() -> None:
    """Idempotent heavy init: population → world model → agents → KOLs → Voronoi."""
    global POP, WM, AG, SOULS, KOLS, RUNNER, SANDBOX, HAWKES, RECSYS_RL, BRAND_STORE
    global SLOC_PROVIDER, OASIS_PROVIDER, VORONOI_PROVIDER, PARTITION, PERSONA_TO_SLOT
    if POP is not None:
        return
    logger.info("[Oransim] bootstrapping…")
    t0 = time.time()
    pop_size = int(os.environ.get("POP_SIZE", "100000"))
    POP = generate_population(N=pop_size, seed=42)
    logger.info(
        f"  population {pop_size} ready  ({time.time()-t0:.1f}s)"
        f"  marginal KL={marginal_fit_report(POP)}"
    )
    WM = PlatformWorldModel(POP)
    AG = StatisticalAgents(POP)
    SOULS = SoulAgentPool(POP, n=int(os.environ.get("SOUL_POOL_N", "100")), seed=7)
    KOLS = generate_kol_library(n_per_platform=30)
    RUNNER = ScenarioRunner(WM, AG)
    SANDBOX = SandboxStore(RUNNER)
    HAWKES = HawkesSimulator(POP, beta=0.9, branching=0.35)
    RECSYS_RL = RecSysRLSimulator(WM)
    BRAND_STORE = BrandMemoryState.empty(POP.N)

    SLOC_PROVIDER = SLOCProvider(AG, SOULS, None, None)
    OASIS_PROVIDER = OASISProvider(SOULS, POP, n_total=10_000, activation_prob=0.05)
    VORONOI_PROVIDER = SLOC_PROVIDER

    bootstrap_default_sources()
    _bootstrap_index()
    logger.info(
        f"  UEB ready: {len(BUS.list_sources())} sources, "
        f"{sum(s['n_items'] for s in BUS.list_sources())} items pre-indexed"
    )

    logger.info(f"  building Voronoi partition ({len(SOULS.idx)} souls)...")
    t1 = time.time()
    PARTITION = voronoi_partition(POP, [int(i) for i in SOULS.idx])
    PERSONA_TO_SLOT = {int(pid): slot for slot, pid in enumerate(SOULS.idx)}
    SLOC_PROVIDER.partition = PARTITION
    SLOC_PROVIDER.persona_to_slot = PERSONA_TO_SLOT
    logger.info(
        f"  done in {time.time()-t1:.1f}s · max territory {PARTITION.weights.max():.3f}"
        f" mean {PARTITION.weights.mean():.4f}"
    )
    logger.info(f"[Oransim] ready in {time.time()-t0:.1f}s")
    logger.info(f"[Oransim] LLM: {llm_info()}")


def _bootstrap_index() -> None:
    import numpy as _np

    persona_texts = [p.full_card() for p in SOULS.personas.values()]
    BUS.index("comment_text", persona_texts)
    BUS.index("competitor_signal", [f"{k.niche}·{k.fan_count}fans·{k.platform}" for k in KOLS])
    BUS.index("kol_audience", [_np.asarray(k.emb) for k in KOLS])
    sample_n = min(5000, POP.N)
    sample_idx = _np.random.default_rng(0).choice(POP.N, size=sample_n, replace=False)
    BUS.index("user_interest", [POP.interest[i] for i in sample_idx])
    user_demo = _np.stack(
        [
            _np.concatenate(
                [
                    _np.eye(6)[POP.age_idx[i]],
                    _np.eye(2)[POP.gender_idx[i]],
                    [POP.income[i] / 9.0],
                    [POP.edu_idx[i] / 4.0],
                    [POP.occ_idx[i] / 7.0],
                    [POP.city_idx[i] / 4.0],
                ]
            )
            for i in sample_idx[:1000]
        ]
    )
    BUS.index("user_demo", list(user_demo))
    try:
        ws = get_world_state()
        if ws and ws.get("events"):
            BUS.index("world_event", ws["events"])
    except Exception:
        pass
