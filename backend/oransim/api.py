"""FastAPI app exposing the full Oransim causal digital-twin stack.

Previously a 1700-line god-file. Endpoints now live in
``oransim.api_routers``; the heavy runtime singletons live in
``oransim.api_state``. This module keeps only the FastAPI instance,
CORS config, lifespan hook, the ``/`` health check, and the
router-include wiring.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import api_state
from .api_routers import adapters as adapters_router
from .api_routers import analysis as analysis_router
from .api_routers import health as health_router
from .api_routers import predict as predict_router
from .api_routers import sandbox as sandbox_router
from .api_routers import ueb as ueb_router
from .api_routers import v2 as v2_router
from .api_routers import ws as ws_router

logger = logging.getLogger("oransim.api")


def _check_multi_worker_state_safety() -> None:
    """Warn on startup if the runtime is about to be sharded across multiple
    workers. Oransim v0.2 stores the ~GB runtime state (population, agents,
    world model, Embedding Bus indexes, brand-memory cache) in
    process-local ``api_state`` / ``embedding_bus.BUS`` singletons. Each
    gunicorn / uvicorn worker therefore holds its own independent copy:
    boot time × N, no cross-worker consistency (a sandbox scenario created
    on worker A is invisible on worker B), and memory × N.

    A shared-state backend (Redis / a single "runtime" worker fronted by
    in-process queues) is Enterprise-only in v0.2. Until it lands in OSS,
    single-worker is the correct deployment for the OSS tier. We surface
    a loud WARNING rather than silently letting the operator hit
    surprising cross-request inconsistency in production.

    Checked env vars: ``WEB_CONCURRENCY`` (gunicorn/uvicorn convention),
    ``WORKERS``, ``UVICORN_WORKERS`` — the first one ≥ 2 trips the warn.
    """
    for var in ("WEB_CONCURRENCY", "WORKERS", "UVICORN_WORKERS"):
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            n = int(raw)
        except ValueError:
            continue
        if n >= 2:
            logger.warning(
                "[Oransim] %s=%d detected — the OSS runtime state "
                "(api_state.* singletons + runtime.embedding_bus.BUS) is "
                "process-local (~GB per worker, no cross-worker sync). "
                "Expect %dx boot time, %dx memory, and inconsistent "
                "sandbox / brand-memory / UEB state across requests. "
                "Run a single worker in production until the shared-state "
                "backend ships, or switch to Enterprise Edition.",
                var,
                n,
                n,
                n,
            )
            return


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    _check_multi_worker_state_safety()
    api_state.bootstrap()
    yield


app = FastAPI(
    title="Oransim",
    description="Causal Digital Twin for Marketing at Scale",
    version="0.2.0a0",
    lifespan=_lifespan,
)

# CORS: two knobs so the browser demo works on localhost, public IP, or
# behind a reverse proxy.
#   OSIM_CORS_ORIGINS        comma-separated exact allowlist
#   OSIM_CORS_ORIGIN_REGEX   regex for demo-port hosts; set empty to disable
# Regex default admits common demo ports (8090/8091/8092/8094/8001) on any host,
# so pointing a public-IP frontend at this backend "just works". For production,
# set a specific allowlist and clear the regex.
_cors_env = os.environ.get(
    "OSIM_CORS_ORIGINS",
    "http://localhost:8090,http://127.0.0.1:8090,http://localhost:8001,http://127.0.0.1:8001",
)
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
_cors_regex = os.environ.get(
    "OSIM_CORS_ORIGIN_REGEX",
    r"^https?://[^/]+:(8090|8091|8092|8094|8001)$",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_regex or None,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(adapters_router.router)
app.include_router(analysis_router.router)
app.include_router(health_router.router)
app.include_router(predict_router.router)
app.include_router(sandbox_router.router)
app.include_router(ueb_router.router)
app.include_router(v2_router.router)
app.include_router(ws_router.router)


@app.get("/")
def root():
    """Root health check."""
    return {
        "name": "Oransim",
        "version": "0.2.0a0",
        "status": "alpha",
        "docs": "/docs",
        "repo": "https://github.com/OranAi-Ltd/oransim",
    }
