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


@asynccontextmanager
async def _lifespan(app_: FastAPI):
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
