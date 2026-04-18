"""FastAPI application entry point.

Phase 1: stub only. Phase 2 will migrate the internal `causaltwin.api`
module here after desensitization.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Oransim",
    description="Causal Digital Twin for Marketing at Scale",
    version="0.1.0a0",
)


@app.get("/")
def root() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "name": "Oransim",
        "version": "0.1.0a0",
        "status": "alpha",
        "docs": "/docs",
        "repo": "https://github.com/oranai/oransim",
    }
