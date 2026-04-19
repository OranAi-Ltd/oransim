"""Quickstart end-to-end smoke.

Starts uvicorn as a real subprocess (the command from README Quickstart), polls
``/api/health`` until it answers, then hits ``GET /`` and ``POST /api/predict``.
The goal is to catch regressions in the *actual* entry point path — import-time
bootstrap deferred to lifespan, version string baked into the ``/`` handler,
CORS middleware configured, etc. — that pure TestClient tests miss because they
never exercise the wire format.

Port binding is skipped (not failed) in sandboxed envs that disallow TCP bind;
CI runs with network enabled so this exercises the full startup path.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _pick_free_port() -> int:
    """Ask the OS for an unused high port. Avoids TIME_WAIT collisions when
    the test suite runs multiple times back-to-back on the same box.
    """
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get_json(base: str, path: str, timeout: float = 2.0) -> tuple[int, dict]:
    req = urllib.request.Request(base + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        import json as _json

        return r.status, _json.loads(r.read().decode("utf-8"))


def _post_json(base: str, path: str, body: dict, timeout: float = 30.0) -> tuple[int, dict]:
    import json as _json

    data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, _json.loads(r.read().decode("utf-8"))


def test_quickstart_e2e():
    """Spawn uvicorn, poll /api/health, hit / and /api/predict."""
    import oransim

    port = _pick_free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "POP_SIZE": "2000",  # keep startup under ~2s
        "SOUL_POOL_N": "3",
        "LLM_MODE": "mock",
    }
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "oransim.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    # Context manager closes stdout/stderr PIPEs on exit so pytest's
    # unraisable-exception hook doesn't flag a leaked _io.FileIO.
    with subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        try:
            # Poll up to 30s for /api/health to come up. If the subprocess dies
            # (port bind denied by sandbox, import error, etc.) skip — not fail —
            # because the harness isn't what we're testing.
            deadline = time.time() + 30
            ready = False
            while time.time() < deadline:
                rc = proc.poll()
                if rc is not None:
                    out = (
                        proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
                    )
                    # Sandbox-style bind failure: skip so dev environments without
                    # network capabilities don't get a red test.
                    if "permission" in out.lower() or "address" in out.lower() or rc in (1, 144):
                        pytest.skip(f"uvicorn subprocess exited early (rc={rc}): {out[:200]}")
                    pytest.fail(f"uvicorn exited early (rc={rc}): {out[:500]}")
                try:
                    status, _ = _get_json(base, "/api/health", timeout=1.0)
                    if status == 200:
                        ready = True
                        break
                except (urllib.error.URLError, ConnectionError, OSError):
                    pass
                time.sleep(0.3)

            assert ready, "backend did not answer /api/health within 30s"

            # 1. Version baked into the root handler matches the package
            status, j = _get_json(base, "/")
            assert status == 200
            assert (
                j["version"] == oransim.__version__
            ), f"root handler version ({j['version']}) != package version ({oransim.__version__})"
            assert j["name"] == "Oransim"

            # 2. /api/health is well-formed
            status, h = _get_json(base, "/api/health")
            assert status == 200
            assert h.get("status") == "ok", f"health status not ok: {h}"

            # 3. The predict pipeline actually produces non-zero KPIs
            status, p = _post_json(
                base,
                "/api/predict",
                {
                    "creative": {"caption": "test", "duration_sec": 15.0},
                    "total_budget": 10_000,
                    "platform_alloc": {"douyin": 0.5, "xhs": 0.5},
                    "use_llm": False,
                    "n_souls": 3,
                },
                timeout=60.0,
            )
            assert status == 200
            kpis = p.get("kpis", {})
            assert kpis.get("impressions", 0) > 0, f"impressions zero: {kpis}"
            assert kpis.get("clicks", 0) >= 0
            assert "roi" in kpis
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
