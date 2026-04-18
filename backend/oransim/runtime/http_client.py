"""OpenAI-compatible HTTP client with retry, jitter, and model fallback.

Hardens the LLM call path against transient upstream failures. Used by
``oransim.agents.soul_llm`` and any other module making synchronous LLM
requests.

Retry policy
------------

- **Retryable**: ``429`` (rate limit), ``500``-``599`` (server), connection
  errors, read timeouts.
- **Non-retryable**: ``400`` (bad request), ``401`` (auth), ``403`` (perms),
  ``404`` (missing).
- Exponential backoff with full jitter: ``sleep(random(0, base * 2^n))``,
  capped at ``max_backoff_s``.

Fallback chain
--------------

If ``LLM_MODEL_FALLBACK`` env var is set (comma-separated), each retry
rotates to the next model name. Useful when the primary model is
throttled but a compatible alternative is available (e.g.,
``gpt-5.4,gpt-4o-mini,deepseek-chat``).

Example:

    from oransim.runtime.http_client import post_json

    resp = post_json(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        body={"model": "gpt-5.4", "messages": [...]},
        max_retries=3,
    )
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))
DEFAULT_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
DEFAULT_BACKOFF_BASE = float(os.environ.get("LLM_BACKOFF_BASE", "0.8"))
DEFAULT_MAX_BACKOFF = float(os.environ.get("LLM_MAX_BACKOFF", "20.0"))

# HTTP codes on which we retry
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
# Codes that definitively indicate a client error — do NOT retry
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 405, 410, 422}


def _should_retry(status: int) -> bool:
    return status in RETRYABLE_STATUS or (500 <= status < 600)


def _fallback_chain(primary_model: str) -> list[str]:
    """Return the ordered model-name sequence to try on successive retries."""
    chain_env = os.environ.get("LLM_MODEL_FALLBACK", "").strip()
    if not chain_env:
        return [primary_model]
    extras = [m.strip() for m in chain_env.split(",") if m.strip()]
    return [primary_model] + [m for m in extras if m != primary_model]


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Full jitter exponential backoff (AWS recommendation)."""
    return min(cap, random.random() * base * (2 ** attempt))


def post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
) -> dict[str, Any]:
    """POST a JSON body with retry + jitter + optional model fallback.

    The ``body['model']`` field, if present, is rotated through the
    ``LLM_MODEL_FALLBACK`` chain on each retry attempt.

    Returns the parsed JSON response on 2xx. Raises the last-encountered
    exception otherwise.
    """
    primary_model = body.get("model") if isinstance(body, dict) else None
    chain = _fallback_chain(primary_model) if primary_model else [None]

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        # Pick model for this attempt (only if primary was set)
        if primary_model is not None:
            body = dict(body)  # avoid mutating caller's dict
            body["model"] = chain[min(attempt, len(chain) - 1)]

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in NON_RETRYABLE_STATUS or attempt == max_retries:
                raise
            if not _should_retry(e.code):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_exc = e
            if attempt == max_retries:
                raise

        sleep_s = _backoff_seconds(attempt, backoff_base, max_backoff)
        time.sleep(sleep_s)

    # Shouldn't reach here, but be defensive
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("post_json exhausted retries without raising")


__all__ = [
    "post_json",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_MAX_BACKOFF",
    "RETRYABLE_STATUS",
    "NON_RETRYABLE_STATUS",
]
