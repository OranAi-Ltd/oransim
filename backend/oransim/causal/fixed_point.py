"""Fixed-point / equilibrium solver for cyclic Structural Causal Models.

The Oransim causal graph (``scm.py``) is cyclic by design — long-term
marketing feedback like ``repurchase → brand_equity → ecpm_bid →
next-cycle impression_dist`` cannot be expressed as a strict DAG without
losing the physics. Pearl's standard 3-step abduction is undefined on
cycles, so we use the **Bongers et al. 2021** generalisation: within the
feedback strongly-connected component (SCC), the intervention
``do(X=x)`` evaluates to the **equilibrium** of the structural system,
not a topological forward pass.

This module provides two solver paths:

  1. :func:`banach_iterate` — generic Banach fixed-point iteration for
     any callable ``f: R^n → R^n``. Converges when ``f`` is a
     contraction; we add a damping coefficient ``α ∈ (0, 1]`` so mild
     non-contractions can still converge (damped Picard).
  2. :func:`solve_linear_scm` — closed-form equilibrium for linear
     structural models ``x = Mx + b``. Solution is
     ``x* = (I - M)^(-1) b`` iff the spectral radius ``ρ(M) < 1``. This
     is the canonical linear-SCM treatment in Bongers 2021 §5 and
     costs a single ``np.linalg.solve`` per call.

Both paths return a :class:`FixedPointResult` with diagnostics
(converged / iteration count / residual / spectral radius where
applicable) so callers can surface convergence quality to the API
response rather than silently returning an unchecked estimate.

The module is intentionally free of Oransim-specific SCM logic; the
:mod:`oransim.causal.scm` module wires this solver to the 25-node
feedback SCC and exposes ``equilibrium_under_do()`` as the end-to-end
public entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class FixedPointResult:
    """Equilibrium solve outcome + diagnostics.

    ``x`` is the equilibrium estimate. ``converged`` is ``True`` iff the
    stopping criterion was met before ``max_iter``. ``residual_inf`` is
    ``||x_final - f(x_final)||_∞`` (or ``0.0`` for closed-form linear
    solves that hit the equation exactly up to numerical precision).
    """

    x: np.ndarray
    converged: bool
    n_iter: int
    residual_inf: float
    spectral_radius: float | None = None  # only set for linear-SCM path
    method: str = ""  # "banach" | "linear_closed_form"


def banach_iterate(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 200,
    damping: float = 1.0,
) -> FixedPointResult:
    """Damped Picard iteration ``x_{n+1} = (1-α) x_n + α f(x_n)``.

    Damping ``α = 1.0`` (default) is plain Banach iteration. Smaller ``α``
    helps non-contractive ``f`` by averaging the old and new iterate.

    Stops when ``||x_{n+1} - x_n||_∞ < tol`` or ``n_iter == max_iter``.
    """
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")

    x = np.asarray(x0, dtype=np.float64).copy()
    for n in range(1, max_iter + 1):
        fx = np.asarray(f(x), dtype=np.float64)
        x_next = (1.0 - damping) * x + damping * fx
        delta = float(np.max(np.abs(x_next - x)))
        x = x_next
        if delta < tol:
            # residual on the raw equation x = f(x)
            residual = float(np.max(np.abs(x - np.asarray(f(x), dtype=np.float64))))
            return FixedPointResult(
                x=x,
                converged=True,
                n_iter=n,
                residual_inf=residual,
                method="banach",
            )
    residual = float(np.max(np.abs(x - np.asarray(f(x), dtype=np.float64))))
    return FixedPointResult(
        x=x,
        converged=False,
        n_iter=max_iter,
        residual_inf=residual,
        method="banach",
    )


def solve_linear_scm(
    M: np.ndarray,
    b: np.ndarray,
    *,
    tol_spectral: float = 1.0,
) -> FixedPointResult:
    """Closed-form equilibrium ``x* = (I - M)^(-1) b`` for linear SCM.

    The linear structural system ``x = M x + b`` has a unique
    equilibrium iff ``I - M`` is invertible. The Banach fixed-point
    theorem additionally guarantees contraction-based convergence of
    :func:`banach_iterate` on the same system iff the spectral radius
    ``ρ(M) < 1``. We compute ``ρ(M)`` and report it so callers can
    distinguish a well-posed equilibrium (``ρ < 1``, iteration also
    converges) from a merely algebraically-solvable one (``ρ ≥ 1``,
    closed-form still works if ``I - M`` is non-singular, but dynamic
    iteration would not converge — typically a sign the model is
    mis-specified).

    Parameters
    ----------
    M : ``(n, n)`` structural coefficient matrix (``x = M x + b``).
    b : ``(n,)`` exogenous input vector.
    tol_spectral : ``ρ(M)`` strictly below this to count as a contraction.
        Default 1.0 (the math); set lower to be conservative.
    """
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = M.shape[0]
    if M.shape != (n, n):
        raise ValueError(f"M must be square, got shape {M.shape}")
    if b.shape != (n,):
        raise ValueError(f"b must be ({n},), got shape {b.shape}")

    # Spectral radius — max |eigenvalue|
    eigenvalues = np.linalg.eigvals(M)
    rho = float(np.max(np.abs(eigenvalues)))

    # Solve (I - M) x = b
    I = np.eye(n, dtype=np.float64)
    try:
        x = np.linalg.solve(I - M, b)
    except np.linalg.LinAlgError:
        # Singular (I - M) — no unique equilibrium
        return FixedPointResult(
            x=np.full(n, np.nan, dtype=np.float64),
            converged=False,
            n_iter=0,
            residual_inf=float("inf"),
            spectral_radius=rho,
            method="linear_closed_form",
        )

    residual = float(np.max(np.abs(x - (M @ x + b))))
    converged = rho < tol_spectral and np.isfinite(residual) and residual < 1e-8
    return FixedPointResult(
        x=x,
        converged=converged,
        n_iter=1,
        residual_inf=residual,
        spectral_radius=rho,
        method="linear_closed_form",
    )
