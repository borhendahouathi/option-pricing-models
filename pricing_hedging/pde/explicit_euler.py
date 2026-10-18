"""Explicit Euler solver for the forward-time Black-Scholes PDE.

Uses the simple 1D S/tau grids and initial/boundary helpers in grid.py.
The time step is rejected unless a sufficient max-norm stability bound holds.
"""

import math
import numpy as np

from pricing_hedging.pde.grid import BoundaryConditions, validate_grid


def check_explicit_stability(S, tau, r, sigma):
    """Check a sufficient, grid-uniform stability condition.

    For r >= 0 and sigma > 0, if
        dt <= 1 / (sigma**2 * (M-1)**2 + r),
    the diagonal of B = I + dt*A is nonnegative. The subordinate norm obeys
        ||B||_infinity <= 1 + dt * r**2 / (4*sigma**2),
    hence ||B**n||_infinity <= exp(r**2*T / (4*sigma**2)).

    The boundary values are handled separately from the interior matrix.
    This certifies stability of homogeneous error propagation, not an
    accuracy tolerance or equality to a market price.
    """
    S, tau = validate_grid(S, tau)
    dt = tau[1] - tau[0]
    if not np.isfinite(r) or r < 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("This stability certificate assumes r >= 0 and sigma > 0")

    M = len(S) - 1
    max_dt = 1.0 / (sigma**2 * (M - 1)**2 + r)
    minimum_N = math.ceil(tau[-1] / max_dt)

    if dt > max_dt * (1 + 1e-12):
        raise ValueError(
            f"Explicit timestep too large: dt={dt:.6g}, "
            f"allowed dt <= {max_dt:.6g}. "
            f"With M={M}, use N >= {minimum_N} (try a safety margin)."
        )

    # Spatial operator coefficients on interior nodes i=1,...,M-1.
    i = np.arange(1, M, dtype=float)
    a = 0.5 * (sigma**2 * i**2 - r * i)
    b = -sigma**2 * i**2 - r
    c = 0.5 * (sigma**2 * i**2 + r * i)

    # Infinity norm of the (M-1)x(M-1) interior update matrix.
    # First-row a_1 and last-row c_(M-1) are boundary contributions,
    # not matrix entries.
    row_sums = np.abs(1 + dt * b)
    row_sums += dt * np.abs(a) * (i > 1)
    row_sums += dt * np.abs(c) * (i < M - 1)
    norm_inf = float(np.max(row_sums))

    C = r**2 / (4 * sigma**2)
    if norm_inf > 1 + C * dt + 1e-10:
        raise AssertionError("Unexpected violation of the stability norm bound")

    return {
        "dt": dt,
        "max_dt": max_dt,
        "minimum_N": minimum_N,
        "matrix_norm_inf": norm_inf,
        "one_step_bound": 1 + C * dt,
        "power_bound_over_T": math.exp(C * tau[-1]),
    }


def explicit_euler(S, tau, K, r, sigma, option_type="call"):
    """Return U[n, i] ~= option value with tau[n] remaining and spot S[i].

    The stability check runs before time stepping. Uses a tridiagonal
    stencil without building a dense matrix or introducing classes.
    """
    check_explicit_stability(S, tau, r, sigma)
    U = BoundaryConditions(S, tau, option_type, r, K)

    M = len(S) - 1
    dt = tau[1] - tau[0]
    i = np.arange(1, M, dtype=float)
    a = 0.5 * (sigma**2 * i**2 - r * i)
    b = -sigma**2 * i**2 - r
    c = 0.5 * (sigma**2 * i**2 + r * i)

    for n in range(len(tau) - 1):
        U[n + 1, 1:-1] = U[n, 1:-1] + dt * (
            a * U[n, :-2]
            + b * U[n, 1:-1]
            + c * U[n, 2:]
        )

    return U
