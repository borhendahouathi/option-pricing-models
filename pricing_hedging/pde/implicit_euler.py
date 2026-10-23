"""Implicit-Euler solver for the forward-time Black--Scholes PDE."""

import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve

from pricing_hedging.pde.grid import BoundaryConditions, Makegrid, validate_grid

def create_matrix_A(N, T, r, sigma, x):
    """Build the spatial Black--Scholes operator on the spot grid ``x``.

    ``N`` and ``T`` are retained for compatibility with the original code;
    the spatial operator itself only depends on ``x``, ``r``, and ``sigma``.
    """
    del N, T
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    A = np.zeros((len(x), len(x)))

    for i in range(1, len(x) - 1):
        s = x[i]
        lower = 0.5 * sigma**2 * s**2 / dx**2 - 0.5 * r * s / dx
        diagonal = -sigma**2 * s**2 / dx**2 - r
        upper = 0.5 * sigma**2 * s**2 / dx**2 + 0.5 * r * s / dx

        A[i, i - 1] = lower
        A[i, i] = diagonal
        A[i, i + 1] = upper

    return A


def solve_euleur_implicit(A, U, dt):
    """Advance one price vector: ``(I - dt * A) U_new = U``."""
    A = np.asarray(A, dtype=float)
    U = np.asarray(U, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if U.ndim != 1 or len(U) != A.shape[0]:
        raise ValueError("U must be a vector compatible with A")

    matrix = np.eye(A.shape[0]) - dt * A
    return solve(matrix, U)


def implicit_euler(S, tau, K, r, sigma, option_type="call"):
    """Return ``U[n, i]`` at time ``tau[n]`` and spot ``S[i]``."""
    S, tau = validate_grid(S, tau)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    # U[0, :] is the payoff.  The left and right columns contain boundaries.
    U = BoundaryConditions(S, tau, option_type, r, K)
    A = create_matrix_A(len(tau) - 1, tau[-1], r, sigma, S)
    dt = tau[1] - tau[0]
    factorization = lu_factor(np.eye(len(S)) - dt * A)

    for n in range(len(tau) - 1):
        rhs = U[n, :].copy()
        # Impose the known boundary values at tau[n + 1].
        rhs[0] = U[n + 1, 0]
        rhs[-1] = U[n + 1, -1]
        U[n + 1, :] = lu_solve(factorization, rhs)

    return U


