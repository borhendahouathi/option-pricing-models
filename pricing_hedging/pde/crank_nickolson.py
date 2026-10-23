"""Crank--Nicolson solver for the forward-time Black--Scholes PDE."""

import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve

from pricing_hedging.pde.grid import BoundaryConditions, validate_grid

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


def solve_euleur_Nikelson(A, U_old, dt, left_boundary, right_boundary):
    """Advance one vector with the Crank--Nicolson update.

    Solves ``(I - dt*A/2) U_new = (I + dt*A/2) U_old`` while imposing
    the known boundary values at the new time level.
    """
    M1 = np.eye(A.shape[0]) - 0.5 * dt * A
    M2 = np.eye(A.shape[0]) + 0.5 * dt * A

    rhs = M2 @ U_old          # old time level only
    rhs[0] = left_boundary    # new time-level boundaries
    rhs[-1] = right_boundary

    return solve(M1, rhs)


def crank_nicolson(S, tau, K, r, sigma, option_type="call"):
    """Return ``U[n, i]`` at time ``tau[n]`` and spot ``S[i]``."""
    S, tau = validate_grid(S, tau)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    # U[0, :] is the payoff.  The left and right columns contain boundaries.
    U = BoundaryConditions(S, tau, option_type, r, K)
    A = create_matrix_A(len(tau) - 1, tau[-1], r, sigma, S)
    dt = tau[1] - tau[0]
    M1 = np.eye(len(S)) - 0.5 * dt * A
    M2 = np.eye(len(S)) + 0.5 * dt * A
    factorization = lu_factor(M1)

    for n in range(len(tau) - 1):
        rhs = M2 @ U[n, :]
        rhs[0] = U[n + 1, 0]
        rhs[-1] = U[n + 1, -1]
        U[n + 1, :] = lu_solve(factorization, rhs)
    return U


