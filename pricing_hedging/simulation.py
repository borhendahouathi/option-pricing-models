"""Functions for simulating full stock-price paths."""

from math import sqrt

import numpy as np


def simulate_gbm_paths(S0, drift, sigma, tau, N, n_paths, seed=None):
    """Simulate full geometric Brownian motion paths with the exact formula."""
    _validate_simulation_inputs(S0, drift, sigma, tau, N, n_paths)

    dt = tau / N
    times = np.linspace(0, tau, N + 1)
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0

    if tau == 0:
        paths[:, 1:] = S0
        return times, paths

    random_generator = np.random.default_rng(seed)
    for i in range(N):
        Z = random_generator.standard_normal(n_paths)
        paths[:, i + 1] = paths[:, i] * np.exp(
            (drift - sigma**2 / 2) * dt + sigma * sqrt(dt) * Z
        )

    return times, paths


def _validate_simulation_inputs(S0, drift, sigma, tau, N, n_paths):
    for name, value in [("S0", S0), ("drift", drift), ("sigma", sigma), ("tau", tau)]:
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if S0 <= 0:
        raise ValueError("S0 must be strictly positive")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if tau < 0:
        raise ValueError("tau cannot be negative")
    for name, value in [("N", N), ("n_paths", n_paths)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

