"""Numerical Greeks and delta hedges derived from Black--Scholes PDE surfaces.

The PDE solvers in :mod:`pricing_hedging.pde` use ``tau = T - t`` and return
``U[n, i]`` at remaining maturity ``tau[n]`` and spot ``S[i]``.
"""

from __future__ import annotations

import numpy as np

from .pde.grid import Makegrid, validate_grid
from .pricing import payoff
from .simulation import simulate_gbm_paths


def pde_greeks(S, tau, U):
    """Return Delta, Gamma and calendar-time Theta on a PDE price surface.

    Delta and Gamma are centered spatial differences, so their values on the
    artificial spot-domain endpoints are ``NaN``.  Since the PDE time axis is
    remaining maturity, calendar-time Theta is ``-dU/dtau``.
    """
    S, tau, U = _validate_surface(S, tau, U, min_time_nodes=3)
    dS = S[1] - S[0]

    delta = np.full(U.shape, np.nan, dtype=float)
    gamma = np.full(U.shape, np.nan, dtype=float)
    delta[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2.0 * dS)
    gamma[:, 1:-1] = (U[:, 2:] - 2.0 * U[:, 1:-1] + U[:, :-2]) / dS**2
    theta = -np.gradient(U, tau, axis=0, edge_order=2)
    return {"delta": delta, "gamma": gamma, "theta": theta}


def interpolate_pde_value(S, tau, U, spots, remaining):
    """Bilinearly interpolate a price surface at spot(s) and remaining time.

    Price interpolation permits the PDE boundaries, unlike Delta interpolation.
    """
    S, tau, U = _validate_surface(S, tau, U)
    return _interpolate_surface(S, tau, U, spots, remaining, S[0], S[-1], "price")


def interpolate_pde_delta(S, tau, delta_grid, spots, remaining):
    """Bilinearly interpolate Delta without extrapolating endpoint derivatives."""
    S, tau, delta_grid = _validate_surface(S, tau, delta_grid, allow_endpoint_nans=True)
    if len(S) < 3:
        raise ValueError("Need at least three spot nodes for PDE Delta")
    return _interpolate_surface(
        S,
        tau,
        delta_grid[:, 1:-1],
        spots,
        remaining,
        S[1],
        S[-2],
        "Delta",
        interpolation_spots=S[1:-1],
    )


def pde_vega_rho(
    S,
    tau,
    K,
    r,
    sigma,
    option_type="call",
    volatility_bump=1e-3,
    rate_bump=1e-4,
    solver="explicit",
):
    """Return Vega and Rho surfaces by central bump-and-reprice.

    Vega and Rho are per unit change in volatility and interest rate,
    respectively, rather than per percentage point.
    """
    S, tau = validate_grid(S, tau)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    if not np.isfinite(r):
        raise ValueError("r must be finite")
    if not np.isfinite(volatility_bump) or not 0 < volatility_bump < sigma:
        raise ValueError("volatility_bump must be in (0, sigma)")
    if not np.isfinite(rate_bump) or not 0 < rate_bump <= r:
        raise ValueError("rate_bump must be positive and no larger than r")

    up_vol = _solve_pde(solver, S, tau, K, r, sigma + volatility_bump, option_type)
    down_vol = _solve_pde(solver, S, tau, K, r, sigma - volatility_bump, option_type)
    up_rate = _solve_pde(solver, S, tau, K, r + rate_bump, sigma, option_type)
    down_rate = _solve_pde(solver, S, tau, K, r - rate_bump, sigma, option_type)
    return {
        "vega": (up_vol - down_vol) / (2.0 * volatility_bump),
        "rho": (up_rate - down_rate) / (2.0 * rate_bump),
    }


def discrete_pde_delta_hedge(paths, times, r, S, tau, U, K, option_type):
    """Delta hedge simulated paths using an already-computed PDE surface."""
    paths, times, r = _validate_hedging_inputs(paths, times, r)
    S, tau, U = _validate_surface(S, tau, U, min_time_nodes=3)
    if not np.isclose(times[-1], tau[-1], rtol=1e-10, atol=1e-12):
        raise ValueError("paths maturity must match the PDE surface maturity")

    greeks = pde_greeks(S, tau, U)
    n_paths, n_dates = paths.shape
    remaining = tau[-1] - times
    initial_price = interpolate_pde_value(S, tau, U, paths[0, 0], remaining[0])

    shares = np.empty((n_paths, n_dates), dtype=float)
    cash = np.empty((n_paths, n_dates), dtype=float)
    portfolio_values = np.empty((n_paths, n_dates), dtype=float)
    shares[:, 0] = interpolate_pde_delta(S, tau, greeks["delta"], paths[:, 0], remaining[0])
    cash[:, 0] = initial_price - shares[:, 0] * paths[:, 0]
    portfolio_values[:, 0] = shares[:, 0] * paths[:, 0] + cash[:, 0]

    for date_index in range(1, n_dates):
        dt = times[date_index] - times[date_index - 1]
        accrued_cash = cash[:, date_index - 1] * np.exp(r * dt)
        if date_index == n_dates - 1:
            shares[:, date_index] = shares[:, date_index - 1]
            cash[:, date_index] = accrued_cash
        else:
            new_delta = interpolate_pde_delta(
                S, tau, greeks["delta"], paths[:, date_index], remaining[date_index]
            )
            shares[:, date_index] = new_delta
            cash[:, date_index] = accrued_cash - (
                new_delta - shares[:, date_index - 1]
            ) * paths[:, date_index]
        portfolio_values[:, date_index] = (
            shares[:, date_index] * paths[:, date_index] + cash[:, date_index]
        )

    terminal_payoff = payoff(K, paths[:, -1], option_type)
    return {
        "times": times,
        "paths": paths,
        "shares": shares,
        "cash": cash,
        "portfolio_values": portfolio_values,
        "initial_option_price": float(initial_price),
        "terminal_payoff": terminal_payoff,
        "hedging_error": portfolio_values[:, -1] - terminal_payoff,
    }


def simulate_pde_delta_hedge(
    S0,
    r,
    realized_sigma,
    hedge_sigma,
    tau,
    K,
    option_type,
    n_rebalances,
    n_paths,
    Smax,
    M,
    N,
    solver="explicit",
    seed=None,
):
    """Build a PDE surface, simulate GBM paths, and apply a PDE Delta hedge."""
    if not np.isfinite(realized_sigma) or realized_sigma <= 0:
        raise ValueError("realized_sigma must be strictly positive")
    if not np.isfinite(hedge_sigma) or hedge_sigma <= 0:
        raise ValueError("hedge_sigma must be strictly positive")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be strictly positive")

    S, pde_tau = Makegrid(Smax, tau, M, N)
    U = _solve_pde(solver, S, pde_tau, K, r, hedge_sigma, option_type)
    times, paths = simulate_gbm_paths(
        S0, r, realized_sigma, tau, n_rebalances, n_paths, seed=seed
    )
    result = discrete_pde_delta_hedge(paths, times, r, S, pde_tau, U, K, option_type)
    result.update(
        {
            "S": S,
            "tau": pde_tau,
            "price_surface": U,
            "solver": _normalise_solver(solver),
            "realized_sigma": float(realized_sigma),
            "hedge_sigma": float(hedge_sigma),
        }
    )
    return result


def _solve_pde(solver, S, tau, K, r, sigma, option_type):
    solver = _normalise_solver(solver)
    if solver == "explicit":
        from .pde.explicit_euler import explicit_euler

        return explicit_euler(S, tau, K, r, sigma, option_type)
    if solver == "implicit":
        from .pde.implicit_euler import implicit_euler

        return implicit_euler(S, tau, K, r, sigma, option_type)

    from .pde.crank_nickolson import crank_nicolson

    return crank_nicolson(S, tau, K, r, sigma, option_type)


def _normalise_solver(solver):
    solver = str(solver).lower()
    aliases = {"explicit": "explicit", "implicit": "implicit", "crank_nicolson": "crank_nicolson"}
    if solver not in aliases:
        raise ValueError("solver must be 'explicit', 'implicit', or 'crank_nicolson'")
    return aliases[solver]


def _validate_surface(S, tau, U, min_time_nodes=2, allow_endpoint_nans=False):
    S, tau = validate_grid(S, tau)
    U = np.asarray(U, dtype=float)
    if len(tau) < min_time_nodes:
        raise ValueError(f"Need at least {min_time_nodes} time nodes")
    if U.shape != (len(tau), len(S)):
        raise ValueError("U must have shape (len(tau), len(S))")
    valid_values = U[:, 1:-1] if allow_endpoint_nans else U
    if not np.all(np.isfinite(valid_values)):
        raise ValueError("U must contain finite values")
    return S, tau, U


def _interpolate_surface(S, tau, surface, spots, remaining, lower, upper, label, interpolation_spots=None):
    spots_array = np.asarray(spots, dtype=float)
    if not np.isfinite(remaining) or remaining < tau[0] - 1e-12 or remaining > tau[-1] + 1e-12:
        raise ValueError("remaining maturity is outside the PDE time grid")
    if not np.all(np.isfinite(spots_array)):
        raise ValueError("spot prices must be finite")
    if np.any(spots_array < lower) or np.any(spots_array > upper):
        raise ValueError(
            f"Spot outside reliable PDE {label} domain [{lower}, {upper}]; enlarge Smax or adjust the spatial grid."
        )
    remaining = float(np.clip(remaining, tau[0], tau[-1]))
    j = min(int(np.searchsorted(tau, remaining, side="right") - 1), len(tau) - 2)
    weight = (remaining - tau[j]) / (tau[j + 1] - tau[j])
    x = S if interpolation_spots is None else interpolation_spots
    at_lower = np.interp(spots_array, x, surface[j])
    at_upper = np.interp(spots_array, x, surface[j + 1])
    result = (1.0 - weight) * at_lower + weight * at_upper
    return float(result) if spots_array.ndim == 0 else result


def _validate_hedging_inputs(paths, times, r):
    paths = np.asarray(paths, dtype=float)
    times = np.asarray(times, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_dates) with at least two dates")
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise ValueError("paths must contain finite, strictly positive spot values")
    if not np.allclose(paths[:, 0], paths[0, 0], rtol=1e-12, atol=1e-12):
        raise ValueError("all paths must share the same initial spot")
    if times.ndim != 1 or len(times) != paths.shape[1]:
        raise ValueError("times must be a one-dimensional grid matching paths")
    if not np.all(np.isfinite(times)) or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite, start at zero, and be strictly increasing")
    if not np.isscalar(r) or not np.isfinite(r):
        raise ValueError("r must be a finite number")
    return paths, times, float(r)
