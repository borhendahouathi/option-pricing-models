"""Black--Scholes Greeks and discrete self-financing delta hedges."""

from __future__ import annotations

from math import exp, sqrt

import numpy as np
from scipy.stats import norm

from .pricing import black_scholes_price, payoff
from .simulation import simulate_gbm_paths


def black_scholes_greeks(sigma, r, tau, K, S0, option_type):
    """Return analytical Black--Scholes Greeks for a European call or put.

    Theta is calendar-time theta, dV/dt with maturity held fixed. Volatility
    and remaining maturity must be strictly positive.
    """
    sigma, r, tau, K, S0, option_type = _validate_greek_inputs(
        sigma, r, tau, K, S0, option_type
    )
    d1, d2 = _d1_d2(S0, K, r, sigma, tau)
    density = norm.pdf(d1)
    discount = exp(-r * tau)
    gamma = density / (S0 * sigma * sqrt(tau))
    vega = S0 * density * sqrt(tau)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -(S0 * density * sigma) / (2 * sqrt(tau)) - r * K * discount * norm.cdf(d2)
        rho = K * tau * discount * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(S0 * density * sigma) / (2 * sqrt(tau)) + r * K * discount * norm.cdf(-d2)
        rho = -K * tau * discount * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


def discrete_delta_hedge(paths, times, r, hedge_sigma, K, option_type):
    """Delta hedge an option along paths with self-financing rebalancing."""
    paths, times, r, hedge_sigma, K, option_type = _validate_hedging_inputs(
        paths, times, r, hedge_sigma, K, option_type
    )
    n_paths, n_dates = paths.shape
    maturity = times[-1]
    initial_spot = float(paths[0, 0])
    initial_price = black_scholes_price(
        hedge_sigma, r, maturity, K, initial_spot, option_type
    )

    shares = np.empty((n_paths, n_dates), dtype=float) #number of shares held in the underlying delta hedge
    cash = np.empty((n_paths, n_dates), dtype=float) #money in the bank account
    portfolio_values = np.empty((n_paths, n_dates), dtype=float) #total value of the portfolio (shares + cash)
    shares[:, 0] = _delta(paths[:, 0], K, r, hedge_sigma, maturity, option_type)
    cash[:, 0] = initial_price - shares[:, 0] * paths[:, 0]
    portfolio_values[:, 0] = shares[:, 0] * paths[:, 0] + cash[:, 0]

    for date_index in range(1, n_dates):
        dt = times[date_index] - times[date_index - 1]
        accrued_cash = cash[:, date_index - 1] * np.exp(r * dt)
        if date_index == n_dates - 1:
            shares[:, date_index] = shares[:, date_index - 1]
            cash[:, date_index] = accrued_cash
        else:
            new_delta = _delta(
                paths[:, date_index], K, r, hedge_sigma,
                maturity - times[date_index], option_type,
            )
            shares[:, date_index] = new_delta
            cash[:, date_index] = accrued_cash - (
                new_delta - shares[:, date_index - 1]
            ) * paths[:, date_index] #To obtain new stock postion we need to exchange cash for stock. 
        portfolio_values[:, date_index] = (
            shares[:, date_index] * paths[:, date_index] + cash[:, date_index]
        ) #because the portfolio is self-financing we only rearrange the cash and stock positions. Xt_i=X(t_i)-: value before update = value after update
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


def simulate_delta_hedge(
    S0, r, realized_sigma, hedge_sigma, tau, K, option_type,
    n_rebalances, n_paths, seed=None,
):
    """Simulate risk-neutral paths and apply a discrete delta hedge."""
    _validate_scalar("realized_sigma", realized_sigma, strictly_positive=True)
    _validate_scalar("hedge_sigma", hedge_sigma, strictly_positive=True)
    _validate_scalar("tau", tau, strictly_positive=True)
    times, paths = simulate_gbm_paths(
        S0, r, realized_sigma, tau, n_rebalances, n_paths, seed=seed
    )
    result = discrete_delta_hedge(paths, times, r, hedge_sigma, K, option_type)
    result["realized_sigma"] = float(realized_sigma)
    result["hedge_sigma"] = float(hedge_sigma)
    return result


def _delta(spot, K, r, sigma, tau, option_type):
    d1, _ = _d1_d2(np.asarray(spot, dtype=float), K, r, sigma, tau)
    call_delta = norm.cdf(d1)
    return call_delta if option_type == "call" else call_delta - 1.0


def _d1_d2(spot, K, r, sigma, tau):
    d1 = (
        np.log(np.asarray(spot, dtype=float) / K)
        + (r + sigma**2 / 2) * tau
    ) / (sigma * np.sqrt(tau))
    return d1, d1 - sigma * np.sqrt(tau)


def _validate_greek_inputs(sigma, r, tau, K, S0, option_type):
    _validate_scalar("sigma", sigma, strictly_positive=True)
    _validate_scalar("r", r)
    _validate_scalar("tau", tau, strictly_positive=True)
    _validate_scalar("K", K, strictly_positive=True)
    _validate_scalar("S0", S0, strictly_positive=True)
    return (
        float(sigma), float(r), float(tau), float(K), float(S0),
        _validate_option_type(option_type),
    )


def _validate_hedging_inputs(paths, times, r, hedge_sigma, K, option_type):
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
    sigma, r, _, K, _, option_type = _validate_greek_inputs(
        hedge_sigma, r, times[-1], K, paths[0, 0], option_type
    )
    return paths, times, r, sigma, K, option_type


def _validate_scalar(name, value, strictly_positive=False):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if strictly_positive and value <= 0:
        raise ValueError(f"{name} must be strictly positive")


def _validate_option_type(option_type):
    option_type = str(option_type).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type
