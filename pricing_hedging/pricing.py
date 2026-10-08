"""Option-pricing functions developed from the original project notebook."""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm


def payoff(K, spot, option_type):
    """Return the call or put payoff for one spot or an array of spots."""
    _validate_positive("K", K)
    option_type = _validate_option_type(option_type)
    spot_values = np.asarray(spot, dtype=float)
    if np.any(~np.isfinite(spot_values)) or np.any(spot_values < 0):
        raise ValueError("spot values must be finite and non-negative")

    if option_type == "call":
        result = np.maximum(spot_values - K, 0.0)
    else:
        result = np.maximum(K - spot_values, 0.0)
    return float(result) if result.ndim == 0 else result


def black_scholes_price(sigma, r, tau, K, S0, option_type):
    """Price a European call or put with the Black-Scholes formula."""
    _validate_inputs(S0, K, r, sigma, tau)
    option_type = _validate_option_type(option_type)

    if tau == 0:
        return payoff(K, S0, option_type)

    discounted_strike = K * exp(-r * tau)
    if sigma == 0:
        if option_type == "call":
            return max(S0 - discounted_strike, 0.0)
        return max(discounted_strike - S0, 0.0)

    d1 = (log(S0 / K) + (r + sigma**2 / 2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)

    if option_type == "call":
        return float(S0 * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def build_binomial_tree(sigma, r, tau, N, S0):
    """Build the full CRR stock tree, following the original BTree function."""
    _validate_inputs(S0, 1.0, r, sigma, tau)
    N = _validate_integer("N", N)

    if tau == 0:
        return np.array([[float(S0)]]), None
    if sigma == 0:
        dt = tau / N
        stock_tree = np.full((N + 1, N + 1), np.nan)
        for j in range(N + 1):
            stock_tree[j, : j + 1] = S0 * exp(r * dt * j)
        return stock_tree, None

    dt = tau / N
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    _validate_probability(p)

    stock_tree = np.full((N + 1, N + 1), np.nan)
    for j in range(N + 1):
        number_of_up_moves = np.arange(j + 1)
        stock_tree[j, : j + 1] = (
            S0 * d ** (j - number_of_up_moves) * u**number_of_up_moves
        )
    return stock_tree, p


def crr_price(
    sigma,
    r,
    tau,
    N,
    S0,
    K,
    option_type,
    exercise="european",
    store_tree=False,
):
    """Price a European or American option with the CRR tree."""
    _validate_inputs(S0, K, r, sigma, tau)
    N = _validate_integer("N", N)
    option_type = _validate_option_type(option_type)
    exercise = _validate_exercise(exercise)

    if tau == 0:
        price = payoff(K, S0, option_type)
        if store_tree:
            return {
                "price": price,
                "risk_neutral_probability": None,
                "stock_tree": np.array([[S0]], dtype=float),
                "option_tree": np.array([[price]], dtype=float),
            }
        return price

    if sigma == 0:
        return _deterministic_tree_price(
            r, tau, N, S0, K, option_type, exercise, store_tree
        )

    dt = tau / N
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    _validate_probability(p)
    p = min(max(p, 0.0), 1.0)
    discount = exp(-r * dt)

    # A standalone price only needs one column, so memory use stays O(N).
    number_of_up_moves = np.arange(N + 1)
    terminal_spots = S0 * d ** (N - number_of_up_moves) * u**number_of_up_moves
    option_values = payoff(K, terminal_spots, option_type)

    stock_tree = option_tree = None
    if store_tree:
        stock_tree, _ = build_binomial_tree(sigma, r, tau, N, S0)
        option_tree = np.full((N + 1, N + 1), np.nan)
        option_tree[N, : N + 1] = option_values

    for j in range(N - 1, -1, -1):
        option_values = discount * (
            (1 - p) * option_values[: j + 1] + p * option_values[1 : j + 2]
        )
        if exercise == "american":
            number_of_up_moves = np.arange(j + 1)
            spots = S0 * d ** (j - number_of_up_moves) * u**number_of_up_moves
            option_values = np.maximum(
                option_values, payoff(K, spots, option_type)
            )
        if store_tree:
            option_tree[j, : j + 1] = option_values

    price = float(option_values[0])
    if store_tree:
        return {
            "price": price,
            "risk_neutral_probability": p,
            "stock_tree": stock_tree,
            "option_tree": option_tree,
        }
    return price


def monte_carlo_price(S0, r, sigma, tau, n_paths, K, option_type, seed=None):
    """Price a European option from directly simulated terminal prices."""
    _validate_inputs(S0, K, r, sigma, tau)
    n_paths = _validate_integer("n_paths", n_paths, minimum=2)
    option_type = _validate_option_type(option_type)

    if tau == 0:
        price = payoff(K, S0, option_type)
        return {
            "price": price,
            "standard_error": 0.0,
            "confidence_interval": (price, price),
            "n_paths": n_paths,
        }

    random_generator = np.random.default_rng(seed)
    Z = random_generator.standard_normal(n_paths)
    terminal_spots = S0 * np.exp(
        (r - sigma**2 / 2) * tau + sigma * sqrt(tau) * Z
    )
    discounted_payoffs = exp(-r * tau) * payoff(K, terminal_spots, option_type)

    price = float(discounted_payoffs.mean())
    standard_error = float(discounted_payoffs.std(ddof=1) / sqrt(n_paths))
    confidence_radius = 1.96 * standard_error
    return {
        "price": price,
        "standard_error": standard_error,
        "confidence_interval": (
            price - confidence_radius,
            price + confidence_radius,
        ),
        "n_paths": n_paths,
    }


def _deterministic_tree_price(r, tau, N, S0, K, option_type, exercise, store_tree):
    """Handle sigma=0 without dividing by u-d=0."""
    dt = tau / N
    discount = exp(-r * dt)
    spots = S0 * np.exp(r * dt * np.arange(N + 1))
    value = payoff(K, spots[-1], option_type)
    values = np.empty(N + 1)
    values[-1] = value

    for j in range(N - 1, -1, -1):
        value *= discount
        if exercise == "american":
            value = max(value, payoff(K, spots[j], option_type))
        values[j] = value

    if not store_tree:
        return float(value)

    stock_tree = np.full((N + 1, N + 1), np.nan)
    option_tree = np.full((N + 1, N + 1), np.nan)
    for j in range(N + 1):
        stock_tree[j, : j + 1] = spots[j]
        option_tree[j, : j + 1] = values[j]
    return {
        "price": float(value),
        "risk_neutral_probability": None,
        "stock_tree": stock_tree,
        "option_tree": option_tree,
    }


def _validate_inputs(S0, K, r, sigma, tau):
    for name, value in [("S0", S0), ("K", K), ("r", r), ("sigma", sigma), ("tau", tau)]:
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    _validate_positive("S0", S0)
    _validate_positive("K", K)
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if tau < 0:
        raise ValueError("tau cannot be negative")


def _validate_positive(name, value):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, strictly positive number")


def _validate_integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _validate_option_type(option_type):
    value = str(option_type).lower()
    if value not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return value


def _validate_exercise(exercise):
    value = str(exercise).lower()
    if value not in {"european", "american"}:
        raise ValueError("exercise must be 'european' or 'american'")
    return value


def _validate_probability(p):
    if p < -1e-14 or p > 1 + 1e-14:
        raise ValueError(
            "CRR risk-neutral probability is outside [0, 1]; "
            "increase N or check r and sigma"
        )
