import math

import numpy as np
import pytest
from scipy.stats import norm

from pricing_hedging.greeks import (
    black_scholes_greeks,
    discrete_delta_hedge,
    simulate_delta_hedge,
)


S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.20
TAU = 1.0


def test_black_scholes_greeks_match_known_call_values():
    greeks = black_scholes_greeks(SIGMA, R, TAU, K, S0, "call")

    assert greeks["delta"] == pytest.approx(0.6368306512, abs=1e-10)
    assert greeks["gamma"] == pytest.approx(0.0187620173, abs=1e-10)
    assert greeks["vega"] == pytest.approx(37.5240347, abs=1e-7)
    assert greeks["theta"] == pytest.approx(-6.4140275, abs=1e-7)
    assert greeks["rho"] == pytest.approx(53.2324815, abs=1e-7)


def test_call_and_put_share_gamma_vega_and_have_expected_signs():
    call = black_scholes_greeks(SIGMA, R, TAU, K, S0, "call")
    put = black_scholes_greeks(SIGMA, R, TAU, K, S0, "put")

    assert call["gamma"] == pytest.approx(put["gamma"])
    assert call["vega"] == pytest.approx(put["vega"])
    assert call["delta"] - put["delta"] == pytest.approx(1.0)
    assert call["rho"] > 0
    assert put["rho"] < 0
    assert call["theta"] < 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0.0, "r": R, "tau": TAU, "K": K, "S0": S0, "option_type": "call"},
        {"sigma": SIGMA, "r": R, "tau": 0.0, "K": K, "S0": S0, "option_type": "call"},
        {"sigma": SIGMA, "r": R, "tau": TAU, "K": 0.0, "S0": S0, "option_type": "call"},
        {"sigma": SIGMA, "r": R, "tau": TAU, "K": K, "S0": S0, "option_type": "straddle"},
    ],
)
def test_greeks_reject_undefined_or_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        black_scholes_greeks(**kwargs)


def test_hedging_rejects_invalid_paths_times_and_volatility():
    paths = np.array([[100.0, 110.0]])
    times = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        discrete_delta_hedge(paths, times, R, 0.0, K, "call")
    with pytest.raises(ValueError):
        discrete_delta_hedge(paths, np.array([0.1, 1.0]), R, SIGMA, K, "call")
    with pytest.raises(ValueError):
        discrete_delta_hedge(np.array([[100.0, -1.0]]), times, R, SIGMA, K, "call")


def test_discrete_hedge_is_self_financing_and_has_expected_terminal_error():
    times = np.array([0.0, 0.5, 1.0])
    paths = np.array([[100.0, 110.0, 120.0]])
    result = discrete_delta_hedge(paths, times, R, SIGMA, K, "call")

    initial_delta = norm.cdf(
        (math.log(S0 / K) + (R + SIGMA**2 / 2) * TAU) / (SIGMA * math.sqrt(TAU))
    )
    initial_cash = result["initial_option_price"] - initial_delta * S0
    rebalanced_delta = black_scholes_greeks(SIGMA, R, 0.5, K, 110.0, "call")["delta"]
    rebalanced_cash = initial_cash * math.exp(R * 0.5) - (rebalanced_delta - initial_delta) * 110.0
    terminal_cash = rebalanced_cash * math.exp(R * 0.5)
    expected_terminal_value = rebalanced_delta * 120.0 + terminal_cash

    assert result["shares"][0, 0] == pytest.approx(initial_delta)
    assert result["cash"][0, 0] == pytest.approx(initial_cash)
    assert result["shares"][0, 1] == pytest.approx(rebalanced_delta)
    assert result["cash"][0, 1] == pytest.approx(rebalanced_cash)
    assert result["shares"][0, 2] == pytest.approx(rebalanced_delta)
    assert result["cash"][0, 2] == pytest.approx(terminal_cash)
    assert result["portfolio_values"][0, -1] == pytest.approx(expected_terminal_value)
    assert result["hedging_error"][0] == pytest.approx(expected_terminal_value - 20.0)


def test_simulated_hedge_is_reproducible():
    first = simulate_delta_hedge(
        S0, R, SIGMA, SIGMA, TAU, K, "call", 24, 2_000, seed=17
    )
    second = simulate_delta_hedge(
        S0, R, SIGMA, SIGMA, TAU, K, "call", 24, 2_000, seed=17
    )

    assert np.array_equal(first["paths"], second["paths"])
    assert np.array_equal(first["hedging_error"], second["hedging_error"])


def test_more_frequent_ideal_rebalancing_reduces_error_dispersion():
    coarse = simulate_delta_hedge(
        S0, R, SIGMA, SIGMA, TAU, K, "call", 12, 10_000, seed=123
    )
    fine = simulate_delta_hedge(
        S0, R, SIGMA, SIGMA, TAU, K, "call", 104, 10_000, seed=123
    )

    assert abs(fine["hedging_error"].mean()) < 0.08
    assert fine["hedging_error"].std(ddof=1) < coarse["hedging_error"].std(ddof=1)


def test_volatility_mismatch_changes_hedging_error_distribution():
    ideal = simulate_delta_hedge(
        S0, R, SIGMA, SIGMA, TAU, K, "call", 52, 10_000, seed=9
    )
    mismatched = simulate_delta_hedge(
        S0, R, 0.30, SIGMA, TAU, K, "call", 52, 10_000, seed=9
    )

    assert np.isfinite(mismatched["hedging_error"]).all()
    assert abs(mismatched["hedging_error"].mean() - ideal["hedging_error"].mean()) > 1.0
