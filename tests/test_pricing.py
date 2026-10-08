import math

import numpy as np
import pytest

from pricing_hedging import (
    black_scholes_price,
    build_binomial_tree,
    crr_price,
    monte_carlo_price,
    simulate_gbm_paths,
)


S0 = 36.0
K = 35.0
R = 0.06
SIGMA = 0.20
TAU = 1.0


def test_black_scholes_reference_and_put_call_parity():
    call = black_scholes_price(SIGMA, R, TAU, K, S0, "call")
    put = black_scholes_price(SIGMA, R, TAU, K, S0, "put")
    assert call == pytest.approx(4.5272965347, abs=1e-10)
    assert call - put == pytest.approx(S0 - K * math.exp(-R * TAU), abs=1e-12)


def test_expiry_and_zero_volatility():
    assert black_scholes_price(SIGMA, R, 0.0, K, S0, "call") == 1.0
    expected = max(S0 - K * math.exp(-R * TAU), 0.0)
    assert black_scholes_price(0.0, R, TAU, K, S0, "call") == pytest.approx(expected)
    assert crr_price(0.0, R, TAU, 10, S0, K, "call") == pytest.approx(expected)


def test_crr_converges_and_tree_storage_is_optional():
    reference = black_scholes_price(SIGMA, R, TAU, K, S0, "call")
    compact_price = crr_price(SIGMA, R, TAU, 1_000, S0, K, "call")
    full = crr_price(SIGMA, R, TAU, 10, S0, K, "call", store_tree=True)
    assert compact_price == pytest.approx(reference, abs=3e-4)
    assert isinstance(compact_price, float)
    assert full["stock_tree"].shape == (11, 11)
    assert full["option_tree"][0, 0] == pytest.approx(full["price"])
    assert np.isnan(full["stock_tree"][0, 1])


def test_build_binomial_tree_matches_original_tree_purpose():
    stock_tree, p = build_binomial_tree(SIGMA, R, TAU, 2, S0)
    assert stock_tree.shape == (3, 3)
    assert stock_tree[0, 0] == S0
    assert 0 < p < 1
    assert stock_tree[2, 1] == pytest.approx(S0)


def test_american_put_is_at_least_european_put():
    european = crr_price(SIGMA, R, TAU, 500, S0, K, "put")
    american = crr_price(
        SIGMA, R, TAU, 500, S0, K, "put", exercise="american"
    )
    assert american >= european


def test_monte_carlo_returns_uncertainty_and_is_reproducible():
    first = monte_carlo_price(S0, R, SIGMA, TAU, 100_000, K, "call", seed=7)
    second = monte_carlo_price(S0, R, SIGMA, TAU, 100_000, K, "call", seed=7)
    reference = black_scholes_price(SIGMA, R, TAU, K, S0, "call")
    assert first == second
    assert first["standard_error"] > 0
    assert first["confidence_interval"][0] < first["price"] < first["confidence_interval"][1]
    assert abs(first["price"] - reference) < 4 * first["standard_error"]


def test_full_path_simulation_shape_and_risk_neutral_mean():
    times, paths = simulate_gbm_paths(S0, R, SIGMA, TAU, 12, 50_000, seed=11)
    assert paths.shape == (50_000, 13)
    assert times.shape == (13,)
    expected_mean = S0 * math.exp(R * TAU)
    assert paths[:, -1].mean() == pytest.approx(expected_mean, rel=0.004)


@pytest.mark.parametrize(
    "call",
    [
        lambda: black_scholes_price(SIGMA, R, TAU, K, 0.0, "call"),
        lambda: black_scholes_price(SIGMA, R, TAU, -1.0, S0, "call"),
        lambda: black_scholes_price(-0.1, R, TAU, K, S0, "call"),
        lambda: black_scholes_price(SIGMA, R, -1.0, K, S0, "call"),
        lambda: black_scholes_price(SIGMA, R, TAU, K, S0, "invalid"),
    ],
)
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
