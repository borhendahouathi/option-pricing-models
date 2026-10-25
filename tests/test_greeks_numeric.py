import numpy as np
import pytest

from pricing_hedging.greeks_numeric import (
    discrete_pde_delta_hedge,
    interpolate_pde_delta,
    interpolate_pde_value,
    pde_greeks,
    pde_vega_rho,
)
from pricing_hedging.pde.explicit_euler import explicit_euler
from pricing_hedging.pde.grid import Makegrid
from pricing_hedging.pricing import black_scholes_price


S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0


@pytest.fixture(scope="module")
def explicit_surface():
    S, tau = Makegrid(400.0, T, 160, 1_500)
    return S, tau, explicit_euler(S, tau, K, R, SIGMA, "call")


def test_pde_greeks_have_endpoint_nans_and_reasonable_delta(explicit_surface):
    S, tau, U = explicit_surface
    greeks = pde_greeks(S, tau, U)

    assert np.isnan(greeks["delta"][:, [0, -1]]).all()
    assert np.isnan(greeks["gamma"][:, [0, -1]]).all()
    assert np.isfinite(greeks["theta"]).all()
    delta = interpolate_pde_delta(S, tau, greeks["delta"], S0, T)
    assert delta == pytest.approx(0.6368306512, abs=0.015)


def test_surface_interpolation_and_delta_domain_guardrails(explicit_surface):
    S, tau, U = explicit_surface
    value = interpolate_pde_value(S, tau, U, S0, T)
    reference = black_scholes_price(SIGMA, R, T, K, S0, "call")
    assert value == pytest.approx(reference, abs=0.08)

    delta = pde_greeks(S, tau, U)["delta"]
    with pytest.raises(ValueError, match="reliable PDE Delta domain"):
        interpolate_pde_delta(S, tau, delta, S[0], T)
    with pytest.raises(ValueError, match="outside the PDE time grid"):
        interpolate_pde_value(S, tau, U, S0, T + 0.1)


def test_pde_hedge_is_self_financing(explicit_surface):
    S, tau, U = explicit_surface
    paths = np.array([[100.0, 110.0, 120.0]])
    times = np.array([0.0, 0.5, 1.0])
    result = discrete_pde_delta_hedge(paths, times, R, S, tau, U, K, "call")

    initial_cash = result["cash"][0, 0]
    rebalanced_cash = result["cash"][0, 1]
    expected_terminal_cash = rebalanced_cash * np.exp(R * 0.5)
    assert np.isfinite(result["shares"]).all()
    assert result["cash"][0, 2] == pytest.approx(expected_terminal_cash)
    assert result["portfolio_values"][0, -1] == pytest.approx(
        result["shares"][0, -1] * 120.0 + expected_terminal_cash
    )
    assert initial_cash == pytest.approx(
        result["initial_option_price"] - result["shares"][0, 0] * 100.0
    )


@pytest.mark.parametrize(
    ("solver", "M", "N"),
    [("explicit", 30, 100), ("implicit", 30, 20), ("crank_nicolson", 30, 20)],
)
def test_pde_vega_rho_supports_all_solvers(solver, M, N):
    S, tau = Makegrid(300.0, T, M, N)
    result = pde_vega_rho(S, tau, K, R, SIGMA, solver=solver)
    assert result["vega"].shape == (N + 1, M + 1)
    assert result["rho"].shape == (N + 1, M + 1)
    assert np.isfinite(result["vega"]).all()
    assert np.isfinite(result["rho"]).all()


def test_pde_vega_rho_rejects_negative_rate_bump_surface():
    S, tau = Makegrid(300.0, T, 30, 100)
    with pytest.raises(ValueError, match="no larger than r"):
        pde_vega_rho(S, tau, K, 0.0, SIGMA)
