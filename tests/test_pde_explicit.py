"""Tests for explicit Euler using the project's Makegrid / BoundaryConditions API."""
import math
import numpy as np
import pytest

from pricing_hedging.pde.grid import Makegrid, BoundaryConditions
from pricing_hedging.pde.explicit_euler import explicit_euler, check_explicit_stability


def bs_price(spot, strike, rate, sigma, T, kind):
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    d1 = (math.log(spot / strike) + (rate + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = spot * Phi(d1) - strike * math.exp(-rate * T) * Phi(d2)
    return call if kind == "call" else call - spot + strike * math.exp(-rate * T)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_grid_boundaries_at_maturity(kind):
    S, taux = Makegrid(400, 1, 200, 2000)
    U = BoundaryConditions(S, taux, kind, 0.05, 400, 1, 100)
    if kind == "call":
        assert U[0, 0] == pytest.approx(0)
        assert U[0, -1] == pytest.approx(300)
        assert U[-1, -1] == pytest.approx(400 - 100 * np.exp(-0.05))
    else:
        assert U[0, 0] == pytest.approx(100)
        assert U[-1, 0] == pytest.approx(100 * np.exp(-0.05))
        assert U[0, -1] == pytest.approx(0)


def test_stability_check():
    S, taux = Makegrid(400, 1, 200, 2000)
    info = check_explicit_stability(S, taux, r=0.05, sigma=0.2)
    assert info["dt"] <= info["max_dt"]
    assert info["matrix_norm_inf"] <= info["one_step_bound"] + 1e-12


def test_reject_oversized_step():
    S, taux = Makegrid(400, 1, 200, 500)
    with pytest.raises(ValueError, match="N >="):
        explicit_euler(S, taux, K=100, r=0.05, sigma=0.2)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_euler_matches_bs(kind):
    S, taux = Makegrid(400, 1, 200, 2000)
    U = explicit_euler(S, taux, 100, 0.05, 0.2, kind)
    assert U.shape == (2001, 201)
    assert np.isfinite(U).all()
    assert np.allclose(U[0], BoundaryConditions(S, taux, kind, 0.05, 400, 1, 100)[0])
    i0 = np.argmin(abs(S - 100))
    assert abs(U[-1, i0] - bs_price(100, 100, 0.05, 0.2, 1, kind)) < 0.04
