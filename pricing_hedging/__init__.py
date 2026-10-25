"""Function-based pricing and simulation tools for the project notebooks."""

from .pricing import (
    black_scholes_price,
    build_binomial_tree,
    crr_price,
    monte_carlo_price,
    payoff,
)
from .simulation import simulate_gbm_paths
from .greeks import (
    black_scholes_greeks,
    discrete_delta_hedge,
    simulate_delta_hedge,
)
from .greeks_numeric import (
    discrete_pde_delta_hedge,
    interpolate_pde_delta,
    interpolate_pde_value,
    pde_greeks,
    pde_vega_rho,
    simulate_pde_delta_hedge,
)

__all__ = [
    "black_scholes_price",
    "build_binomial_tree",
    "crr_price",
    "monte_carlo_price",
    "payoff",
    "simulate_gbm_paths",
    "black_scholes_greeks",
    "discrete_delta_hedge",
    "simulate_delta_hedge",
    "pde_greeks",
    "interpolate_pde_value",
    "interpolate_pde_delta",
    "pde_vega_rho",
    "discrete_pde_delta_hedge",
    "simulate_pde_delta_hedge",
]
