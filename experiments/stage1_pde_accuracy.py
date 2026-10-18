"""Run from repository root: python -m experiments.stage1_pde_accuracy"""
from pricing_hedging.pde.grid import Makegrid
from pricing_hedging.pde.explicit_euler import (
    check_explicit_stability,
    explicit_euler,
)

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
Smax, M, N = 400.0, 200, 2000

S, taux = Makegrid(Smax, T, M, N)
info = check_explicit_stability(S, taux, r, sigma)
U = explicit_euler(S, taux, K, r, sigma, option_type="call")

index = abs(S - S0).argmin()
print(f"Explicit Euler call price: {U[-1, index]:.6f}")
print(f"Sufficient stability condition: dt={info['dt']:.6g} <= {info['max_dt']:.6g}")
print(f"Interior update matrix infinity norm: {info['matrix_norm_inf']:.8f}")
