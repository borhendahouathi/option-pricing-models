"""Grid refinement while maintaining the sufficient explicit stability bound.

Run from project root: python -m experiments.stage1_convergence
"""

import math
import numpy as np

from pricing_hedging.pde.grid import Makegrid
from pricing_hedging.pde.explicit_euler import explicit_euler

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
S_max = 400.0
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
d1 = (math.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)
exact = S0 * Phi(d1) - K * math.exp(-r * T) * Phi(d2)

print("M       N       dS         price         abs_error")
errors = []
for M in (40, 80, 160, 320):
    # 20% margin beneath the sufficient maximum dt.
    N = math.ceil(1.2 * T * (sigma**2 * (M - 1)**2 + r))
    S, tau = Makegrid(S_max, T, M, N)
    U = explicit_euler(S, tau, K, r, sigma)
    price = U[-1, int(np.argmin(abs(S - S0)))]
    error = abs(price - exact)
    errors.append(error)
    print(f"{M:<7} {N:<7} {S[1]-S[0]:<10.5f} {price:<13.7f} {error:.8f}")

print("Observed error ratios on successive M doublings:",
      [round(float(errors[j - 1] / errors[j]), 3) for j in range(1, len(errors))])
print("Note: this comparison includes domain error at fixed S_max; it does")
print("not separately certify the boundary truncation error.")
