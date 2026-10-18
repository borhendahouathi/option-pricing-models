
from pricing_hedging.pde.grid import (
    BoundaryConditions,
    Makegrid 
)

# Parameters
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1

S_max = 4 * max(S0, K)

M = 5  # Spatial intervals
N = 5  # Time intervals

# Create the grid
S, tau = Makegrid(S_max, T, M, N)

# Initialize our call price matrix
U_call = BoundaryConditions(S, tau, "call", r, K)
U_put = BoundaryConditions(S, tau, "put", r, K)
# Grid spacing
dS = S[1] - S[0]
dt = tau[1] - tau[0]

print("dS =", dS)
print("dt =", dt)
print("Call option prices:")
print(U_call)
print("Put option prices:")
print(U_put)

