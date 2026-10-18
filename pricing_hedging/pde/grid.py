
import numpy as np

from pricing_hedging.pricing import payoff


def validate_grid(S, tau):
    """Return validated one-dimensional, uniform spot and tau grids."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)

    if S.ndim != 1 or len(S) < 3 or tau.ndim != 1 or len(tau) < 2:
        raise ValueError("Need 1D grids with M >= 2 and N >= 1")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(tau)):
        raise ValueError("Grid values must be finite")
    if abs(S[0]) > 1e-12 or abs(tau[0]) > 1e-12:
        raise ValueError("S and tau grids must start at zero")

    dS = S[1] - S[0]
    dt = tau[1] - tau[0]
    if dS <= 0 or dt <= 0:
        raise ValueError("Grid spacings must be positive")
    if not np.allclose(np.diff(S), dS, rtol=1e-10, atol=1e-12):
        raise ValueError("S grid must be uniform")
    if not np.allclose(np.diff(tau), dt, rtol=1e-10, atol=1e-12):
        raise ValueError("tau grid must be uniform")

    return S, tau



def Makegrid(Smax, T, M, N):
    S = np.linspace(0, Smax, M + 1)
    taux = np.linspace(0, T, N + 1)
    return S, taux


def BoundaryConditions(S, taux, option_type, r, K):
    # Boundary conditions for the option price grid
    S, taux = validate_grid(S, taux)
    option_type = str(option_type).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    M = len(S) - 1
    N = len(taux) - 1
    Smax = S[-1]
    U = np.zeros((N + 1, M + 1))

    if S[-1] < K * np.exp(max(0.0, -r * taux[-1])):
        raise ValueError("Choose a larger S_max")
    
    # Set boundary conditions for call option
    if option_type == "call":
        U[0, :] = payoff(K, S, option_type)  # Payoff at tau = 0
        U[:, 0] = 0  # Call option value is 0 when stock price is 0
        U[:, M] = Smax - K * np.exp(-r * taux)
    # Set boundary conditions for put option
    else:
        U[0, :] = payoff(K, S, option_type)  # Payoff at tau = 0
        U[:, 0] = K * np.exp(-r * taux)
        U[:, M] = 0  # Put option value is 0 when stock price is very high
    
    return U

