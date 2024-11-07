import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Heston model parameters
kappa = 2.0    # Speed of mean reversion
theta = 0.04   # Long-term variance
xi = 0.1       # Volatility of volatility
rho = -0.7     # Correlation between the stock price and volatility
r = 0.05       # Risk-free rate (Ensure this is defined)

def simulate_heston(S0, V0, r, kappa, theta, xi, rho, T, N):
    dt = T / N
    S = np.zeros(N)
    V = np.zeros(N)
    S[0] = S0
    V[0] = V0

    for t in range(1, N):
        Z1 = norm.rvs()
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * norm.rvs()
        V[t] = V[t-1] + kappa * (theta - V[t-1]) * dt + xi * np.sqrt(V[t-1] * dt) * Z1
        S[t] = S[t-1] * np.exp((r - 0.5 * V[t-1]) * dt + np.sqrt(V[t-1] * dt) * Z2)

    return S, V

# Initial conditions
S0 = 100
V0 = 0.04
T = 1.0
N = 252

# Simulate
S, V = simulate_heston(S0, V0, r, kappa, theta, xi, rho, T, N)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, T, N), S)
plt.title('Stock Price Simulation')
plt.xlabel('Time')
plt.ylabel('Stock Price')

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, T, N), V)
plt.title('Variance Simulation')
plt.xlabel('Time')
plt.ylabel('Variance')

plt.tight_layout()
plt.show()
