import numpy as np
import simdkalman

np.random.seed(0)

# ---- Simulate time series of 200 observations
n = 200
rand = lambda: np.random.normal(size=n)
x_true = np.cumsum(rand()*0.1)

# Measurement: true value is corrupted by noise whose magnitude
# depends on t (and this magnitude is known)
t = np.linspace(0, 1, num=n)
noise_level = (t*(1-t)*4)**2 + 0.1
y = x_true + rand()*noise_level

# ---- Define dynamic model for reconstruction: random walk
A = np.eye(1)
Q = np.eye(1) * 0.1**2

# ---- Measurement model: different noise level for each sample
H = np.eye(1)
R_list = [sigma**2 * np.eye(1) for sigma in noise_level]

# ---- Kalman filter forward pass
# store posterior distributions for smoothing
m_list = []
P_list = []

# initial values
m = np.array([[y[0]]])
P = np.eye(1) * 5**2

for i in range(n):
    m, P = simdkalman.primitives.predict(m, P, A, Q)
    m, P = simdkalman.primitives.update(m, P, H, R_list[i], y[np.newaxis, i])
    m_list.append(m)
    P_list.append(P)

# ---- Backward pass: Kalman smoother
# store smoothed means (x) and variances
smoothed_x = np.zeros(n)
smoothed_var = np.zeros(n)

smoothed_x[-1] = m[0]
smoothed_var[-1] = P[0,0]

for i in range(n)[-2::-1]:
    m, P = simdkalman.primitives.smooth(m_list[i], P_list[i], A, Q, m, P)
    smoothed_x[i] = m[0]
    smoothed_var[i] = P[0,0]

# ---- Visualize results
import matplotlib.pyplot as plt
plt.plot(x_true, 'r', label='true x')
plt.plot(y, 'kx', alpha=0.2, label='measurements y')
plt.plot(smoothed_x, 'b', label='smoothed x')
stdev = np.sqrt(smoothed_var)
plt.plot(smoothed_x + stdev*2, 'b--', label='95% confidence')
plt.plot(smoothed_x - stdev*2, 'b--')
plt.legend()
plt.show()
