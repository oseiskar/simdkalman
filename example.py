import simdkalman
import numpy as np
import numpy.random as random

kf = simdkalman.KalmanFilter(
    state_transition = np.array([[1,1],[0,1]]),
    process_noise = np.diag([0.1, 0.01]),
    measurement_model = np.array([[1,0]]),
    measurement_noise = 1.0)

# simulate 100 random walk time series
rand = lambda: random.normal(size=(100, 200))
data = np.cumsum(np.cumsum(rand()*0.02, axis=1) + rand(), axis=1) + rand()*3

# introduce 10% of NaNs denoting missing values
data[random.uniform(size=data.shape) < 0.1] = np.nan

# fit noise parameters to data with the EM algorithm (optional)
kf = kf.em(data, n_iter=10)

r = kf.compute(
    data,
    n_test = 0, # predict this many new observations
    initial_value = np.array([[0],[0]]),
    initial_covariance = 1.0)

import matplotlib.pyplot as plt

# show the first 3 smoothed time series
for i in range(3):
    plt.plot(data[i,:], 'b.', label="data")

    hidden_level = r.smoothed_means[i,:,0]
    stdev = np.sqrt(r.smoothed_covariances[i,:,0,0])

    plt.plot(hidden_level, 'r-', label="hidden level")
    plt.plot(hidden_level - stdev, 'k--', label="67% confidence")
    plt.plot(hidden_level + stdev, 'k--')
    plt.title("time series %d" % (i+1))
    plt.legend()
    plt.show()
