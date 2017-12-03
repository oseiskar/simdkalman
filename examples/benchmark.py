"""
Benchmark against pykalman
"""

import numpy as np
import numpy.random as random
import time
import simdkalman
import pykalman # pip install pykalman
import filterpy.kalman # pip install filterpy

N_SERIES = 100
N_MEAS = 200

# define model
state_transition = np.array([[1,1],[0,1]])
process_noise = np.diag([0.1, 0.01])
observation_model = np.array([[1,0]])
observation_noise = np.eye(1)

initial_values = np.zeros((N_SERIES,2))
initial_covariance = np.eye(2)

# generate example data: 100 random walk time series
rand = lambda: random.normal(size=(N_SERIES, N_MEAS))
data = np.cumsum(np.cumsum(rand()*0.02, axis=1) + rand(), axis=1) + rand()*3

def time_computation(func):
    start = time.time()
    r = func()
    end = time.time()
    print("%.2f ms" % ((end - start)*1000))
    return r

def compute_simd():
    kf = simdkalman.KalmanFilter(
        state_transition,
        process_noise,
        observation_model,
        observation_noise)

    r = kf.smooth(data,
        initial_value = initial_values[...,np.newaxis],
        initial_covariance = initial_covariance,
        observations = False)

    return (r.states.mean, r.states.cov)

def compute_non_simd():

    mean = np.empty((N_SERIES,N_MEAS,2))
    cov = np.empty((N_SERIES,N_MEAS,2,2))

    for j in range(N_SERIES):
        kf = simdkalman.KalmanFilter(
            state_transition,
            process_noise,
            observation_model,
            observation_noise)

        r = kf.smooth(data[j,:],
            initial_value = initial_values[j,:],
            initial_covariance = initial_covariance,
            observations = False)

        mean[j,:,:] = r.states.mean
        cov[j,:,:,:] = r.states.cov

    return mean, cov

def compute_pykalman():

    mean = np.empty((N_SERIES,N_MEAS,2))
    cov = np.empty((N_SERIES,N_MEAS,2,2))

    for j in range(N_SERIES):

        kf = pykalman.KalmanFilter(
            transition_matrices = state_transition,
            transition_covariance = process_noise,
            observation_matrices = observation_model,
            observation_covariance = observation_noise,
            initial_state_mean = initial_values[j,:],
            initial_state_covariance = initial_covariance,
            n_dim_state = 2,
            n_dim_obs = 1)

        mean[j,:,:], cov[j,:,:,:] = kf.smooth(data[j,:][np.newaxis,:])

    return mean, cov

def compute_filterpy():
    mean = np.empty((N_SERIES,N_MEAS,2))
    cov = np.empty((N_SERIES,N_MEAS,2,2))

    kf = filterpy.kalman.KalmanFilter(dim_x=2, dim_z=1)

    kf.F = state_transition
    kf.Q = process_noise
    kf.H = observation_model
    kf.R = observation_noise

    for j in range(N_SERIES):
        kf.x = initial_values[j,:]
        kf.P = initial_covariance

        zs = list(data[j,:])
        (mus, covs, _, _) = kf.batch_filter(zs)
        # don't know what the last value is, maybe pairwise covariances?
        (xs, Ps, _, _) = kf.rts_smoother(mus, covs)
        mean[j,:,:] = np.vstack([x[np.newaxis,:] for x in xs])
        cov[j,:,:,:] = np.vstack([P[np.newaxis,...] for P in Ps])

    return mean, cov

print("simdkalman")
mean_simd, cov_simd = time_computation(compute_simd)

print("simdkalman (non-simd)")
mean_simd, cov_simd = time_computation(compute_non_simd)

print("pykalman")
mean_pk, cov_pk = time_computation(compute_pykalman)

print("filterpy")
mean_fp, cov_fp = time_computation(compute_filterpy)

print("difference (simdkalman vs pykalman)")
print("mean: %g" % (np.linalg.norm(mean_simd - mean_pk) / np.linalg.norm(mean_pk)))
print("cov: %g" % (np.linalg.norm(cov_simd - cov_pk) / np.linalg.norm(cov_pk)))

import matplotlib.pyplot as plt
plt.plot(data[0,:], 'b.')
plt.plot(mean_simd[0,:,0], label="simdkalman")
plt.plot(mean_pk[0,:,0], label="pykalman")
plt.plot(mean_fp[0,:,0], label="filterpy")
plt.legend()
plt.show()
