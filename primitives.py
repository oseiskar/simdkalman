import numpy as np

def ddot(A, B):
    "Matrix multiplication over last two axes"
    return np.matmul(A, B)
    #return np.einsum('...ij,...jk->...ik', A, B)

def ddot_t_right(A, B):
    "Matrix multiplication over last 2 axes with right operand transposed"
    return np.einsum('...ij,...kj->...ik', A, B)

def douter(a, b):
    return a * b.transpose((0,2,1))

def dinv(A):
    return np.linalg.inv(A)

def autoshape(func):
    def to_3d_array(v):
        if len(v.shape) == 1:
            return v[np.newaxis,:,np.newaxis]
        elif len(v.shape) == 2:
            return v[np.newaxis,...]
        else:
            return v

    def reshaped_func(*args, **kwargs):
        any_tensor = any([len(x.shape) > 2 for x in args])
        outputs = func(*[to_3d_array(a) for a in args], **kwargs)
        if not any_tensor:
            outputs = [mat[0,...] for mat in outputs]
        return outputs

    return reshaped_func

@autoshape
def predict(mean, covariance, state_transition, process_noise):

    n = mean.shape[1]

    assert(covariance.shape[-2:] == (n,n))
    assert(covariance.shape[-2:] == (n,n))
    assert(process_noise.shape[-2:] == (n,n))
    assert(state_transition.shape[-2:] == (n,n))

    # mp = A * m
    prior_mean = ddot(state_transition, mean)
    # Pp = A * P * A.t + Q
    prior_cov = ddot(state_transition, ddot_t_right(covariance, state_transition)) + process_noise

    return prior_mean, prior_cov

@autoshape
def _update(prior_mean, prior_covariance, observation_model, observation_noise, measurement, log_likelihood=False):

    n = prior_mean.shape[1]
    m = observation_model.shape[1]

    assert(measurement.shape[-2:] == (m,1))
    assert(prior_covariance.shape[-2:] == (n,n))
    assert(observation_model.shape[-2:] == (m,n))
    assert(observation_noise.shape[-2:] == (m,m))

    # y - H * mp
    v = measurement - ddot(observation_model, prior_mean)

    # H * Pp * H.t + R
    S = ddot(observation_model, ddot_t_right(prior_covariance, observation_model)) + observation_noise
    invS = dinv(S)

    # Kalman gain: Pp * H.t * invS
    K = ddot(ddot_t_right(prior_covariance, observation_model), invS)

    # K * v + mp
    posterior_mean = ddot(K, v) + prior_mean

    # Pp - K * H * Pp
    posterior_covariance = prior_covariance - ddot(K, ddot(observation_model, prior_covariance))

    # inv-chi2 test var
    # outlier_test = np.sum(v * ddot(invS, v), axis=0)
    if log_likelihood:
        l = np.ravel(ddot(v.transpose((0,2,1)), ddot(invS, v)))
        l += np.log(np.linalg.det(S))
        l *= -0.5
        return posterior_mean, posterior_covariance, K, l
    else:
        return posterior_mean, posterior_covariance, K

    return posterior_mean, posterior_covariance

def update(prior_mean, prior_covariance, observation_model, observation_noise, measurement):
    return  _update(prior_mean, prior_covariance, observation_model, observation_noise, measurement)[:2]

@autoshape
def priv_smooth(posterior_mean, posterior_covariance, state_transition, process_noise, next_smooth_mean, next_smooth_covariance):

    n = posterior_mean.shape[1]

    assert(posterior_covariance.shape[-2:] == (n,n))
    assert(process_noise.shape[-2:] == (n,n))
    assert(state_transition.shape[-2:] == (n,n))

    assert(next_smooth_mean.shape == posterior_mean.shape)
    assert(next_smooth_covariance.shape == posterior_covariance.shape)

    # re-predict a priori estimates for the next state
    # A * m
    mp = ddot(state_transition, posterior_mean)
    # A * P * A.t + Q
    Pp = ddot(state_transition, ddot_t_right(posterior_covariance, state_transition)) + process_noise

    # Kalman smoothing gain: P * A.t * inv(Pp)
    C = ddot(ddot_t_right(posterior_covariance, state_transition), dinv(Pp))

    # m + C * (ms - mp)
    smooth_mean = posterior_mean + ddot(C, next_smooth_mean - mp)
    # P + C * (Ps - Pp) * C.t
    smooth_covariance = posterior_covariance + ddot(C, ddot_t_right(next_smooth_covariance - Pp, C))

    return smooth_mean, smooth_covariance, C

def smooth(posterior_mean, posterior_covariance, state_transition, process_noise, next_smooth_mean, next_smooth_covariance):
    return priv_smooth(posterior_mean, posterior_covariance, state_transition, process_noise, next_smooth_mean, next_smooth_covariance)[:2]

@autoshape
def expected_observation(mean, observation_model):

    n = mean.shape[1]
    m = observation_model.shape[1]
    assert(observation_model.shape[-2:] == (m,n))

    # H * m
    return ddot(observation_model, mean)

@autoshape
def observation_covariance(covariance, observation_model, observation_noise):

    n = covariance.shape[1]
    m = observation_model.shape[1]
    assert(covariance.shape[-2:] == (n,n))
    assert(observation_model.shape[-2:] == (m,n))

    # H * P * H^T + R
    return ddot(observation_model,
        ddot_t_right(covariance, observation_model)) + observation_noise

@autoshape
def priv_update_with_nan_check(
        prior_mean,
        prior_covariance,
        observation_model,
        observation_noise,
        measurement,
        log_likelihood=False):

    tup = _update(
        prior_mean,
        prior_covariance,
        observation_model,
        observation_noise,
        measurement,
        log_likelihood=log_likelihood)

    m1, P1, K = tup[:3]

    is_nan = np.ravel(np.any(np.isnan(m1), axis=1))

    m1[is_nan,...] = prior_mean[is_nan,...]
    P1[is_nan,...] = prior_covariance[is_nan,...]
    K[is_nan,...] = 0

    if log_likelihood:
        l = tup[-1]
        l[is_nan] = 0
        return m1, P1, K, l
    else:
        return m1, P1, K

def update_with_nan_check(
        prior_mean,
        prior_covariance,
        observation_model,
        observation_noise,
        measurement):

    return priv_update_with_nan_check(
        prior_mean,
        prior_covariance,
        observation_model,
        observation_noise,
        measurement)[:2]

def ensure_matrix(x, dim=1):
    # pylint: disable=W0702,W0104
    try:
        y = np.array(x)
        y.shape[0]
        x = y
    except:
        x = np.eye(dim)*x
    return x
