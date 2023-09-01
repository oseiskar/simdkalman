"""
Low-level Kalman filter computation steps with multi-dimensional input arrays.
Unlike with the `KalmanFilter <index.html#simdkalman.KalmanFilter>`_ class,
all inputs must be numpy arrays. However, their dimensions can flexibly vary
form 1 to 3 as long as they are reasonable from the point of view of matrix
multiplication and numpy broadcasting rules. Matrix operations are applied on
the *last* two axes of the arrays.
"""
import numpy as np
from functools import wraps

# work around some numpy glitches associated with different versions
from numpy.lib import NumpyVersion
_HAVE_MATMUL = NumpyVersion(np.__version__) >= '1.10.0'
_EINSUM_OPTS = {}
if NumpyVersion(np.__version__) == '1.14.0':
    # https://github.com/numpy/numpy/issues/10343
    _EINSUM_OPTS = { 'optimize': False }

def ddot(A, B):
    "Matrix multiplication over last two axes"
    if _HAVE_MATMUL:
        return np.matmul(A, B)
    else:
        return np.einsum('...ij,...jk->...ik', A, B)

def ddot_t_right(A, B):
    "Matrix multiplication over last 2 axes with right operand transposed"
    return np.matmul(A, np.swapaxes(B, -1, -2))

def douter(a, b):
    "Outer product, last two axes"
    return a * b.transpose((0,2,1))

def dinv(A):
    "Matrix inverse applied to last two axes"
    return np.linalg.inv(A)

def autoshape(func):
    "Automatically shape arguments and return values"
    def to_3d_array(v):
        if len(v.shape) == 1:
            return v[np.newaxis,:,np.newaxis]
        elif len(v.shape) == 2:
            return v[np.newaxis,...]
        else:
            return v

    @wraps(func)
    def reshaped_func(*args, **kwargs):
        any_tensor = any([len(x.shape) > 2 for x in args])
        outputs = func(*[to_3d_array(a) for a in args], **kwargs)
        if not any_tensor:
            outputs = [mat[0,...] for mat in outputs]
        return outputs

    return reshaped_func

@autoshape
def predict(mean, covariance, state_transition, process_noise):
    """
    Kalman filter prediction step

    :param mean: :math:`{\\mathbb E}[x_{j-1}]`,
        the filtered mean form the previous step
    :param covariance: :math:`{\\rm Cov}[x_{j-1}]`,
        the filtered covariance form the previous step
    :param state_transition: matrix :math:`A`
    :param process_noise: matrix :math:`Q`

    :rtype: ``(prior_mean, prior_cov)`` predicted mean and covariance
        :math:`{\\mathbb E}[x_j]`, :math:`{\\rm Cov}[x_j]`
    """

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

    return posterior_mean, posterior_covariance, K

def update(prior_mean, prior_covariance, observation_model, observation_noise, measurement):
    """
    Kalman filter update step

    :param prior_mean: :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_{j-1}]`,
        the prior mean of :math:`x_j`
    :param prior_covariance: :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_{j-1}]`,
        the prior covariance of :math:`x_j`
    :param observation_model: matrix :math:`H`
    :param observation_noise: matrix :math:`R`
    :param measurement: observation :math:`y_j`

    :rtype: ``(posterior_mean, posterior_covariance)``
        posterior mean and covariance
        :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_j]`,
        :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_j]`
        after observing :math:`y_j`
    """
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
    """
    Kalman smoother backwards step

    :param posterior_mean: :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_j]`,
        the filtered mean of :math:`x_j`
    :param posterior_covariance: :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_j]`,
        the filtered covariance of :math:`x_j`
    :param state_transition: matrix :math:`A`
    :param process_noise: matrix :math:`Q`
    :param next_smooth_mean:
        :math:`{\\mathbb E}[x_{j+1}|y_1,\\ldots,y_T]`
    :param next_smooth_covariance:
        :math:`{\\rm Cov}[x_{j+1}|y_1,\\ldots,y_T]`

    :rtype: ``(smooth_mean, smooth_covariance, smoothing_gain)``
        smoothed mean :math:`{\\mathbb E}[x_j|y_1,\\ldots,y_T]`,
        and covariance :math:`{\\rm Cov}[x_j|y_1,\\ldots,y_T]`
    """
    return priv_smooth(posterior_mean, posterior_covariance, state_transition, process_noise, next_smooth_mean, next_smooth_covariance)[:2]

@autoshape
def predict_observation(mean, covariance, observation_model, observation_noise):
    """
    Compute probability distribution of the observation :math:`y`, given
    the distribution of :math:`x`.

    :param mean: :math:`{\\mathbb E}[x]`
    :param covariance: :math:`{\\rm Cov}[x]`
    :param observation_model: matrix :math:`H`
    :param observation_noise: matrix :math:`R`

    :rtype: mean :math:`{\\mathbb E}[y]` and covariance :math:`{\\rm Cov}[y]`
    """

    n = mean.shape[1]
    m = observation_model.shape[1]
    assert(observation_model.shape[-2:] == (m,n))
    assert(covariance.shape[-2:] == (n,n))
    assert(observation_model.shape[-2:] == (m,n))

    # H * m
    obs_mean = ddot(observation_model, mean)

    # H * P * H^T + R
    obs_cov = ddot(observation_model,
        ddot_t_right(covariance, observation_model)) + observation_noise

    return obs_mean, obs_cov

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
    """
    Kalman filter update with a check for NaN observations. Like ``update`` but
    returns ``(prior_mean, prior_covariance)`` if ``measurement`` is NaN
    """

    return priv_update_with_nan_check(
        prior_mean,
        prior_covariance,
        observation_model,
        observation_noise,
        measurement)[:2]

def ensure_matrix(x, dim=1):
    # pylint: disable=W0702,W0104,E1136
    try:
        y = np.array(x)
        y.shape[0]
        x = y
    except:
        x = np.eye(dim)*x
    return x
