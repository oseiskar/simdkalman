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
def _update(prior_mean, prior_covariance, measurement_model, measurement_noise, measurement, compute_log_likelihood=False):

    n = prior_mean.shape[1]
    m = measurement_model.shape[1]

    assert(measurement.shape[-2:] == (m,1))
    assert(prior_covariance.shape[-2:] == (n,n))
    assert(measurement_model.shape[-2:] == (m,n))
    assert(measurement_noise.shape[-2:] == (m,m))

    # y - H * mp
    v = measurement - ddot(measurement_model, prior_mean)

    # H * Pp * H.t + R
    S = ddot(measurement_model, ddot_t_right(prior_covariance, measurement_model)) + measurement_noise
    invS = dinv(S)

    # Kalman gain: Pp * H.t * invS
    K = ddot(ddot_t_right(prior_covariance, measurement_model), invS)

    # K * v + mp
    posterior_mean = ddot(K, v) + prior_mean

    # Pp - K * H * Pp
    posterior_covariance = prior_covariance - ddot(K, ddot(measurement_model, prior_covariance))

    # inv-chi2 test var
    # outlier_test = np.sum(v * ddot(invS, v), axis=0)
    if compute_log_likelihood:
        l = np.ravel(ddot(v.transpose((0,2,1)), ddot(invS, v)))
        l += np.log(np.linalg.det(S))
        l *= -0.5
        return posterior_mean, posterior_covariance, K, l
    else:
        return posterior_mean, posterior_covariance, K

    return posterior_mean, posterior_covariance

def update(prior_mean, prior_covariance, measurement_model, measurement_noise, measurement):
    return  _update(prior_mean, prior_covariance, measurement_model, measurement_noise, measurement)[:2]

@autoshape
def _smooth(posterior_mean, posterior_covariance, state_transition, process_noise, next_smooth_mean, next_smooth_covariance):

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
    return _smooth(posterior_mean, posterior_covariance, state_transition, process_noise, next_smooth_mean, next_smooth_covariance)[:2]

@autoshape
def expected_observation(mean, measurement_model):

    n = mean.shape[1]
    m = measurement_model.shape[1]
    assert(measurement_model.shape[-2:] == (m,n))

    # H * m
    return ddot(measurement_model, mean)

@autoshape
def _update_with_nan_check(
        prior_mean,
        prior_covariance,
        measurement_model,
        measurement_noise,
        measurement,
        compute_log_likelihood=False):

    tup = _update(
        prior_mean,
        prior_covariance,
        measurement_model,
        measurement_noise,
        measurement,
        compute_log_likelihood=compute_log_likelihood)

    m1, P1, K = tup[:3]

    is_nan = np.ravel(np.any(np.isnan(m1), axis=1))

    m1[is_nan,...] = prior_mean[is_nan,...]
    P1[is_nan,...] = prior_covariance[is_nan,...]
    K[is_nan,...] = 0

    if compute_log_likelihood:
        l = tup[-1]
        l[is_nan] = 0
        return m1, P1, K, l
    else:
        return m1, P1, K

def update_with_nan_check(
        prior_mean,
        prior_covariance,
        measurement_model,
        measurement_noise,
        measurement):

    return _update_with_nan_check(
        prior_mean,
        prior_covariance,
        measurement_model,
        measurement_noise,
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

class KalmanFilter(object):
    # pylint: disable=W0232
    class Result:
        pass

    def __init__(self,
        state_transition,
        process_noise,
        measurement_model,
        measurement_noise):

        n_obs = 1
        state_transition = ensure_matrix(state_transition)
        n_states = state_transition.shape[0]

        process_noise = ensure_matrix(process_noise, n_states)
        measurement_model = ensure_matrix(measurement_model)
        measurement_noise = ensure_matrix(measurement_noise, n_obs)

        assert(state_transition.shape[-2:] == (n_states, n_states))
        assert(process_noise.shape[-2:] == (n_states, n_states))
        assert(measurement_model.shape[-2:] == (n_obs, n_states))
        assert(measurement_noise.shape[-2:] == (n_obs, n_obs))

        self.state_transition = state_transition
        self.process_noise = process_noise
        self.measurement_model = measurement_model
        self.measurement_noise = measurement_noise

    def predict_next(self, m, P):
        return predict(m, P, self.state_transition, self.process_noise)

    def update_with_nan_check(self, m, P, y, compute_log_likelihood=False):
        return _update_with_nan_check(m, P, self.measurement_model, self.measurement_noise, y,
            compute_log_likelihood=compute_log_likelihood)

    def expected_observation(self, m):
        return expected_observation(m, self.measurement_model)

    def smooth_current(self, m, P, ms, Ps):
        return _smooth(m, P, self.state_transition, self.process_noise, ms, Ps)

    def compute(self,
        training_matrix,
        n_test,
        initial_value = None,
        initial_covariance = None,
        compute_smoother = True,
        store_filtered = False,
        store_means = True,
        store_covariances = True,
        store_observations = True,
        store_likelihoods = False,
        store_gains = False,
        compute_log_likelihood = False,
        verbose = False):

        # pylint: disable=W0201
        result = KalmanFilter.Result()

        n_vars = training_matrix.shape[0]
        n_measurements = training_matrix.shape[1]
        n_states = self.state_transition.shape[0]

        if initial_value is None:
            initial_value = np.zeros((n_states, 1))

        if initial_covariance is None:
            initial_covariance = ensure_matrix(
                np.trace(ensure_matrix(self.measurement_model))*(5**2), n_states)

        initial_covariance = ensure_matrix(initial_covariance, n_states)
        initial_value = ensure_matrix(initial_value)
        assert(initial_value.shape[-2:] == (n_states, 1))
        assert(initial_covariance.shape[-2:] == (n_states, n_states))

        if len(initial_value.shape) == 2:
            initial_value = np.vstack([initial_value[np.newaxis,...]]*n_vars)

        if len(initial_covariance.shape) == 2:
            initial_covariance = np.vstack([initial_covariance[np.newaxis,...]]*n_vars)

        m = initial_value
        P = initial_covariance

        keep_filtered = store_filtered or compute_smoother

        if compute_log_likelihood:
            result.log_likelihood = np.zeros((n_vars,))
            if store_likelihoods:
                result.log_likelihoods = np.empty((n_vars, n_measurements))

        if keep_filtered:
            if store_observations:
                filtered_observations = np.empty(training_matrix.shape)
            filtered_means = np.empty((n_vars, n_measurements, n_states))
            filtered_covariances = np.empty((n_vars, n_measurements, n_states, n_states))

        if store_gains:
            result.filtered_gains = np.empty((n_vars, n_measurements, n_states, n_states))

        for j in range(n_measurements):
            if verbose:
                print('filtering %d/%d' % (j+1, n_measurements))

            y = training_matrix[:,j].reshape((n_vars, 1, 1))

            tup = self.update_with_nan_check(m, P, y, compute_log_likelihood)
            m, P, K = tup[:3]
            if compute_log_likelihood:
                l = tup[-1]
                result.log_likelihood += l
                if store_likelihoods:
                    result.log_likelihoods[:,j] = l

            if keep_filtered:
                if store_observations:
                    filtered_observations[:,j] = np.ravel(self.expected_observation(m))
                filtered_means[:,j,:] = m[...,0]
                filtered_covariances[:,j,:,:] = P

            if store_gains:
                result.filtered_gains[:,j,:,:] = K

            m, P = self.predict_next(m, P)

        if compute_smoother:
            # lazy trick to keep last filtered = last smoothed
            if store_observations:
                result.smoothed_observations = 1*filtered_observations
            if store_means:
                result.smoothed_means = 1*filtered_means
                if store_covariances:
                    result.smoothed_covariances = 1*filtered_covariances
            if store_gains:
                result.smoothed_gains = np.zeros((n_vars, n_measurements, n_states, n_states))
                result.pairwise_covariances = np.zeros((n_vars, n_measurements, n_states, n_states))

            ms = filtered_means[:,-1,:][...,np.newaxis]
            Ps = filtered_covariances[:,-1,:,:]

            for j in range(n_measurements)[-2::-1]:
                if verbose:
                    print('smoothing %d/%d' % (j+1, n_measurements))
                m0 = filtered_means[:,j,:][...,np.newaxis]
                P0 = filtered_covariances[:,j,:,:]

                PsNext = Ps
                ms, Ps, Cs = self.smooth_current(m0, P0, ms, Ps)

                if store_observations:
                    result.smoothed_observations[:,j] = np.ravel(self.expected_observation(ms))
                if store_means:
                    result.smoothed_means[:,j,:] = ms[...,0]
                    if store_covariances:
                        result.smoothed_covariances[:,j,:,:] = Ps
                if store_gains:
                    result.smoothed_gains[:,j,:,:] = Cs
                    result.pairwise_covariances[:,j,:,:] = ddot_t_right(PsNext, Cs)

        if store_filtered:
            if store_observations:
                result.filtered_observations = filtered_observations
            if store_means:
                result.filtered_means = filtered_means
                if store_covariances:
                    result.filtered_covariances = filtered_covariances

        if n_test > 0:
            if store_observations:
                result.predicted_observations = np.empty((n_vars, n_test))
            if store_means:
                result.predicted_means = np.empty((n_vars, n_test, n_states))
                if store_covariances:
                    result.predicted_covariances = np.empty((n_vars, n_test, n_states, n_states))

            for j in range(n_test):
                if verbose:
                    print('predicting %d/%d' % (j+1, n_test))
                if store_observations:
                    result.predicted_observations[:,j] = np.ravel(self.expected_observation(m))
                if store_means:
                    result.predicted_means[:,j,:] = m[...,0]
                    if store_covariances:
                        result.predicted_covariances[:,j,:,:] = P

                m, P = self.predict_next(m, P)

        return result

    def em_process_noise(self, result, verbose=False):
        n_vars, n_measurements, n_states = result.smoothed_means.shape

        res = np.zeros((n_vars, n_states, n_states))

        for j in range(n_measurements):
            if verbose:
                print('computing ML process noise, step %d/%d' % (j+1, n_measurements))

            ms1 = result.smoothed_means[:,j,:][...,np.newaxis]
            Ps1 = result.smoothed_covariances[:,j,...]

            if j > 0:
                # pylint: disable=E0601
                V1 = result.pairwise_covariances[:,j,...]
                err = ms1 - ddot(self.state_transition, ms0)
                Vt1tA = ddot_t_right(V1, self.state_transition)
                res += douter(err, err) \
                    + ddot(self.state_transition, ddot_t_right(Ps0, self.state_transition)) \
                    + Ps1 - Vt1tA - Vt1tA.transpose((0,2,1))

            ms0 = ms1
            Ps0 = Ps1

        return (1.0 / (n_measurements - 1)) * res

    def em_observation_noise(self, result, training_matrix, verbose=False):
        n_vars, n_measurements, _ = result.smoothed_means.shape

        res = np.zeros((n_vars,))
        n_not_nan = np.zeros((n_vars,))

        for j in range(n_measurements):
            if verbose:
                print('computing ML observation noise, step %d/%d' % (j+1, n_measurements))

            ms = result.smoothed_means[:,j,:][...,np.newaxis]
            Ps = result.smoothed_covariances[:,j,...]

            y = training_matrix[:,j].reshape((n_vars, 1, 1))
            not_nan = np.ravel(~np.isnan(y))
            n_not_nan += not_nan
            err = y - ddot(self.measurement_model, ms)

            r = douter(err, err) + ddot(self.measurement_model, ddot_t_right(Ps, self.measurement_model))
            res[not_nan] += np.ravel(r)[not_nan]

        res /= np.maximum(n_not_nan, 1)

        return res.reshape((n_vars,1,1))

    def em(self, training_matrix, n_iter=5, initial_value=None, initial_covariance=None, verbose=False):

        if n_iter <= 0:
            return self

        n_vars, _ = training_matrix.shape
        n_states = self.state_transition.shape[0]

        if initial_value is None:
            initial_value = np.zeros((n_vars, n_states, 1))

        if verbose:
            print("--- EM algorithm %d iteration(s) to go" % n_iter)
            print(" * E step")

        e_step = self.compute(
            training_matrix,
            n_test = 0,
            initial_value = initial_value,
            initial_covariance = initial_covariance,
            compute_smoother = True,
            store_filtered = False,
            store_means = True,
            store_covariances = True,
            store_observations = False,
            store_likelihoods = False,
            store_gains = True,
            compute_log_likelihood = False,
            verbose = verbose)

        if verbose:
            print(" * M step")

        process_noise = self.em_process_noise(e_step, verbose=verbose)
        observation_noise = self.em_observation_noise(e_step, training_matrix, verbose=verbose)
        initial_value, initial_covariance = em_initial_state(e_step, initial_value)

        new_model = KalmanFilter(
            self.state_transition,
            process_noise,
            self.measurement_model,
            observation_noise)

        return new_model.em(training_matrix, n_iter-1, initial_value, initial_covariance, verbose)

def em_initial_state(result, initial_means):

    x0 = result.smoothed_means[:,0,:][...,np.newaxis]
    P0 = result.smoothed_covariances[:,0,...]
    x0_x0 = P0 + douter(x0, x0)

    m = x0
    P = x0_x0 - douter(initial_means, x0) - douter(x0, initial_means) + douter(initial_means, initial_means)

    return m, P
