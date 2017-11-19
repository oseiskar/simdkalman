import numpy as np
# pylint: disable=W0401,W0614
from primitives import *

class KalmanFilter(object):
    # pylint: disable=W0232
    class Result:
        pass

    def __init__(self,
        state_transition,
        process_noise,
        observation_model,
        observation_noise):

        n_obs = 1
        state_transition = ensure_matrix(state_transition)
        n_states = state_transition.shape[0]

        process_noise = ensure_matrix(process_noise, n_states)
        observation_model = ensure_matrix(observation_model)
        observation_noise = ensure_matrix(observation_noise, n_obs)

        assert(state_transition.shape[-2:] == (n_states, n_states))
        assert(process_noise.shape[-2:] == (n_states, n_states))
        assert(observation_model.shape[-2:] == (n_obs, n_states))
        assert(observation_noise.shape[-2:] == (n_obs, n_obs))

        self.state_transition = state_transition
        self.process_noise = process_noise
        self.observation_model = observation_model
        self.observation_noise = observation_noise

    def predict_next(self, m, P):
        return predict(m, P, self.state_transition, self.process_noise)

    def update_with_nan_check(self, m, P, y, compute_log_likelihood=False):
        return priv_update_with_nan_check(m, P,
            self.observation_model, self.observation_noise, y,
            compute_log_likelihood=compute_log_likelihood)

    def expected_observation(self, m):
        return expected_observation(m, self.observation_model)

    def smooth_current(self, m, P, ms, Ps):
        return priv_smooth(m, P,
            self.state_transition, self.process_noise, ms, Ps)

    def predict(self,
        training_matrix,
        n_test,
        observations = True,
        means = True,
        covariances = True,
        initial_value = None,
        initial_covariance = None,
        verbose = False):

        r = self.compute(
            training_matrix,
            n_test,
            initial_value,
            initial_covariance,
            compute_smoother = False,
            means = means,
            covariances = covariances,
            observations = observations,
            verbose = verbose)

        # pylint: disable=W0201
        r1 = KalmanFilter.Result()
        if observations:
            r1.observations = r.predicted.observations
        if means:
            r1.means = r.predicted.means
        if covariances:
            r1.covariances = r.predicted.covariances
        return r1

    def smooth(self,
        training_matrix,
        observations = True,
        means = True,
        covariances = True,
        initial_value = None,
        initial_covariance = None,
        verbose = False):

        r = self.compute(
            training_matrix,
            0,
            initial_value,
            initial_covariance,
            compute_smoother = True,
            means = means,
            covariances = covariances,
            observations = observations,
            verbose = verbose)

        # pylint: disable=W0201
        r1 = KalmanFilter.Result()
        if observations:
            r1.observations = r.smoothed.observations
        if means:
            r1.means = r.smoothed.means
        if covariances:
            r1.covariances = r.smoothed.covariances
        return r1

    def compute(self,
        training_matrix,
        n_test,
        initial_value = None,
        initial_covariance = None,
        compute_smoother = True,
        filtered = False,
        means = True,
        covariances = True,
        observations = True,
        likelihoods = False,
        gains = False,
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
                np.trace(ensure_matrix(self.observation_model))*(5**2), n_states)

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

        keep_filtered = filtered or compute_smoother
        if filtered or gains:
            result.filtered = KalmanFilter.Result()

        if compute_log_likelihood:
            result.log_likelihood = np.zeros((n_vars,))
            if likelihoods:
                result.log_likelihoods = np.empty((n_vars, n_measurements))

        if keep_filtered:
            if observations:
                filtered_observations = np.empty(training_matrix.shape)
            filtered_means = np.empty((n_vars, n_measurements, n_states))
            filtered_covariances = np.empty((n_vars, n_measurements, n_states, n_states))

        if gains:
            result.filtered.gains = np.empty((n_vars, n_measurements, n_states, n_states))

        for j in range(n_measurements):
            if verbose:
                print('filtering %d/%d' % (j+1, n_measurements))

            y = training_matrix[:,j].reshape((n_vars, 1, 1))

            tup = self.update_with_nan_check(m, P, y, compute_log_likelihood)
            m, P, K = tup[:3]
            if compute_log_likelihood:
                l = tup[-1]
                result.log_likelihood += l
                if likelihoods:
                    result.log_likelihoods[:,j] = l

            if keep_filtered:
                if observations:
                    filtered_observations[:,j] = np.ravel(self.expected_observation(m))
                filtered_means[:,j,:] = m[...,0]
                filtered_covariances[:,j,:,:] = P

            if gains:
                result.filtered.gains[:,j,:,:] = K

            m, P = self.predict_next(m, P)

        if compute_smoother:
            result.smoothed = KalmanFilter.Result()

            # lazy trick to keep last filtered = last smoothed
            if observations:
                result.smoothed.observations = 1*filtered_observations
            if means:
                result.smoothed.means = 1*filtered_means
            if covariances:
                result.smoothed.covariances = 1*filtered_covariances
            if gains:
                result.smoothed.gains = np.zeros((n_vars, n_measurements, n_states, n_states))
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

                if observations:
                    result.smoothed.observations[:,j] = np.ravel(self.expected_observation(ms))
                if means:
                    result.smoothed.means[:,j,:] = ms[...,0]
                if covariances:
                    result.smoothed.covariances[:,j,:,:] = Ps
                if gains:
                    result.smoothed.gains[:,j,:,:] = Cs
                    result.pairwise_covariances[:,j,:,:] = ddot_t_right(PsNext, Cs)

        if filtered:
            if observations:
                result.filtered.observations = filtered_observations
            if means:
                result.filtered.means = filtered_means
            if covariances:
                result.filtered.covariances = filtered_covariances

        if n_test > 0:
            result.predicted = KalmanFilter.Result()
            if observations:
                result.predicted.observations = np.empty((n_vars, n_test))
            if means:
                result.predicted.means = np.empty((n_vars, n_test, n_states))
            if covariances:
                result.predicted.covariances = np.empty((n_vars, n_test, n_states, n_states))

            for j in range(n_test):
                if verbose:
                    print('predicting %d/%d' % (j+1, n_test))
                if observations:
                    result.predicted.observations[:,j] = np.ravel(self.expected_observation(m))
                if means:
                    result.predicted.means[:,j,:] = m[...,0]
                if covariances:
                    result.predicted.covariances[:,j,:,:] = P

                m, P = self.predict_next(m, P)

        return result

    def em_process_noise(self, result, verbose=False):
        n_vars, n_measurements, n_states = result.smoothed.means.shape

        res = np.zeros((n_vars, n_states, n_states))

        for j in range(n_measurements):
            if verbose:
                print('computing ML process noise, step %d/%d' % (j+1, n_measurements))

            ms1 = result.smoothed.means[:,j,:][...,np.newaxis]
            Ps1 = result.smoothed.covariances[:,j,...]

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
        n_vars, n_measurements, _ = result.smoothed.means.shape

        res = np.zeros((n_vars,))
        n_not_nan = np.zeros((n_vars,))

        for j in range(n_measurements):
            if verbose:
                print('computing ML observation noise, step %d/%d' % (j+1, n_measurements))

            ms = result.smoothed.means[:,j,:][...,np.newaxis]
            Ps = result.smoothed.covariances[:,j,...]

            y = training_matrix[:,j].reshape((n_vars, 1, 1))
            not_nan = np.ravel(~np.isnan(y))
            n_not_nan += not_nan
            err = y - ddot(self.observation_model, ms)

            r = douter(err, err) + ddot(self.observation_model, ddot_t_right(Ps, self.observation_model))
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
            filtered = False,
            means = True,
            covariances = True,
            observations = False,
            likelihoods = False,
            gains = True,
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
            self.observation_model,
            observation_noise)

        return new_model.em(training_matrix, n_iter-1, initial_value, initial_covariance, verbose)

def em_initial_state(result, initial_means):

    x0 = result.smoothed.means[:,0,:][...,np.newaxis]
    P0 = result.smoothed.covariances[:,0,...]
    x0_x0 = P0 + douter(x0, x0)

    m = x0
    P = x0_x0 - douter(initial_means, x0) - douter(x0, initial_means) + douter(initial_means, initial_means)

    return m, P
