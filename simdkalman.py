import numpy as np
# pylint: disable=W0401,W0614
from primitives import *

class Gaussian:
    def __init__(self, mean, cov):
        self.mean = mean
        if cov is not None:
            self.cov = cov

    @staticmethod
    def zeros(n_states, n_vars, n_measurements, cov=True):
        mean = np.zeros((n_vars, n_measurements, n_states))
        if cov:
            cov = np.zeros((n_vars, n_measurements, n_states, n_states))
        else:
            cov = None
        return Gaussian(mean, cov)

    @staticmethod
    def empty(n_states, n_vars, n_measurements, cov=True):
        mean = np.empty((n_vars, n_measurements, n_states))
        if cov:
            cov = np.empty((n_vars, n_measurements, n_states, n_states))
        else:
            cov = None
        return Gaussian(mean, cov)

    def unvectorize_state(self):
        n_states = self.mean.shape[-1]
        assert(n_states == 1)

        mean = self.mean
        cov = self.cov

        mean = mean[...,0]
        if cov is not None:
            cov = cov[...,0,0]

        return Gaussian(mean, cov)

    def unvectorize_vars(self):
        n_vars = self.mean.shape[0]
        assert(n_vars == 1)

        mean = self.mean
        cov = self.cov

        mean = mean[0,...]
        if cov is not None:
            cov = cov[0,...]

        return Gaussian(mean, cov)

    def __str__(self):
        s = "mean:\n  %s" % str(self.mean).replace("\n", "\n  ")
        if self.cov is not None:
            s += "\ncov:\n  %s" % str(self.cov).replace("\n", "\n  ")
        return s

class KalmanFilter(object):
    # pylint: disable=W0232
    class Result:
        def __str__(self):
            s = ""
            for k,v in self.__dict__.items():
                if len(s) > 0:
                    s += "\n"
                s += "%s:\n" % k
                s += "  " + str(v).replace("\n", "\n  ")
            return s

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

    def update_with_nan_check(self, m, P, y, log_likelihood=False):
        return priv_update_with_nan_check(m, P,
            self.observation_model, self.observation_noise, y,
            log_likelihood=log_likelihood)

    def expected_observation(self, m):
        return expected_observation(m, self.observation_model)

    def observation_covariance(self, P):
        return observation_covariance(P,
            self.observation_model, self.observation_noise)

    def smooth_current(self, m, P, ms, Ps):
        return priv_smooth(m, P,
            self.state_transition, self.process_noise, ms, Ps)

    def predict(self,
        training_matrix,
        n_test,
        states = True,
        observations = True,
        covariances = True,
        initial_value = None,
        initial_covariance = None,
        verbose = False):

        return self.compute(
            training_matrix,
            n_test,
            initial_value,
            initial_covariance,
            smoothed = False,
            states = states,
            covariances = covariances,
            observations = observations,
            verbose = verbose).predicted

    def smooth(self,
        training_matrix,
        observations = True,
        states = True,
        covariances = True,
        initial_value = None,
        initial_covariance = None,
        verbose = False):

        return self.compute(
            training_matrix,
            0,
            initial_value,
            initial_covariance,
            smoothed = True,
            states = states,
            covariances = covariances,
            observations = observations,
            verbose = verbose).smoothed

    def compute(self,
        training_matrix,
        n_test,
        initial_value = None,
        initial_covariance = None,
        smoothed = True,
        filtered = False,
        states = True,
        covariances = True,
        observations = True,
        likelihoods = False,
        gains = False,
        log_likelihood = False,
        verbose = False):

        # pylint: disable=W0201
        result = KalmanFilter.Result()

        training_matrix = ensure_matrix(training_matrix)
        single_sequence = len(training_matrix.shape) == 1
        if single_sequence:
            training_matrix = training_matrix[np.newaxis,:]

        n_vars = training_matrix.shape[0]
        n_measurements = training_matrix.shape[1]
        n_states = self.state_transition.shape[0]
        n_obs = 1 # should also allow 3d training data...

        def empty_gaussian(
            n_states=n_states,
            n_measurements=n_measurements,
            cov=covariances):
            return Gaussian.empty(n_states, n_vars, n_measurements, cov)

        def auto_flat_observations(obs_gaussian):
            r = obs_gaussian
            if n_obs == 1:
                r = r.unvectorize_state()
            if single_sequence:
                r = r.unvectorize_vars()
            return r

        def auto_flat_states(obs_gaussian):
            if single_sequence:
                return obs_gaussian.unvectorize_vars()
            return obs_gaussian

        if initial_value is None:
            initial_value = np.zeros((n_states, 1))
        initial_value = ensure_matrix(initial_value)
        if len(initial_value.shape) == 1:
            initial_value = initial_value.reshape((n_states, 1))

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

        keep_filtered = filtered or smoothed
        if filtered or gains:
            result.filtered = KalmanFilter.Result()

        if log_likelihood:
            result.log_likelihood = np.zeros((n_vars,))
            if likelihoods:
                result.log_likelihoods = np.empty((n_vars, n_measurements))

        if keep_filtered:
            if observations:
                filtered_observations = empty_gaussian(n_states=n_obs)
            filtered_states = empty_gaussian(cov=True)

        if gains:
            result.filtered.gains = np.empty((n_vars, n_measurements, n_states, n_states))

        for j in range(n_measurements):
            if verbose:
                print('filtering %d/%d' % (j+1, n_measurements))

            y = training_matrix[:,j].reshape((n_vars, 1, 1))

            tup = self.update_with_nan_check(m, P, y, log_likelihood)
            m, P, K = tup[:3]
            if log_likelihood:
                l = tup[-1]
                result.log_likelihood += l
                if likelihoods:
                    result.log_likelihoods[:,j] = l

            if keep_filtered:
                if observations:
                    filtered_observations.mean[:,j,:] = \
                        self.expected_observation(m)[...,0]
                    if covariances:
                        filtered_observations.cov[:,j,:,:] = \
                            self.observation_covariance(P)

                filtered_states.mean[:,j,:] = m[...,0]
                filtered_states.cov[:,j,:,:] = P

            if gains:
                result.filtered.gains[:,j,:,:] = K

            m, P = self.predict_next(m, P)

        if smoothed:
            result.smoothed = KalmanFilter.Result()
            if states:
                result.smoothed.states = empty_gaussian()

                # lazy trick to keep last filtered = last smoothed
                result.smoothed.states.mean = 1*filtered_states.mean
                if covariances:
                    result.smoothed.states.cov = 1*filtered_states.cov

            if observations:
                result.smoothed.observations = empty_gaussian(n_states=n_obs)
                result.smoothed.observations.mean = 1*filtered_observations.mean
                if covariances:
                    result.smoothed.observations.cov = 1*filtered_observations.cov

            if gains:
                result.smoothed.gains = np.zeros((n_vars, n_measurements, n_states, n_states))
                result.pairwise_covariances = np.zeros((n_vars, n_measurements, n_states, n_states))

            ms = filtered_states.mean[:,-1,:][...,np.newaxis]
            Ps = filtered_states.cov[:,-1,:,:]

            for j in range(n_measurements)[-2::-1]:
                if verbose:
                    print('smoothing %d/%d' % (j+1, n_measurements))
                m0 = filtered_states.mean[:,j,:][...,np.newaxis]
                P0 = filtered_states.cov[:,j,:,:]

                PsNext = Ps
                ms, Ps, Cs = self.smooth_current(m0, P0, ms, Ps)

                if states:
                    result.smoothed.states.mean[:,j,:] = ms[...,0]
                    if covariances:
                        result.smoothed.states.cov[:,j,:,:] = Ps

                if observations:
                    result.smoothed.observations.mean[:,j,:] = \
                        self.expected_observation(ms)[...,0]
                    if covariances:
                        result.smoothed.observations.cov[:,j,:,:] = \
                            self.observation_covariance(Ps)

                if gains:
                    result.smoothed.gains[:,j,:,:] = Cs
                    result.pairwise_covariances[:,j,:,:] = ddot_t_right(PsNext, Cs)

        if filtered:
            if states:
                result.filtered.states = Gaussian(filtered_states.mean, None)
                if covariances:
                    result.filtered.states.cov = filtered_states.cov
                result.filtered.states = auto_flat_states(result.filtered.states)
            if observations:
                result.filtered.observations = auto_flat_observations(
                    filtered_observations)

        if smoothed:
            if observations:
                result.smoothed.observations = auto_flat_observations(
                    result.smoothed.observations)
            if states:
                result.smoothed.states = auto_flat_states(
                    result.smoothed.states)

        if n_test > 0:
            result.predicted = KalmanFilter.Result()
            if observations:
                result.predicted.observations = empty_gaussian(
                    n_measurements=n_test,
                    n_states=n_obs)
            if states:
                result.predicted.states = empty_gaussian(n_measurements=n_test)

            for j in range(n_test):
                if verbose:
                    print('predicting %d/%d' % (j+1, n_test))
                if states:
                    result.predicted.states.mean[:,j,:] = m[...,0]
                    if covariances:
                        result.predicted.states.cov[:,j,:,:] = P
                if observations:
                    result.predicted.observations.mean[:,j,:] = \
                        self.expected_observation(m)[...,0]
                    if covariances:
                        result.predicted.observations.cov[:,j,:,:] = \
                            self.observation_covariance(P)

                m, P = self.predict_next(m, P)

            if observations:
                result.predicted.observations = auto_flat_observations(
                    result.predicted.observations)
            if states:
                result.predicted.states = auto_flat_states(result.predicted.states)

        return result

    def em_process_noise(self, result, verbose=False):
        n_vars, n_measurements, n_states = result.smoothed.states.mean.shape

        res = np.zeros((n_vars, n_states, n_states))

        for j in range(n_measurements):
            if verbose:
                print('computing ML process noise, step %d/%d' % (j+1, n_measurements))

            ms1 = result.smoothed.states.mean[:,j,:][...,np.newaxis]
            Ps1 = result.smoothed.states.cov[:,j,...]

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
        n_vars, n_measurements, _ = result.smoothed.states.mean.shape

        res = np.zeros((n_vars,))
        n_not_nan = np.zeros((n_vars,))

        for j in range(n_measurements):
            if verbose:
                print('computing ML observation noise, step %d/%d' % (j+1, n_measurements))

            ms = result.smoothed.states.mean[:,j,:][...,np.newaxis]
            Ps = result.smoothed.states.cov[:,j,...]

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

        training_matrix = ensure_matrix(training_matrix)
        if len(training_matrix.shape) == 1:
            training_matrix = training_matrix[np.newaxis,:]

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
            smoothed = True,
            filtered = False,
            states = True,
            observations = True,
            covariances = True,
            likelihoods = False,
            gains = True,
            log_likelihood = False,
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

    x0 = result.smoothed.states.mean[:,0,:][...,np.newaxis]
    P0 = result.smoothed.states.cov[:,0,...]
    x0_x0 = P0 + douter(x0, x0)

    m = x0
    P = x0_x0 - douter(initial_means, x0) - douter(x0, initial_means) + douter(initial_means, initial_means)

    return m, P
