import unittest
import simdkalman
import numpy as np

class TestWithMatrices(unittest.TestCase):
    def assertMatrixEqual(self, matA, matB, epsilon=0.0):
        self.assertSequenceEqual(matA.shape, matB.shape)

        if epsilon > 0:
            self.assertTrue(np.sum(np.abs(matA - matB)) < epsilon)
        else:
            self.assertSequenceEqual(list(np.ravel(matA)), list(np.ravel(matB)))

class TestKalman(TestWithMatrices):

    def test_ensure_matrix(self):
        self.assertMatrixEqual(simdkalman.ensure_matrix(3), np.eye(1)*3)
        self.assertMatrixEqual(simdkalman.ensure_matrix([[3.0]]), np.eye(1)*3)
        self.assertMatrixEqual(simdkalman.ensure_matrix(np.eye(1)), np.eye(1))
        self.assertMatrixEqual(simdkalman.ensure_matrix(np.eye(3)), np.eye(3))

    def test_ddot(self):

        vec = lambda *args: np.array(args)[np.newaxis,:,np.newaxis]
        self.assertMatrixEqual(vec(1,2,3), np.array([[[1],[2],[3]]]))

        self.assertMatrixEqual(\
            simdkalman.ddot(np.diag([1,-1,0])[np.newaxis,...], vec(1,2,3)), \
            vec(1,-2,0))

        stack_mats = lambda *args: np.vstack([a[np.newaxis,...] for a in args])
        self.assertMatrixEqual(stack_mats(np.eye(2), np.eye(2)*3)[0,...], np.eye(2))

        self.assertMatrixEqual( \
            simdkalman.ddot( \
                stack_mats(np.eye(2)*1, np.array([[2,0],[1,1]]), np.eye(2)*3), \
                np.vstack([vec(1,2), vec(3,4), vec(5,6)])), \
            np.vstack([vec(1,2), vec(6, 7), vec(15, 18)]))

    def test_t_ddot(self):

        vec = lambda *args: np.array(args)[np.newaxis,:,np.newaxis]
        T =  lambda x: x.transpose((0,2,1))
        stack_mats = lambda *args: np.vstack([a[np.newaxis,...] for a in args])

        self.assertMatrixEqual( \
            simdkalman.ddot_t_right( \
                T(np.vstack([vec(1,2), vec(3,4), vec(5,6)])),
                stack_mats(np.eye(2)*1, np.array([[2,0],[1,1]]), np.eye(2)*3)), \
            T(np.vstack([vec(1,2), vec(6, 7), vec(15, 18)])))

    def test_predict(self):

        mean = np.array([[1],[2],[3]])
        covariance = np.eye(3)*2

        state_transition = np.eye(3)*0.5
        process_noise = np.eye(3)*0.1

        m1, P1 = simdkalman.predict(mean, covariance, state_transition, process_noise)

        self.assertSequenceEqual(m1.shape, (3,1))
        self.assertSequenceEqual(P1.shape, (3,3))

        # should be diagonal
        self.assertTrue(np.linalg.norm(P1 - np.diag(np.diag(P1))) < 1e-6)

    def test_update(self):
        prior_mean = np.array([[1],[2],[3]])
        prior_covariance = np.eye(3)*2

        measurement_model = np.ones((2,3))
        measurement_noise = np.eye(2)*0.1

        measurement = np.array([[3],[4]])

        m, P = simdkalman.update(prior_mean, prior_covariance, measurement_model, measurement_noise, measurement)

        self.assertSequenceEqual(m.shape, (3,1))
        self.assertSequenceEqual(P.shape, (3,3))

    def test_smooth(self):

        mean = np.array([[1],[2],[3]])
        covariance = np.eye(3)*2

        next_mean =  np.array([[2],[4],[6]])
        next_covariance = np.eye(3)

        state_transition = np.eye(3)*0.5
        process_noise = np.eye(3)*0.1

        ms, Ps = simdkalman.smooth(
            mean,
            covariance,
            state_transition,
            process_noise,
            next_mean,
            next_covariance)

        self.assertSequenceEqual(ms.shape, (3,1))
        self.assertSequenceEqual(Ps.shape, (3,3))

        # should be diagonal
        self.assertTrue(np.linalg.norm(Ps - np.diag(np.diag(Ps))) < 1e-6)


    def test_update_with_nan_check(self):
        prior_mean = np.array([[1],[2],[3]])
        prior_covariance = np.eye(3)*2

        measurement_model = np.ones((2,3))
        measurement_noise = np.eye(2)*0.1

        measurement = np.array([[3],[np.nan]])

        m, P = simdkalman.update_with_nan_check(
            prior_mean,
            prior_covariance,
            measurement_model,
            measurement_noise,
            measurement)

        self.assertSequenceEqual(m.shape, (3,1))
        self.assertSequenceEqual(P.shape, (3,3))
        self.assertMatrixEqual(m, prior_mean)
        self.assertMatrixEqual(P, prior_covariance)

    def test_vectorized(self):

        mean = np.zeros((3,2,1))
        mean[0,...] = np.array([[[1],[10]]]) # state 1
        mean[1,...] = np.array([[[2],[20]]]) # state 2
        mean[2,...] = np.array([[[3],[30]]]) # state 3

        stack_mats = lambda arr: np.vstack([a[np.newaxis,...] for a in arr])

        covariance = stack_mats([np.eye(2)]*3)
        state_transition = stack_mats([np.eye(2)]*3)
        process_noise = stack_mats([np.eye(2)]*3)*0.1

        m1, P1 = simdkalman.predict(
            mean,
            covariance,
            state_transition,
            process_noise)

        self.assertMatrixEqual(m1, mean)
        self.assertSequenceEqual(P1.shape, (3,2,2))

        measurement_model = stack_mats([np.ones((1,2))]*3)
        measurement_noise = stack_mats([np.eye(1)*0.1]*3)

        measurement = np.array([[[2]], [[np.nan]], [[33]]])

        m, P = simdkalman.update_with_nan_check(
            m1,
            P1,
            measurement_model,
            measurement_noise,
            measurement)

        self.assertSequenceEqual(m.shape, (3,2,1))
        self.assertSequenceEqual(P.shape, (3,2,2))

        self.assertMatrixEqual(m[1,...], m1[1,...])
        self.assertMatrixEqual(m[2,...], mean[2,...], epsilon=1e-6)
        self.assertMatrixEqual(P[1,...], P1[1,...])

    def test_semi_vectorized(self):

        mean = np.zeros((3,2,1))
        mean[0,...] = np.array([[[1],[10]]]) # state 1
        mean[1,...] = np.array([[[2],[20]]]) # state 2
        mean[2,...] = np.array([[[3],[30]]]) # state 3

        stack_mats = lambda arr: np.vstack([a[np.newaxis,...] for a in arr])

        covariance = stack_mats([np.eye(2)]*3)

        state_transition = np.eye(2)
        process_noise = np.eye(2)*0.1

        m1, P1 = simdkalman.predict(
            mean,
            covariance,
            state_transition,
            process_noise)

        self.assertMatrixEqual(m1, mean)
        self.assertSequenceEqual(P1.shape, (3,2,2))

        measurement_model = np.ones((1,2))
        measurement_noise = np.eye(1)*0.1

        measurement = np.array([[[2]], [[np.nan]], [[33]]])

        m, P = simdkalman.update_with_nan_check(
            m1,
            P1,
            measurement_model,
            measurement_noise,
            measurement)

        self.assertSequenceEqual(m.shape, (3,2,1))
        self.assertSequenceEqual(P.shape, (3,2,2))

        self.assertMatrixEqual(m[1,...], m1[1,...])
        self.assertMatrixEqual(m[2,...], mean[2,...], epsilon=1e-6)
        self.assertMatrixEqual(P[1,...], P1[1,...])


    def test_one_dimensional(self):
        mean = np.array([[1]])
        covariance = np.array([[1]])
        state_transition = np.array([[2]])
        process_noise = np.array([[0.1]])

        m1, P1 = simdkalman.predict(
            mean,
            covariance,
            state_transition,
            process_noise)

        self.assertMatrixEqual(m1, mean*2, epsilon=1e-6)
        self.assertSequenceEqual(P1.shape, (1,1))

        measurement_model = np.array([[1]])
        measurement_noise = np.array([[0.2]])

        measurement = np.array([[1]])

        m, P = simdkalman.update_with_nan_check(
            m1,
            P1,
            measurement_model,
            measurement_noise,
            measurement)

        self.assertSequenceEqual(m.shape, (1,1))
        self.assertSequenceEqual(P.shape, (1,1))

    def test_train_and_predict_vectorized_kalman_filter_ema(self):
        training_matrix = np.ones((5,10))

        smoother = simdkalman.KalmanFilter(
            state_transition = 1,
            process_noise = 0.1,
            measurement_model = 1,
            measurement_noise = 0.1)

        r = smoother.compute_matrix(
            training_matrix,
            n_test = 4,
            initial_value = 0,
            initial_covariance = 1.0)

        self.assertSequenceEqual(r.predicted_observations.shape, (5,4))
        self.assertSequenceEqual(r.smoothed_observations.shape, training_matrix.shape)
        self.assertSequenceEqual(r.predicted_means.shape, (5,4,1))
        self.assertSequenceEqual(r.smoothed_means.shape, (5,10,1))
        self.assertSequenceEqual(r.predicted_covariances.shape, (5,4,1,1))
        self.assertSequenceEqual(r.smoothed_covariances.shape, (5,10,1,1))

    def test_train_and_predict_vectorized_kalman_filter_2_states(self):
        training_matrix = np.ones((5,10))

        smoother = simdkalman.KalmanFilter(
            state_transition = np.eye(2),
            process_noise = 0.1,
            measurement_model = np.array([[1,1]]),
            measurement_noise = 0.1)

        r = smoother.compute_matrix(
            training_matrix,
            n_test = 4,
            initial_value = np.array([[0],[0]]),
            initial_covariance = 1.0,
            smooth = True,
            store_gains = True)

        self.assertSequenceEqual(r.predicted_observations.shape, (5,4))
        self.assertSequenceEqual(r.smoothed_observations.shape, training_matrix.shape)
        self.assertSequenceEqual(r.predicted_means.shape, (5,4,2))
        self.assertSequenceEqual(r.smoothed_means.shape, (5,10,2))
        self.assertSequenceEqual(r.predicted_covariances.shape, (5,4,2,2))
        self.assertSequenceEqual(r.smoothed_covariances.shape, (5,10,2,2))

        A = smoother.em_process_noise(r)

        self.assertSequenceEqual(A.shape, (5,2,2))
        A0 = A[0,...]
        self.assertMatrixEqual(A0, A0.T)
        self.assertTrue(min(np.linalg.eig(A0)[0]) > 0)

        B = smoother.em_observation_noise(r, training_matrix)
        self.assertSequenceEqual(B.shape, (5,1,1))
        self.assertTrue(min(list(B)) > 0)


    def test_em_algorithm(self):
        training_matrix = np.ones((5,10))

        smoother = simdkalman.KalmanFilter(
            state_transition = np.eye(2),
            process_noise = 0.1,
            measurement_model = np.array([[1,1]]),
            measurement_noise = 0.1)

        r = smoother.em(training_matrix, n_iter=5, verbose=False)

        self.assertSequenceEqual(r.process_noise.shape, (5,2,2))
        A0 = r.process_noise[0,...]
        self.assertMatrixEqual(A0, A0.T)
        self.assertTrue(min(np.linalg.eig(A0)[0]) > 0)

        B = r.measurement_noise
        self.assertSequenceEqual(B.shape, (5,1,1))
        self.assertTrue(min(list(B)) > 0)

if __name__ == '__main__':
    unittest.main()
