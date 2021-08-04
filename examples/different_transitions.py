# Example of updating with different H,y etc

from simdkalman import KalmanFilter
import numpy as np 

# This demonstrates that different transition matrices can be applied
# to different time-series. However for this to work a couple of small changes
# need to be made to the dimension checks in the KalmanFilter constructor and 
# also the prediction primitive. 

if __name__=='__main__':

   # Process noise
    Q = np.zeros((2,3,3))
    Q[0,:,:] = np.eye(3)
    Q[1,:,:] = np.eye(3)

    # Prior state means
    m0 = np.zeros((2,3,1))
    m0[0,:,:] = np.array([[1.5, 1.5, 1.5]]).transpose()
    m0[1,:,:] = np.array([[1.5, 1.5, 1.5]]).transpose()

    # Prior covariances
    P0 = np.zeros((2, 3, 3))
    P0[0, :, :] = np.eye(3)
    P0[1, :, :] = np.eye(3)

    # Observations
    y_update = np.zeros((2, 1, 1))
    y_update[0, :, :] = np.array([[-1.4]])
    y_update[1, :, :] = np.array([[-1.4]])

    # Measurement var
    R = np.zeros((2, 1, 1))
    R[0, :, :] = np.eye(1)
    R[1, :, :] = 0.66*np.eye(1)

    # Observation equations
    H0 = np.array([[0.8, 0.2, 0]])
    H1 = np.array([[0.8, 0.2, 0]])
    H0.shape = (1, 3)
    H1.shape = (1, 3)
    H = np.zeros((2, 1, 3))
    H[0, :, :] = H0
    H[1, :, :] = H1

    # Transition(s)
    testing_different_transitions = True
    A0 = np.array([[0.5, 0.3, 0.2], [1, 0, 0], [0, 1, 0]])
    if testing_different_transitions:
        A = np.zeros((2,3,3) )
        A1 = np.array([[0.4, 0.4, 0.2], [1, 0, 0], [0, 1, 0]])
        A[0,:,:] = A0
        A[1,:,:] = A1
    else:
        A = A0

    kf = KalmanFilter(
        state_transition=A,  # A
        process_noise=Q,  # Q
        observation_model=H,  # H
        observation_noise=R)  # R
    results = kf.predict(data=y_update,n_test=1,initial_value=m0, initial_covariance=P0,
                         states=True, covariances=True)
    m1 = results.states.mean
    P1 = results.states.cov
    print(m1)
