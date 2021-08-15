"""
In this simulation, there are multiple moving "trackable" objects and multiple
"sensors", which measure the noisy distance to each object. The goal is to
estimate the positions of the objects based on the noisy distance data.
"""
import numpy as np
import simdkalman

class ExampleEKF:
    """
    Vectorized EKF model for estimating the positions of the trackables
    based on noisy distance data
    """
    def __init__(self, simulation):
        # these parameters are assumed to be known and fixed
        self.n_trackables = len(simulation.trackables)
        self.sensors = simulation.sensor_positions

        # the state per trackable is the xy-position
        STATE_DIM = 2

        self.m = np.zeros((self.n_trackables, STATE_DIM, 1))
        self.P = np.zeros((self.n_trackables, STATE_DIM, STATE_DIM))

        # initial position guesses: at origin, lots of noise
        INITIAL_POS_NOISE = 10.0
        for i in range(self.n_trackables):
            self.P[i, ...] = np.eye(STATE_DIM) * INITIAL_POS_NOISE

        # measurements are distances (1-d) to each sensor
        OBS_DIM = len(self.sensors)
        OBS_NOISE = 0.1

        # A simple random walk model is assumed for the time evolution of
        # the estimated positions. Extending to something more sophisticated,
        # e.g., with estimated velocities in the state, can give better results
        RANDOM_WALK_NOISE = 0.05

        # these matrices are fixed in this example, but could also vary
        # dynamically in other models
        self.R = np.zeros((self.n_trackables, OBS_DIM, OBS_DIM))
        self.A = np.zeros((self.n_trackables, STATE_DIM, STATE_DIM))
        self.Q = np.zeros((self.n_trackables, STATE_DIM, STATE_DIM))

        for i in range(self.n_trackables):
            self.A[i, ...] = np.eye(STATE_DIM)
            self.Q[i, ...] = np.eye(STATE_DIM) * RANDOM_WALK_NOISE**2

            self.R[i, ...] = np.eye(OBS_DIM) * OBS_NOISE**2

    def predict(self):
        self.m, self.P = simdkalman.primitives.predict(self.m, self.P, self.A, self.Q)

    def update(self, observations):
        # EKF update: different H on every step, formed in a vectorized manner

        # axes: m: (obj_i, xy, dummy), sensors: (sensor_i, xy)
        # -> est_deltas: (obj_i, sensor_i, xy)
        est_deltas = self.m[:, np.newaxis, :, 0] - self.sensors[np.newaxis, :, :]
        est_distances = np.sqrt(np.sum(est_deltas**2, axis=2))[..., np.newaxis]

        # Jacobian matrix
        # D dist = D sqrt(v.v) = 1/2 * 1/sqrt(v.v) * (v.Dv + Dv.v) = v.Dv / dist
        # => grad dist = v / dist = normalize(v)
        H = est_deltas / est_distances

        # EKF update as using the fully linear KF update equations
        y_lin = observations - est_distances + simdkalman.primitives.ddot(H, self.m)

        self.m, self.P = simdkalman.primitives.update(self.m, self.P, H, self.R, y_lin)

class Simulation:
    """
    Simulate movement of the "trackables". This is not vectorized and does not
    aim to be very efficients.
    """
    def __init__(self):
        # the trackables move along circular arcs
        def point_on_circle(center, radius, theta):
            return center + np.array([np.sin(theta), np.cos(theta)]) * radius

        class Trackable:
            def __init__(self):
                self.arc_center = np.random.normal(size=2) * 0.6
                self.arc_radius = np.random.rand() * 0.5 + 0.5
                self.arc_theta = np.random.rand() * 2.0 * np.pi
                self.arc_vel = np.random.rand() * 0.15 + 0.15
                if np.random.rand() > 0.5: self.arc_vel *= -1
                self.move(0)

            def move(self, delta_t):
                self.arc_theta += self.arc_vel * delta_t

            @property
            def position(self):
                return point_on_circle(self.arc_center, self.arc_radius, self.arc_theta)

        N_TRACKABLES = 2
        N_SENSORS = 3
        ARC_LEN_DEG = 80
        ARC_START = 90
        ARC_RADIUS = 2.5

        self.trackables = [Trackable() for _ in range(N_TRACKABLES)]
        self.sensor_positions = np.array([
            point_on_circle(
                center = np.array([0, 0]),
                radius = ARC_RADIUS,
                theta = (ARC_LEN_DEG * i / (N_SENSORS-1) + ARC_START) * np.pi / 180)
            for i in range(N_SENSORS)
        ])

        self.time = 0

    def simulate_step(self):
        DELTA_T = 0.2
        T_MAX = 10
        MEASUREMENT_NOISE = 0.05

        self.time += DELTA_T
        if self.time > T_MAX: return None

        # store for visualization
        true_positions = []
        observations = np.zeros((
            len(self.trackables),
            len(self.sensor_positions),
            1))

        for obj_i, obj  in enumerate(self.trackables):
            obj.move(DELTA_T)
            true_positions.append(obj.position)

            for sensor_i, sensor_position in enumerate(self.sensor_positions):
                true_distance = np.linalg.norm(sensor_position - obj.position)
                observations[obj_i, sensor_i, 0] = true_distance \
                    + np.random.normal() * MEASUREMENT_NOISE

        return (true_positions, observations)

def uncertainty_ellipse_95(mean, cov):
    """
    Compute the points on the arc of an 95% uncertainty ellipse based on
    a mean and covariance matrix of an xy-position
    """
    N = 30
    theta = np.linspace(0, 2*np.pi, num=N)
    circle = np.vstack([c[np.newaxis, :] for c in (np.sin(theta), np.cos(theta))])

    u, s_vec, _ = np.linalg.svd(cov)
    # see https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    scale_for_95 = 2.0 * np.sqrt(5.991)
    s_mat = np.diag(np.sqrt(s_vec)) * scale_for_95

    return np.dot(u, np.dot(s_mat, circle)).transpose() + mean[np.newaxis, :]

# ---------- Initialize
np.random.seed(100)

simulation = Simulation()
ekf = ExampleEKF(simulation)

# run simulation step by step and store the intermediary results for visualization
true_trajectories = []
estimated_trajectories = []
estimated_uncertainties = []

# ---------- Run simulation and estimation
while True:
    simulated = simulation.simulate_step()
    if simulated is None: break

    true_positions, observations = simulated
    true_trajectories.append(true_positions)

    ekf.predict()
    ekf.update(observations)

    estimated_trajectories.append(ekf.m[:, :, 0])
    for i in range(ekf.P.shape[0]):
        # flat list of ellipses, not separating trackables
        estimated_uncertainties.append(
            uncertainty_ellipse_95(ekf.m[i, :, 0], ekf.P[i, ...]))

# ---------- Visualize results
import matplotlib.pyplot as plt

# uncertainty ellipses at the bottom
for ell in estimated_uncertainties:
    plt.plot(ell[:, 0], ell[:, 1], color='blue', lw=1, alpha=0.1)

true_trajectories = np.array(true_trajectories)
estimated_trajectories = np.array(estimated_trajectories)
for i in range(len(true_trajectories[0])):
    kwargs = {}
    if i == 0: kwargs['label'] = 'true trajectories'
    plt.plot(true_trajectories[:, i, 0], true_trajectories[:, i, 1], 'k', **kwargs)
    if i == 0: kwargs['label'] = 'estimated trajectories'
    plt.plot(estimated_trajectories[:, i, 0], estimated_trajectories[:, i, 1], 'bx', alpha=0.5, **kwargs)

plt.scatter(
    simulation.sensor_positions[:, 0],
    simulation.sensor_positions[:, 1],
    color='red', label='sensors')
plt.legend()
plt.axis('equal')
plt.show()
