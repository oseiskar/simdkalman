import numpy as np
from simdkalman.primitives import predict, update

# define model
state_transition = np.array([[1,1],[0,1]])
process_noise = np.eye(2)*0.01
observation_model = np.array([[1,0]])
observation_noise = np.array([[1.0]])

# initial state
m = np.array([0, 1])
P = np.eye(2)

# predict next state
m, P = predict(m, P, state_transition, process_noise)

# first observation
y = np.array([4])
m, P = update(m, P, observation_model, observation_noise, y)

# predict second state
m, P = predict(m, P, state_transition, process_noise)

print('mean')
print(m)

print('cov')
print(P)
