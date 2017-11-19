
# SIMD Kalman

Fast Kalman filters in Python leveraging single-instruction multiple-data
vectorization. That is, running _n_ similar Kalman filters on _n_
independent series of observations. Not to be confused with SIMD processor
instructions.

```python
import simdkalman

kf = simdkalman.KalmanFilter(
    state_transition = np.array([[1,1],[0,1]]),
    process_noise = np.diag([0.1, 0.01]),
    observation_model = np.array([[1,0]]),
    observation_noise = 1.0)

data = numpy.random.normal(size=(200, 1000))

# smooth and explain existing data
smoothed = kf.smooth(data)
# predict new data
pred = kf.predict(data, 15)
```
See `example.py` for more details.

According to `benchmark.py`. This can be up to 100x faster than
[pykalman](https://pykalman.github.io/).

### TODO
 - [ ] documentation
 - [ ] multi-dimensional observations
 - [ ] PyPI package
