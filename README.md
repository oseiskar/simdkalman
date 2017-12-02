
# SIMD Kalman [![Build Status](https://travis-ci.org/oseiskar/simdkalman.svg?branch=master)](https://travis-ci.org/oseiskar/simdkalman)

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

### Development

 1. Create virtualenv
    * Python 2: `virtualenv venvs/python2`
    * Python 3: `python3 -m venv venvs/python3`
 1. Activate virtualenv: `source venvs/pythonNNN/bin/activate`
 1. Install locally `pip install -e .[dev]`
 1. `./run-tests.sh`
 1. `deactivate` virtualenv

### Distribution

(personal howto)

Once:

 1. create an account in https://testpypi.python.org/pypi and
    https://pypi.python.org/pypi
 1. create `~/.pypirc` as described [here](https://packaging.python.org/guides/migrating-to-pypi-org)
 1. `sudo pip install twine`
 1. create testing virutalenvs:
    * `virtualenv venvs/test-python2`
    * `python3 -m venv venvs/test-python3`

Each distribution:

    python setup.py bdist_wheel
    # test PyPI site
    twine upload --repository testpypi dist/simdkalman-VERSION*
    # the real thing
    twine upload dist/simdkalman-VERSION*

Test installation from the test site with

    source venvs/test-pythonNNN/bin/activate
    pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple \
        simdkalman --upgrade
    pip install matplotlib
    python examples/example.py
    deactivate

### TODO

 - [ ] multi-dimensional observations
 - [ ] documentation site
 - [ ] support numpy versions < 1.10
