simdkalman documentation
======================================

.. include:: DESCRIPTION.rst

**Installation**: ``pip install simdkalman``

**Source code**:  https://github.com/oseiskar/simdkalman

**License**: `MIT <https://github.com/oseiskar/simdkalman/blob/master/LICENSE.txt>`_

.. image:: example.png

Terminology
------------

.. automodule:: simdkalman

For a complete code example with figures, see:
https://github.com/oseiskar/simdkalman/blob/master/examples/example.py

Class documentation
-------------------

.. autoclass:: simdkalman.KalmanFilter
   :members:

Primitives
----------

.. automodule:: simdkalman.primitives

.. autofunction:: predict(mean, covariance, state_transition, process_noise)
.. autofunction:: update
.. autofunction:: update_with_nan_check
.. autofunction:: smooth
.. autofunction:: expected_observation(mean, observation_model)
.. autofunction:: observation_covariance(covariance, observation_model, observation_noise)

Change log
-----------

See https://github.com/oseiskar/simdkalman/releases
