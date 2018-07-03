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

For complete code examples with figures, see:
https://github.com/oseiskar/simdkalman/blob/master/examples/ and
`this Gist <https://gist.github.com/oseiskar/5e8937af96efdfe5f8a6994764b07081>`_.

Using multi-dimensional observations is demonstrated in
`this example <https://github.com/oseiskar/simdkalman/blob/master/examples/multi_dimensional_observations.py>`_.

Class documentation
-------------------

.. autoclass:: simdkalman.KalmanFilter
   :members:


Primitives
-------------

The ``simdkalman.primitives`` module contains low-level Kalman filter computation
steps with multi-dimensional input arrays. See `this page <primitives.html>`_
for full documentation.


Change log
-----------

See https://github.com/oseiskar/simdkalman/releases
