simdkalman primitives module
----------------------------

.. automodule:: simdkalman.primitives

For example, to apply a prediction step to 3 different states :math:`x` of
dimension 2 with 3 different state transition matrices :math:`A`, one could do

.. testcode ::

    from simdkalman.primitives import predict
    import numpy

    # different states
    m1, m2, m3 = [numpy.array([[0],[i]]) for i in range(3)]
    # with the same covariance (initially)
    P = numpy.eye(2)

    # different transition matrices
    A1, A2, A3 = [numpy.eye(2)*i for i in range(3)]
    # same process noise
    Q = numpy.eye(2)*0.01

    # stack correctly
    m = numpy.vstack([
        m1[numpy.newaxis, ...],
        m2[numpy.newaxis, ...],
        m3[numpy.newaxis, ...]
    ])
    A = numpy.vstack([
        A1[numpy.newaxis, ...],
        A2[numpy.newaxis, ...],
        A3[numpy.newaxis, ...]
    ])

    # predict
    m, P = predict(m, P, A, Q)

    print(m.shape)
    print(P.shape)

.. testoutput ::

    (3, 2, 1)
    (3, 2, 2)


.. autofunction:: predict(mean, covariance, state_transition, process_noise)
.. autofunction:: update
.. autofunction:: update_with_nan_check
.. autofunction:: smooth
.. autofunction:: predict_observation(mean, covariance, observation_model, observation_noise)
