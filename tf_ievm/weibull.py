import tensorflow as tf


@tf.function
def cdf(data, shapes, scales, dtype=tf.float32):
    return (tf.constant(1., dtype=dtype)
            -
            tf.exp(tf.negative(
                tf.pow(
                    tf.divide(data, scales),
                    shapes))))


@tf.function
def fit(tail, max_iter=100, tolerance=1e-6, shape_init=1., eps=tf.keras.backend.epsilon(), dtype=tf.float32):
    """
    Estimate a Weibull distribution on the given tail, i.e. will give the shape and scale parameter.

    Parameters
    ----------
    tail : tf.Tensor, shape (tail_size,).
           This is the tail for a single sample. It does not contain any inf values.
    max_iter : int, default is 100.
               The maximum number of Newton iterations to fit a Weibull distribution.
    tolerance : float, default is 1e-6.
                Parameter tolerance to check for convergence of the shape parameter during Weibull estimation.
    shape_init : float, default is 1. The initial shape value.
    eps : float, default is tf.keras.backend.epsilon(), which is 1e-7.
          Stop the Newton iterations in case the derivative of the objective w.r.t. the shape parameter is < 'eps'.
    dtype : TensorFlow floating point type, default is tf.float32.
            The floating point type.

    Returns
    -------
    shapes : tf.Tensor, shape (1,).
             The estimated shape parameter of the Weibull distribution.
    scale : tf.Tensor, shape (1,).
            The estimated scale parameter of the Weibull distribution.
    """
    tail = tf.cast(tail, dtype)
    tolerance = tf.constant(tolerance, dtype)
    eps = tf.constant(eps, dtype)

    converged = tf.constant(False, dtype=tf.bool, shape=(1,))
    # shape_init could be a Tensor, then tf.constant does not work, therefore we multiply here
    shape = tf.math.multiply(tf.ones(shape=(1,), dtype=dtype), shape_init)
    log_tail = tf.math.log(tail)

    def condition(_converged, _shape):
        return tf.logical_not(_converged)

    def body(_converged, _shape):
        f, df = _objective(tail, log_tail, _shape, dtype=dtype)

        # Check if df is lower than precision or is nan:
        # True: forward previous result
        # False: compute new shape
        reached_precision = tf.logical_or(tf.logical_not(tf.greater(df, eps)), tf.math.is_nan(df))
        new_shape = tf.cond(reached_precision, true_fn=lambda: _shape,
                            false_fn=lambda: tf.maximum(_shape - (f / df), tf.constant(1., dtype)),
                            name="if_not_reached_precision_compute_shape")

        # Check convergence
        delta = tf.abs(_shape - new_shape)
        _converged = tf.logical_or(reached_precision, tf.less(delta, tolerance))
        return _converged, new_shape

    # Newton iterations.
    # Parallel iterations should have no effect since a subsequent iteration depends on the previous.
    converged, shape = tf.while_loop(
        condition, body, (converged, shape), parallel_iterations=1, maximum_iterations=max_iter,
        name="while_not_converged_do_newton")

    scale = _compute_scale(tail, shape, dtype=dtype)
    return shape, scale


@tf.function
def fit_vectorized(tails, tail_sizes, max_iter=100, tolerance=1e-6, shape_init=1., eps=tf.keras.backend.epsilon(),
                   dtype=tf.float32):
    """
    Estimate Weibull distributions on given tails, i.e. will give the shape and scale parameters.
    Estimations are calculated in a vectorized manner which may lead to unnecessary computations.
    However, this version is much faster than the sample-wise version with tf.map_fn().
    Once an estimate is converged it will not be replaced by following computations, i.e. the result is the same as
    the sample-wise version.

    Parameters
    ----------
    tails : tf.Tensor, shape (n_samples, tail_size).
            The tensor contains multiple tails and may contain inf values in entries where the tail is incomplete.
    tail_sizes : tf.Tensor shape (n_samples, 1). The actual tail size for each sample, ignoring the inf entries.
    max_iter : int, default is 100.
               The maximum number of Newton iterations to fit a Weibull distribution.
    tolerance : float, default is 1e-6.
                Parameter tolerance to check for convergence of the shape parameter during Weibull estimation.
    shape_init : float, default is 1. The initial shape value.
    eps : float, default is tf.keras.backend.epsilon(), which is 1e-7.
          Stop the Newton iterations in case the derivative of the objective w.r.t. the shape parameter is < 'eps'.
    dtype : TensorFlow floating point type, default is tf.float32.
            The floating point type.

    Returns
    -------
    shapes : tf.Tensor, shape (n_samples, 1).
             The estimated shape parameters of the Weibull distribution.
    scale : tf.Tensor, shape (n_samples, 1).
            The estimated scale parameters of the Weibull distribution.
    """
    tails = tf.cast(tails, dtype)
    tail_sizes = tf.cast(tail_sizes, dtype)
    tolerance = tf.constant(tolerance, dtype)
    eps = tf.constant(eps, dtype)

    n_samples = tf.shape(tails)[0]
    converged = tf.fill((n_samples, 1), False, name="converged")
    shapes = tf.cast(tf.fill((n_samples, 1), shape_init, name="shapes"), dtype)
    inf_mask = tf.math.is_inf(tails)

    # Replace infs by log(1) which is 0 and has no effect in the following computations.
    log_tails = tf.math.log(tf.where(inf_mask, tf.constant(1., dtype), tails))

    # Replace infs by 0 such that they have no effect in the following computations.
    inf_replaced_tails = tf.where(inf_mask, tf.constant(0., dtype), tails)

    def condition(_converged, _shapes):
        return tf.logical_not(tf.reduce_all(_converged))  # stop if all converged

    def body(_converged, _shapes):
        f, df = _objective_vectorized(inf_replaced_tails, log_tails, tail_sizes, _shapes, dtype=dtype)

        # Check if df is lower than precision or contains nans:
        # True: forward previous result
        # False: compute new shape
        reached_precision = tf.logical_or(tf.logical_not(tf.greater(df, eps)), tf.math.is_nan(df))
        new_shapes = tf.where(reached_precision, _shapes, tf.maximum(_shapes - (f / df), tf.constant(1., dtype)))

        # Check convergence
        deltas = tf.abs(_shapes - new_shapes)
        new_converged = tf.logical_or(reached_precision, tf.less(deltas, tolerance))

        # Otherwise the tensor shapes cannot be resolved in graph execution.
        new_shapes = tf.ensure_shape(new_shapes, _shapes.shape)
        new_converged = tf.ensure_shape(new_converged, _converged.shape)
        return new_converged, new_shapes

    # Newton iterations.
    # Parallel iterations should have no effect since a subsequent iteration depends on the previous.
    converged, shapes = tf.while_loop(
        condition, body, (converged, shapes), parallel_iterations=1, maximum_iterations=max_iter,
        name="while_not_converged_do_newton")

    scales = _compute_scale_vectorized(inf_replaced_tails, tail_sizes, shapes, dtype=dtype)
    return shapes, scales


@tf.function
def _objective(tail, log_tail, shape, dtype=tf.float32):
    # auxiliary variables
    xk = tf.pow(tail, shape)
    sum_xk = tf.reduce_sum(xk)
    sum_xk_logx = tf.reduce_sum(xk * log_tail)

    # objective
    f = (sum_xk_logx / sum_xk
         -
         tf.reduce_mean(log_tail)
         -
         tf.constant(1., dtype) / shape)

    # Derivative w.r.t. shape.

    # df = (
    #     tf.reduce_sum(xk * tf.square(log_tail)) / sum_xk
    #     -
    #     tf.square(sum_xk_logx) / tf.square(sum_xk)
    #     +
    #     tf.constant(1., dtype) / tf.square(shape)
    # )

    # This version is slightly faster as we can replace one division by a multiplication.
    df = (
            (sum_xk * tf.reduce_sum(xk * tf.square(log_tail)) - tf.square(sum_xk_logx))
            /
            tf.square(sum_xk)
            +
            tf.constant(1., dtype) / tf.square(shape)
    )
    return f, df


@tf.function
def _objective_vectorized(inf_replaced_tails, log_tails, tail_sizes, shapes, dtype=tf.float32):
    # auxiliary variables

    # Replace infs by 0, which has no effect in the following computations.
    xk = tf.pow(inf_replaced_tails, shapes)
    sum_xk = tf.reduce_sum(xk, axis=1, keepdims=True)
    sum_xk_logx = tf.reduce_sum(xk * log_tails, axis=1, keepdims=True)

    # objective
    f = (sum_xk_logx / sum_xk
         -
         tf.reduce_sum(log_tails, axis=1, keepdims=True) / tail_sizes
         -
         tf.constant(1., dtype) / shapes)

    # Derivative w.r.t. shape.

    # df = (
    #     tf.reduce_sum(xk * tf.square(log_tails), axis=1, keepdims=True) / sum_xk
    #     -
    #     tf.square(sum_xk_logx) / tf.square(sum_xk)
    #     +
    #     tf.constant(1., dtype) / tf.square(shapes)
    # )

    # This version is slightly faster as we can replace one division by a multiplication.
    df = (
            (sum_xk * tf.reduce_sum(xk * tf.square(log_tails), axis=1, keepdims=True) - tf.square(sum_xk_logx))
            /
            tf.square(sum_xk)
            +
            tf.constant(1., dtype) / tf.square(shapes)
    )
    return f, df


@tf.function
def _compute_scale(tail, shape, dtype=tf.float32):
    return tf.pow(tf.reduce_mean(tf.pow(tail, shape)), tf.constant(1., dtype) / shape)


@tf.function
def _compute_scale_vectorized(inf_replaced_tails, tail_sizes, shapes, dtype=tf.float32):
    return tf.pow(
        tf.divide(
            tf.reduce_sum(
                tf.pow(inf_replaced_tails, shapes), axis=1, keepdims=True),
            tail_sizes),
        tf.constant(1., dtype) / shapes)
