import numpy as np
import tensorflow as tf


@tf.function
def euclidean_distances(a: tf.Tensor, b: tf.Tensor = None, squared: bool = False) -> tf.Tensor:
    """
    This implementation is based on sklearn:
    Considering the rows of a (and b) as vectors, compute the
    distance matrix between each pair of vectors.

    The euclidean distance between a pair of row vector a and b is computed as:

        dist(a, b) = sqrt(dot(a, a) - 2 * dot(a, b) + dot(b, b))

    According to scikit-learn, it is not the most precise way as it can suffer from catastrophic cancellation.

    For more details see:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html

    Parameters
    ----------
    a : 2D Tensor, shape: (n_samples_a, n_features).
    b : 2D Tensor, default is None, shape: (n_samples_b, n_features).
        If b is equal to a, you should only provide the argument a.
        For performance reasons we do not check the equality of a and b.
    squared : bool, default is False. Return squared Euclidean distances.

    Returns
    -------
    distances : 2D Tensor, shape (n_samples_a, n_samples_b).
    """
    aa = tf.reduce_sum(tf.square(a), axis=1)

    _b = a
    bb = aa
    if b is not None:
        _b = b
        bb = tf.reduce_sum(tf.square(b), axis=1)

    dists = -2. * tf.matmul(a, tf.transpose(_b))
    dists += aa[:, tf.newaxis]
    dists += bb[tf.newaxis, :]
    dists = tf.maximum(dists, 0.)

    # enforce zeros on the diagonal if a = b
    if b is None:
        diag = tf.zeros((1, tf.shape(a)[0]), dtype=dists.dtype)
        dists = tf.linalg.set_diag(dists[tf.newaxis, :], diag)
        dists = tf.squeeze(dists, axis=0)

    if not squared:
        dists = tf.sqrt(dists)

    return dists


@tf.function
def compute_batch_count(n_samples: tf.Tensor, batch_size: tf.Tensor):
    n_batches = tf.cast(n_samples / batch_size, tf.int32)
    is_rest = tf.not_equal(tf.math.mod(n_samples, batch_size), tf.constant(0, tf.int32))
    return tf.where(is_rest, tf.add(n_batches, 1), n_batches)


@tf.function
def remove_infs(tensor, ensure_1d=True):
    """
    Remove inf values from the given tensor. Note that the output tensor will be the flattened version of the input.

    Parameters
    ----------
    tensor : Tensor of any shape.
    ensure_1d : bool, default is True. Check whether the Tensor is 1d.

    Returns
    -------
    Will remove all inf entries and return the other entries flattened.
    """
    mask = tf.math.logical_not(tf.math.is_inf(tensor))
    if ensure_1d:
        mask = tf.ensure_shape(mask, [None])
    return tf.boolean_mask(tensor, mask)


def convert_tf_tensor_array_2_numpy(tensor_array):
    # TensorArray behaves different in (non-)eager execution.
    result = None
    if tf.is_tensor(tensor_array):
        result = tensor_array.numpy()
    elif tensor_array.size().numpy() > 0:
        result = []
        for i in range(tensor_array.size().numpy()):
            result.append(tensor_array.read(i).numpy())
        result = np.concatenate(result, axis=0)
    return result
