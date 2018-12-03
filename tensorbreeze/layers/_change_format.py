import tensorflow as tf


def to_nchw(tensor, name='to_nchw'):
    assert len(tensor.shape) == 4
    return tf.transpose(tensor, perm=(0, 3, 1, 2), name=name, conjugate=False)


def to_nhwc(tensor, name='to_nhwc'):
    assert len(tensor.shape) == 4
    return tf.transpose(tensor, perm=(0, 2, 3, 1), name=name, conjugate=False)
