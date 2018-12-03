import tensorflow as tf
from .. import layers


def add_stem_ops(x, dim, data_format='channels_first', trainable=True):
    norm_axis = 1 if data_format == 'channels_first' else -1

    x = layers.pad2d(x, 3)
    x = tf.layers.conv2d(
        x,
        dim,
        kernel_size=7,
        strides=2,
        padding='valid',
        data_format=data_format,
        use_bias=False,
        trainable=trainable,
        name='conv1'
    )
    x = tf.layers.batch_normalization(
        x,
        axis=norm_axis,
        momentum=0.1,
        epsilon=1e-5,
        trainable=False,
        name='bn1'
    )
    return tf.nn.relu(x)


def add_maxpool_ops(x, data_format='channels_first'):
    x = layers.pad2d(x, 1)
    return tf.layers.max_pooling2d(
        x,
        3,
        strides=2,
        padding='valid',
        data_format=data_format,
        name='pool1'
    )
