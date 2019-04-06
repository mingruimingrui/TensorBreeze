import tensorflow as tf
from .. import layers


def add_stem_ops(x, filters, data_format='channels_first', trainable=True, freeze_bn=True):
    norm_axis = 1 if data_format == 'channels_first' else -1

    x = layers.pad2d(x, 3)
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=7,
        strides=2,
        padding='valid',
        data_format=data_format,
        use_bias=False,
        trainable=trainable,
        name='conv1'
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=norm_axis,
        momentum=0.1,
        epsilon=1e-5,
        trainable=trainable and not freeze_bn,
        name='bn1'
    )(x)
    return tf.nn.relu(x)


def add_maxpool_ops(x, data_format='channels_first'):
    x = layers.pad2d(x, 1)
    return tf.keras.layers.MaxPooling2D(
        3,
        strides=2,
        padding='valid',
        data_format=data_format,
        name='pool1'
    )(x)
