import tensorflow as tf


def add_classifier_ops(x, num_classes, data_format='channels_first'):
    x = tf.layers.average_pooling2d(
        x,
        7,
        strides=1,
        padding='valid',
        data_format=data_format,
        name='layer5_pool'
    )
    x = tf.layers.flatten(x)
    x = tf.layers.dense(
        x,
        num_classes,
        activation='softmax',
        name='fc'
    )
    return x
