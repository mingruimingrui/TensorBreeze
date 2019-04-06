"""
Methods to add output heads for resnet
"""

import tensorflow as tf


def add_head_ops(
    x,
    num_classes,
    data_format='channels_first',
    activation='softmax',
    config_for_torch=True
):
    """
    config_for_torch would ensure that final classifier naming is consistent
    """
    if config_for_torch:
        assert x.shape[-2:] == (7, 7), \
            'For head to borrow torch weights, it has to have the correct shape'
        x = tf.keras.layers.AveragePooling2D(
            7,
            strides=1,
            padding='valid',
            data_format=data_format,
            name='layer5_pool'
        )(x)

    else:
        x = tf.keras.layers.AveragePooling2D(
            x.shape[-2:],
            strides=1,
            padding='valid',
            data_format=data_format,
            name='final_pool'
        )(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        num_classes,
        activation=activation,
        name='fc'
    )(x)

    return x
