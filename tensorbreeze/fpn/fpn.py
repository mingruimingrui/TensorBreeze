import tensorflow as tf

from .config import make_config
from .. import layers

data_format = 'channels_first'


def add_fpn_ops(input_features, config_file=None, **kwargs):
    """
    FPN layer as defined in https://arxiv.org/abs/1612.03144
    """
    config = make_config(config_file, **kwargs)

    # Produce features in this order
    # MAX_INPUT_LEVEL
    # MAX_INPUT_LEVEL - 1
    # ...
    # MIN_LEVEL + 1
    # MIN_LEVEL
    # MAX_INPUT_LEVEL + 1
    # ...
    # MAX_LEVEL - 1
    # MAX_LEVEL

    # Create dicts for easy access of tensors later
    reduced_features = dict()
    output_features = dict()

    # Create ranges of levels for looping later
    input_levels = range(config.MIN_INPUT_LEVEL, config.MAX_INPUT_LEVEL + 1)
    output_levels_1 = range(config.MIN_LEVEL, config.MAX_INPUT_LEVEL + 1)
    output_levels_2 = range(config.MAX_INPUT_LEVEL + 1, config.MAX_LEVEL + 1)
    all_output_levels = range(config.MIN_LEVEL, config.MAX_LEVEL + 1)

    # Convert input_features into dict
    assert len(input_features) == len(input_levels), \
        'Number of input_feautres does not match number of levels required for FPN'

    input_features = dict(zip(input_levels, input_features))

    # Produce MAX_INPUT_LEVEL to MIN_LEVEL
    for level in output_levels_1[::-1]:
        reduced_features[level] = tf.layers.conv2d(
            input_features[level],
            config.FEATURE_SIZE,
            kernel_size=1,
            strides=1,
            padding='valid',
            data_format=data_format,
            use_bias=True,
            trainable=True,
            name='conv_reduce/{}'.format(level)
        )

        inner = reduced_features[level]

        if level < config.MAX_INPUT_LEVEL:
            # Resize and add the higher level features
            inner_shape = tf.shape(inner)
            if data_format == 'channels_first':
                inner_shape = inner_shape[-2:]
            else:
                inner_shape = inner_shape[1:3]

            inner_higher = reduced_features[level + 1]

            if data_format == 'channels_first':
                inner_higher = layers.to_nhwc(inner_higher)

            inner_higher = tf.image.resize_images(
                inner_higher,
                size=inner_shape,
                method=tf.image.ResizeMethod.BILINEAR,
                align_corners=False
            )

            if data_format == 'channels_first':
                inner_higher = layers.to_nchw(inner_higher)

            inner = tf.math.add(inner, inner_higher)

        inner = layers.pad2d(inner, 1)
        output_features[level] = tf.layers.conv2d(
            inner,
            config.FEATURE_SIZE,
            kernel_size=3,
            strides=1,
            padding='valid',
            data_format=data_format,
            use_bias=True,
            trainable=True,
            name='conv/{}'.format(level)
        )

    # Produce (MAX_INPUT_LEVEL + 1) to MAX_LEVEL
    for level in output_levels_2:
        inner = output_features[level - 1]
        inner = tf.nn.relu(inner)
        inner = layers.pad2d(inner, 1)
        output_features[level] = tf.layers.conv2d(
            inner,
            config.FEATURE_SIZE,
            kernel_size=3,
            strides=2,
            padding='valid',
            data_format=data_format,
            use_bias=True,
            trainable=True,
            name='conv/{}'.format(level)
        )

    return [output_features[level] for level in all_output_levels], config
