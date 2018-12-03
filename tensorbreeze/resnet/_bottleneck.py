import tensorflow as tf
from .. import layers


def add_bottleneck_ops_1(
    x_in,
    dim_inner,
    dim_out,
    stride=1,
    data_format='channels_first',
    trainable=True,
    downsample=False
):
    """
    Default bottleneck layer for resnet18 and resnet34
    """
    channel_axis = 1 if data_format == 'channels_first' else -1

    residual = x_in
    if downsample:
        with tf.variable_scope('downsample'):
            residual = tf.layers.conv2d(
                x_in,
                dim_out,
                kernel_size=1,
                strides=stride,
                padding='valid',
                data_format=data_format,
                use_bias=False,
                trainable=trainable,
                name='0'
            )
            residual = tf.layers.batch_normalization(
                residual,
                axis=channel_axis,
                momentum=0.1,
                epsilon=1e-5,
                trainable=False,
                name='1'
            )

    # 3x3 BN ReLU
    x_a = layers.pad2d(x_in, 1)
    x_a = tf.layers.conv2d(
        x_a,
        dim_inner,
        kernel_size=3,
        strides=stride,
        padding='valid',
        data_format=data_format,
        use_bias=False,
        trainable=trainable,
        name='conv1'
    )
    x_a = tf.layers.batch_normalization(
        x_a,
        axis=channel_axis,
        momentum=0.1,
        epsilon=1e-5,
        trainable=False,
        name='bn1'
    )
    x_a = tf.nn.relu(x_a)

    # 3x3 BN
    x_b = layers.pad2d(x_a, 1)
    x_b = tf.layers.conv2d(
        x_b,
        dim_out,
        kernel_size=3,
        strides=1,
        padding='valid',
        data_format=data_format,
        use_bias=False,
        trainable=trainable,
        name='conv2'
    )
    x_b = tf.layers.batch_normalization(
        x_b,
        axis=channel_axis,
        momentum=0.1,
        epsilon=1e-5,
        trainable=False,
        name='bn2'
    )

    # Sum and ReLU
    x_out = tf.math.add(residual, x_b, name='sum')
    return tf.nn.relu(x_out)


def add_bottleneck_ops_2(
    x_in,
    dim_inner,
    dim_out,
    stride=1,
    data_format='channels_first',
    trainable=True,
    downsample=False
):
    """
    Default bottleneck layer for resnet50, resnet101, resnet152
    """
    channel_axis = 1 if data_format == 'channels_first' else -1

    residual = x_in
    if downsample:
        with tf.variable_scope('downsample'):
            residual = tf.layers.conv2d(
                x_in,
                dim_out,
                kernel_size=1,
                strides=stride,
                padding='valid',
                data_format=data_format,
                use_bias=False,
                trainable=trainable,
                name='0'
            )
            residual = tf.layers.batch_normalization(
                residual,
                axis=channel_axis,
                momentum=0.1,
                epsilon=1e-5,
                trainable=False,
                name='1'
            )

    # 1x1 BN ReLU
    x_a = tf.layers.conv2d(
        x_in,
        dim_inner,
        kernel_size=1,
        strides=1,
        padding='valid',
        data_format=data_format,
        use_bias=False,
        trainable=trainable,
        name='conv1'
    )
    x_a = tf.layers.batch_normalization(
        x_a,
        axis=channel_axis,
        momentum=0.1,
        epsilon=1e-5,
        trainable=False,
        name='bn1'
    )
    x_a = tf.nn.relu(x_a)

    # 3x3 BN ReLU
    x_b = layers.pad2d(x_a, 1)
    x_b = tf.layers.conv2d(
        x_b,
        dim_inner,
        kernel_size=3,
        strides=stride,
        padding='valid',
        data_format=data_format,
        use_bias=False,
        trainable=trainable,
        name='conv2'
    )
    x_b = tf.layers.batch_normalization(
        x_b,
        axis=channel_axis,
        momentum=0.1,
        epsilon=1e-5,
        trainable=False,
        name='bn2'
    )
    x_b = tf.nn.relu(x_b)

    # 1x1 BN
    x_c = tf.layers.conv2d(
        x_b,
        dim_out,
        kernel_size=1,
        strides=1,
        padding='valid',
        data_format=data_format,
        use_bias=False,
        trainable=trainable,
        name='conv3'
    )
    x_c = tf.layers.batch_normalization(
        x_c,
        axis=channel_axis,
        momentum=0.1,
        epsilon=1e-5,
        trainable=False,
        name='bn3'
    )

    # Sum and ReLU
    x_out = tf.math.add(residual, x_c, name='sum')
    return tf.nn.relu(x_out)


def get_add_bottleneck_ops(resnet_type):
    """
    Get the add bottleneck ops function based on the type of ResNet
    """
    if resnet_type in {'resnet18', 'resnet34'}:
        return add_bottleneck_ops_1
    else:
        return add_bottleneck_ops_2
