import tensorflow as tf


def add_encoder_ops(
    x,
    feature_size=256,
    pool_method='avg',
    data_format='channels_first'
):
    assert pool_method in {'avg', 'max'}

    # Identify size of input image/feature tensor
    if data_format == 'channels_first':
        x_size = x.shape[2:]
    else:
        x_size = x.shape[1:3]

    # Determine pooling function to use
    if pool_method == 'avg':
        pooling_fn = tf.keras.layers.AveragePooling2D
    else:
        pooling_fn = tf.keras.layers.MaxPooling2D

    # Define the encoder ops
    x = pooling_fn(
        pool_size=x_size,
        strides=1,
        padding='valid',
        data_format=data_format,
        name='final_pool'
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        feature_size,
        name='fc'
    )(x)
    return x
