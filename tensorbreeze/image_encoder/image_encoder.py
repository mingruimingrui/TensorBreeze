import tensorflow as tf

from .config import make_config
from ..resnet import add_resnet_ops
from ._encoder import add_encoder_ops

data_format = 'channels_first'


def add_image_encoder_ops(x, config_file, **kwargs):
    """
    Image encoder which encodes an image into an embedding
    """
    config = make_config(config_file, **kwargs)

    with tf.variable_scope('backbone'):
        features, _ = add_resnet_ops(x, **config.BACKBONE)

    with tf.variable_scope('encoder'):
        embedding = add_encoder_ops(
            x,
            feature_size=config.TARGET.FEATURE_SIZE,
            pool_method=config.TARGET.POOL_METHOD,
            data_format=data_format
        )

    return embedding, config
