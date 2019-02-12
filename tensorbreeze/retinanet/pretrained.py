import logging
import tensorflow as tf
from ..resnet import load_pretrained_weights as load_pretrained_resnet_weights

logger = logging.getLogger(__name__)


def load_pretrained_weights(backbone_type, sess=None, verbosity=0):
    """
    Load Backbone weights from a pretrained torchvision model and insert said
    weights into associated tensors in session

    Ensure that parameters have already been initalized at this point

    Args:
        backbone_type: The type of backbone used
        sess: Session containing graph wherer backbone tensors and ops resides
        verbosity: Level of logging to use
            0 - Only Errors
            1 - Start and end logged
            2 - Skipped blobs logged
    """
    if sess is None:
        sess = tf.get_default_session()

    if verbosity >= 1:
        logger.info('Loading pretrained weights')

    with tf.variable_scope('backbone'):
        if 'resnet' in backbone_type:
            load_pretrained_resnet_weights(backbone_type, sess, verbosity=0)
        else:
            raise ValueError(
                '{} is not a valid backbone type'.format(backbone_type))

    if verbosity >= 1:
        logger.info('Pretrained weights loaded')
