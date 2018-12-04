import tensorflow as tf

from .config import make_config
from ..resnet import add_resnet_ops
from ..fpn import add_fpn_ops
from ..layers import compute_anchors
from ._head import (
    add_comb_head_ops,
    add_cls_head_ops,
    add_reg_head_ops
)

data_format = 'channels_first'


def add_retinanet_ops(x, config_file=None, **kwargs):
    """
    RetinaNet as described in https://arxiv.org/abs/1708.02002
    """
    config = make_config(config_file, **kwargs)

    with tf.variable_scope('backbone'):
        features, _ = add_resnet_ops(x, **config.BACKBONE)

    with tf.variable_scope('fpn'):
        pyramid_features, _ = add_fpn_ops(features, **config.FPN)

    # Generate anchors at each feature level
    anchors = compute_anchors(
        pyramid_features,
        ratios=config.ANCHOR.RATIOS,
        scales_per_octave=config.ANCHOR.SCALES_PER_OCTAVE,
        min_feature_level=config.FPN.MIN_LEVEL,
        size_mult=config.ANCHOR.SIZE_MULT,
        stride_mult=config.ANCHOR.STRIDE_MULT,
        data_format=data_format,
        name='compute_anchors'
    )

    # Generate classification and regression outputs at each feature level
    num_anchors = len(config.ANCHOR.RATIOS) * config.ANCHOR.SCALES_PER_OCTAVE

    if config.COMBINED.USE:
        with tf.variable_scope('detector'):
            cls_output, reg_output = add_comb_head_ops(
                pyramid_features,
                num_anchors=num_anchors,
                num_classes=config.TARGET.NUM_CLASSES,
                num_layers=config.COMBINED.NUM_LAYERS,
                feature_size=config.COMBINED.FEATURE_SIZE,
                use_class_specific_bbox=config.TARGET.CLASS_SPECIFIC_BBOX,
                use_bg_predictor=config.TARGET.BG_PREDICTOR,
                prior_prob=config.INITIALIZATION.PRIOR_PROB,
                data_format=data_format
            )

    else:
        with tf.variable_scope('classifier'):
            cls_output = add_cls_head_ops(
                pyramid_features,
                num_anchors=num_anchors,
                num_classes=config.TARGET.NUM_CLASSES,
                num_layers=config.CLASSIFIER.NUM_LAYERS,
                feature_size=config.CLASSIFIER.FEATURE_SIZE,
                use_bg_predictor=config.TARGET.BG_PREDICTOR,
                prior_prob=config.INITIALIZATION.PRIOR_PROB,
                data_format=data_format
            )
        with tf.variable_scope('regressor'):
            reg_output = add_reg_head_ops(
                pyramid_features,
                num_anchors=num_anchors,
                num_classes=config.TARGET.NUM_CLASSES,
                num_layers=config.REGRESSOR.NUM_LAYERS,
                feature_size=config.REGRESSOR.FEATURE_SIZE,
                use_class_specific_bbox=config.TARGET.CLASS_SPECIFIC_BBOX,
                prior_prob=config.INITIALIZATION.PRIOR_PROB,
                data_format=data_format
            )

    #

    return (anchors), config
