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
from ._loss import (
    add_target_ops,
    add_loss_ops
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

    return (anchors, cls_output, reg_output), config


def add_retinanet_train_ops(image_tensor, annotations_tensors, config_file=None, **kwargs):
    (anchors, cls_output, reg_output), config = \
        add_retinanet_ops(image_tensor, config_file=config_file, **kwargs)

    cls_target, reg_target, anchor_states = add_target_ops(
        annotations_tensors,
        anchors,
        num_classes=config.TARGET.NUM_CLASSES,
        use_class_specific_bbox=config.TARGET.CLASS_SPECIFIC_BBOX,
        positive_overlap=config.TARGET.POSITIVE_OVERLAP,
        negative_overlap=config.TARGET.NEGATIVE_OVERLAP
    )

    loss_dict = add_loss_ops(
        cls_output=cls_output,
        reg_output=reg_output,
        cls_target=cls_target,
        reg_target=reg_target,
        anchor_states=anchor_states,
        use_focal_loss=config.LOSS.USE_FOCAL,
        focal_alpha=config.LOSS.FOCAL_ALPHA,
        focal_gamma=config.LOSS.FOCAL_GAMMA,
        reg_weight=config.LOSS.REG_WEIGHT,
        reg_beta=config.LOSS.REG_BETA,
        use_bg_predictor=config.TARGET.BG_PREDICTOR
    )

    return loss_dict, config


def add_retinanet_eval_ops(x, config_file=None, **kwargs):
    (anchors, cls_output, reg_output), config = \
        add_retinanet_ops(x, config_file=None, **kwargs)
