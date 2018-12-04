"""
Random note
The design of the methods anchor and compute_anchors
are supposed to be low-level and high-level in nature
This is because I wanted to make anchor more easily generalizable
mean-while compute_anchors is most likely only going to be used in
object detection
"""

from __future__ import division

import tensorflow as tf
from ..utils import anchors as utils_anchors


def anchor(
    feature_shape,
    ratios=[0.5, 1., 2.],
    scales=[2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)],
    size=32,
    stride=8
):
    """
    Anchor generator for a single feature level

    Also used to store the reference anchors at each moving window
    """
    anchors_at_window = utils_anchors.generate_anchors_at_window(
        base_size=size,
        ratios=ratios,
        scales=scales
    )
    return utils_anchors.shift_anchors(
        feature_shape,
        stride=stride,
        anchors=anchors_at_window
    )


def compute_anchors(
    features,
    ratios=[0.5, 1., 2.],
    scales_per_octave=3,
    min_feature_level=2,
    size_mult=4.0,
    stride_mult=1.0,
    data_format='channels_first'
):
    """
    Multi level feature generator
    Used to generate anchors for each feature level of a convnet
    """
    assert data_format in {'channels_first', 'channels_last'}

    scales = [2 ** (i / scales_per_octave) for i in range(scales_per_octave)]
    all_anchor_levels = range(min_feature_level, min_feature_level + len(features))

    anchors = dict()
    for level, feature in enumerate(features, min_feature_level):
        size = size_mult * (2 ** level)
        stride = stride_mult * (2 ** level)

        feature_shape = tf.shape(feature)

        if data_format == 'channels_first':
            feature_shape = feature_shape[-2:]
        else:
            feature_shape = feature_shape[1:3]

        anchors[level] = anchor(
            feature_shape,
            ratios=ratios,
            scales=scales,
            size=size,
            stride=stride
        )

    return [anchors[level] for level in all_anchor_levels]
