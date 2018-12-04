"""
Script to store anchor related functions
that uses tf tensors
"""

from __future__ import division

import numpy as np
import tensorflow as tf


def compute_overlap(bboxes1, bboxes2):
    """
    Parameters
    ----------
    bboxes1: (N, 4) Tensor
    bboxes2: (K, 4) Tensor
    Returns
    -------
    overlaps: (N, K) Tensor of overlap between boxes and query_boxes
    """
    bboxes1_x1 = bboxes1[..., 0:1]
    bboxes1_y1 = bboxes1[..., 1:2]
    bboxes1_x2 = bboxes1[..., 2:3]
    bboxes1_y2 = bboxes1[..., 3:4]

    bboxes2_x1 = bboxes2[..., 0:1]
    bboxes2_y1 = bboxes2[..., 1:2]
    bboxes2_x2 = bboxes2[..., 2:3]
    bboxes2_y2 = bboxes2[..., 3:4]

    inter_x1 = tf.maximum(bboxes1_x1, tf.transpose(bboxes2_x1))
    inter_y1 = tf.maximum(bboxes1_y1, tf.transpose(bboxes2_y1))
    inter_x2 = tf.minimum(bboxes1_x2, tf.transpose(bboxes2_x2))
    inter_y2 = tf.minimum(bboxes1_y2, tf.transpose(bboxes2_y2))

    inter_area = tf.maximum(inter_x2 - inter_x1, 0) * tf.maximum(inter_y2 - inter_y1, 0)

    area1 = (bboxes1_x2 - bboxes1_x1) * (bboxes1_y2 - bboxes1_y1)
    area2 = (bboxes2_x2 - bboxes2_x1) * (bboxes2_y2 - bboxes2_y1)
    area2 = tf.transpose(area2)

    return inter_area / (area1 + area2 - inter_area)


def generate_anchors_at_window(
    base_size=32,
    ratios=[0.5, 1., 2.],
    scales=[2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)],
    dtype=tf.float32,
    name='anchors',
    as_variable=False
):
    """ Generate anchors based on a size a set of ratios and scales
    w.r.t a reference window
    """
    if not isinstance(ratios, np.ndarray):
        ratios = np.array(ratios)
    if not isinstance(scales, np.ndarray):
        scales = np.array(scales)

    num_ratios = len(ratios)
    num_scales = len(scales)
    num_anchors = num_ratios * num_scales
    tiled_scales = np.array(scales.copy().tolist() * num_ratios)
    repeated_ratios = ratios.repeat(num_scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4), dtype=dtype.name)
    anchors[:, 2] = base_size * tiled_scales
    anchors[:, 3] = base_size * tiled_scales

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / repeated_ratios)
    anchors[:, 3] = anchors[:, 2].copy() * repeated_ratios

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] = anchors[:, 0::2].copy() - anchors[:, 2:3].copy() / 2
    anchors[:, 1::2] = anchors[:, 1::2].copy() - anchors[:, 3:4].copy() / 2

    if as_variable:
        return tf.Variable(anchors, name=name, dtype=dtype, trainable=False)
    else:
        return tf.constant(anchors, name=name, dtype=dtype)


def shift_anchors(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size """
    dtype = anchors.dtype

    shape = tf.cast(shape, dtype=dtype)
    shift_x = (tf.range(0, shape[1]) + 0.5) * stride
    shift_y = (tf.range(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1])
    shift_y = tf.reshape(shift_y, [-1])

    shifts = tf.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=-1)

    # add A, {anchors (1, A, 4)} to
    # cell K, {shifts (K, 1, 4)} to get
    # shift anchors (K, A, 4)
    # reshape to (K * A, 4) shifted anchors
    A = tf.reshape(anchors, [1, tf.shape(anchors)[0], 4])
    K = tf.transpose(tf.reshape(shifts, [1, tf.shape(shifts)[0], 4]), [1, 0, 2])
    all_anchors = A + K
    all_anchors = tf.reshape(all_anchors, [-1, 4])

    return all_anchors


def bbox_transform(anchors, gt_boxes, mean=0.0, std=0.2):
    """ Compute bounding-box regression targets for an image """
    anchor_widths = anchors[..., 2] - anchors[..., 0]
    anchor_heights = anchors[..., 3] - anchors[..., 1]

    targets_dx1 = (gt_boxes[..., 0] - anchors[..., 0]) / anchor_widths
    targets_dy1 = (gt_boxes[..., 1] - anchors[..., 1]) / anchor_heights
    targets_dx2 = (gt_boxes[..., 2] - anchors[..., 2]) / anchor_widths
    targets_dy2 = (gt_boxes[..., 3] - anchors[..., 3]) / anchor_heights

    targets = tf.stack([
        targets_dx1,
        targets_dy1,
        targets_dx2,
        targets_dy2
    ], axis=-1)
    targets = (targets - mean) / std

    return targets


def bbox_transform_inv(boxes, deltas, mean=0.0, std=0.2):
    """ Applies deltas (usually regression results) to boxes (usually anchors).
    Before applying the deltas to the boxes, the normalization that was
    previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are
    unnormalized in this function and then applied to the boxes.
    Args
        boxes : tf.Tensor of shape (N, 4), where N the number of boxes and
            4 values for (x1, y1, x2, y2).
        deltas: tf.Tensor of same shape as boxes. These deltas
            (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas
            (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas
            (defaults to [0.2, 0.2, 0.2, 0.2]).
    Returns
        A tf.Tensor of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    width = boxes[..., 2] - boxes[..., 0]
    height = boxes[:, :, 3] - boxes[..., 1]

    x1 = boxes[..., 0] + (deltas[..., 0] * std + mean) * width
    y1 = boxes[..., 1] + (deltas[..., 1] * std + mean) * height
    x2 = boxes[..., 2] + (deltas[..., 2] * std + mean) * width
    y2 = boxes[..., 3] + (deltas[..., 3] * std + mean) * height

    pred_boxes = tf.stack([x1, y1, x2, y2], axis=-1)

    return pred_boxes


def anchor_targets_bbox(
    anchors,
    annotations,
    num_classes,
    mask_shape=None,
    use_class_specific_bbox=False,
    positive_overlap=0.5,
    negative_overlap=0.4,
):
    """ Generate anchor targets for bbox detection.
    This is very pooly implemented, hope to rewrite
    Args
        anchors: tf.Tensor of shape (A, 4) in the (x1, y1, x2, y2) format.
        annotations: tf.Tensor of shape (N, 5) in the
            (x1, y1, x2, y2, label) format.
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used
            to mark the relevant part of the image.
        use_class_specific_bbox: Should each class have it's own bbox?
        negative_overlap: IoU overlap for negative anchors
            (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors
            (all anchors with overlap > positive_overlap are positive).
    Returns
        cls_target: tf.Tensor containing the classification target at
            each anchor position shape will be (A, num_classes)
        bbox_target: tf.Tensor containing the detection bbox at each
            anchor position shape will be (A, 4) if not using class specific
            bbox or (A, 4 * num_classes) if using class specific bbox
        anchor_states: anchor_states: tf.Tensor of shape (N,) containing
            the state of each anchor (-1 for ignore, 0 for bg, 1 for fg).
    """
    dtype = annotations.dtype
    overlaps = compute_overlap(anchors, annotations)

    max_overlaps_inds = tf.argmax(overlaps, axis=1)
    max_overlaps = tf.reduce_max(overlaps, reduction_indices=[1])

    bbox_cls_target = tf.gather(annotations, max_overlaps_inds, axis=0)
    anchor_states = tf.ones_like(bbox_cls_target[..., :1])
    anchor_states_bbox_cls_target = tf.concat([anchor_states, bbox_cls_target], 1)

    negative_bbox_cls_target = tf.zeros_like(bbox_cls_target) - 1

    negative_target = tf.zeros_like(anchor_states)
    ignore_target = negative_target - 1

    negative_target = tf.concat([negative_target, negative_bbox_cls_target], 1)
    ignore_target = tf.concat([ignore_target, negative_bbox_cls_target], 1)

    targets = tf.where(
        max_overlaps >= positive_overlap,
        anchor_states_bbox_cls_target,
        ignore_target
    )

    targets = tf.where(
        max_overlaps <= negative_overlap,
        negative_target,
        targets
    )

    anchor_states = targets[..., 0]
    bbox_target = targets[..., 1:-1]
    cls_target = targets[..., -1]

    anchor_states = tf.identity(anchor_states, name='anchor_states')

    if use_class_specific_bbox:
        raise NotImplementedError()
    else:
        bbox_target = tf.identity(bbox_target, name='bbox_target')

    cls_target = tf.one_hot(
        tf.cast(cls_target, tf.uint8),
        depth=num_classes,
        on_value=1.0,
        off_value=0.0,
        dtype=dtype,
        name='cls_target'
    )

    return cls_target, bbox_target, anchor_states
