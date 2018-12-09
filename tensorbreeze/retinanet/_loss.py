import tensorflow as tf
from .. import layers
from ..utils import anchors as utils_anchors


def add_target_ops(
    annotations_batch,
    anchors,
    num_classes,
    use_class_specific_bbox=False,
    positive_overlap=0.5,
    negative_overlap=0.4
):
    """
    Compute the classification and regression targets given a set of anchors
    and annotations
    """
    with tf.variable_scope('target/'):
        cls_batch = []
        reg_batch = []
        states_batch = []

        for annotations in annotations_batch:
            cls_target, bbox_target, anchor_states = utils_anchors.anchor_targets_bbox(
                anchors,
                annotations,
                num_classes=num_classes,
                use_class_specific_bbox=use_class_specific_bbox,
                positive_overlap=positive_overlap,
                negative_overlap=negative_overlap
            )
            reg_target = utils_anchors.bbox_transform(anchors, bbox_target)

            cls_batch.append(cls_target)
            reg_batch.append(reg_target)
            states_batch.append(anchor_states)

        cls_batch = tf.stack(cls_batch, 0)
        reg_batch = tf.stack(reg_batch, 0)
        states_batch = tf.stack(states_batch, 0)

    return cls_batch, reg_batch, states_batch


def add_loss_ops(
    cls_output,
    reg_output,
    cls_target,
    reg_target,
    anchor_states,
    use_focal_loss=True,
    focal_alpha=0.25,
    focal_gamma=2.0,
    reg_weight=1.0,
    reg_beta=0.11,
    use_bg_predictor=False
):
    """
    Compute the losses given classification and regression output and targets
    """
    with tf.variable_scope('detection_loss/'):
        pos_anchors = tf.greater(anchor_states, 0.5)
        non_neg_anchors = tf.greater_equal(anchor_states, 0)
        num_pos_anchors = tf.reduce_sum(tf.cast(pos_anchors, cls_output.dtype))
        num_pos_anchors = tf.maximum(num_pos_anchors, 10)

        # Compute reg loss
        with tf.variable_scope('reg_loss/'):
            reg_output = tf.boolean_mask(reg_output, pos_anchors)
            reg_target = tf.boolean_mask(reg_target, pos_anchors)
            reg_loss = tf.losses.huber_loss(
                labels=reg_target,
                predictions=reg_output,
                delta=reg_beta,
                reduction=tf.losses.Reduction.SUM,
            ) / num_pos_anchors
            reg_loss = tf.identity(reg_loss, name='value')

        if use_bg_predictor:
            with tf.variable_scope('bg_loss/'):
                bg_output = cls_output[..., -1]
                bg_output = tf.boolean_mask(bg_output, non_neg_anchors)
                bg_target = 1 - tf.boolean_mask(anchor_states, non_neg_anchors)
                bg_loss = layers.focal_loss(
                    labels=bg_target,
                    preds=bg_output,
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    reduction=tf.losses.Reduction.SUM
                ) / num_pos_anchors
                bg_loss = tf.identity(bg_loss, 'value')

            with tf.variable_scope('cls_loss/'):
                cls_output = cls_output[..., :-1]
                cls_output = tf.boolean_mask(cls_output, pos_anchors)
                cls_target = tf.boolean_mask(cls_target, pos_anchors)
                cls_loss = layers.focal_loss(
                    labels=cls_target,
                    preds=cls_output,
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    reduction=tf.losses.Reduction.SUM
                ) / num_pos_anchors
                cls_loss = tf.identity(cls_loss, 'value')

            # Compute total loss and form dict
            total_loss = bg_loss + cls_loss + reg_loss * reg_weight
            total_loss = tf.identity(total_loss, 'total_loss/value')
            loss_dict = {
                'bg_loss': bg_loss,
                'cls_loss': cls_loss,
                'reg_loss': reg_loss,
                'total_loss': total_loss
            }

        else:
            with tf.variable_scope('cls_loss/'):
                cls_target = tf.boolean_mask(cls_target, non_neg_anchors)
                cls_output = tf.boolean_mask(cls_output, non_neg_anchors)
                cls_loss = layers.focal_loss(
                    labels=cls_target,
                    preds=cls_output,
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    reduction=tf.losses.Reduction.SUM
                ) / num_pos_anchors
                cls_loss = tf.identity(cls_loss, 'value')

            # Compute total loss and form dict
            total_loss = cls_loss + reg_loss * reg_weight
            total_loss = tf.identity(total_loss, 'total_loss/value')
            loss_dict = {
                'cls_loss': cls_loss,
                'reg_loss': reg_loss,
                'total_loss': total_loss
            }

    return loss_dict
