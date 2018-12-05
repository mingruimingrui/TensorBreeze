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
    pos_anchors = tf.equal(anchor_states, 1)
    non_neg_anchors = tf.not_equal(anchor_states, -1)
    num_pos_anchors = tf.reduce_sum(tf.cast(pos_anchors, cls_output.dtype))

    # Compute reg loss
    with tf.variable_scope('reg_loss/'):
        reg_loss = tf.losses.huber_loss(
            labels=reg_target,
            predictions=reg_output,
            delta=reg_beta,
            reduction=tf.losses.Reduction.NONE
        )
        reg_loss = tf.reduce_sum(reg_loss, -1)

    zeros = tf.zeros_like(reg_loss)

    with tf.variable_scope('reg_loss/'):
        reg_loss = tf.where(
            pos_anchors,
            reg_loss,
            zeros,
            name='filtered'
        )
        reg_loss = tf.reduce_mean(reg_loss, name='value')

    if use_bg_predictor:
        bg_output = cls_output[..., -1]
        bg_target = tf.where(
            tf.equal(anchor_states, 0),
            tf.ones_like(zeros),
            zeros
        )
        cls_output = cls_output[..., :-1]

        with tf.variable_scope('bg_loss/'):
            bg_loss = layers.sigmoid_focal_loss(
                labels=bg_target,
                logits=bg_output,
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction=tf.losses.Reduction.NONE
            )
            bg_loss = tf.where(
                non_neg_anchors,
                bg_loss,
                zeros
            )
            bg_loss = tf.reduce_sum(bg_loss) / tf.maximum(num_pos_anchors, 10)
            bg_loss = tf.identity(bg_loss, 'value')

        with tf.variable_scope('cls_loss/'):
            cls_loss = layers.sigmoid_focal_loss(
                labels=cls_target,
                logits=cls_output,
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction=tf.losses.Reduction.NONE
            )
            cls_loss = tf.reduce_sum(cls_loss, -1)
            cls_loss = tf.where(
                pos_anchors,
                cls_loss,
                zeros
            )
            cls_loss = tf.reduce_sum(cls_loss) / num_pos_anchors
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
            cls_loss = layers.sigmoid_focal_loss(
                labels=cls_target,
                logits=cls_output,
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction=tf.losses.Reduction.NONE
            )
            cls_loss = tf.reduce_sum(cls_loss, -1)
            cls_loss = tf.where(
                non_neg_anchors,
                cls_loss,
                zeros
            )
            cls_loss = tf.reduce_sum(cls_loss)
            cls_loss = tf.reduce_sum(cls_loss) / tf.maximum(num_pos_anchors, 10)
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
