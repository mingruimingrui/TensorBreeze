import tensorflow as tf


# def sigmoid_focal_loss(
#     labels,
#     logits,
#     alpha,
#     gamma,
#     scope='',
#     reduction=tf.losses.Reduction.SUM
# ):
#     eps = 1e-5
#     with tf.variable_scope(scope):
#         predictions = tf.nn.sigmoid(logits)
#         predictions = tf.clip_by_value(predictions, eps, 1.0 - eps)
#         zeros = tf.zeros_like(predictions, dtype=predictions.dtype)

#         # For poitive prediction, only need consider front part loss, back part is 0;
#         # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
#         pos_p_sub = tf.where(labels > zeros, labels - predictions, zeros)

#         # For negative prediction, only need consider back part loss, front part is 0;
#         # target_tensor > zeros <=> z=1, so negative coefficient = 0.
#         neg_p_sub = tf.where(labels > zeros, zeros, predictions)

#         per_entry_cross_ent = \
#             (alpha - 1) * (neg_p_sub ** gamma) * tf.log(1.0 - predictions) - \
#             alpha * (pos_p_sub ** gamma) * tf.log(predictions)

#         if reduction == tf.losses.Reduction.SUM:
#             return tf.reduce_sum(per_entry_cross_ent)
#         elif reduction == tf.losses.Reduction.NONE:
#             return per_entry_cross_ent
#         else:
#             raise NotImplementedError()


def focal_loss(
    labels,
    preds,
    alpha=0.25,
    gamma=2.0,
    scope='',
    reduction=tf.losses.Reduction.SUM
):
    """
    FL(p_t) = - alpha * log(p_t) * ((1 - p_t) ** gamma)
    FL(p_t) = alpha * BCE(p_t) * focal_weight
    where
    focal_weight = (1 - p_t) ** gamma
    """
    eps = 1e-5
    with tf.variable_scope(scope):
        preds = tf.clip_by_value(preds, eps, 1.0 - eps)
        p_t = tf.where(
            tf.greater(labels, 0.5),
            preds,
            1 - preds
        )

        focal_weight = (1 - p_t) ** gamma
        bce_loss = -tf.log(p_t)

        focal_loss = alpha * bce_loss * focal_weight

        if reduction == tf.losses.Reduction.SUM:
            return tf.reduce_sum(focal_loss)
        elif reduction == tf.losses.Reduction.NONE:
            return focal_loss
        else:
            raise NotImplementedError()
