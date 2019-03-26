import tensorflow as tf


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
        elif reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE:
            return focal_loss / tf.shape(preds)[0]
        else:
            msg = '{} has not been implemented for focal_loss'.format(
                reduction)
            raise NotImplementedError(msg)
