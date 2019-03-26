import tensorflow as tf


def triplet_margin_loss(
    anchor,
    positive,
    negative,
    margin=0.1,
    p=2,
    use_cosine=False,
    swap=False,
    eps=1e-6,
    scope='',
    reduction=tf.losses.Reduction.SUM
):
    """
    Computes the triplet margin loss

    Args:
        anchor: The tensor containing the anchor embeddings
        postiive: The tensor containg the positive embeddings
        negative: The tensor containg the negative embeddings

        The shapes of anchor, positive and negative must all be equal

        margin: The margin in the triplet loss
        p: The norm degree for pairwise distances Options: 1, 2 Default: 2
        use_cosine: Should cosine distance be used?
        swap: Should we swap anchor and positive to get the harder negative?
        eps: A value used to prevent numerical instability
        reduction: The reduction method to use
    """
    assert anchor.shape == positive.shape == negative.shape
    assert p in {1, 2}

    if use_cosine:
        def dist_fn(labels, preds):
            return tf.losses.cosine_distance(
                labels, preds,
                reduction=tf.losses.Reduction.NONE
            )

    elif p == 2:
        def dist_fn(labels, preds):
            return tf.losses.mean_squared_error(
                labels, preds,
                reduction=tf.losses.Reduction.NONE
            )

    elif p == 1:
        def dist_fn(labels, preds):
            return tf.losses.absolute_difference(
                labels, preds,
                reduction=tf.losses.Reduction.NONE
            )

    else:
        raise NotImplementedError()

    with tf.variable_scope(scope):
        pdist = dist_fn(anchor, positive)
        ndist = dist_fn(anchor, negative)

        if swap:
            # ndist_2 is the distance between postive and negative
            ndist_2 = dist_fn(positive, negative)
            ndist = tf.maximum(ndist, ndist_2)

        loss = tf.maximum(pdist - ndist + margin, 0)

        if reduction == tf.losses.Reduction.SUM:
            return tf.sum(loss)
        elif reduction == tf.losses.Reduction.MEAN:
            return tf.reduce_mean(loss)
        elif reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE:
            return tf.sum() / tf.shape(anchor)[0]
        elif reduction == tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS:
            return tf.sum(loss) / tf.sum(tf.greater(loss, 0))
        else:
            msg = '{} has not been implemented for triplet_margin_loss'.format(
                reduction)
            raise NotImplementedError(msg)
