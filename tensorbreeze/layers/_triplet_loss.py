import tensorflow as tf
import numpy as np


def add_fixed_semi_random_triplet_loss(
    embedding,
    cls_per_batch,
    img_per_cls,
    dist_type='euclidean',
    margin=1.0,
    # swap=False,
    k1=1,
    k2=1,
    scope='',
    reduction=tf.losses.Reduction.SUM
):
    """
    Fixed semi random triplet loss which can be used to select
    triplets given an embedding tensor. Requires that the class per minibatch
    and image per class are constants

    Able to intelligently select the hardest negative or randomly sample
    from the top negatives

    Args:
        embedding: A 2D embedding tensor of the shape
            (batch_size, embedding_size). Do note that `batch_size` should be
            cls_per_batch * img_per_cls
        cls_per_batch: An integer representing the number of unique classes in
            `embedding`
        img_per_cls: An integer representing the number of unique images for
            each class in `embedding`
        dist_type: Either one of 'cosine' or 'euclidean'
        margin: The margin to be used in margin triplet loss
        # swap: Should the anchor and positive sample be swapped to find the
        #     hardest negative?
        k1: Hardest positive will be sampled from the top k1 samples
            (-1 for all)
        k2: Hardest negative will be sampled from the top k2 samples
            (-1 for all)
        reduction: A tf.losses.Reduction value
    """
    def gather_dists(pairwise_dists, row_ids, col_ids):
        return tf.gather(
            params=tf.gather(
                params=pairwise_dists,
                indices=row_ids,
                axis=0
            ),
            indices=col_ids,
            axis=1
        )

    with tf.variable_scope(scope):
        # Ensure that variables are valid
        mb_size = embedding.shape[0]
        assert len(embedding.shape) == 2
        assert mb_size == cls_per_batch * img_per_cls
        assert dist_type in {'cosine', 'euclidean'}

        # Constraint k1 and k2
        max_k1 = img_per_cls - 1
        max_k2 = mb_size - img_per_cls
        k1 = max_k1 if k1 == -1 else min(k1, max_k1)
        k2 = max_k2 if k2 == -1 else min(k2, max_k2)

        # Compute all pairwise distances
        if dist_type == 'euclidean':
            row_norm = tf.reduce_sum(tf.pow(embedding, 2), axis=1)
            pairwise_dists = tf.matmul(embedding, tf.transpose(embedding))
            pairwise_dists *= -2
            pairwise_dists += row_norm[:, tf.newaxis]
            pairwise_dists += row_norm[tf.newaxis, :]

        else:
            embedding /= tf.sqrt(
                tf.reduce_sum(
                    tf.pow(embedding, 2),
                    axis=1
                )
            )[:, tf.newaxis]

            pairwise_dists = 1 - tf.matmul(embedding, tf.transpose(embedding))

        print(embedding, pairwise_dists)

        # Gather pos and neg sample for each anchor
        all_pos_dist = []
        all_neg_dist = []

        for i in range(cls_per_batch):
            # Gather pos and neg dists for each sample of class
            pos_ids_start = i * img_per_cls
            pos_ids_end = (i + 1) * img_per_cls

            pos_ids = list(range(pos_ids_start, pos_ids_end))
            neg_ids = list(range(0, pos_ids_start)) + \
                list(range(pos_ids_end, mb_size))

            pos_dists = gather_dists(pairwise_dists, pos_ids, pos_ids)
            neg_dists = gather_dists(pairwise_dists, pos_ids, neg_ids)

            # Sort and retrieve top k
            # top_k_pos_dists will be (img_per_cls, k1) shaped
            # top_k_neg_dists will be (img_per_cls, k2) shaped
            # top_k_pos_dists, _ = tf.math.top_k(pos_dists, k1, sorted=True)
            # top_k_neg_dists, _ = tf.math.top_k(neg_dists, k2, sorted=False)

            top_k_pos_dists = tf.contrib.framework.sort(
                pos_dists,
                axis=-1,
                direction='DESCENDING'
            )[:, :k1]
            top_k_neg_dists = tf.contrib.framework.sort(
                neg_dists,
                axis=-1,
                direction='ASCENDING'
            )[:, :k2]

            # Pre-determine which of the top k to use for each entry
            pos_choice = np.random.randint(0, k1, size=img_per_cls)
            neg_choice = np.random.randint(0, k2, size=img_per_cls)

            pos_dist = tf.gather_nd(
                top_k_pos_dists,
                list(zip(range(img_per_cls), pos_choice))
            )
            neg_dist = tf.gather_nd(
                top_k_neg_dists,
                list(zip(range(img_per_cls), neg_choice))
            )

            all_pos_dist.append(pos_dist)
            all_neg_dist.append(neg_dist)

        all_pos_dist = tf.concat(all_pos_dist, axis=0)
        all_neg_dist = tf.concat(all_neg_dist, axis=0)

        # Compute triplet loss
        triplet_loss = all_pos_dist - all_neg_dist + margin
        triplet_loss = tf.maximum(triplet_loss, 0)

        if reduction == tf.losses.Reduction.SUM:
            return tf.reduce_sum(triplet_loss)
        elif reduction == tf.losses.Reduction.NONE:
            return triplet_loss
        elif reduction == tf.losses.Reduction.MEAN:
            return tf.reduce_mean(triplet_loss)
        else:
            raise NotImplementedError()
