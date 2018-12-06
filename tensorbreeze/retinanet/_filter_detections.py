import tensorflow as tf
from ..utils import anchors as utils_anchors


def add_nms_ops(
    boxes,
    scores,
    labels,
    apply_nms=True,
    pre_nms_top_n=1000,
    post_nms_top_n=300,
    nms_thresh=0.5,
    score_thresh=0.3
):
    """
    Perform NMS and output the boxes, scores and labels
    """
    labels_is_fixed = isinstance(labels, int)

    # Remove inds with low score
    inds_keep = tf.greater_equal(scores, score_thresh)
    boxes = tf.boolean_mask(boxes, inds_keep)
    scores = tf.boolean_mask(scores, inds_keep)
    if not labels_is_fixed:
        labels = tf.boolean_mask(labels, inds_keep)

    # Sort scores and keep only pre_nms_top_n
    order = tf.contrib.framework.argsort(scores, direction='DESCENDING')
    order = order[:pre_nms_top_n]
    boxes = tf.gather(boxes, order)
    scores = tf.gather(scores, order)
    if not labels_is_fixed:
        labels = tf.gather(labels, order)

    if apply_nms:
        # Apply nms
        inds_keep = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=post_nms_top_n,
            iou_threshold=nms_thresh,
            score_threshold=score_thresh
        )
        boxes = tf.gather(boxes, inds_keep)
        scores = tf.gather(scores, inds_keep)
        if not labels_is_fixed:
            labels = tf.gather(labels, inds_keep)

    if labels_is_fixed:
        labels = tf.ones_like(scores) * labels

    return boxes, scores, labels


def filter_detections(
    anchors,
    cls_output_batch,
    reg_output_batch,
    num_classes,
    batch_size,
    apply_nms=True,
    class_specific_nms=True,
    pre_nms_top_n=1000,
    post_nms_top_n=300,
    nms_thresh=0.5,
    score_thresh=0.3,
    bg_thresh=0.7,
    use_bg_predictor=False
):
    """
    Apply bbox transform inv and filter detections with NMS
    """
    bbox_output_batch = utils_anchors.bbox_transform_inv(
        anchors[None, ...], reg_output_batch
    )

    all_detections = []
    for i in range(batch_size):
        cls_output = cls_output_batch[i]
        reg_output = reg_output_batch[i]
        bbox_output = bbox_output_batch[i]

        if use_bg_predictor:
            inds_keep = tf.less(cls_output[:, -1], bg_thresh)
            cls_output = tf.boolean_mask(cls_output[:, :-1], inds_keep)
            reg_output = tf.boolean_mask(reg_output, inds_keep)

        with tf.device('/cpu:0'):
            detections = {
                'boxes': [],
                'scores': [],
                'labels': []
            }

            if class_specific_nms:
                for c in range(num_classes):
                    filtered_output = add_nms_ops(
                        boxes=bbox_output,
                        scores=cls_output[..., c],
                        labels=c,
                        apply_nms=apply_nms,
                        pre_nms_top_n=pre_nms_top_n,
                        post_nms_top_n=post_nms_top_n,
                        nms_thresh=nms_thresh,
                        score_thresh=score_thresh,
                    )
                    detections['boxes'].append(filtered_output[0])
                    detections['scores'].append(filtered_output[1])
                    detections['labels'].append(filtered_output[2])

                detections['boxes'] = tf.concat(detections['boxes'], axis=0)
                detections['scores'] = tf.concat(detections['scores'], axis=0)
                detections['labels'] = tf.concat(detections['labels'], axis=0)

            else:
                scores = tf.reduce_max(cls_output, axis=-1)
                labels = tf.argmax(cls_output, axis=-1)
                filtered_output = add_nms_ops(
                    boxes=bbox_output,
                    scores=scores,
                    labels=labels,
                    apply_nms=apply_nms,
                    pre_nms_top_n=pre_nms_top_n,
                    post_nms_top_n=post_nms_top_n,
                    nms_thresh=nms_thresh,
                    score_thresh=score_thresh,
                )
                detections['boxes'] = filtered_output[0]
                detections['scores'] = filtered_output[1]
                detections['labels'] = filtered_output[2]

            all_detections.append(detections)

    return all_detections
