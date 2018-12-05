import tensorflow as tf
from .. import layers


def comb_head(
    x,
    num_anchors,
    num_classes,
    num_layers=4,
    feature_size=256,
    use_class_specific_bbox=False,
    use_bg_predictor=False,
    prior_prob=0.01,
    data_format='channels_first',
    reuse=False
):
    """
    """
    # Infer some variables
    batch_size = tf.shape(x)[0]

    total_num_classes = num_classes
    if use_bg_predictor:
        total_num_classes += 1

    total_num_bbox = 4
    if use_class_specific_bbox:
        total_num_bbox *= num_classes

    split_point = num_anchors * num_classes
    if use_bg_predictor:
        bg_split_point = num_anchors * num_classes
        split_point = num_anchors * total_num_classes

    # Apply conv layers
    for i in range(num_layers):
        x = layers.pad2d(x, 1)
        x = tf.layers.conv2d(
            x,
            feature_size,
            kernel_size=3,
            strides=1,
            padding='valid',
            data_format=data_format,
            use_bias=True,
            trainable=True,
            name='head/{}'.format(i * 2),
            reuse=reuse
        )
        x = tf.nn.relu(x)

    x = layers.pad2d(x, 1)
    x = tf.layers.conv2d(
        x,
        num_anchors * (total_num_classes + total_num_bbox),
        kernel_size=3,
        strides=1,
        padding='valid',
        data_format=data_format,
        use_bias=True,
        trainable=True,
        name='head/{}'.format(num_layers * 2),
        reuse=reuse
    )

    # Perform some surgery to extract the classification and regression
    # outputs (and background output)
    if use_bg_predictor:
        classification = x[:, :bg_split_point]
        background = x[:, bg_split_point:split_point]
    else:
        classification = x[:, :split_point]

    regression = x[:, split_point:]

    if data_format == 'channels_first':
        classification = layers.to_nhwc(classification)
        regression = layers.to_nhwc(regression)
        if use_bg_predictor:
            background = layers.to_nhwc(background)

    classification = tf.reshape(classification, [batch_size, -1, num_classes])
    regression = tf.reshape(regression, [batch_size, -1, total_num_bbox])
    if use_bg_predictor:
        background = tf.reshape(background, [batch_size, -1, 1])
        classification = tf.concat([classification, background], axis=-1)

    return classification, regression


def add_comb_head_ops(
    features,
    num_anchors,
    num_classes,
    num_layers=4,
    feature_size=256,
    use_class_specific_bbox=False,
    use_bg_predictor=False,
    prior_prob=0.01,
    data_format='channels_first'
):
    """
    """
    cls_output = []
    reg_output = []
    reuse = False

    for feature in features:
        classification, regression = comb_head(
            feature,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_layers=num_layers,
            feature_size=feature_size,
            use_class_specific_bbox=use_class_specific_bbox,
            use_bg_predictor=use_bg_predictor,
            prior_prob=0.01,
            data_format=data_format,
            reuse=reuse
        )
        cls_output.append(classification)
        reg_output.append(regression)
        reuse = True

    cls_output = tf.concat(cls_output, axis=1, name='cls_output')
    reg_output = tf.concat(reg_output, axis=1, name='reg_output')

    return cls_output, reg_output


def cls_head(
    x,
    num_anchors,
    num_classes,
    num_layers=4,
    feature_size=256,
    use_bg_predictor=False,
    prior_prob=0.01,
    data_format='channels_first',
    reuse=False
):
    """
    """
    # Infer some variables
    batch_size = tf.shape(x)[0]

    total_num_classes = num_classes
    if use_bg_predictor:
        total_num_classes += 1

    if use_bg_predictor:
        bg_split_point = num_anchors * num_classes

    # Apply conv layers
    for i in range(num_layers):
        x = layers.pad2d(x, 1)
        x = tf.layers.conv2d(
            x,
            feature_size,
            kernel_size=3,
            strides=1,
            padding='valid',
            data_format=data_format,
            use_bias=True,
            trainable=True,
            name='head/{}'.format(i * 2),
            reuse=reuse
        )
        x = tf.nn.relu(x)

    x = layers.pad2d(x, 1)
    x = tf.layers.conv2d(
        x,
        num_anchors * (total_num_classes),
        kernel_size=3,
        strides=1,
        padding='valid',
        data_format=data_format,
        use_bias=True,
        trainable=True,
        name='head/{}'.format(num_layers * 2),
        reuse=reuse
    )

    # Perform some surgery to extract the classification outputs
    # (and background output)
    if use_bg_predictor:
        classification = x[:, :bg_split_point]
        background = x[:, bg_split_point:]
    else:
        classification = x

    if data_format == 'channels_first':
        classification = layers.to_nhwc(classification)
        if use_bg_predictor:
            background = layers.to_nhwc(background)

    classification = tf.reshape(classification, [batch_size, -1, num_classes])
    if use_bg_predictor:
        background = tf.reshape(background, [batch_size, -1, 1])
        classification = tf.concat([classification, background], axis=-1)

    return classification


def add_cls_head_ops(
    features,
    num_anchors,
    num_classes,
    num_layers=4,
    feature_size=256,
    use_bg_predictor=False,
    prior_prob=0.01,
    data_format='channels_first'
):
    """
    """
    cls_output = []
    reuse = False

    for feature in features:
        classification = cls_head(
            feature,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_layers=num_layers,
            feature_size=feature_size,
            use_bg_predictor=use_bg_predictor,
            prior_prob=0.01,
            data_format=data_format,
            reuse=reuse
        )
        cls_output.append(classification)
        reuse = True

    cls_output = tf.concat(cls_output, axis=1, name='cls_output')

    return cls_output


def reg_head(
    x,
    num_anchors,
    num_classes,
    num_layers=4,
    feature_size=256,
    use_class_specific_bbox=False,
    prior_prob=0.01,
    data_format='channels_first',
    reuse=False
):
    """
    """
    # Infer some variables
    batch_size = tf.shape(x)[0]

    total_num_bbox = 4
    if use_class_specific_bbox:
        total_num_bbox *= num_classes

    # Apply conv layers
    for i in range(num_layers):
        x = layers.pad2d(x, 1)
        x = tf.layers.conv2d(
            x,
            feature_size,
            kernel_size=3,
            strides=1,
            padding='valid',
            data_format=data_format,
            use_bias=True,
            trainable=True,
            name='head/{}'.format(i * 2),
            reuse=reuse
        )
        x = tf.nn.relu(x)

    x = layers.pad2d(x, 1)
    x = tf.layers.conv2d(
        x,
        num_anchors * (total_num_bbox),
        kernel_size=3,
        strides=1,
        padding='valid',
        data_format=data_format,
        use_bias=True,
        trainable=True,
        name='head/{}'.format(num_layers * 2),
        reuse=reuse
    )

    # Perform some surgery to extract the regression outputs
    regression = x

    if data_format == 'channels_first':
        regression = layers.to_nhwc(regression)

    regression = tf.reshape(regression, [batch_size, -1, total_num_bbox])

    return regression


def add_reg_head_ops(
    features,
    num_anchors,
    num_classes,
    num_layers=4,
    feature_size=256,
    use_class_specific_bbox=False,
    prior_prob=0.01,
    data_format='channels_first'
):
    """
    """
    reg_output = []
    reuse = False

    for feature in features:
        regression = reg_head(
            feature,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_layers=num_layers,
            feature_size=feature_size,
            use_class_specific_bbox=use_class_specific_bbox,
            prior_prob=0.01,
            data_format=data_format,
            reuse=reuse
        )
        reg_output.append(regression)
        reuse = True

    reg_output = tf.concat(reg_output, axis=1, name='reg_output')

    return reg_output
