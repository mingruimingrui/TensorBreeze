import tensorflow as tf


def add_layer_ops(
    x,
    dim_inner,
    dim_out,
    add_block,
    num_blocks,
    stride=1,
    data_format='channels_first',
    trainable=True,
    freeze_bn=True
):
    """
    Dynamic resnet layer builder that builds bottleneck layers

    Args:
        add_block: An add_bottleneck_ops function,
        num_blocks: Number of bottleneck blocks in this layer
    """
    channel_axis = 1 if data_format == 'channels_first' else -1
    dim_in = x.shape[channel_axis]

    for i in range(num_blocks):
        is_first_block = i == 0

        with tf.variable_scope(str(i)):
            x = add_block(
                x,
                dim_inner=dim_inner,
                dim_out=dim_out,
                stride=stride if is_first_block else 1,
                data_format=data_format,
                trainable=trainable,
                downsample=is_first_block and dim_in != dim_out,
                freeze_bn=freeze_bn
            )

    return x
