import tensorflow as tf


def pad2d(
    x,
    paddings=(0),
    mode='CONSTANT',
    name=None,
    constant_values=0
):
    """
    2D padding for 2D convnets
    Wrapper around tf.pad

    Args:
        paddings: Either an int or a tuple
        refer to the rest of the variables in
        https://www.tensorflow.org/api_docs/python/tf/pad
    """
    if not isinstance(paddings, (tuple, list)):
        paddings = (int(paddings),) * 2
    assert len(paddings) == 2

    padding = (
        (0, 0),
        (0, 0),
        (paddings[0],) * 2,
        (paddings[1],) * 2
    )

    return tf.pad(
        x,
        padding,
        mode=mode,
        name=name,
        constant_values=constant_values
    )
