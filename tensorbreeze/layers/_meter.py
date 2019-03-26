from collections import Mapping
import tensorflow as tf


def meter(
    loss,
    decay=0.01,
    name=None,
    init_value=0
):
    """ Meter for recording loss """
    with tf.variable_scope('meter'):
        x = tf.Variable(
            initial_value=init_value,
            trainable=False,
            validate_shape=False,
            dtype=tf.float32
        )
        sess = tf.get_default_session()
        sess.run(x.initializer)

        y = x * (1 - decay) + decay * loss
        y = tf.assign(x, y, validate_shape=False, name=name)

    return y, y.op


def meter_dict(
    loss_dict,
    decay=0.01,
    names=None,
    init_values=0
):
    """ Meter for recording loss dict """
    meter_dict = {}
    meter_dict_ops = {}

    for key, loss in loss_dict.items():
        if isinstance(init_values, Mapping):
            init_value = init_values[key]
        else:
            init_value = init_values

        if isinstance(names, Mapping):
            name = names[key]
        else:
            name = None

        meter_dict[key], meter_dict_ops[key] = add_meter_ops(
            loss,
            decay=decay,
            name=name,
            init_value=init_value
        )

    return meter_dict, meter_dict_ops
