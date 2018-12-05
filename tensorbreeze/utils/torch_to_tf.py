"""
Helper functions to assist in conversion of pytorch models into TensorBreeze
models
"""
from __future__ import absolute_import

from collections import OrderedDict
import tensorflow as tf

TORCH_DELIMITER = '.'
TF_DELIMITER = '/'
TF_POSTFIX = ':0'

ext_map = {
    'weight': 'kernel',
    'bias': 'bias'
}

bn_ext_map = {
    'weight': 'gamma',
    'bias': 'beta',
    'running_mean': 'moving_mean',
    'running_var': 'moving_variance',
    'num_batches_tracked': 'num_batches_tracked'
}


def split_base_ext(name, delim=TORCH_DELIMITER):
    """
    Splits a name by a delimitter then splits the returning list into
    basename (as list) and ext
    """
    name_split = name.split(delim)
    return name_split[:-1], name_split[-1]


def reorder_kernel_weight(torch_weight):
    """ Reorder a torch kernel weight into a tf format """
    len_shape = len(torch_weight.shape)
    transpose_target = list(range(len_shape))
    transpose_target = transpose_target[2:] + transpose_target[:2][::-1]
    return torch_weight.transpose(transpose_target)


def check_if_bn(torch_name, torch_state_dict):
    """ Determine if weight is part of a batch norm layer """
    basename, ext = split_base_ext(torch_name, TORCH_DELIMITER)
    equiv_moving_mean = TORCH_DELIMITER.join(basename + ['running_mean'])
    is_bn = equiv_moving_mean in torch_state_dict
    return is_bn


def check_if_kernel(torch_name, torch_state_dict):
    """ Determine if weight is a FC or conv kernel """
    basename, ext = split_base_ext(torch_name, TORCH_DELIMITER)
    return ext == 'weight'


def convert_name(torch_name, scoped=True, is_bn=False):
    """
    Converts a torch variable name into a TensorBreeze name

    Args:
        is_bn: Perform variable name conversion of BN weight
    """
    basename, torch_ext = split_base_ext(torch_name, TORCH_DELIMITER)

    if is_bn:
        tf_ext = bn_ext_map[torch_ext]
    else:
        if torch_ext in ext_map:
            tf_ext = ext_map[torch_ext]
        else:
            tf_ext = torch_ext

    tf_name = TF_DELIMITER.join(basename + [tf_ext]) + TF_POSTFIX

    if scoped:
        scope_name = tf.get_variable_scope().name
        if scope_name != '':
            return '{}{}{}'.format(scope_name, TF_DELIMITER, tf_name)

    return tf_name


def convert_state_dict(torch_state_dict, scoped=True):
    """
    Converts a torch state dict into a TensorBreeze `state dict`
    """
    tf_state_dict = OrderedDict()

    for torch_name, torch_value in torch_state_dict.items():
        is_bn = check_if_bn(torch_name, torch_state_dict)
        is_kernel = check_if_kernel(torch_name, torch_state_dict)

        tf_name = convert_name(torch_name, scoped, is_bn=is_bn)
        tf_value = torch_value.cpu().data.numpy()

        if is_kernel:
            tf_value = reorder_kernel_weight(tf_value)

        tf_state_dict[tf_name] = tf_value

    return tf_state_dict
