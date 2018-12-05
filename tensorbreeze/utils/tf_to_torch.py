"""
Helper functions to assist in conversion of TensorBreeze models into pytorch
models
"""
from __future__ import absolute_import

from collections import OrderedDict
import torch

TORCH_DELIMITER = '.'
TF_DELIMITER = '/'
TF_POSTFIX_SEP = ':'

ext_map = {
    'kernel': 'weight',
    'bias': 'bias'
}

bn_ext_map = {
    'gamma': 'weight',
    'beta': 'bias',
    'moving_mean': 'running_mean',
    'moving_variance': 'running_var',
    'num_batches_tracked': 'num_batches_tracked'
}


def split_base_ext(name, delim=TF_DELIMITER):
    """
    Splits a name by a delimitter then splits the returning list into
    basename (as list) and ext
    """
    name = name.split(TF_POSTFIX_SEP)[0]
    name_split = name.split(delim)
    return name_split[:-1], name_split[-1]


def reorder_kernel_weight(tf_weight):
    """ Reorder a tf kernel weight into a torch format """
    len_shape = len(tf_weight.shape)
    transpose_target = list(range(len_shape))
    transpose_target = transpose_target[2:][::-1] + transpose_target[:2]
    return tf_weight.permute(transpose_target)


def check_if_bn(tf_name):
    """ Determine if weight is part of a batch norm layer """
    basename, ext = split_base_ext(tf_name, TF_DELIMITER)
    return ext in bn_ext_map


def check_if_kernel(tf_name):
    """ Determine if weight is a FC or conv kernel """
    basename, ext = split_base_ext(tf_name, TF_DELIMITER)
    return ext == 'kernel'


def convert_name(tf_name, is_bn=False):
    """
    Converts a TensorBreeze variable name into a pytorch name

    Args:
        is_bn: Perform variable name conversion of BN weight
    """
    basename, tf_ext = split_base_ext(tf_name, TF_DELIMITER)

    if is_bn:
        torch_ext = bn_ext_map[tf_ext]
    else:
        if tf_ext in ext_map:
            torch_ext = ext_map[tf_ext]
        else:
            torch_ext = tf_ext

    torch_name = TORCH_DELIMITER.join(basename + [torch_ext])
    return torch_name


def convert_state_dict(tf_state_dict):
    """
    Converts a TensorBreeze `state dict` into a torch state dict
    """
    torch_state_dict = OrderedDict()

    for tf_name, tf_value in tf_state_dict.items():
        is_bn = check_if_bn(tf_name)
        is_kernel = check_if_kernel(tf_name)

        torch_name = convert_name(tf_name, is_bn=is_bn)
        torch_value = torch.Tensor(tf_value)

        if is_kernel:
            torch_value = reorder_kernel_weight(torch_value)

        torch_state_dict[torch_name] = torch_value

    return torch_state_dict
