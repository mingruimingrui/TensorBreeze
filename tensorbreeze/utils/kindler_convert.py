from collections import OrderedDict

import torch

from .torch_to_tf import convert_state_dict as convert_from_torch_state_dict
from .weights_io import load_weights_from_state_dict


def convert_from_kindler_state_dict(kindler_state_dict):
    """ Converts a Kindler state dict into Tensorbreeze """
    tf_state_dict = convert_from_torch_state_dict(kindler_state_dict)
    tb_state_dict = OrderedDict()
    for weight_name, value in tf_state_dict.items():

        if 'layers/0/0' in weight_name:
            weight_name = weight_name.replace('layers/0/0', 'conv1')

        if 'layers/0/1' in weight_name:
            weight_name = weight_name.replace('layers/0/1', 'bn1')

        if 'layers/1/' in weight_name:
            weight_name = weight_name.replace('layers/1/', 'layer')

        if 'layers/' in weight_name:
            weight_name = weight_name.replace('layers/', 'layer')

        tb_state_dict[weight_name] = value

    return tb_state_dict


def load_weights_from_kindler_state_dict(kindler_state_dict, sess=None):
    tb_state_dict = convert_from_kindler_state_dict(kindler_state_dict)
    load_weights_from_state_dict(tb_state_dict, sess=sess)


def load_weights_from_kindler_file(kindler_file, sess=None):
    kindler_file_contents = torch.load(kindler_file)

    # Retrieve kindler_state_dict from file
    if isinstance(kindler_file_contents, OrderedDict):
        kindler_state_dict = kindler_file_contents

    elif 'model' in kindler_file_contents:
        kindler_state_dict = kindler_file_contents['model']

    elif hasattr(kindler_file_contents, 'state_dict'):
        kindler_state_dict = kindler_file_contents.cpu().state_dict()

    else:
        raise Exception(
            'Unable to load state dict from file automatically. '
            'Suggested to manually find state dict from file'
        )

    load_weights_from_kindler_state_dict(kindler_state_dict, sess=sess)
