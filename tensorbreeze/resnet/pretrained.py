import torchvision
import tensorflow as tf

from .config import valid_resnet_types
from ..utils.torch_to_tf import convert_state_dict
from ..utils.weights_io import load_weights_from_state_dict


def load_pretrained_weights(resnet_type, sess=None, verbosity=0):
    """
    Load ResNet weights from a pretrained torchvision model and insert said
    weights into associated tensors in session

    Ensure that parameters have already been initialized at this point

    Args:
        sess: Session containing graph where ResNet tensors and ops resides
        resnet_types: The type of Resnet
        verbosity: Level of logging to use
            0 - Only Errors
            1 - Start and end logged
            2 - Skipped blobs logged
    """
    assert resnet_type in valid_resnet_types, \
        '{} resnet_type is invalid'.foramt(resnet_type)

    if sess is None:
        sess = tf.get_default_session()

    if verbosity >= 1:
        print('Loading pretrained weights for {}'.format(resnet_type))

    # Load a torch model and extract its state_dict
    torch_model_fn = getattr(torchvision.models, resnet_type)
    torch_model = torch_model_fn(pretrained=True)
    torch_state_dict = torch_model.state_dict()
    tf_state_dict = convert_state_dict(torch_state_dict)

    # Load weights into sess
    load_weights_from_state_dict(tf_state_dict, sess)
