"""
ResNet config system
"""
import logging
from ..utils.config_system import ConfigSystem

logger = logging.getLogger(__name__)

_C = ConfigSystem()
config = _C

valid_resnet_types = {
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
}

resnet_type_to_channel_sizes = {
    'resnet18': [64, 128, 256, 512],
    'resnet34': [64, 128, 256, 512],
    'resnet50': [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
    'resnet152': [256, 512, 1024, 2048]
}


# --------------------------------------------------------------------------- #
# ResNet options
# --------------------------------------------------------------------------- #

# The type of ResNet to use
# For valid ResNet types refer to above
_C.TYPE = 'resnet50'

# The number of classes to classify
_C.NUM_CLASSES = 1000

# If true, uses resnet as a feature extractor, FC layer will not be added
_C.NO_TOP = False

# Last conv layer to output (based on number of pooling)
# eg. 5 will mean the last layer of conv for resnet
_C.LAST_CONV = 5

# Layers below this will be frozen
_C.FREEZE_AT = 0

# Freeze the batch norm layers?
_C.FREEZE_BN = True

# Use group normalization?
_C.USE_GN = False

# The number of groups to use for group normalization
_C.GN_NUM_GROUPS = 32

# --------------------------------------------------------------------------- #
# End of options
# --------------------------------------------------------------------------- #
_C.immutable(True)


def validate_config(config):
    """
    Check validity of configs
    """
    assert isinstance(config.NUM_CLASSES, int) and config.NUM_CLASSES > 0, \
        'NUM_CLASSES must a positive int'

    assert config.TYPE in valid_resnet_types, \
        '{} is invalid backbone type'.format(config.TYPE)

    assert config.LAST_CONV in {2, 3, 4, 5}, \
        'Currently only [2, 3, 4, 5] are accepted values for config.BACKBONE.LAST_CONV'

    assert config.FREEZE_AT in {0, 1, 2, 3, 4, 5}, \
        'FREEZE_AT must be a one of [0, 1, 2, 3, 4, 5]'

    if config.FREEZE_BN and config.USE_GN:
        logger.warn('Normalization layers will not be frozen if using group normalization')

    if not config.NO_TOP:
        assert config.LAST_CONV == 5, \
            'In this version of CaffeBrewer, if fully connected layers are used, LAST_CONV must be 5'

    assert config.FREEZE_BN, \
        'In this version of CaffeBrewer, batch norm layers must be frozen'

    assert not config.USE_GN, \
        'In this version of CaffeBrewer, group norma has not been implemented'


def make_config(config_file=None, **kwargs):
    """ Wrapper around ConfigSystem.make_config """
    return _C.make_config(config_file, validate_config, **kwargs)
