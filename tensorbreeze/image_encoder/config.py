"""
ImageEncoder config system
"""
import logging
from ..utils.config_system import ConfigSystem

from ..resnet.config import config as backbone_config
from ..resnet.config import validate_config as validate_backbone_config

logger = logging.getLogger(__name__)

_C = ConfigSystem()
config = _C


# --------------------------------------------------------------------------- #
# Backbone options
# --------------------------------------------------------------------------- #
_C.BACKBONE = backbone_config.clone()
_C.BACKBONE.immutable(False)
# Refer to kindler.backbone.config for full list of BACKBONE

# --------------------------------------------------------------------------- #
# Target options
# --------------------------------------------------------------------------- #
_C.TARGET = ConfigSystem()
# Target options are here to control how model outputs are to be produced

# The output feature size of the encoder network
_C.TARGET.FEATURE_SIZE = 256

# Pooling method to use. Choice of ["avg", "max"]
_C.TARGET.POOL_METHOD = 'avg'

# --------------------------------------------------------------------------- #
# End of options
# --------------------------------------------------------------------------- #
_C.immutable(True)


def validate_config(config):
    """
    Check validity of configs
    """
    config.BACKBONE.update({'NO_TOP': True})

    # Validate backbone config
    validate_backbone_config(config.BACKBONE)

    assert config.TARGET.FEATURE_SIZE >= 1, \
        'TARGET.FEATURE_SIZE has to be a positive integer'

    assert config.TARGET.POOL_METHOD in ['avg', 'max'], \
        'TARGET.POOL_METHOD has to be in ["avg", "max"]'


def make_config(config_file=None, **kwargs):
    """ Wrapper around ConfigSystem.make_config """
    return _C.make_config(config_file, validate_config, **kwargs)
