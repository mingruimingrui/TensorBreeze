"""
A collection of random but useful misc functions which does not really belong
anywhere
"""
from __future__ import absolute_import

import os
import yaml
from collections import Mapping


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def deep_cast_dict(obj):
    if isinstance(obj, Mapping):
        return {k: deep_cast_dict(v) for k, v in obj.items()}
    else:
        return obj


def prettify(obj):
    """ Transfroms a jsonfiable object into pretty string """
    return yaml.safe_dump(deep_cast_dict(obj), default_flow_style=False)
