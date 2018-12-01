"""
Helper functions for working with tf contexts
"""

import contextlib
import tensorflow as tf


@contextlib.contextmanager
def Session(target='', graph=None, config=None, allow_growth=True):
    if config is None:
        config = tf.ConfigProto()
    if allow_growth:
        config.gpu_options.allow_growth = True
    with tf.Session(target=target, graph=graph, config=config) as sess:
        yield sess


@contextlib.contextmanager
def NamedSession(scope_name, target='', graph=None, config=None, allow_growth=True):
    """ Named session manager """
    with Session(target, graph, config, allow_growth) as sess:
        with tf.name_scope(scope_name):
            yield sess


@contextlib.contextmanager
def NamedDeviceScope(scope_name, device_name):
    with tf.name_scope(scope_name):
        with tf.device(device_name):
            yield


@contextlib.contextmanager
def DeviceSession(device_name, target='', graph=None, config=None, allow_growth=True):
    with Session(target, graph, config, allow_growth) as sess:
        with tf.device(device_name):
            yield sess


@contextlib.contextmanager
def NamedDeviceSession(scope_name, device_name, graph=None, config=None, allow_growth=True):
    with NamedSession(scope_name, graph, config, allow_growth) as sess:
        with tf.device(device_name):
            yield sess
