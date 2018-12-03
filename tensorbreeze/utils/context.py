"""
Helper functions for working with tf contexts
"""

import contextlib


@contextlib.contextmanager
def combine_contexts(*args):
    """
    Are you tired of writing nested contexts?

    with tf.Session(config=config) as sess:
        with tf.device('/device/GPU:0'):
            ...

    Just nest one time!

    with combine_contexts(tf.Session, {'config': config}, device, ['/device/GPU:0']) as (sess, _):
        ...

    """
    def is_callable(o):
        return hasattr(o, '__call__')

    assert len(args) > 0, \
        'No contexts provided'

    first_context_fn = args[0]
    first_context_args = []
    first_context_kwargs = {}

    remaining_args = args[1:]
    next_arg = remaining_args[0]

    assert is_callable(first_context_fn), \
        '{} is not callable, expecting a context'

    while len(remaining_args) > 0 and not is_callable(next_arg):
        if isinstance(next_arg, list):
            first_context_args += next_arg
        elif isinstance(next_arg, dict):
            first_context_kwargs.update(next_arg)
        else:
            raise Exception('Expecting a list or a dict, got {}'.format(type(next_arg)))

        next_arg = remaining_args[0]
        remaining_args = remaining_args[1:]

    with first_context_fn(*first_context_args, **first_context_kwargs) as first_context:
        if len(remaining_args) > 0:
            with combine_contexts(*remaining_args) as remaining_contexts:
                yield [first_context] + remaining_contexts
        else:
            yield [first_context]


# @contextlib.contextmanager
# def Session(target='', graph=None, config=None, allow_growth=True):
#     if config is None:
#         config = tf.ConfigProto()
#     if allow_growth:
#         config.gpu_options.allow_growth = True
#     with tf.Session(target=target, graph=graph, config=config) as sess:
#         yield sess


# @contextlib.contextmanager
# def NamedSession(scope_name, target='', graph=None, config=None, allow_growth=True):
#     """ Named session manager """
#     with Session(target, graph, config, allow_growth) as sess:
#         with tf.name_scope(scope_name):
#             yield sess


# @contextlib.contextmanager
# def NamedDeviceScope(scope_name, device_name):
#     with tf.name_scope(scope_name):
#         with tf.device(device_name):
#             yield


# @contextlib.contextmanager
# def DeviceSession(device_name, target='', graph=None, config=None, allow_growth=True):
#     with Session(target, graph, config, allow_growth) as sess:
#         with tf.device(device_name):
#             yield sess


# @contextlib.contextmanager
# def NamedDeviceSession(scope_name, device_name, graph=None, config=None, allow_growth=True):
#     with NamedSession(scope_name, graph, config, allow_growth) as sess:
#         with tf.device(device_name):
#             yield sess
