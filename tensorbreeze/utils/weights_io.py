import tensorflow as tf
import logging
from six.moves import cPickle as pickle

logger = logging.getLogger(__name__)

PKL_PROTOCOL = 2


def save_weights_to_state_dict(variables=None, scope=None):
    """ Save all weights to a state dict """
    if variables is None:
        variables = tf.global_variables(scope=scope)
    state_dict = {v.name: v for v in variables}
    return state_dict


def load_weights_from_state_dict(state_dict, sess=None):
    """
    Load weights from state dict into current session
    Tensors should already been built
    """
    if sess is None:
        sess = tf.get_default_session()

    # Identify existing tensors
    existing_variables = tf.global_variables()
    existing_variables = {v.name: v for v in existing_variables}

    # Assign appropriate weights to tensors
    none_loaded = True
    for name, value in state_dict.items():
        if name in existing_variables:
            none_loaded = False
            assign_op = existing_variables[name].assign(value)
            sess.run(assign_op)

    if none_loaded:
        logger.warning('No weights were loaded, check if the variables has been created')


def save_state_dict_to_file(state_dict, filename):
    """ Save state dict to a file """
    with open(filename, 'wb') as f:
        pickle.dump(state_dict, f, protocol=PKL_PROTOCOL)


def load_state_dict_from_file(filename):
    with open(filename, 'rb') as f:
        state_dict = pickle.load(f)
    return state_dict


def save_weights_to_file(filename, variables=None, scope=None):
    """ Save weights (as state_dict) to a file """
    state_dict = save_weights_to_state_dict(variables, scope)
    return save_state_dict_to_file(state_dict, filename)


def load_weights_from_file(filename, sess=None):
    """ Load weights from a file """
    state_dict = load_state_dict_from_file(filename)
    return load_weights_from_state_dict(state_dict, sess)