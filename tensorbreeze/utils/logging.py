from __future__ import absolute_import

import warnings
import logging

default_format = '%(asctime)s %(levelname)-4.4s %(filename)s:%(lineno)d: %(message)s'
warnings.warn('tensorbreeze.utils.logging is depreciated and will be removed soon')


def setup_logger(
    name,
    filename,
    level=logging.INFO,
    format=default_format,
    log_to_stdout=False
):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt=format,
        datefmt='%m-%d %H:%M:%S'
    )

    # File handler
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if log_to_stdout:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
