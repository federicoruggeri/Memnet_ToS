"""

@Author:

@Date: 25/09/18

"""

import logging
import os
import sys
from logging import FileHandler

import const_define as cd

log_file = cd.NAME_LOG

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


class SimpleLevelFilter(object):
    """
    Simple logging filter
    """

    def __init__(self, level):
        self._level = level

    def filter(self, log_record):
        """
        Filters log message according to filter level

        :param log_record: message to log
        :return: True if message level is less than or equal to filter level
        """

        return log_record.levelno <= self._level


def get_logger(name, log_path=None):
    """
    Returns a logger instance that handles info, debug, warning and error messages.

    :param name: logger name
    :return: logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    if log_path is None:
        log_path = os.path.join(cd.PATH_LOG, cd.NAME_LOG)

    if not os.path.isdir(cd.PATH_LOG):
        os.makedirs(cd.PATH_LOG)

    trf_handler = FileHandler(log_path)
    trf_handler.setLevel(logging.DEBUG)
    trf_handler.setFormatter(formatter)
    logger.addHandler(trf_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(SimpleLevelFilter(logging.WARNING))
    logger.addHandler(stdout_handler)

    return logger
