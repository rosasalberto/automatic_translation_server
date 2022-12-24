"""
Module name: logging

This module contains functions for setting up custom loggers and basic loggers.

Functions:
    setup_custom_logger(name:str): This function sets up a custom logger with the specified name.
    get_basic_logger(): This function sets up a basic logger.

"""

import logging
import sys
import time


def setup_custom_logger(name: str):
    """
    Sets up a custom logger with the specified name.

    Parameters:
        name (str): The name of the logger.

    Returns:
        logger: The custom logger.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s     %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.Formatter.converter = time.gmtime
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)
    return logger


def get_basic_logger():
    """
    Sets up a basic logger.
    """
    logging.basicConfig(
        format="%(asctime)s  %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
