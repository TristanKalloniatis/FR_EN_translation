from datetime import datetime
import logging
from sys import stdout
import os

if not os.path.exists('log_files/'):
    os.mkdir('log_files/')

def write_log(message, log_object):
    timestamp = datetime.now()
    log_object.info('[{0}]: {1}'.format(timestamp, message))
    return


def create_logger(log_filename):
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, stream=stdout)
    logger.addHandler(logging.FileHandler('log_files/log_{0}_{1}.txt'.format(log_filename, datetime.now())))
    return logger
