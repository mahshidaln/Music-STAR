import os
import sys
import time
import numpy
import shutil
import logging
import matplotlib
from pathlib import Path
from scipy.io import wavfile
from datetime import timedelta
from matplotlib import pyplot as plt


def setup_logger(logger_name, filename):
    """Logger for data preprocessing"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def train_logger(opt, path: Path):
    """Logger for training"""

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if hasattr(opt, 'rank'):
        filepath = path / f'main_{opt.rank}.log'
    else:
        filepath = path / 'main.log'

    if hasattr(opt, 'rank') and opt.rank != 0:
        sys.stdout = open(path / f'stdout_{opt.rank}.log', 'w')
        sys.stderr = open(path / f'stderr_{opt.rank}.log', 'w')

    # Safety check
    if filepath.exists() and not opt.checkpoint:
        logging.warning("Experiment already exists!")

    class TrainLogFormatter:
        def __init__(self):
            self.start_time = time.time()

        def format(self, record):
            elapsed_seconds = round(record.created - self.start_time)

            prefix = "%s - %s - %s" % (
                time.strftime('%x %X'),
                timedelta(seconds=elapsed_seconds),
                record.levelname
            )
            message = record.getMessage()
            message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
            return "%s - %s" % (prefix, message)

    # Create log formatter
    formatter = TrainLogFormatter()

    # Create logger and set level to debug
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # create console handler and set level to info
    if hasattr(opt, 'rank') and opt.rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
     
    # reset logger elapsed time
    formatter.start_time = time.time()
    logger.info(opt)
    return logger


def mu_law(x, mu=255):
    x = numpy.clip(x, -1, 1)
    x_mu = numpy.sign(x) * numpy.log(1 + mu*numpy.abs(x))/numpy.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16') #map to [0,255]


def inv_mu_law(x, mu=255.0):
    x = numpy.array(x).astype(numpy.float32)
    y = 2. * (x - (mu+1.)/2.) / (mu+1.) #map to [-1,1]
    return numpy.sign(y) * (1./mu) * ((1. + mu)**numpy.abs(y) - 1.)


def copy_files(files, from_path, to_path: Path):
    for f in files:
        out_file_path = to_path / f.relative_to(from_path)
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, out_file_path)


def save_audio(x, path, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, x)


def save_wav_image(wav, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 5))
    plt.plot(wav)
    plt.savefig(path)