"""
Logging class/functions, inspired/copied from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
"""
import os
import torch
import pathlib
import datetime
import tensorflow as tf

from logzero import logger
from torch.utils.tensorboard import SummaryWriter


class TFTBLogger(object):
    """Tensorflow Tensorboard logger"""
    def __init__(self, config):
        """Create a summary writer logging to log_dir."""
        self.config = config
        self.log_dir = os.path.join(config["log_dir"],
                                    "TB_" + config["run_name"] + f"_{datetime.date.today().strftime('%d_%m_%Y')}")
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if config["clean_log_dir"] and len(os.listdir(self.log_dir)) > 0:  # Clears dir of old event files if True
            for f in os.listdir(self.log_dir):
                try:
                    os.remove(os.path.join(self.log_dir, f))
                    logger.info("Successfully cleaned TB log directory")
                except Exception as e:
                    logger.info(e)
                    logger.info("Continuing run without deleting some files from log directory")
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(name=tag, data=values, step=step, buckets=bins)
            self.writer.flush()

    def close(self):
        self.writer.close()


class PTTBLogger(object):
    def __init__(self, config):
        """Create a summary writer logging to log_dir."""
        self.config = config
        self.log_dir = os.path.join(config["log_dir"],
                                    "TB_" + config["run_name"] + f"_{datetime.date.today().strftime('%d_%m_%Y')}")
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if config["clean_log_dir"] and len(os.listdir(self.log_dir)) > 0:  # Clears dir of old event files if True
            for f in os.listdir(self.log_dir):
                try:
                    os.remove(os.path.join(self.log_dir, f))
                    logger.info(f"Successfully cleaned TB log directory {self.log_dir}")
                except Exception as e:
                    logger.info(e)
                    logger.info(f"Continuing run without deleting some files from log directory {self.log_dir}")
        self.writer = SummaryWriter(log_dir=self.log_dir)  # For some reason I need to import it directly...

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def histo_summary(self, tag, value, step):
        self.writer.add_histogram(tag, value, step)
        self.writer.flush()

    def close(self):
        self.writer.close()
