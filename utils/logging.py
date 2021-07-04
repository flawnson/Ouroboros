"""
Logging class/functions, inspired/copied from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
"""
import os
import wandb
import shutil
import pathlib
import functools
import os.path as osp
from datetime import datetime
import tensorflow as tf
import json

import logzero
from logzero import logger
from torch.utils.tensorboard import SummaryWriter


def allow_logging(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.config["logging"]:
            pass
        else:
            return func(self, *args, **kwargs)
    return wrapper


def clean_log_dir(config, log_dir):
    if config["clean_log_dir"] and os.path.isdir(log_dir) and len(os.listdir(log_dir)) > 0:  # Clears dir of old event files if True
        try:
            shutil.rmtree(log_dir, ignore_errors=True)  # If ignore error left False, raises PermissionError with log.log
            logger.info("Successfully cleaned TB log directory")
        except Exception as e:
            logger.exception(e)
            logger.info("Continuing run without deleting some files from log directory")


class TFTBLogger(object):
    """Tensorflow Tensorboard logger"""
    def __init__(self, config):
        """Create a summary writer logging to log_dir."""
        self.config = config
        if self.config["logging"]:
            run_name = "TB_" + config["run_name"] + f"_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
            self.log_dir = os.path.join(config["log_dir"], "events")
            clean_log_dir(config, self.log_dir)
            pathlib.Path(self.log_dir, run_name).mkdir(parents=True, exist_ok=True)
            self.writer = tf.summary.create_file_writer(os.path.join(self.log_dir, run_name))

            #save the config file
            json.dump(config, open(osp.join(self.log_dir, run_name, "config.json"),
                                   'w',
                                   encoding='utf-8'),
                      ensure_ascii=False, indent=4)

            #create log file for logzero
            logzero.logfile(osp.join(self.log_dir, run_name, "log.log"))

    @allow_logging
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    @allow_logging
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(name=tag, data=values, step=step, buckets=bins)
            self.writer.flush()

    @allow_logging
    def close(self):
        self.writer.close()


class PTTBLogger(object):
    def __init__(self, config):
        """
        Main Logger.
        Create a summary writer logging to log_dir.
        Create a config.json and logzero log file.
        """
        self.config = config
        if self.config["logging"]:
            run_name = "TB_" + config["run_name"] + f"_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
            self.log_dir = os.path.join(config["log_dir"], "events")
            clean_log_dir(config, self.log_dir)
            pathlib.Path(self.log_dir, run_name).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, run_name))  # For some reason I need to import it directly...

            #save the config file
            with open(osp.join(self.log_dir, run_name, "config.json"), 'w', encoding='utf-8') as o:
                json.dump(config, o, ensure_ascii=False, indent=4)

            #create log file for logzero
            logzero.logfile(osp.join(self.log_dir, run_name, "log.log"))

    @allow_logging
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    @allow_logging
    def histo_summary(self, tag, value, step):
        self.writer.add_histogram(tag, value, step)
        self.writer.flush()

    @allow_logging
    def close(self):
        self.writer.close()


class WandBLogger(object):
    def __init__(self, config):
        self.config = config
        self.logger = wandb.init(name=config["wandb_logging"]["run_name"],
                      project=config["wandb_logging"]["project"],
                      entity=config["wandb_logging"]["entity"],
                      config=config) if config["logging"] else None

