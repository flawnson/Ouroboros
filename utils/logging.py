"""
Logging class/functions, inspired/copied from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
"""
import os
import pathlib
import tensorflow as tf


class TBLogger(object):

    def __init__(self, config):
        """Create a summary writer logging to log_dir."""
        self.config = config
        self.log_dir = os.path.join(config["log_dir"], config["run_name"])
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if config["clean_log_dir"] and len(os.listdir(self.log_dir)) > 0:  # Clears dir of old event files if True
            for f in os.listdir(self.log_dir):
                os.remove(os.path.join(self.log_dir, f))
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
