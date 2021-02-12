import sys

import tensorflow as tf


def print_one_line(*args):
    tf.print("\033[K", end="")
    tf.print("\r", *args, sep="", end=" ", output_stream=sys.stdout)
