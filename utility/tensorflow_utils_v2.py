"""

@Authors:

@Date: 27/09/2019

"""

import numpy as np
import tensorflow as tf


def add_gradient_noise(t, stddev=1e-3, name="add_gradient_noise"):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """

    with tf.name_scope(name) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random.normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    Positional encoding adopted in the Transformer implementation
    as described in https://www.tensorflow.org/tutorials/text/transformer
    """

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
