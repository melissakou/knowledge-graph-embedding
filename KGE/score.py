import numpy as np
import tensorflow as tf


def p_norm(x, y, params={}):
    assert params.get("p") is not None, "'p' should be given in score_params when using p_norm"
    p = params["p"]
    if p == np.inf:
        return -tf.reduce_max(tf.abs(x - y), axis=-1)
    else:
        return -tf.pow(tf.clip_by_value(tf.reduce_sum(tf.pow(tf.abs(x - y), p), axis=-1), 1e-9, np.inf), 1 / p)
        
def squared_euclidean(x, y, params={}):
    return -tf.reduce_sum(tf.pow(x - y, 2), axis=-1)


def dot(x, y, params={}):
    return tf.reduce_sum(x * y, axis=-1)