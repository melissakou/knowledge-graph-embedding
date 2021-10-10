import numpy as np
import tensorflow as tf


def Lp_distance(x, y, params={}):
    assert params.get("p") is not None, "'p' should be given in score_params when using Lp_distance"
    p = params["p"]
    if p == np.inf:
        return -tf.reduce_max(tf.abs(x - y), axis=-1)
    else:
        return -tf.pow(tf.clip_by_value(tf.reduce_sum(tf.pow(tf.abs(x - y), p), axis=-1), 1e-9, np.inf), 1 / p)
        
def Lp_distance_pow(x, y, params={}):
    assert params.get("p") is not None, "'p' should be given in score_params when using Lp_distance_pow"
    p = params["p"]

    return tf.pow(Lp_distance(x, y, params={"p": p}), 2)

def dot(x, y, params={}):
    return tf.reduce_sum(x * y, axis=-1)