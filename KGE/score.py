import tensorflow as tf

def p_norm(x, y, params):
    assert params.get("p") is not None, "'p' should be given in score_params when using p_norm"
    return -tf.norm(x - y, ord=params["p"], axis=1)

def dot(x, y, params):
    return tf.reduce_sum(x * y, axis=1)
