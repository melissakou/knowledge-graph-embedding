import numpy as np
import tensorflow as tf

def pairwise_hinge_loss(pos_score, neg_score, params):
    assert params.get("margin") is not None, "'margin' should be given in loss_params when using pairwise_loss"
    pos_score = tf.repeat(pos_score, int(neg_score.shape[0] / pos_score.shape[0]))
    return tf.reduce_sum(tf.clip_by_value(params["margin"] + neg_score - pos_score, 0, np.inf))

def binary_cross_entropy_loss(pos_score, neg_score, params):
    pos_ll = tf.reduce_sum(tf.math.log(tf.sigmoid(pos_score)))
    neg_ll = tf.reduce_sum(tf.math.log(tf.sigmoid(-neg_score)))

    return -(pos_ll + neg_ll)