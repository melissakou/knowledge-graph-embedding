import numpy as np
import tensorflow as tf

def pairwise_hinge_loss(pos_score, neg_score, params):
    assert params.get("margin") is not None, "'margin' should be given in loss_params when using pairwise_loss"
    pos_score = tf.repeat(pos_score, int(neg_score.shape[0] / pos_score.shape[0]))
    return tf.reduce_sum(tf.clip_by_value(params["margin"] + neg_score - pos_score, 0, np.inf)) / pos_score.shape[0]


def binary_cross_entropy_loss(pos_score, neg_score, params):
    pos_ll = tf.reduce_sum(tf.math.log_sigmoid(pos_score))
    neg_ll = tf.reduce_sum(tf.math.log_sigmoid(-neg_score))

    return -(pos_ll + neg_ll) / pos_score.shape[0]


def self_adversarial_negative_sampling_loss(pos_score, neg_score, params):
    assert params.get("margin") is not None, "'margin' should be given in loss_params when using self_adversarial_negative_sampling_loss"
    assert params.get("temperature") is not None, "'temperature' should be given in loss_params when using self_adversarial_negative_sampling_loss"

    neg_score = tf.reshape(neg_score, (pos_score.shape[0], int(neg_score.shape[0] / pos_score.shape[0])))
    neg_prob = tf.stop_gradient(tf.nn.softmax(params["temperature"] * neg_score, axis=-1))

    pos_ll = tf.reduce_sum(tf.math.log_sigmoid(pos_score + params["margin"]))
    neg_ll = tf.reduce_sum(neg_prob * tf.math.log_sigmoid(- neg_score - params["margin"]))

    return -(pos_ll + neg_ll) / pos_score.shape[0]


def square_error_loss(pos_score, neg_score, params):
    pos_loss = tf.reduce_sum(tf.pow(pos_score - 1.0, 2))
    neg_loss = tf.reduce_sum(tf.pow(neg_score - 0.0, 2))

    return (pos_loss + neg_loss) / 2 / pos_score.shape[0]