import numpy as np
import tensorflow as tf

def normalized_embeddings(X, p, value, axis):
    norm = tf.pow(tf.reduce_sum(tf.pow(tf.abs(X), p), axis=axis, keepdims=True), 1/p)
    return X / norm * value


def soft_constraint(X, p, value, axis):
    norm = tf.pow(tf.reduce_sum(tf.pow(tf.abs(X), p), axis=axis, keepdims=True), 1/p)
    return tf.reduce_sum(tf.clip_by_value(tf.pow(norm, p) - value, 0, np.inf))


def clip_constraint(X, p, value, axis):
    norm = tf.pow(tf.reduce_sum(tf.pow(tf.abs(X), p), axis=axis, keepdims=True), 1/p)
    mask = tf.cast(norm<value, X.dtype)
    return mask * X + (1 - mask) * (X / tf.clip_by_value(norm, 1e-9, np.inf) * value)


def Lp_regularization(X, p, axis):
    return tf.reduce_sum(tf.pow(tf.abs(X), p), axis=axis)