import numpy as np
import tensorflow as tf

def normalized_embeddings(X, p, value, axis):
    """ Normalized embeddings

    Normalized :code:`X` into :code:`p`-norm equals :code:`value`. 

    Parameters
    ----------
    X : tf.Tensor
        Tensor to be normalized
    p : int
        p-norm
    value : float
        restrict value
    axis : int or tuple
        along what axis

    Returns
    -------
    tf.Tensor
        normalized tensor with same shape as :code:`X`
    """

    if p == np.inf:
        norm = tf.reduce_max(tf.abs(X), axis=axis, keepdims=True)
    else:
        norm = tf.pow(tf.reduce_sum(tf.pow(tf.abs(X), p), axis=axis, keepdims=True), 1/p)

    return X / norm * value


def soft_constraint(X, p, value, axis):
    """ Soft constraint

    Soft constraint that described in `TransH <https://ojs.aaai.org/index.php/AAAI/article/view/8870>`_:

    .. math::
        regularization ~ term =
            \sum \left[ \left\| \\textbf{X} \\right\|_p^2 - value \\right]_+

    where :math:`[x]_+ = max(0,x)`

    Parameters
    ----------
    X : tf.Tensor
        Tensor to be constraint
    p : int
        p-norm
    value : float
        restrict value
    axis : int or tuple
        along what axis

    Returns
    -------
    tf.Tensor
        regularization term
    """

    if p == np.inf:
        norm = tf.reduce_max(tf.abs(X), axis=axis, keepdims=True)
    else:
        norm = tf.pow(tf.reduce_sum(tf.pow(tf.abs(X), p), axis=axis, keepdims=True), 1/p)
        
    return tf.reduce_sum(tf.clip_by_value(tf.pow(norm, p) - value, 0, np.inf))


def clip_constraint(X, p, value, axis):
    """ Clip embeddings

    If :code:`X`'s :code:`p`-norm exceeds :code:`value`, clip the value that let
    :code:`p`-norm of :code:`X` equals :code:`value.`

    Parameters
    ----------
    X : tf.Tensor
        Tensor to be constraint
    p : int
        p-norm
    value : float
        restrict value
    axis : int or tuple
        along what axis

    Returns
    -------
    tf.Tensor
        constraint tensor with same shape as :code:`X`
    """

    if p == np.inf:
        norm = tf.reduce_max(tf.abs(X), axis=axis, keepdims=True)
    else:
        norm = tf.pow(tf.reduce_sum(tf.pow(tf.abs(X), p), axis=axis, keepdims=True), 1/p)
        
    mask = tf.cast(norm<value, X.dtype)
    return mask * X + (1 - mask) * (X / tf.clip_by_value(norm, 1e-9, np.inf) * value)


def Lp_regularization(X, p, axis):
    """Standard Lp-regularization

    The standard Lp-regularization:
    
    ..math ::
        regularization ~ term = 
            \sum \left\| X \\right\|_p^p

    Parameters
    ----------
    X : tf.Tensor
        Tensor to be regularized
    p : int
        p-norm
    axis : int or tuple
        along what axis

    Returns
    -------
    tf.Tensor
        constraint term
    """

    return tf.reduce_sum(tf.pow(tf.abs(X), p), axis=axis)