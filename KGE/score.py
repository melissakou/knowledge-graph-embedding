""" Different score functions that you can choose when training translating-based models.

Score functions measure the distance bewtween :math:`translation` and :math:`predicate`
in translating-based models, for example,
`TransE <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
uses :py:mod:`L1 or L2-distance <LpDistance>`,
`TransH <https://ojs.aaai.org/index.php/AAAI/article/view/8870>`_,
`TransR <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/>`_,
`TransD <https://aclanthology.org/P15-1067.pdf>`_ use :py:mod:`squared L2-distance <LpDistancePow>`.

You can change the score function to try any possibility in a very easy way:

.. code-block:: python

    from KGE.models.translating_based.TransE import TransE
    from KGE.score import LpDistancePow

    model = TransE(
        embedding_params={"embedding_size": 10},
        negative_ratio=10,
        corrupt_side="h+t",
        score_fn=LpDistancePow(p=2) # specifying score function you want
    )
"""

import numpy as np
import tensorflow as tf

class Score:
    """A base module for score.
    """

    def __init__(self):
        """ Initialize score.
        """
        raise NotImplementedError("subclass of Score should implement __init__() to init score parameters")

    def __call__(self, x, y):
        """ Calculate score.

        Parameters
        ----------
        x : tf.Tensor
        y : tf.Tensor
        """
        raise NotImplementedError("subclass of Score should implement __call__() to calculate score")


class LpDistance(Score):
    """An implementation of negative Lp-distance.

    Score between :math:`\\textbf{x}` and :math:`\\textbf{y}` is defined as
    :math:`- \left\| \\textbf{x} - \\textbf{y} \\right\|_p`
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, x, y):
        if self.p == np.inf:
            return -tf.reduce_max(tf.abs(x - y), axis=-1)
        else:
            return -tf.pow(tf.clip_by_value(tf.reduce_sum(tf.pow(tf.abs(x - y), self.p), axis=-1), 1e-9, np.inf), 1.0 / self.p)

class LpDistancePow(Score):
    """An implementation of negative squared Lp-distance.

    Score between :math:`\\textbf{x}` and :math:`\\textbf{y}` is defined as
    :math:`- \left\| \\textbf{x} - \\textbf{y} \\right\|_p^2`
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, x, y):
        return -tf.pow(LpDistance(p=self.p)(x, y), 2)
        
class Dot(Score):
    """An implementation of dot product.

    Score between :math:`\\textbf{x}` and :math:`\\textbf{y}` is defined as
    :math:`\\textbf{x} \cdot \\textbf{y}`
    """
    
    def __init__(self):
        pass

    def __call__(self, x, y):
        return tf.reduce_sum(x * y, axis=-1)