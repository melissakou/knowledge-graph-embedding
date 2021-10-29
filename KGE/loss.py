"""Different loss functions that you can choose when training Knowledge Graph Embedding Model.

Different Knowledge Graph Embedding Models use different loss functions, the default
setting for each KGE model is according to the original paper described,  for example,
:py:mod:`TransE <KGE.models.translating_based.TransE>` using 
:py:mod:`Pairwise Hinge Loss <PairwiseHingeLoss>`,
:py:mod:`RotatE <KGE.models.translating_based.RotatE>` using
:py:mod:`Self Adversarial Negative Sampling Loss <SelfAdversarialNegativeSamplingLoss>`. 

You can change the loss function to try any possibility in a very easy way:

.. code-block:: python

    from KGE.models.translating_based.TransE import TransE
    from KGE.loss import SelfAdversarialNegativeSamplingLoss

    model = TransE(
        embedding_params={"embedding_size": 10},
        negative_ratio=10,
        corrupt_side="h+t",
        loss_fn=SelfAdversarialNegativeSamplingLoss(margin=3, temperature=1) # specifying loss function you want
    )
"""

import numpy as np
import tensorflow as tf

class Loss:
    """A base module for loss.
    """

    def __init__(self):
        """ Initialize loss
        """
        raise NotImplementedError("subclass of Loss should implement __init__() to init loss parameters")

    def __call__(self, pos_score, neg_score):
        """ Calculate loss.

        Parameters
        ----------
        pos_score : tf.Tensor
            score of postive triplets, with shape :code:`(n,)`
        neg_score : tf.Tensor
            score of negative triplets, with shape :code:`(n,)`
        """
        raise NotImplementedError("subclass of Loss should implement __call__() to calculate loss")

class PairwiseHingeLoss(Loss):
    """An implementation of Pairwise Hinge Loss / Margin Ranking Loss.

    Pairwise Hinge Loss or Margin Ranking Loss is a common loss function that used in many
    models such as `UM <https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf>`_,
    `SE <https://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/viewFile/3659/3898>`_,
    `TransE <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_,
    `TransH <https://ojs.aaai.org/index.php/AAAI/article/view/8870>`_,
    `TransR <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/>`_,
    `TransD <https://aclanthology.org/P15-1067.pdf>`_,
    `DistMult <https://arxiv.org/abs/1412.6575>`_.

    For each **pair** of postive triplet :math:`(h,r,t)_i^+` and negative triplet :math:`(h,r,t)_i^-`,
    Pairwise Hinge Loss compare the difference of scores between postivie triplet and negative triplet:

    .. math::
        \Delta_i = f\left( (h,r,t)_i^- \\right) - f\left( (h,r,t)_i^+ \\right)
    
    Since the socre of triplet :math:`f(h,r,t)` measures how plausible :math:`(h,r,t)` is, so
    :math:`\Delta_i < 0` is favorable. If the difference :math:`\Delta_i` does not execeed the
    given margin :math:`\gamma`, Pairwise Hinge Loss penalize this pair:

    .. math::
        \mathscr{L} = \sum_i max \left( 0, \gamma + \Delta_i \\right)
    """

    def __init__(self, margin):
        
        self.margin = margin
    
    def __call__(self, pos_score, neg_score):
        
        pos_score = tf.repeat(pos_score, int(neg_score.shape[0] / pos_score.shape[0]))
        return tf.reduce_sum(tf.clip_by_value(self.margin + neg_score - pos_score, 0, np.inf)) / pos_score.shape[0]


class PairwiseLogisticLoss(Loss):
    """An implementation of Pairwise Logistic Loss.

    Described in `Loss Functions in Knowledge Graph Embedding Models <http://ceur-ws.org/Vol-2377/paper_1.pdf>`_.

    For each **pair** of postive triplet :math:`(h,r,t)_i^+` and negative triplet :math:`(h,r,t)_i^-`,
    Pairwise Logistic Loss compare the difference of scores between postivie triplet and negative triplet:

    .. math::
        \Delta_i = f\left( (h,r,t)_i^- \\right) - f\left( (h,r,t)_i^+ \\right)

    and define the Pairwise Logistic Loss as:

    .. math::
        \mathscr{L} = \sum_i log(1+exp(\Delta_i))

    Pairwise Logistic Loss is a smooth version of :py:func:`Pairwise Hinge Loss <KGE.loss.pairwise_hinge_loss>` while
    :math:`\gamma = 0`, you can view function graph `here <https://www.desmos.com/calculator/vvrrjhqbgu>`_ to
    campare these two functions.
    """
    
    def __init__(self):
        
        pass

    def __call__(self, pos_score, neg_score):
        
        pos_score = tf.repeat(pos_score, int(neg_score.shape[0] / pos_score.shape[0]))
        return tf.reduce_sum(tf.math.log(1 + tf.exp(neg_score - pos_score)))


class BinaryCrossEntropyLoss(Loss):
    """An implementation of Binary Cross Entropy Loss.

    Binary Cross Entropy Loss is commonly used in binary classification problem.
    In KGE, we can also turn the problem into a binary classification problem
    that classifies triplet into positive or negative :math:`y_i = 1~or~0` with
    the triplet score as logit:  :math:`logit_i = f\left( (h,r,t)_i \\right)`

    .. math::
        \\begin{aligned}
        \mathscr{L} &= - \sum_i y_i log(\hat{y}_i) + (1-y_i) log(1-\hat{y}_i)

                    &= - \sum_i log\left[\sigma(f((h,r,t)_i^+))\\right] - \sum_i log\left[1-\sigma(f((h,r,t)_i^-))\\right]

                    &= - \sum_i log\left[\sigma(f((h,r,t)_i^+))\\right] - \sum_i log\left[\sigma(-f((h,r,t)_i^-))\\right]
        \end{aligned}
    """

    def __init__(self):
        
        pass

    def __call__(self, pos_score, neg_score):

        pos_ll = tf.reduce_sum(tf.math.log_sigmoid(pos_score))
        neg_ll = tf.reduce_sum(tf.math.log_sigmoid(-neg_score))

        return -(pos_ll + neg_ll) / pos_score.shape[0]


class SelfAdversarialNegativeSamplingLoss(Loss):
    """ An implementation of Self Adversarial Negative Sampling Loss.

    Described in `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space <https://arxiv.org/pdf/1902.10197v1.pdf>`_.

    Self Adversarial Negative Sampling Loss samples negative triples according to the current embedding model. Specifically, it sample
    negative triples from the following distribution:

    .. math::
        p\left((h,r,t)_{i,j}^- \\vert (h,r,t)_i^+ \\right) = 
            \\frac{exp~ \\alpha f((h,r,t)_{i.j}^-)}{\sum_k exp~ \\alpha f((h,r,t)_{i,k}^-)}

    where :math:`(h,r,t)_i^+` denotes i-th positive triplet, :math:`(h,r,t)_{i,j}^-` denotes j-th negative triplet generate from i-th 
    positive triplet, :math:`\\alpha` is the temperature of sampling.

    Since the sampling procedure may be costly, Self Adversarial Negative Sampling Loss treats the above probability as the weight of
    the negative sample. Therefore, the final negative sampling loss with self-adversarial training takes the following form:

    .. math::
        \mathscr{L} =
            - \sum_i log~ \sigma(\gamma + f((h,r,t)_i^+))
            - \sum_i \sum_j p\left( (h,r,t)_{i,j}^- \\right) log~ \sigma(-\gamma - f((h,r,t)_{i,j}^-))
    """

    def __init__(self, margin, temperature):
        self.margin = margin
        self.temperature = temperature

    def __call__(self, pos_score, neg_score):
        
        neg_score = tf.reshape(neg_score, (pos_score.shape[0], int(neg_score.shape[0] / pos_score.shape[0])))
        neg_prob = tf.stop_gradient(tf.nn.softmax(self.temperature * neg_score, axis=-1))

        pos_ll = tf.reduce_sum(tf.math.log_sigmoid(pos_score + self.margin))
        neg_ll = tf.reduce_sum(neg_prob * tf.math.log_sigmoid(- neg_score - self.margin))

        return -(pos_ll + neg_ll) / pos_score.shape[0]


class SquareErrorLoss(Loss):
    """ An implementation of Square Error Loss.

    Square Error Loss is a loss function used in
    `RESCAL <https://icml.cc/2011/papers/438_icmlpaper.pdf>`_, it computes
    the squared difference between triplet scores :math:`f((h,r,t)_i)` and
    labels (:math:`y_i = 1~or~0`):

    .. math::
        \mathscr{L} = \sum_i \left( f((h,r,t)_i) - y_i \\right)^2
    """

    def __init__(self):
        pass

    def __call__(self, pos_score, neg_score):
        pos_loss = tf.reduce_sum(tf.pow(pos_score - 1.0, 2))
        neg_loss = tf.reduce_sum(tf.pow(neg_score - 0.0, 2))

        return (pos_loss + neg_loss) / 2 / pos_score.shape[0]