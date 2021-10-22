"""An implementation of DistMult
"""

import numpy as np
import tensorflow as tf
from ..base_model.SemanticModel import SemanticModel
from ...loss import PairwiseHingeLoss
from ...ns_strategy import UniformStrategy
from ...constraint import normalized_embeddings, Lp_regularization

class DistMult(SemanticModel):
    """An implementation of DistMult from `[yang 2014] <https://arxiv.org/abs/1412.6575>`_.

    DistMult simplified RESCAL by restricting :math:`\\textbf{R}_i` to diagonal matrix.

    The score of :math:`(h,r,t)` is defined by a bilinear function:

    .. math::
        f(h,r,t) = \\textbf{e}_h^{T} \\textbf{R}_{r} \\textbf{e}_t
                 = \\textbf{e}_h^{T} diag(\\textbf{R}_{r}) \\textbf{e}_t
                 = \sum_i (\\textbf{e}_h^{T})_i diag(\\textbf{R}_{r})_i (\\textbf{e}_t)_i

    
    where :math:`\\textbf{e}_i \in \mathbb{R}^k` are vector representations of
    the entities, and :math:`\\textbf{R}_i \in \mathbb{R}^{k \\times k}` is a
    diagonal matrix associated with the relation.

    If :code:`constraint=True` given in :py:func:`__init__`,
    renormalized :math:`\left\| \\textbf{e}_i \\right\|_2 = 1` to have unit length every iteration
    and conduct L2-regularization on :math:`\\textbf{R}` described in
    `original DistMult paper <https://arxiv.org/abs/1412.6575>`_:

    .. math::
        regularization~term = \lambda \\times \sum_{i}  {\left\| \\textbf{R}_i \\right\|}_F^2
                            = \lambda \\times \sum_i {\left\| diag(\\textbf{R}_i) \\right\|}_2^2
    """

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 loss_fn=PairwiseHingeLoss(margin=1), ns_strategy=UniformStrategy,
                 constraint=True, constraint_weight=1.0, n_workers=1):
        """Initialized DistMult

        Parameters
        ----------
        embedding_params : dict
            embedding dimension parameters, should have key :code:`'embedding_size'` for embedding dimension :math:`k`
        negative_ratio : int
            number of negative sample
        corrupt_side : str
            corrupt from which side while trainging, can be :code:`'h'`, :code:`'t'`, or :code:`'h+t'`
        loss_fn : class, optional
            loss function class :py:mod:`KGE.loss.Loss`, by default :py:mod:`KGE.loss.PairwiseHingeLoss`
        ns_strategy : function, optional
            negative sampling strategy, by default :py:func:`KGE.ns_strategy.uniform_strategy`
        constraint : bool, optional
            conduct constraint or not, by default :code:`True`
        constraint_weight : float, optional
            regularization weight :math:`\lambda`, by default 1.0
        n_workers : int, optional
            number of workers for negative sampling, by default 1
        """
        
        super(DistMult, self).__init__(embedding_params, negative_ratio, corrupt_side, 
                                       loss_fn, ns_strategy, n_workers)
        self.constraint = constraint
        self.constraint_weight = constraint_weight

    def _init_embeddings(self, seed):
        """Initialized the DistMult embeddings.

        If :code:`model_weight_initial` not given in :py:func:`train`, initialized embeddings randomly,  
        otherwise, initialized from :code:`model_weight_initial`. 

        Parameters
        ----------
        seed : int
            random seed
        """

        if self._model_weights_initial is None:
            assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using RESCAL"
            
            limit = np.sqrt(6.0 / self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"]]),
                name="entities_embedding", dtype=np.float32
            )

            limit = np.sqrt(6.0 / self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            rel_inter = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]]),
                name="relations_interaction", dtype=np.float32
            )       

            self.model_weights = {"ent_emb": ent_emb, "rel_inter": rel_inter}
        else:
            self._check_model_weights(self._model_weights_initial)
            self.model_weights = self._model_weights_initial

    def _check_model_weights(self, model_weights):
        """Check the model_weights have necessary keys and dimensions.

        Parameters
        ----------
        model_weights : dict
            model weights to check.
        """

        assert model_weights.get("ent_emb") is not None, "entity embedding should be given in model_weights with key 'ent_emb'"
        assert model_weights.get("rel_inter") is not None, "relation interaction matrix should be given in model_weights with key 'rel_inter'"
        assert list(model_weights["ent_emb"].shape) == [len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"]], \
            "shape of 'ent_emb' should be (len(metadata['ind2ent']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_inter"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]], \
            "shape of 'rel_inter' should be (len(metadata['ind2rel']), embedding_params['embedding_size'])"
    
    def score_hrt(self, h, r, t):
        """ Score the triplets :math:`(h,r,t)`.

        If :code:`h` is :code:`None`, score all entities: :math:`(h_i, r, t)`. \n
        If :code:`t` is :code:`None`, score all entities: :math:`(h, r, t_i)`. \n
        :code:`h` and :code:`t` should not be :code:`None` simultaneously.

        Parameters
        ----------
        h : tf.Tensor or np.ndarray or None
            index of heads with shape :code:`(n,)`
        r : tf.Tensor or np.ndarray
            index of relations with shape :code:`(n,)`
        t : tf.Tensor or np.ndarray or None
            index of tails with shape :code:`(n,)`

        Returns
        -------
        tf.Tensor
            triplets scores with shape :code:`(n,)`
        """

        h,r,t = super(DistMult, self).score_hrt(h,r,t)

        h_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], h)
        t_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], t)
        r_inter = tf.nn.embedding_lookup(self.model_weights["rel_inter"], r)

        return tf.reduce_sum(h_emb * r_inter * t_emb, axis=-1)

    def _constraint_loss(self, X):
        """Perform constraint if necessary.

        Parameters
        ----------
        X : batch_data
            batch data

        Returns
        -------
        tf.Tensor
            regularization term with shape (1,)
        """

        if self.constraint:
            self.model_weights["ent_emb"].assign(normalized_embeddings(X=self.model_weights["ent_emb"], p=2, axis=1, value=1))
            r_inter = tf.nn.embedding_lookup(self.model_weights["rel_inter"], X[:, 1])
            return self.constraint_weight * tf.reduce_mean(Lp_regularization(r_inter, p=2, axis=-1))
        else:
            return 0