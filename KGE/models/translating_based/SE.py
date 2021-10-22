"""An implementation of SE (Structured Embedding)
"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import LpDistance
from ...loss import PairwiseHingeLoss
from ...ns_strategy import UniformStrategy
from ...constraint import normalized_embeddings

logging.getLogger().setLevel(logging.INFO)

class SE(TranslatingModel):
    """An implementation of SE (Structured Embedding) from `[brodes 2011] <https://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/viewFile/3659/3898>`_.

    SE uses two matrices :math:`\\textbf{R}_r^{head}` and :math:`\\textbf{R}_r^{head}`
    to project head and tail entities for given relation :math:`r`.

    The score of :math:`(h,r,t)` is:

    .. math::
        f(h,r,t) = s(\\textbf{R}_r^{head} \\textbf{e}_h, \\textbf{R}_r^{tail} \\textbf{e}_t)

    
    where :math:`\\textbf{e}_i \in \mathbb{R}^k` are vector representations of the entities,
    :math:`\\textbf{R}_i^{head} \in \mathbb{R}^{k \\times k}` and :math:`\\textbf{R}_i^{tail} \in \mathbb{R}^{k \\times k}`
    are relation specific projection matrices for head and tail entities. \n
    and :math:`s` is a scoring function (:py:mod:`KGE.score`) that scores the plausibility of matching between :math:`(translation, predicate)`. \n
    By default, using :py:mod:`KGE.score.LpDistance`, negative L1-distance: 
    
    .. math::
        s(\\textbf{R}_r^{head} \\textbf{e}_h, \\textbf{R}_r^{tail} \\textbf{e}_t) =
            - \left\| \\textbf{R}_r^{head} \\textbf{e}_h - \\textbf{R}_r^{tail} \\textbf{e}_t \\right\|_1

    You can change to L2-distance by giving :code:`score_fn=LpDistance(p=2)` in :py:func:`__init__`,
    or change any score function you like by specifying :code:`score_fn` in :py:func:`__init__`.

    If :code:`constraint=True` given in :py:func:`__init__`,
    renormalized :math:`\left\| \\textbf{e}_i \\right\|_2 = 1` to have unit length every iteration described in
    `original SE paper <https://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/viewFile/3659/3898>`_.
    """

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=LpDistance(p=1), loss_fn=PairwiseHingeLoss(margin=1),
                 ns_strategy=UniformStrategy, constraint=True, n_workers=1):
        """Initialized SE

        Parameters
        ----------
        embedding_params : dict
            embedding dimension parameters, should have key :code:`'embedding_size'` for embedding dimension :math:`k`
        negative_ratio : int
            number of negative sample
        corrupt_side : str
            corrupt from which side while trainging, can be :code:`'h'`, :code:`'t'`, or :code:`'h+t'`
        score_fn : function, optional
            scoring function, by default :py:mod:`KGE.score.LpDistance`
        loss_fn : class, optional
            loss function class :py:mod:`KGE.loss.Loss`, by default :py:mod:`KGE.loss.PairwiseHingeLoss`
        ns_strategy : function, optional
            negative sampling strategy, by default :py:func:`KGE.ns_strategy.uniform_strategy`
        constraint : bool, optional
            conduct constraint or not, by default True
        n_workers : int, optional
            number of workers for negative sampling, by default 1
        """
        super(SE, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                 score_fn, loss_fn, ns_strategy, n_workers)
        self.constraint = constraint
        
    def _init_embeddings(self, seed):
        """Initialized the SE embeddings.

        If :code:`model_weight_initial` not given in :py:func:`train`, initialized embeddings randomly,  
        otherwise, initialized from :code:`model_weight_initial`. 

        Parameters
        ----------
        seed : int
            random seed
        """

        if self._model_weights_initial is None:
            assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using SE"
            
            limit = np.sqrt(6.0 / self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"]]),
                name="entities_embedding", dtype=np.float32
            )

            limit = np.sqrt(3.0 / self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            rel_proj_h = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]]),
                name="relations_projector_h", dtype=np.float32
            )

            rel_proj_t = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]]),
                name="relations_projector_t", dtype=np.float32
            )         

            self.model_weights = {"ent_emb": ent_emb, "rel_proj_h": rel_proj_h, "rel_proj_t": rel_proj_t}
        else:
            self._check_model_weights(self._model_weights_initial)
            self.model_weights = self._model_weights_initial

    def _check_model_weights(self, model_weights):
        """Check the model_weights have necessary keys and dimensions

        Parameters
        ----------
        model_weights : dict
            model weights to check.
        """

        assert model_weights.get("ent_emb") is not None, "entity embedding should be given in model_weights with key 'ent_emb'"
        assert model_weights.get("rel_proj_h") is not None, "relation projection matrix(head) should be given in model_weights with key 'rel_proj_h'"
        assert model_weights.get("rel_proj_t") is not None, "relation projection matrix(tail) should be given in model_weights with key 'rel_proj_t'"
        assert list(model_weights["ent_emb"].shape) == [len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"]], \
            "shape of 'ent_emb' should be (len(metadata['ind2ent']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_proj_h"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]], \
            "shape of 'rel_proj_h' should be (len(metadata['ind2rel']), embedding_params['embedding_size'], embedding_params['embedding_size'])"
        assert list(model_weights["rel_proj_t"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]], \
            "shape of 'rel_proj_t' should be (len(metadata['ind2rel']), embedding_params['embedding_size'], embedding_params['embedding_size'])"

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
        
        h,r,t = super(SE, self).score_hrt(h,r,t)

        h_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], h), axis=-1)
        t_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], t), axis=-1)

        rel_proj_h = tf.nn.embedding_lookup(self.model_weights["rel_proj_h"], r)
        rel_proj_t = tf.nn.embedding_lookup(self.model_weights["rel_proj_t"], r)

        return self.score_fn(tf.squeeze(tf.matmul(rel_proj_h, h_emb)), tf.squeeze(tf.matmul(rel_proj_t, t_emb)))

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

        return 0