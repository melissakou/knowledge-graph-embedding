"""An implementation of TransD
"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import LpDistancePow
from ...loss import PairwiseHingeLoss
from ...ns_strategy import UniformStrategy
from ...constraint import clip_constraint

logging.getLogger().setLevel(logging.INFO)

class TransD(TranslatingModel):
    """An implementation of TransD from `[ji 2015] <https://aclanthology.org/P15-1067.pdf>`_.

    TransD models entities and relations in distinct embedding spaces like TransR,
    but unlike TransR which projects entities embeddings to relation space using single
    projection matrix :math:`\\textbf{M}_i` for each relation, TransD consturcts two
    projection matrices **dynamically**, these two projection matrices are determined by
    both entities and relations, so called **TransD**.
    
    In TransD, each entity and relation are represented by two vectors:
    :math:`\\textbf{e}_i \in \mathbb{R}^k, \\textbf{r}_i \in \mathbb{R}^d`
    capture the meaning of entity and relation,
    :math:`\\tilde{\\textbf{e}}_i \in \mathbb{R}^k, \\tilde{\\textbf{r}}_i \in \mathbb{R}^d`
    used to construct projection matrices:

    .. math::
        \mathbf{M}_{rh} = \\tilde{\\textbf{r}}_r \\tilde{\\textbf{e}}_h^T + \mathbf{I}^{d \\times k}

        \mathbf{M}_{rt} = \\tilde{\\textbf{r}}_r \\tilde{\\textbf{e}}_t^T + \mathbf{I}^{d \\times k}
    
    These two constructed projection matrices are used to project embedding vectors to relation
    space similar with TransR:

    .. math::
        {\\textbf{e}_h}_{\perp} = \\textbf{M}_{rh} \\textbf{e}_h

        {\\textbf{e}_t}_{\perp} = \\textbf{M}_{rt} \\textbf{e}_t

    and expecting the projected entity embeddings can be connected by the relation embeddings in
    the relation spaces:
    
    .. math::
        {\\textbf{e}_h}_{\perp} + \\textbf{r}_r \\approx {\\textbf{e}_t}_{\perp}
    
    The score of :math:`(h,r,t)` is:

    .. math::
        f(h,r,t) = s({\\textbf{e}_h}_{\perp} + \\textbf{r}_r, {\\textbf{e}_t}_{\perp})

    where :math:`s` is a scoring function (:py:mod:`KGE.score`) that scores the plausibility of matching between
    :math:`(translation, predicate)`. \n
    By default, using :py:func:`KGE.score.LpDistancePow`, negative squared L2-distance: 
    
    .. math::
        s({\\textbf{e}_h}_{\perp} + \\textbf{r}_r, {\\textbf{e}_t}_{\perp}) =
            - \left\| {\\textbf{e}_h}_{\perp} + \\textbf{r}_r - {\\textbf{e}_t}_{\perp} \\right\|_2^2

    You can change to L1-distance by giving :code:`score_fn=LpDistancePow(p=1)` in :py:func:`__init__`,
    or change any score function you like by specifying :code:`score_fn` in :py:func:`__init__`.

    If :code:`constraint=True` given in :py:func:`__init__`, conduct following constraints: \n
    1. :math:`\left\| \\textbf{e}_h \\right\|_2 \leq 1`  and :math:`\left\| \\textbf{r}_r \\right\|_2 \leq 1` and :math:`\left\| \\textbf{e}_t \\right\|_2 \leq 1` \n
    2. :math:`\left\| {\\textbf{e}_h}_{\perp} \\right\|_2 \leq 1` and  :math:`\left\| {\\textbf{e}_t}_{\perp} \\right\|_2 \leq 1` \n
    
    Since the `original TransD paper <https://aclanthology.org/P15-1067.pdf>`_ dose not specify how
    they conduct these constraints, here we use :py:func:`KGE.constraint.clip_constraint` which restrict the tensor's
    norm does not exceeds some value, if exceeds, clip the tensor norm to given threshold value.
    """

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=LpDistancePow(p=2), loss_fn=PairwiseHingeLoss(margin=1),
                 ns_strategy=UniformStrategy, constraint=True, n_workers=1):
        """Initialized TransR

        Parameters
        ----------
        embedding_params : dict
            embedding dimension parameters, should have following keys: \n
            :code:`'ent_embedding_size'` for entity embedding dimension :math:`k`
            :code:`'rel_embedding_size'` for relation embedding dimension :math:`d`
        negative_ratio : int
            number of negative sample
        corrupt_side : str
            corrupt from which side while trainging, can be :code:`'h'`, :code:`'t'`, or :code:`'h+t'`
        score_fn : function, optional
            scoring function, by default :py:mod:`KGE.score.LpDistancePow`
        loss_fn : class, optional
            loss function class :py:mod:`KGE.loss.Loss`, by default :py:mod:`KGE.loss.PairwiseHingeLoss`
        ns_strategy : function, optional
            negative sampling strategy, by default :py:func:`KGE.ns_strategy.uniform_strategy`
        constraint : bool, optional
            conduct constraint or not, by default True
        n_workers : int, optional
            number of workers for negative sampling, by default 1
        """

        super(TransD, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                     score_fn, loss_fn, ns_strategy, n_workers)
        self.constraint = constraint
        
    def _init_embeddings(self, seed):
        """Initialized the TransD embeddings.

        If :code:`model_weight_initial` not given in :py:func:`train`, initialized embeddings randomly,  
        otherwise, initialized from :code:`model_weight_initial`. 

        Parameters
        ----------
        seed : int
            random seed
        """

        if self._model_weights_initial is None:
            assert self.embedding_params.get("ent_embedding_size") is not None, "'ent_embedding_size' should be given in embedding_params when using TransR"
            assert self.embedding_params.get("rel_embedding_size") is not None, "'rel_embedding_size' should be given in embedding_params when using TransR"
                
            limit = np.sqrt(6.0 / self.embedding_params["ent_embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2ent"]), self.embedding_params["ent_embedding_size"]]),
                name="entities_embedding", dtype=np.float32
            )
            ent_proj = tf.Variable(
                uniform_initializer([len(self.metadata["ind2ent"]), self.embedding_params["ent_embedding_size"]]),
                name="entities_projection", dtype=np.float32
            )

            limit = np.sqrt(6.0 / self.embedding_params["rel_embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            rel_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["rel_embedding_size"]]),
                name="relations_embedding", dtype=np.float32
            )
            rel_proj = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["rel_embedding_size"]]),
                name="relations_projection", dtype=np.float32
            )    

            self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb, "ent_proj": ent_proj, "rel_proj": rel_proj}
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
        assert model_weights.get("rel_emb") is not None, "relation embedding should be given in model_weights with key 'rel_emb'"
        assert model_weights.get("ent_proj") is not None, "entity projection vector should be given in model_weights with key 'ent_proj'"
        assert model_weights.get("rel_proj") is not None, "relation projection vector should be given in model_weights with key 'rel_proj'"
        assert list(model_weights["ent_emb"].shape) == [len(self.metadata["ind2ent"]), self.embedding_params["ent_embedding_size"]], \
            "shape of 'ent_emb' should be (len(metadata['ind2ent']), embedding_params['ent_embedding_size'])"
        assert list(model_weights["rel_emb"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["rel_embedding_size"]], \
            "shape of 'rel_emb' should be (len(metadata['ind2rel']), embedding_params['rel_embedding_size'])"
        assert list(model_weights["ent_proj"].shape) == [len(self.metadata["ind2ent"]), self.embedding_params["ent_embedding_size"]], \
            "shape of 'ent_proj' should be (len(metadata['ind2ent']), embedding_params['ent_embedding_size'])"
        assert list(model_weights["rel_proj"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["rel_embedding_size"]], \
            "shape of 'rel_proj' should be (len(metadata['ind2rel']), embedding_params['rel_embedding_size'])"

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

        h,r,t = super(TransD, self).score_hrt(h,r,t)

        h_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], h), axis=-1)
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], r)
        t_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], t), axis=-1)

        h_proj = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_proj"], h), axis=-1)
        r_proj = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["rel_proj"], r), axis=-1)
        t_proj = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_proj"], t), axis=-1)

        if len(h_proj.shape) < 3:
            h_proj = tf.expand_dims(h_proj, 0)
        if len(r_proj.shape) < 3:
            r_proj = tf.expand_dims(r_proj, 0)
        if len(t_proj.shape) < 3:
            t_proj = tf.expand_dims(t_proj, 0)

        diag_matrix = tf.eye(num_rows=self.embedding_params["rel_embedding_size"],
                             num_columns=self.embedding_params["ent_embedding_size"])

        h_proj_matrix = tf.matmul(r_proj, tf.transpose(h_proj, perm=[0, 2, 1])) + diag_matrix
        t_proj_matrix = tf.matmul(r_proj, tf.transpose(t_proj, perm=[0, 2, 1])) + diag_matrix

        h_proj = tf.squeeze(tf.matmul(h_proj_matrix, h_emb))
        t_proj = tf.squeeze(tf.matmul(t_proj_matrix, t_emb))

        if self.constraint:
            h_proj = clip_constraint(X=h_proj, p=2, axis=-1, value=1)
            t_proj = clip_constraint(X=t_proj, p=2, axis=-1, value=1)
        
        return self.score_fn(h_proj + r_emb, t_proj)

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
            self.model_weights["ent_emb"].assign(clip_constraint(X=self.model_weights["ent_emb"], p=2, axis=-1, value=1))
            self.model_weights["rel_emb"].assign(clip_constraint(X=self.model_weights["rel_emb"], p=2, axis=-1, value=1))

        return 0