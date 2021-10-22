"""An implementation of TransR
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

class TransR(TranslatingModel):
    """An implementation of TransR from `[lin 2015] <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/>`_.

    Both TransE and TransH assume embeddings of entities and relations are in the same embedding space :math:`\mathbb{R}_k`.
    But relations and entities are completely different objects, it may be not capable to represent them in the same
    semantic space. To address this issue, TransH models entities and relations in distinct embedding spaces, i.e., entity
    space and relation spaces. \n
    TransH represents each entity as :math:`\\textbf{e}_i \in \mathbb{R}^k` and each relation as
    :math:`\\textbf{r}_i \in \mathbb{R}^d`, the dimensions of entity embeddings and relation embeddings are not necessarily
    identical. For each relation, TransH set a projection matrix :math:`\\textbf{M}_i \in \mathbb{R}^{k \\times d}`, which
    projects entities from entity space to relation space, expecting the projected entity embeddings can be connected by
    the relation embeddings in the relation spaces:
    
    .. math::
        {\\textbf{e}_h}_{\perp} + \\textbf{r}_r \\approx {\\textbf{e}_t}_{\perp}

        {\\textbf{e}_h}_{\perp} = \\textbf{e}_h \\textbf{M}_r

        {\\textbf{e}_t}_{\perp} = \\textbf{e}_t \\textbf{M}_r

    where :math:`\\textbf{e}_i \in \mathbb{R}^k` are vector representations of the entities,
    :math:`\\textbf{r}_i \in \mathbb{R}^d` are vector representations of the relations,
    and :math:`\\textbf{M}_i \in \mathbb{R}^{k \\times d}` are relation projection matrix.
    
    The score of :math:`(h,r,t)` is:

    .. math::
        f(h,r,t) = s({\\textbf{e}_h}_{\perp} + \\textbf{r}_r, {\\textbf{e}_t}_{\perp})

    where :math:`s` is a scoring function (:py:mod:`KGE.score`) that scores the plausibility of matching between
    :math:`(translation, predicate)`. \n
    By default, using :py:mod:`KGE.score.LpDistancePow`, negative squared L2-distance: 
    
    .. math::
        s({\\textbf{e}_h}_{\perp} + \\textbf{r}_r, {\\textbf{e}_t}_{\perp}) =
            - \left\| {\\textbf{e}_h}_{\perp} + \\textbf{r}_r - {\\textbf{e}_t}_{\perp} \\right\|_2^2

    You can change to L1-distance by giving :code:`score_fn=LpDistancePow(p=1)` in :py:func:`__init__`,
    or change any score function you like by specifying :code:`score_fn` in :py:func:`__init__`.

    If :code:`constraint=True` given in :py:func:`__init__`, conduct following constraints: \n
    1. :math:`\left\| \\textbf{e}_h \\right\|_2 \leq 1`  and :math:`\left\| \\textbf{r}_r \\right\|_2 \leq 1` and :math:`\left\| \\textbf{e}_t \\right\|_2 \leq 1` \n
    2. :math:`\left\| \\textbf{e}_h \\textbf{M}_r \\right\|_2 \leq 1` and  :math:`\left\| \\textbf{e}_t \\textbf{M}_r \\right\|_2 \leq 1` \n
    
    Since the `original TransR paper <https://ojs.aaai.org/index.php/AAAI/article/view/8870>`_ dose not specify how
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

        super(TransR, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                     score_fn, loss_fn, ns_strategy, n_workers)
        self.constraint = constraint
        
    def _init_embeddings(self, seed):
        """Initialized the TransR embeddings.

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

            limit = np.sqrt(6.0 / self.embedding_params["rel_embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            rel_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["rel_embedding_size"]]),
                name="relations_embedding", dtype=np.float32
            )

            rel_proj = tf.Variable(
                tf.eye(num_rows=self.embedding_params["ent_embedding_size"], num_columns=self.embedding_params["rel_embedding_size"], batch_shape=[len(self.metadata["ind2rel"])]),
                name="relations_projector", dtype=np.float32
            )     

            self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb, "rel_proj": rel_proj}
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
        assert model_weights.get("rel_proj") is not None, "relation projection matrix should be given in model_weights with key 'rel_proj'"
        assert list(model_weights["ent_emb"].shape) == [len(self.metadata["ind2ent"]), self.embedding_params["ent_embedding_size"]], \
            "shape of 'ent_emb' should be (len(metadata['ind2ent']), embedding_params['ent_embedding_size'])"
        assert list(model_weights["rel_emb"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["rel_embedding_size"]], \
            "shape of 'rel_emb' should be (len(metadata['ind2rel']), embedding_params['rel_embedding_size'])"
        assert list(model_weights["rel_proj"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["ent_embedding_size"], self.embedding_params["rel_embedding_size"]], \
            "shape of 'rel_emb' should be (len(metadata['ind2rel']), embedding_params['ent_embedding_size'], embedding_params['rel_embedding_size'])"

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

        h,r,t = super(TransR, self).score_hrt(h,r,t)

        h_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], h), axis=-1)
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], r)
        t_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], t), axis=-1)

        r_proj = tf.nn.embedding_lookup(self.model_weights["rel_proj"], r)

        h_proj = tf.squeeze(tf.matmul(h_emb, r_proj, transpose_a=True))
        t_proj = tf.squeeze(tf.matmul(t_emb, r_proj, transpose_a=True))

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
