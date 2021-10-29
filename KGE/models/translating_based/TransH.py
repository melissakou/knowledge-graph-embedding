"""An implementation of TransH
"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import LpDistancePow
from ...loss import PairwiseHingeLoss
from ...ns_strategy import UniformStrategy
from ...constraint import normalized_embeddings, soft_constraint

logging.getLogger().setLevel(logging.INFO)

class TransH(TranslatingModel):
    """An implementation of TransH from `[wang 2014] <https://ojs.aaai.org/index.php/AAAI/article/view/8870>`_.

    TransH overcomes the problems of TransE in modeling reflexive/one-to-many/many-to-one/many-to-many relations
    by enabling an entity to have distributed representations when involved in different relations.
    TransH represents each relation :math:`r` the relation-specific translation vector :math:`\\textbf{r}_r`
    in the relation-specific hyperplane :math:`\\textbf{w}_r`, and project head and tail embeddings on to this
    hyperplane, expecting the projected embeddings can be connected by the relation tranalation vector
    :math:`\\textbf{r}_r`:
    
    .. math::
        {\\textbf{e}_h}_{\perp} + \\textbf{r}_r \\approx {\\textbf{e}_t}_{\perp}

        {\\textbf{e}_h}_{\perp} = \\textbf{e}_h - \\textbf{w}_r^T \\textbf{e}_h\\textbf{w}_r

        {\\textbf{e}_t}_{\perp} = \\textbf{e}_t - \\textbf{w}_r^T \\textbf{e}_t\\textbf{w}_r


    where :math:`\\textbf{e}_i \in \mathbb{R}^k` are vector representations of the entities,
    :math:`\\textbf{r}_i \in \mathbb{R}^k` are relation translation vectors,
    and :math:`\\textbf{w}_i \in \mathbb{R}^k` are relation hyperplanes.
    
    The score of :math:`(h,r,t)` is:

    .. math::
        f(h,r,t) = s({\\textbf{e}_h}_{\perp} + \\textbf{r}_r, {\\textbf{e}_t}_{\perp})

    where :math:`s` is a scoring function (:py:mod:`KGE.score`) that scores the plausibility of matching between :math:`(translation, predicate)`. \n
    By default, using :py:mod:`KGE.score.LpDistancePow`, negative squared L2-distance: 
    
    .. math::
        s({\\textbf{e}_h}_{\perp} + \\textbf{r}_r, {\\textbf{e}_t}_{\perp}) =
            - \left\| {\\textbf{e}_h}_{\perp} + \\textbf{r}_r - {\\textbf{e}_t}_{\perp} \\right\|_2^2

    You can change to L1-distance by giving :code:`score_fn=LpDisrancePow(p=1)` in :py:func:`__init__`,
    or change any score function you like by specifying :code:`score_fn` in :py:func:`__init__`.

    If :code:`constraint=True` given in :py:func:`__init__`, conduct following constraints: \n
    1. renormalized :math:`\left\| \\textbf{w}_i \\right\|_2 = 1` to have unit length every iteration \n
    2. :math:`\left\| \\textbf{e}_i \\right\|_2 \leq 1` \n
    3. :math:`\left| \mathbf{w}_{r}^T \mathbf{r}_{r} \\right| /\left\|\mathbf{r}_{r}\\right\|_2 \leq \epsilon` to guarantees the translation vector :math:`\\textbf{r}_r` is in the hyperplane \n
    constraint 2 & 3 are realized by :py:func:`soft constraint <KGE.constraint.soft_constraint>` described in
    `original TransH paper <https://ojs.aaai.org/index.php/AAAI/article/view/8870>`_:

    .. math::
        regularization~term = \lambda
            \left\{ \sum_i \left[\| \\textbf{e}_i \|_{2}^{2}-1 \\right]_+ + \sum_i \left[ \\frac{\left(\ \\textbf{w}_{i}^T \\textbf{r}_{i} \\right)^2}{\left\| \\textbf{r}_{i} \\right\|_2^2}-\epsilon^2 \\right]_{+} \\right\}
    """

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=LpDistancePow(p=2), loss_fn=PairwiseHingeLoss(margin=1),
                 ns_strategy=UniformStrategy, constraint=True, constraint_weight=1.0, n_workers=1):
        """Initialized TransH

        Parameters
        ----------
        embedding_params : dict
            embedding dimension parameters, should have key :code:`'embedding_size'` for embedding dimension :math:`k`
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
        constraint_weight : float, optional
            regularization weight :math:`\lambda`, by default 1.0
        n_workers : int, optional
            number of workers for negative sampling, by default 1
        """
        super(TransH, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                     score_fn, loss_fn, ns_strategy, n_workers)
        self.constraint = constraint
        self.constraint_weight = constraint_weight
        
    def _init_embeddings(self, seed):
        """Initialized the TransH embeddings.

        If :code:`model_weight_initial` not given in :py:func:`train`, initialized embeddings randomly,  
        otherwise, initialized from :code:`model_weight_initial`. 

        Parameters
        ----------
        seed : int
            random seed
        """

        if self._model_weights_initial is None:
            assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using TransH"
                
            limit = np.sqrt(6.0 / self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"]]),
                name="entities_embedding", dtype=np.float32
            )
            rel_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]]),
                name="relations_embedding", dtype=np.float32
            )
            rel_hyper = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]]),
                name="relations_hyperplane", dtype=np.float32
            )

            self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb, "rel_hyper": rel_hyper}
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
        assert model_weights.get("rel_hyper") is not None, "relation hyperplane should be given in model_weights with key 'rel_hyper'"
        assert list(model_weights["ent_emb"].shape) == [len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"]], \
            "shape of 'ent_emb' should be (len(metadata['ind2ent']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_emb"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]], \
            "shape of 'rel_emb' should be (len(metadata['ind2rel']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_hyper"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]], \
            "shape of 'rel_hyper' should be (len(metadata['ind2rel']), embedding_params['embedding_size'])"

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

        h,r,t = super(TransH, self).score_hrt(h,r,t)
        
        h_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], h)
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], r)
        r_hyper = tf.nn.embedding_lookup(self.model_weights["rel_hyper"], r)
        t_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], t)

        h_emb = tf.expand_dims(h_emb, axis=-1)
        r_hyper = tf.expand_dims(r_hyper, axis=-1)
        t_emb = tf.expand_dims(t_emb, axis=-1)

        h_proj = tf.squeeze(h_emb - tf.multiply(tf.matmul(r_hyper, h_emb, transpose_a=True), r_hyper))
        t_proj = tf.squeeze(t_emb - tf.multiply(tf.matmul(r_hyper, t_emb, transpose_a=True), r_hyper))

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
            self.model_weights["rel_hyper"].assign(normalized_embeddings(X=self.model_weights["rel_hyper"], p=2, axis=1, value=1))
            scale = soft_constraint(self.model_weights["ent_emb"], p=2, axis=-1, value=1)
            orthogonal = tf.matmul(tf.expand_dims(self.model_weights["rel_hyper"], axis=-1),
                                   tf.expand_dims(self.model_weights["rel_emb"], axis=-1),
                                   transpose_a=True)
            orthogonal = tf.pow(tf.squeeze(orthogonal) / tf.norm(self.model_weights["rel_emb"], axis=-1), 2) - 1e-18
            orthogonal = tf.reduce_sum(tf.clip_by_value(orthogonal, 0, np.inf))

            return self.constraint_weight * (scale + orthogonal)
        else:
            return 0
