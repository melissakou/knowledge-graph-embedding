"""An implementation of RotatE
"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import LpDistance
from ...loss import SelfAdversarialNegativeSamplingLoss
from ...ns_strategy import UniformStrategy

logging.getLogger().setLevel(logging.INFO)

class RotatE(TranslatingModel):
    """An implementation of RotatE from `[sun 2019] <https://arxiv.org/abs/1902.10197v1>`_.

    RotatE represents both entities and relations as embedding vectors in the complex space,
    and models the relation as an element-wise **rotation** from the head to tail:
    
    .. math::
        \\textbf{e}_h \circ \\textbf{r}_r \\approx \\textbf{e}_t

    where :math:`\\textbf{e}_i, \\textbf{r}_i \in \mathbb{C}^k` are vector representations
    of the entities and relations. and :math:`\circ` is the Hadmard (element-wise) product.
    
    The score of :math:`(h,r,t)` is:

    .. math::
        f(h,r,t) = s(\\textbf{e}_h \circ \\textbf{r}_r, \\textbf{e}_t)

    where :math:`s` is a scoring function (:py:mod:`KGE.score`) that scores the plausibility of matching between :math:`(translation, predicate)`. \n
    By default, using :py:mod:`KGE.score.LpDistance`, negative L1-distance: 
    
    .. math::
        s(\\textbf{e}_h \circ \\textbf{r}_r, \\textbf{e}_t) = 
            - \left\| \\textbf{e}_h \circ \\textbf{r}_r - \\textbf{e}_t \\right\|_1

    You can change to L2-distance by giving :code:`score_fn=LpDistance(p=2)` in :py:func:`__init__`,
    or change any score function you like by specifying :code:`score_fn` in :py:func:`__init__`.

    RotatE constrains the modulus of each element of :math:`\\textbf{r} \in \mathbb{C}^k` to 1,
    i.e., :math:`r_i \in \mathbb{C}` to be :math:`\left| r_i \\right| = 1`.
    By doing this, :math:`r_i` is of the form :math:`e^{i\\theta_{r,i}}`
    """

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=LpDistance(p=1),
                 loss_fn=SelfAdversarialNegativeSamplingLoss(margin=3, temperature=1),
                 ns_strategy=UniformStrategy, n_workers=1):
        """Initialized RotatE

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
            loss function class :py:mod:`KGE.loss.Loss`, by default :py:mod:`KGE.loss.SelfAdversarialNegativeSamplingLoss`
        ns_strategy : function, optional
            negative sampling strategy, by default :py:func:`KGE.ns_strategy.uniform_strategy`
        n_workers : int, optional
            number of workers for negative sampling, by default 1
        """

        super(RotatE, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                     score_fn, loss_fn, ns_strategy, n_workers)
        
    def _init_embeddings(self, seed):
        """Initialized the RotatE embeddings.

        If :code:`model_weight_initial` not given in :py:func:`train`, initialized embeddings randomly,  
        otherwise, initialized from :code:`model_weight_initial`. 

        Parameters
        ----------
        seed : int
            random seed
        """

        if self._model_weights_initial is None:
            assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using TransE"
            
            if hasattr(self.loss_fn, "margin"):
                margin = self.loss_fn.margin
            else:
                margin = 6.0
            
            self.limit = (margin + 2.0) / self.embedding_params["embedding_size"]
            uniform_initializer = tf.initializers.RandomUniform(minval=-self.limit, maxval=self.limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"], 2]),
                name="entities_embedding", dtype=np.float32
            )

            uniform_initializer = tf.initializers.RandomUniform(minval=-self.limit, maxval=self.limit, seed=seed)
            rel_emb = tf.Variable(
                uniform_initializer([len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]]),
                name="relations_embedding", dtype=tf.float32
            )
            self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb}
        else:
            self._check_model_weights(self.__model_weights_initial)
            self.model_weights = self.__model_weights_initial

    def _check_model_weights(self, model_weights):
        """Check the model_weights have necessary keys and dimensions

        Parameters
        ----------
        model_weights : dict
            model weights to check.
        """

        assert model_weights.get("ent_emb") is not None, "entity embedding should be given in model_weights with key 'ent_emb'"
        assert model_weights.get("rel_emb") is not None, "relation embedding should be given in model_weights with key 'rel_emb'"
        assert list(model_weights["ent_emb"].shape) == [len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"], 2], \
            "shape of 'ent_emb' should be (len(metadata['ind2ent']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_emb"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]], \
            "shape of 'rel_emb' should be (len(metadata['ind2rel']), embedding_params['embedding_size'])"

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

        h,r,t = super(RotatE, self).score_hrt(h,r,t)

        h_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], h)
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], r)
        t_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], t)

        if len(h_emb.shape) == 2:
            h_emb = tf.expand_dims(h_emb, 0)
        if len(t_emb.shape) == 2:
            t_emb = tf.expand_dims(t_emb, 0)

        # normalize to [-pi, pi] to ensure sin & cos functions are one-to-one
        r_emb = r_emb / self.limit * np.pi
        
        hadamard = tf.multiply(tf.complex(h_emb[:,:,0], h_emb[:,:,1]),
                               tf.complex(tf.math.cos(r_emb), tf.math.sin(r_emb)))
        
        return self.score_fn(hadamard, tf.complex(t_emb[:,:,0], t_emb[:,:,1]))
    
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

        return 0