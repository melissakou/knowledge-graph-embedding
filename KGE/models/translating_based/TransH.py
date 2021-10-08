"""TransH"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import squared_euclidean
from ...loss import pairwise_hinge_loss
from ...ns_strategy import uniform_strategy
from ...constraint import normalized_embeddings, soft_constraint

logging.getLogger().setLevel(logging.INFO)

class TransH(TranslatingModel):

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=squared_euclidean, score_params=None, loss_fn=pairwise_hinge_loss, loss_param={"margin": 1},
                 ns_strategy=uniform_strategy, constraint=True, constraint_weight=1, n_workers=1):
        super(TransH, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                     score_fn, score_params, loss_fn, loss_param,
                                     ns_strategy, constraint, n_workers)
        self.constraint_weight = constraint_weight
        
    def _init_embeddings(self, seed):

        if self.__model_weights_initial is None:
            assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using TransH"
                
            limit = np.sqrt(6.0 / self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["embedding_size"]]),
                name="entities_embedding", dtype=np.float32
            )
            rel_emb = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"]]),
                name="relations_embedding", dtype=np.float32
            )
            rel_hyper = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"]]),
                name="relations_hyperplane", dtype=np.float32
            )

            self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb, "rel_hyper": rel_hyper}
        else:
            self._check_model_weights(self.__model_weights_initial)
            self.model_weights = self.__model_weights_initial

    def _check_model_weights(self, model_weights):
        assert model_weights.get("ent_emb") is not None, "entity embedding should be given in model_weights with key 'ent_emb'"
        assert model_weights.get("rel_emb") is not None, "relation embedding should be given in model_weights with key 'rel_emb'"
        assert model_weights.get("rel_hyper") is not None, "relation hyperplane should be given in model_weights with key 'rel_hyper'"
        assert list(model_weights["ent_emb"].shape) == [len(self.meta_data["ind2ent"]), self.embedding_params["embedding_size"]], \
            "shape of 'ent_emb' should be (len(meta_data['ind2ent']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_emb"].shape) == [len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"]], \
            "shape of 'rel_emb' should be (len(meta_data['ind2rel']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_hyper"].shape) == [len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"]], \
            "shape of 'rel_hyper' should be (len(meta_data['ind2rel']), embedding_params['embedding_size'])"

    def score_hrt(self, h, r, t):
        if h is None:
            h = np.arange(len(self.meta_data["ind2ent"]))
        if t is None:
            t = np.arange(len(self.meta_data["ind2ent"]))
        
        h_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], h)
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], r)
        r_hyper = tf.nn.embedding_lookup(self.model_weights["rel_hyper"], r)
        t_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], t)

        h_emb = tf.expand_dims(h_emb, axis=-1)
        r_hyper = tf.expand_dims(r_hyper, axis=-1)
        t_emb = tf.expand_dims(t_emb, axis=-1)

        h_proj = tf.squeeze(h_emb - tf.multiply(tf.matmul(r_hyper, h_emb, transpose_a=True), r_hyper))
        t_proj = tf.squeeze(t_emb - tf.multiply(tf.matmul(r_hyper, t_emb, transpose_a=True), r_hyper))

        return self.score_fn(h_proj + r_emb, t_proj, self.score_params)


    def _constraint_loss(self, X):
        if self.constraint:
            self.model_weights["rel_hyper"].assign(normalized_embeddings(X=self.model_weights["rel_hyper"], p=2, axis=1, value=1))
            scale = soft_constraint(self.model_weights["ent_emb"], p=2, axis=1, value=1)
            orthogonal = tf.matmul(tf.expand_dims(self.model_weights["rel_hyper"], axis=-1),
                                tf.expand_dims(self.model_weights["rel_emb"], axis=-1),
                                transpose_a=True)
            orthogonal = tf.pow(tf.squeeze(orthogonal) / tf.norm(self.model_weights["rel_emb"], axis=1), 2) - 1e-18
            orthogonal = tf.reduce_sum(tf.clip_by_value(orthogonal, 0, np.inf))

            return self.constraint_weight * (scale + orthogonal)
        else:
            return 0
