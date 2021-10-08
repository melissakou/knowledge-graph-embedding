"""TransR"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import squared_euclidean
from ...loss import pairwise_hinge_loss
from ...ns_strategy import uniform_strategy
from ...constraint import clip_constraint

logging.getLogger().setLevel(logging.INFO)

class TransR(TranslatingModel):

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=squared_euclidean, score_params=None, loss_fn=pairwise_hinge_loss, loss_param={"margin": 1},
                 ns_strategy=uniform_strategy, constraint=True, n_workers=1):
        super(TransR, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                     score_fn, score_params, loss_fn, loss_param,
                                     ns_strategy, constraint, n_workers)
        
    def _init_embeddings(self, seed):

        if self.__model_weights_initial is None:
            assert self.embedding_params.get("ent_embedding_size") is not None, "'ent_embedding_size' should be given in embedding_params when using TransR"
            assert self.embedding_params.get("rel_embedding_size") is not None, "'rel_embedding_size' should be given in embedding_params when using TransR"
                
            limit = np.sqrt(6.0 / self.embedding_params["ent_embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["ent_embedding_size"]]),
                name="entities_embedding", dtype=np.float32
            )

            limit = np.sqrt(6.0 / self.embedding_params["rel_embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            rel_emb = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["rel_embedding_size"]]),
                name="relations_embedding", dtype=np.float32
            )

            rel_proj = tf.Variable(
                tf.eye(num_rows=self.embedding_params["ent_embedding_size"], num_columns=self.embedding_params["rel_embedding_size"], batch_shape=[len(self.meta_data["ind2rel"])]),
                name="relations_projector", dtype=np.float32
            )     

            self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb, "rel_proj": rel_proj}
        else:
            self._check_model_weights(self.__model_weights_initial)
            self.model_weights = self.__model_weights_initial

    def _check_model_weights(self, model_weights):
        assert model_weights.get("ent_emb") is not None, "entity embedding should be given in model_weights with key 'ent_emb'"
        assert model_weights.get("rel_emb") is not None, "relation embedding should be given in model_weights with key 'rel_emb'"
        assert model_weights.get("rel_proj") is not None, "relation projection matrix should be given in model_weights with key 'rel_proj'"
        assert list(model_weights["ent_emb"].shape) == [len(self.meta_data["ind2ent"]), self.embedding_params["ent_embedding_size"]], \
            "shape of 'ent_emb' should be (len(meta_data['ind2ent']), embedding_params['ent_embedding_size'])"
        assert list(model_weights["rel_emb"].shape) == [len(self.meta_data["ind2rel"]), self.embedding_params["rel_embedding_size"]], \
            "shape of 'rel_emb' should be (len(meta_data['ind2rel']), embedding_params['rel_embedding_size'])"
        assert list(model_weights["rel_proj"].shape) == [len(self.meta_data["ind2rel"]), self.embedding_params["ent_embedding_size"], self.embedding_params["rel_embedding_size"]], \
            "shape of 'rel_emb' should be (len(meta_data['ind2rel']), embedding_params['ent_embedding_size'], embedding_params['rel_embedding_size'])"

    def score_hrt(self, h, r, t):
        if h is None:
            h = np.arange(len(self.meta_data["ind2ent"]))
        if t is None:
            t = np.arange(len(self.meta_data["ind2ent"]))

        h_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], h), axis=-1)
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], r)
        t_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], t), axis=-1)

        r_proj = tf.nn.embedding_lookup(self.model_weights["rel_proj"], h)

        h_proj = tf.squeeze(tf.matmul(h_emb, r_proj, transpose_a=True))
        t_proj = tf.squeeze(tf.matmul(t_emb, r_proj, transpose_a=True))

        if self.constraint:
            h_proj = clip_constraint(X=h_proj, p=2, axis=-1, value=1)
            t_proj = clip_constraint(X=t_proj, p=2, axis=-1, value=1)

        return self.score_fn(h_proj + r_emb, t_proj, self.score_params)

    def _constraint_loss(self, X):
        if self.constraint:
            self.model_weights["ent_emb"].assign(clip_constraint(X=self.model_weights["ent_emb"], p=2, axis=-1, value=1))
            self.model_weights["rel_emb"].assign(clip_constraint(X=self.model_weights["rel_emb"], p=2, axis=-1, value=1))

        return 0
