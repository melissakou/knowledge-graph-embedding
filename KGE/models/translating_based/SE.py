"""SE"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import p_norm
from ...loss import pairwise_hinge_loss
from ...ns_strategy import uniform_strategy
from ...constraint import normalized_embeddings

logging.getLogger().setLevel(logging.INFO)

class SE(TranslatingModel):

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=p_norm, score_params={"p": 1}, loss_fn=pairwise_hinge_loss, loss_param={"margin": 1},
                 ns_strategy=uniform_strategy, constraint=True, n_workers=1):
        super(SE, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                 score_fn, score_params, loss_fn, loss_param,
                                 ns_strategy, constraint, n_workers)
        
    def _init_embeddings(self, seed):

        if self.__model_weights_initial is None:
            assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using SE"
            
            limit = np.sqrt(6.0 / self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["embedding_size"]]),
                name="entities_embedding", dtype=np.float32
            )

            limit = np.sqrt(3.0 / self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            rel_proj_h = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]]),
                name="relations_projector_h", dtype=np.float32
            )

            rel_proj_t = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]]),
                name="relations_projector_t", dtype=np.float32
            )         

            self.model_weights = {"ent_emb": ent_emb, "rel_proj_h": rel_proj_h, "rel_proj_t": rel_proj_t}
        else:
            self._check_model_weights(self.__model_weights_initial)
            self.model_weights = self.__model_weights_initial

    def _check_model_weights(self, model_weights):
        assert model_weights.get("ent_emb") is not None, "entity embedding should be given in model_weights with key 'ent_emb'"
        assert model_weights.get("rel_proj_h") is not None, "relation projection matrix(head) should be given in model_weights with key 'rel_proj_h'"
        assert model_weights.get("rel_proj_t") is not None, "relation projection matrix(tail) should be given in model_weights with key 'rel_proj_t'"
        assert list(model_weights["ent_emb"].shape) == [len(self.meta_data["ind2ent"]), self.embedding_params["embedding_size"]], \
            "shape of 'ent_emb' should be (len(meta_data['ind2ent']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_proj_h"].shape) == [len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]], \
            "shape of 'rel_proj_h' should be (len(meta_data['ind2rel']), embedding_params['embedding_size'], embedding_params['embedding_size'])"
        assert list(model_weights["rel_proj_t"].shape) == [len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]], \
            "shape of 'rel_proj_t' should be (len(meta_data['ind2rel']), embedding_params['embedding_size'], embedding_params['embedding_size'])"

    def score_hrt(self, h, r, t):
        if h is None:
            h = np.arange(len(self.meta_data["ind2ent"]))
        if t is None:
            t = np.arange(len(self.meta_data["ind2ent"]))

        h_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], h), axis=-1)
        t_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], t), axis=-1)

        rel_proj_h = tf.nn.embedding_lookup(self.model_weights["rel_proj_h"], r)
        rel_proj_t = tf.nn.embedding_lookup(self.model_weights["rel_proj_t"], r)

        return self.score_fn(tf.squeeze(tf.matmul(rel_proj_h, h_emb)), tf.squeeze(tf.matmul(rel_proj_t, t_emb)), self.score_params)

    def _constraint_loss(self, X):
        if self.constraint:
            self.model_weights["ent_emb"].assign(normalized_embeddings(X=self.model_weights["ent_emb"], p=2, axis=1, value=1))

        return 0