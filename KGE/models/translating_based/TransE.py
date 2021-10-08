"""TransE"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import p_norm
from ...loss import pairwise_hinge_loss
from ...ns_strategy import uniform_strategy
from ...constraint import normalized_embeddings

logging.getLogger().setLevel(logging.INFO)

class TransE(TranslatingModel):

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=p_norm, score_params={"p": 2}, loss_fn=pairwise_hinge_loss, loss_param={"margin": 1},
                 ns_strategy=uniform_strategy, constraint=True, n_workers=1):
        super(TransE, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                     score_fn, score_params, loss_fn, loss_param,
                                     ns_strategy, constraint, n_workers)
        
    def _init_embeddings(self, seed):

        if self.__model_weights_initial is None:
            assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using TransE"
            
            limit = 6.0 / np.sqrt(self.embedding_params["embedding_size"])
            uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
            ent_emb = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["embedding_size"]]),
                name="entities_embedding", dtype=np.float32
            )
            rel_emb = tf.Variable(
                uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"]]),
                name="relations_embedding", dtype=np.float32
            )

            self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb}
        else:
            self._check_model_weights(self.__model_weights_initial)
            self.model_weights = self.__model_weights_initial

        if self.constraint:
            self.model_weights["rel_emb"].assign(normalized_embeddings(X=rel_emb, p=2, value=1, axis=1))

    def _check_model_weights(self, model_weights):
        assert model_weights.get("ent_emb") is not None, "entity embedding should be given in model_weights with key 'ent_emb'"
        assert model_weights.get("rel_emb") is not None, "relation embedding should be given in model_weights with key 'rel_emb'"
        assert list(model_weights["ent_emb"].shape) == [len(self.meta_data["ind2ent"]), self.embedding_params["embedding_size"]], \
            "shape of 'ent_emb' should be (len(meta_data['ind2ent']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_emb"].shape) == [len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"]], \
            "shape of 'rel_emb' should be (len(meta_data['ind2rel']), embedding_params['embedding_size'])"

    def score_hrt(self, h, r, t):
        if h is None:
            h = np.arange(len(self.meta_data["ind2ent"]))
        if t is None:
            t = np.arange(len(self.meta_data["ind2ent"]))
        
        h_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], h)
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], r)
        t_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], t)

        return self.score_fn(h_emb + r_emb, t_emb, self.score_params)

    def _constraint_loss(self, X):
        if self.constraint:
            self.model_weights["ent_emb"].assign(normalized_embeddings(X=self.model_weights["ent_emb"], p=2, axis=1, value=1))

        return 0