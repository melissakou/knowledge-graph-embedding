"""RotatE"""

import logging
import numpy as np
import tensorflow as tf
from ..base_model.TranslatingModel import TranslatingModel
from ...score import p_norm
from ...loss import self_adversarial_negative_sampling_loss
from ...ns_strategy import uniform_strategy

logging.getLogger().setLevel(logging.INFO)

class RotatE(TranslatingModel):

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn=p_norm, score_params={"p": 1},
                 loss_fn=self_adversarial_negative_sampling_loss, loss_param={"margin":3, "temperature": 1},
                 ns_strategy=uniform_strategy, constraint=True, n_workers=1):
        super(RotatE, self).__init__(embedding_params, negative_ratio, corrupt_side,
                                     score_fn, score_params, loss_fn, loss_param,
                                     ns_strategy, constraint, n_workers)
        
    def _init_embeddings(self, seed):

        if self.__model_weights_initial is None:
            assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using TransE"
            
            self.limit = (self.loss_params["margin"] + 2.0) / self.embedding_params["embedding_size"]
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
        assert model_weights.get("ent_emb") is not None, "entity embedding should be given in model_weights with key 'ent_emb'"
        assert model_weights.get("rel_emb") is not None, "relation embedding should be given in model_weights with key 'rel_emb'"
        assert list(model_weights["ent_emb"].shape) == [len(self.metadata["ind2ent"]), self.embedding_params["embedding_size"], 2], \
            "shape of 'ent_emb' should be (len(metadata['ind2ent']), embedding_params['embedding_size'])"
        assert list(model_weights["rel_emb"].shape) == [len(self.metadata["ind2rel"]), self.embedding_params["embedding_size"]], \
            "shape of 'rel_emb' should be (len(metadata['ind2rel']), embedding_params['embedding_size'])"

    def score_hrt(self, h, r, t):
        if h is None:
            h = np.arange(len(self.metadata["ind2ent"]))
        if t is None:
            t = np.arange(len(self.metadata["ind2ent"]))

        h_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], h)
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], r)
        t_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], t)

        if len(h_emb.shape) == 2:
            h_emb = tf.expand_dims(h_emb, 0)
        if len(t_emb.shape) == 2:
            t_emb = tf.expand_dims(t_emb, 0)

        # normalize to [-pi, pi] to ensure sin & cos functions are one-to-one
        # r_emb = (r_emb - tf.reduce_min(r_emb)) / (tf.reduce_max(r_emb) - tf.reduce_min(r_emb)) * 2 * np.pi - np.pi
        r_emb = r_emb / self.limit * np.pi
        
        hadamard = tf.multiply(tf.complex(h_emb[:,:,0], h_emb[:,:,1]),
                               tf.complex(tf.math.cos(r_emb), tf.math.sin(r_emb)))
        
        return self.score_fn(hadamard, tf.complex(t_emb[:,:,0], t_emb[:,:,1]), self.score_params)  
    
    def _constraint_loss(self, X):
        return 0