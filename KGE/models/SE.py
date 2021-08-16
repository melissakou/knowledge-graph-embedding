import logging
import numpy as np
import tensorflow as tf
from KGE.models.KGEModel import KGEModel

logging.getLogger().setLevel(logging.INFO)

class SE(KGEModel):
        
    def _init_embeddings(self, seed):

        assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using SE"
         
        limit = 6.0 / np.sqrt(self.embedding_params["embedding_size"])
        uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
        ent_emb = tf.Variable(
            uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["embedding_size"]]),
            name="entities_embedding", dtype=np.float32
        )

        xavier_initializer = tf.initializers.GlorotUniform(seed = seed)
        rel_proj_h = tf.Variable(
            xavier_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]]),
            name="relations_projector_h", dtype=np.float32
        )

        rel_proj_t = tf.Variable(
            xavier_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"], self.embedding_params["embedding_size"]]),
            name="relations_projector_t", dtype=np.float32
        )         

        self.model_weights = {"ent_emb": ent_emb, "rel_proj_h": rel_proj_h, "rel_proj_t": rel_proj_t}
    
    def translate(self, X):
        h_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 0]), axis=-1)
        t_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 2]), axis=-1)

        rel_proj_h = tf.nn.embedding_lookup(self.model_weights["rel_proj_h"], X[:, 1])
        rel_proj_t = tf.nn.embedding_lookup(self.model_weights["rel_proj_t"], X[:, 1])

        return tf.squeeze(tf.matmul(rel_proj_h, h_emb)), tf.squeeze(tf.matmul(rel_proj_t, t_emb))