import logging
import numpy as np
import tensorflow as tf
from KGE.models.KGEModel import KGEModel

logging.getLogger().setLevel(logging.INFO)

class TransH(KGEModel):
        
    def _init_embeddings(self, seed):

        assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using TransH"
         
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
        rel_hyper = tf.Variable(
            uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["embedding_size"]]),
            name="relations_hyperplane", dtype=np.float32
        )

        self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb, "rel_hyper": rel_hyper}
    
    def translate(self, X):
        h_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 0])
        r_emb = tf.nn.embedding_lookup(self.model_weights["rel_emb"], X[:, 1])
        r_hyper = tf.nn.embedding_lookup(self.model_weights["rel_hyper"], X[:, 1])
        t_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 2])

        h_emb = tf.expand_dims(h_emb, axis=-1)
        r_hyper = tf.expand_dims(r_hyper, axis=-1)
        t_emb = tf.expand_dims(t_emb, axis=-1)

        h_proj = tf.squeeze(tf.multiply(tf.matmul(r_hyper, h_emb, transpose_a=True), h_emb))
        t_proj = tf.squeeze(tf.multiply(tf.matmul(r_hyper, t_emb, transpose_a=True), t_emb))

        return h_proj + r_emb, t_proj