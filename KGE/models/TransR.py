import logging
import numpy as np
import tensorflow as tf
from KGE.models.KGEModel import KGEModel

logging.getLogger().setLevel(logging.INFO)

class TransR(KGEModel):
        
    def _init_embeddings(self, seed):

        assert self.embedding_params.get("ent_embedding_size") is not None, "'ent_embedding_size' should be given in embedding_params when using TransR"
        assert self.embedding_params.get("rel_embedding_size") is not None, "'rel_embedding_size' should be given in embedding_params when using TransR"
         
        limit = 6.0 / np.sqrt(self.embedding_params["ent_embedding_size"])
        uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
        ent_emb = tf.Variable(
            uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["ent_embedding_size"]]),
            name="entities_embedding", dtype=np.float32
        )

        limit = 6.0 / np.sqrt(self.embedding_params["rel_embedding_size"])
        uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
        rel_emb = tf.Variable(
            uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["rel_embedding_size"]]),
            name="relations_embedding", dtype=np.float32
        )

        xavier_initializer = tf.initializers.GlorotUniform(seed = seed)
        rel_proj = tf.Variable(
            xavier_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["rel_embedding_size"], self.embedding_params["ent_embedding_size"]]),
            name="relations_projector", dtype=np.float32
        )     

        self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb, "rel_proj": rel_proj}
    
    def translate(self, X):
        h_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 0]), axis=-1)
        r_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["rel_emb"], X[:, 1]), axis=-1)
        t_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 2]), axis=-1)

        r_proj = tf.nn.embedding_lookup(self.model_weights["rel_proj"], X[:, 1])

        h_proj = tf.matmul(r_proj, h_emb)
        t_proj = tf.matmul(r_proj, t_emb)


        return tf.squeeze(h_proj + r_emb), tf.squeeze(t_proj)