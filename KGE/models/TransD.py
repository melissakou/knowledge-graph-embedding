import logging
import numpy as np
import tensorflow as tf
from KGE.models.KGEModel import KGEModel

logging.getLogger().setLevel(logging.INFO)

class TransD(KGEModel):
        
    def _init_embeddings(self, seed):

        assert self.embedding_params.get("ent_embedding_size") is not None, "'ent_embedding_size' should be given in embedding_params when using TransR"
        assert self.embedding_params.get("rel_embedding_size") is not None, "'rel_embedding_size' should be given in embedding_params when using TransR"
         
        limit = 6.0 / np.sqrt(self.embedding_params["ent_embedding_size"])
        uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
        ent_emb = tf.Variable(
            uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["ent_embedding_size"]]),
            name="entities_embedding", dtype=np.float32
        )
        ent_proj = tf.Variable(
            uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["ent_embedding_size"]]),
            name="entities_projection", dtype=np.float32
        )

        limit = 6.0 / np.sqrt(self.embedding_params["rel_embedding_size"])
        uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
        rel_emb = tf.Variable(
            uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["rel_embedding_size"]]),
            name="relations_embedding", dtype=np.float32
        )
        rel_proj = tf.Variable(
            uniform_initializer([len(self.meta_data["ind2rel"]), self.embedding_params["rel_embedding_size"]]),
            name="relations_projection", dtype=np.float32
        )    

        self.model_weights = {"ent_emb": ent_emb, "rel_emb": rel_emb, "ent_proj": ent_proj, "rel_proj": rel_proj}
    
    def translate(self, X):
        h_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 0]), axis=-1)
        r_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["rel_emb"], X[:, 1]), axis=-1)
        t_emb = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 2]), axis=-1)

        h_proj = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_proj"], X[:, 0]), axis=-1)
        r_proj = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["rel_proj"], X[:, 1]), axis=-1)
        t_proj = tf.expand_dims(tf.nn.embedding_lookup(self.model_weights["ent_proj"], X[:, 2]), axis=-1)

        diag_matrix = tf.linalg.set_diag(
            input=tf.zeros([self.embedding_params["rel_embedding_size"], self.embedding_params["ent_embedding_size"]]),
            diagonal=[1.0] * min(self.embedding_params["rel_embedding_size"], self.embedding_params["ent_embedding_size"])
        )
        h_proj_matrix = tf.matmul(r_proj, tf.transpose(h_proj, perm=[0, 2, 1])) + diag_matrix
        t_proj_matrix = tf.matmul(r_proj, tf.transpose(t_proj, perm=[0, 2, 1])) + diag_matrix

        h_proj = tf.matmul(h_proj_matrix, h_emb)
        t_proj = tf.matmul(t_proj_matrix, t_emb)


        return tf.squeeze(h_proj + r_emb), tf.squeeze(t_proj)