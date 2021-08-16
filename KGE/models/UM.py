import logging
import numpy as np
import tensorflow as tf
from KGE.models.KGEModel import KGEModel

logging.getLogger().setLevel(logging.INFO)

class UM(KGEModel):
        
    def _init_embeddings(self, seed):
         
         assert self.embedding_params.get("embedding_size") is not None, "'embedding_size' should be given in embedding_params when using UM"

         limit = 6.0 / np.sqrt(self.embedding_params["embedding_size"])
         uniform_initializer = tf.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)
         ent_emb = tf.Variable(
             uniform_initializer([len(self.meta_data["ind2ent"]), self.embedding_params["embedding_size"]]),
             name="entities_embedding", dtype=np.float32
         )

         self.model_weights = {"ent_emb": ent_emb}
    
    def translate(self, X):
        h_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 0])
        t_emb = tf.nn.embedding_lookup(self.model_weights["ent_emb"], X[:, 2])

        return h_emb, t_emb