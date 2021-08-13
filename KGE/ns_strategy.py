import numpy as np
import tensorflow as tf
import multiprocessing as mp

from itertools import repeat
from functools import partial

def ns_with_same_type(x, meta_data, negative_ratio):
    sample_pool = meta_data["type2inds"][meta_data["ind2type"][x]]
    sample_pool = np.delete(sample_pool, np.where(sample_pool == x), axis=0)
    sample_entities = np.random.choice(sample_pool, size=negative_ratio)

    return sample_entities

def uniform_strategy(X, sample_pool, negative_ratio, params):
    sample_index = tf.random.uniform(
        shape=[X.shape[0] * negative_ratio, 1],
        minval=0, maxval=len(sample_pool), dtype=sample_pool.dtype
    )
    sample_entities = tf.gather_nd(sample_pool, sample_index)

    return sample_entities

def typed_strategy(X, sample_pool, negative_ratio, params):

    if params["side"] == "h":
        ref_type = X[:, 0].numpy()
    elif params["side"] == "t":
        ref_type = X[:, 2].numpy()

    if params["n_workers"] > 1:
        pool = mp.Pool(params["n_workers"])
        sample_entities = pool.starmap(
            ns_with_same_type,
            zip(ref_type, repeat(params["meta_data"]), repeat(negative_ratio))
        )
        pool.close()
    else:
        sample_entities = list(map(
            lambda x: ns_with_same_type(x, params["meta_data"], negative_ratio),
            ref_type
        ))

    sample_entities = tf.constant(np.concatenate(sample_entities), dtype=X.dtype)

    return sample_entities
    

    


