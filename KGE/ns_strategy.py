import numpy as np
import tensorflow as tf
from itertools import repeat

from .utils import ns_with_same_type

from tensorflow.python.ops.gen_math_ops import Neg

class NegativeSampler:
    def __init__(self):
        raise NotImplementedError("subclass of NegativeSampler should implement __init__() to init class")

    def __call__(self):
        raise NotImplementedError("subclass of NegativeSampler should implement __call_() to perform negative sampling")


class UniformStrategy(NegativeSampler):
    def __init__(self, sample_pool):
        self.sample_pool = sample_pool

    def __call__(self, X, negative_ratio, side):
        self.sample_pool = tf.cast(self.sample_pool, X.dtype)
        sample_index = tf.random.uniform(
            shape=[X.shape[0] * negative_ratio, 1],
            minval=0, maxval=len(self.sample_pool), dtype=self.sample_pool.dtype
        )
        sample_entities = tf.gather_nd(self.sample_pool, sample_index)

        return sample_entities

class TypedStrategy(NegativeSampler):
    def __init__(self, pool, metadata):
        self.pool = pool
        self.metadata = metadata

    def __call__(self, X, negative_ratio, side):

        if side == "h":
            ref_type = X[:, 0].numpy()
        elif side == "t":
            ref_type = X[:, 2].numpy()

        if self.pool is not None:
            sample_entities = self.pool.starmap(
                ns_with_same_type,
                zip(ref_type, repeat(self.metadata), repeat(negative_ratio))
            )
        else:
            sample_entities = list(map(
                lambda x: ns_with_same_type(x, self.metadata, negative_ratio),
                ref_type
            ))

        sample_entities = tf.constant(np.concatenate(sample_entities), dtype=X.dtype)

        return sample_entities