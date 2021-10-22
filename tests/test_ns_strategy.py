import random
import unittest
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from data import val, metadata
from KGE.ns_strategy import UniformStrategy, TypedStrategy

negative_ratio = 2
val = tf.constant(val)

class TestUniformStrategy(unittest.TestCase):

    def setUp(self):
        self.ns_sampler = UniformStrategy(tf.range(len(metadata["ind2ent"])))
    
    def test_call(self):
        negative_h = self.ns_sampler(X=val, negative_ratio=negative_ratio, side="h")
        negative_t = self.ns_sampler(X=val, negative_ratio=negative_ratio, side="t")
        self.assertEqual(negative_h.shape, len(val) * negative_ratio)
        self.assertEqual(negative_t.shape, len(val) * negative_ratio)
        self.assertEqual(negative_h.dtype, val.dtype)
        self.assertEqual(negative_t.dtype, val.dtype)


class TestTypedStrategy(unittest.TestCase):

    def setUp(self):
        pool = mp.Pool(2)
        metadata["ind2type"] = ['A'] * (len(metadata["ind2ent"]) // 2) \
                                + ['B'] * (len(metadata["ind2ent"]) - len(metadata["ind2ent"]) // 2)
        metadata["type2inds"] = {}
        all_type = np.unique(metadata["ind2type"])
        for t in all_type:
            indices = [i for (i, ti) in enumerate(metadata["ind2type"]) if ti == t]
            metadata["type2inds"][t] = np.array(indices)

        self.h_type = [metadata["ind2type"][x] for x in tf.repeat(val[:,0], negative_ratio)]
        self.t_type = [metadata["ind2type"][x] for x in tf.repeat(val[:,2], negative_ratio)]

        self.ns_sampler = TypedStrategy(pool=None, metadata=metadata)
        self.ns_sampler_pool = TypedStrategy(pool=pool, metadata=metadata)
    
    def test_call(self):
        negative_h = self.ns_sampler(X=val, negative_ratio=negative_ratio, side="h")
        negative_t = self.ns_sampler(X=val, negative_ratio=negative_ratio, side="t")
        self.assertEqual(negative_h.shape, len(val) * negative_ratio)
        self.assertEqual(negative_t.shape, len(val) * negative_ratio)
        self.assertEqual(negative_h.dtype, val.dtype)
        self.assertEqual(negative_t.dtype, val.dtype)
        self.assertEqual(self.h_type, [metadata["ind2type"][x] for x in negative_h])
        self.assertEqual(self.t_type, [metadata["ind2type"][x] for x in negative_t])

        negative_h = self.ns_sampler_pool(X=val, negative_ratio=negative_ratio, side="h")
        negative_t = self.ns_sampler_pool(X=val, negative_ratio=negative_ratio, side="t")
        self.assertEqual(negative_h.shape, len(val) * negative_ratio)
        self.assertEqual(negative_t.shape, len(val) * negative_ratio)
        self.assertEqual(negative_h.dtype, val.dtype)
        self.assertEqual(negative_t.dtype, val.dtype)
        self.assertEqual(self.h_type, [metadata["ind2type"][x] for x in negative_h])
        self.assertEqual(self.t_type, [metadata["ind2type"][x] for x in negative_t])


if __name__ == "__main__":
    unittest.main()
    