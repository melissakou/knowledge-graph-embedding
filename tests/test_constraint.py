import scipy
import unittest
import numpy as np
import tensorflow as tf

from KGE.constraint import normalized_embeddings, soft_constraint, clip_constraint, Lp_regularization

X = tf.random.uniform([10, 5])
value = 1

class TestNormalizedEmbeddings(unittest.TestCase):

    def test_normalized_embeddings(self):
        X_norm_1 = normalized_embeddings(X=X, p=1, value=value, axis=-1)
        X_norm_2 = normalized_embeddings(X=X, p=2, value=value, axis=-1)
        X_norm_inf = normalized_embeddings(X=X, p=np.inf, value=value, axis=-1)
        self.assertEqual(X.shape, X_norm_1.shape)
        self.assertEqual(X.shape, X_norm_2.shape)
        self.assertEqual(X.shape, X_norm_inf.shape)
        self.assertTrue(tf.reduce_all(tf.abs(tf.norm(X_norm_1, 1, axis=-1)-value) <= 1e-6))
        self.assertTrue(tf.reduce_all(tf.abs(tf.norm(X_norm_2, 2, axis=-1)-value) <= 1e-6))
        self.assertTrue(tf.reduce_all(tf.abs(tf.norm(X_norm_inf, np.inf, axis=-1)-value) <= 1e-6))


class TestSoftConstraint(unittest.TestCase):

    def test_soft_constraint(self):
        constraint_1 = soft_constraint(X=X, p=1, value=value, axis=-1)
        constraint_2 = soft_constraint(X=X, p=2, value=value, axis=-1)
        constraint_inf = soft_constraint(X=X, p=np.inf, value=value, axis=-1)
        self.assertEqual(len(constraint_1.shape), 0)
        self.assertEqual(len(constraint_2.shape), 0)
        self.assertEqual(len(constraint_inf.shape), 0)
        self.assertGreaterEqual(constraint_1.numpy(), 0)
        self.assertGreaterEqual(constraint_2.numpy(), 0)
        self.assertGreaterEqual(constraint_inf.numpy(), 0)
        self.assertTrue(tf.math.is_finite(constraint_1))
        self.assertTrue(tf.math.is_finite(constraint_2))
        self.assertTrue(tf.math.is_finite(constraint_inf))


class TestClipConstraint(unittest.TestCase):

    def test_clip_constraint(self):
        X_norm_1 = clip_constraint(X=X, p=1, value=value, axis=-1)
        X_norm_2 = clip_constraint(X=X, p=2, value=value, axis=-1)
        X_norm_inf = clip_constraint(X=X, p=np.inf, value=value, axis=-1)
        self.assertEqual(X.shape, X_norm_1.shape)
        self.assertEqual(X.shape, X_norm_2.shape)
        self.assertEqual(X.shape, X_norm_inf.shape)
        self.assertTrue(tf.reduce_all(tf.norm(X_norm_1, 1, axis=-1)-value <= 1e-6))
        self.assertTrue(tf.reduce_all(tf.norm(X_norm_2, 2, axis=-1)-value <= 1e-6))
        self.assertTrue(tf.reduce_all(tf.norm(X_norm_inf, np.inf, axis=-1)-value <= 1e-6))


class TestLpRegularization(unittest.TestCase):

    def test_Lp_regularization(self):
        constraint_1 = Lp_regularization(X=X, p=1, axis=-1)
        constraint_2 = Lp_regularization(X=X, p=2, axis=-1)
        self.assertEqual(constraint_1.shape, 10)
        self.assertEqual(constraint_2.shape, 10)
        self.assertTrue(tf.reduce_all(constraint_1 >= 0))
        self.assertTrue(tf.reduce_all(constraint_1 >= 0))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(constraint_1)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(constraint_2)))


if __name__ == "__main__":
    unittest.main()


