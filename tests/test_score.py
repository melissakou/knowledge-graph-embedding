import scipy
import unittest
import numpy as np
import tensorflow as tf
from KGE.score import LpDistance, LpDistancePow, Dot

x = tf.random.uniform([10, 5], -1, 1)
y = tf.random.uniform([10, 5], -1, 1)
x_comp = tf.complex(x, x)
y_comp = tf.complex(y, y)

class TestLpDistance(unittest.TestCase):

    def setUp(self):
        self.score_1 = LpDistance(p=1)
        self.score_2 = LpDistance(p=2)
        self.score_inf = LpDistance(p=np.inf)

    def test_call(self):
        scores = [self.score_1(x, y), self.score_2(x, y), self.score_inf(x, y),
                  self.score_1(x_comp, y_comp), self.score_2(x_comp, y_comp), self.score_inf(x_comp, y_comp)]
        [self.assertTrue(tf.reduce_all(s <= 0)) for s in scores]
        [self.assertEqual(len(s), 10) for s in scores]


class TestLpDistancePow(unittest.TestCase):

    def setUp(self):
        self.score_1 = LpDistancePow(p=1)
        self.score_2 = LpDistancePow(p=2)
        self.score_inf = LpDistancePow(p=np.inf)

    def test_call(self):
        scores = [self.score_1(x, y), self.score_2(x, y), self.score_inf(x, y),
                  self.score_1(x_comp, y_comp), self.score_2(x_comp, y_comp), self.score_inf(x_comp, y_comp)]
        [self.assertTrue(tf.reduce_all(s <= 0)) for s in scores]
        [self.assertTrue(tf.reduce_all(tf.math.is_finite(s))) for s in scores]
        [self.assertEqual(len(s), 10) for s in scores]


class TestDot(unittest.TestCase):

    def setUp(self):
        self.score = Dot()

    def test_call(self):
        score = self.score(x, y)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(score)))
        self.assertEqual(len(score), 10)


if __name__ == "__main__":
    unittest.main()
