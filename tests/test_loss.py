import scipy
import unittest
import tensorflow as tf
from KGE.loss import PairwiseHingeLoss, PairwiseLogisticLoss
from KGE.loss import BinaryCrossEntropyLoss, SelfAdversarialNegativeSamplingLoss
from KGE.loss import SquareErrorLoss

test_sample = 10
negative_ratio = 2
pos_scores = tf.random.uniform([test_sample])
neg_scores = tf.random.uniform([test_sample * negative_ratio])

class TestPairwiseHingeLoss(unittest.TestCase):

    def setUp(self):
        self.loss = PairwiseHingeLoss(margin=1)

    def test_call(self):
        loss = self.loss(pos_scores, neg_scores)
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(tf.math.is_finite(loss))
        self.assertTrue(loss >= 0)


class TestPairwiseLogisticLoss(unittest.TestCase):

    def setUp(self):
        self.loss = PairwiseLogisticLoss()

    def test_call(self):
        loss = self.loss(pos_scores, neg_scores)
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(tf.math.is_finite(loss))
        self.assertTrue(loss >= 0)


class TestBinaryCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.loss = BinaryCrossEntropyLoss()

    def test_call(self):
        loss = self.loss(pos_scores, neg_scores)
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(tf.math.is_finite(loss))
        self.assertTrue(loss >= 0)


class TestSelfAdversarialNegativeSamplingLoss(unittest.TestCase):

    def setUp(self):
        self.loss = SelfAdversarialNegativeSamplingLoss(margin=1.0, temperature=1.0)

    def test_call(self):
        loss = self.loss(pos_scores, neg_scores)
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(tf.math.is_finite(loss))
        self.assertTrue(loss >= 0)


class TestSquareErrorLoss(unittest.TestCase):

    def setUp(self):
        self.loss = SquareErrorLoss()

    def test_call(self):
        loss = self.loss(pos_scores, neg_scores)
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(tf.math.is_finite(loss))
        self.assertTrue(loss >= 0)





if __name__ == "__main__":
    unittest.main()


