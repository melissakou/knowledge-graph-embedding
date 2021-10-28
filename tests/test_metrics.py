import random
import unittest

from KGE.metrics import mean_reciprocal_rank, mean_rank, median_rank
from KGE.metrics import geometric_mean_rank, harmonic_mean_rank, std_rank, hits_at_k

ranks = [random.randint(1, 1000) for _ in range(10)]

class TestRankMetrics(unittest.TestCase):
    def test_mean_reciprocal_rank(self):
        metric = mean_reciprocal_rank(ranks)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric > 0 and metric <= 1)

    def test_mean_rank(self):
        metric = mean_rank(ranks)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric >= 1)

    def test_median_rank(self):
        metric = median_rank(ranks)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric >= 1)

    def test_geometric_mean_rank(self):
        metric = geometric_mean_rank(ranks)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric >= 1)

    def test_harmonic_mean_rank(self):
        metric = harmonic_mean_rank(ranks)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric >= 1)

    def test_std_rank(self):
        metric = std_rank(ranks)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric >= 0)

    def test_hits_at_k(self):
        metric = hits_at_k(ranks, 1)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric <= 1)

        metric = hits_at_k(ranks, 5)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric <= 1)

        metric = hits_at_k(ranks, 10)
        self.assertIsInstance(metric, float)
        self.assertTrue(metric <= 1)




if __name__ == "__main__":
    unittest.main()


