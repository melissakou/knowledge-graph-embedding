import unittest
import numpy as np
import tensorflow as tf

from data import train, val, metadata
from KGE.utils import rmtree
from KGE.data_utils import convert_kg_to_index
from KGE.models.translating_based.RotatE import RotatE
from KGE.models.translating_based.SE import SE
from KGE.models.translating_based.TransD import TransD
from KGE.models.translating_based.TransE import TransE
from KGE.models.translating_based.TransH import TransH
from KGE.models.translating_based.TransR import TransR
from KGE.models.translating_based.UM import UM
from KGE.models.semantic_based.DistMult import DistMult
from KGE.models.semantic_based.RESCAL import RESCAL


class TestBaseModel(unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = TransE(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)

    def test_train(self):
        try:
            self.model.train(train_X=train, val_X=val, metadata=metadata, epochs=1, batch_size=2)
            rmtree(self.model.log_path)
        except Exception as e:
            self.fail("Unexpected exception %s" % e)

    def test_get_rank(self):
        test_sample = convert_kg_to_index(np.array([['DaVinci', 'is_a', 'Person']]),
                                          ent2ind=metadata["ent2ind"], rel2ind=metadata["rel2ind"])
        rank = self.model.get_rank(x=test_sample, positive_X=None, corrupt_side="h")
        rank_filter = self.model.get_rank(x=test_sample, positive_X=np.concatenate((train, val), axis=0), corrupt_side="h")
        self.assertIsInstance(rank, np.int_)
        self.assertIsInstance(rank_filter, np.int_)
        self.assertGreaterEqual(rank, 1)
        self.assertGreaterEqual(rank_filter, 1)
        self.assertLessEqual(rank_filter, rank)

    def test_evaluate(self):
        eval_result = self.model.evaluate(eval_X=val, corrupt_side="h", positive_X=None)
        eval_result_filter = self.model.evaluate(eval_X=val, corrupt_side="h",
                                                 positive_X=np.concatenate((train, val), axis=0))
        metrics = ["mean_rank", "mean_reciprocal_rank", "median_rank", "geometric_mean_rank",
                   "harmonic_mean_rank", "hit@1", "hit@3", "hit@10"]
        assert_test = [self.assertLessEqual, self.assertGreaterEqual, self.assertLessEqual, self.assertLessEqual,
                       self.assertLessEqual, self.assertGreaterEqual, self.assertGreaterEqual, self.assertGreaterEqual]
        
        [test(eval_result_filter[m], eval_result[m]) for test, m in zip(assert_test, metrics)]

class BaseTestModels:

    def test_score_hrt(self):
        # test score batch
        scores = self.model.score_hrt(h=val[:,0], r=val[:,1], t = val[:,2])
        self.assertEqual(len(scores), len(val))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(scores)))

        # test score head
        scores = self.model.score_hrt(h=None, r=val[0,1], t=val[0,2])
        self.assertEqual(len(scores), len(metadata["ind2ent"]))

        # test score tail
        scores = self.model.score_hrt(h=val[0,0], r=val[0,1], t=None)
        self.assertEqual(len(scores), len(metadata["ind2ent"]))


class TestRotatE(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = RotatE(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.assertTrue("ent_emb" in self.model.model_weights)
        self.assertTrue("rel_emb" in self.model.model_weights)
        
        

class TestSE(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = SE(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
        self.assertTrue("ent_emb" in self.model.model_weights)
        self.assertTrue("rel_proj_h" in self.model.model_weights)
        self.assertTrue("rel_proj_t" in self.model.model_weights)

class TestTransD(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"ent_embedding_size": 16, "rel_embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = TransD(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
        self.assertTrue("ent_emb" in self.model.model_weights)
        self.assertTrue("rel_emb" in self.model.model_weights)
        self.assertTrue("ent_proj" in self.model.model_weights)
        self.assertTrue("rel_proj" in self.model.model_weights)

class TestTransE(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = TransE(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
        self.assertTrue("ent_emb" in self.model.model_weights)
        self.assertTrue("rel_emb" in self.model.model_weights)


class TestTransH(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = TransH(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
        self.assertTrue("ent_emb" in self.model.model_weights)
        self.assertTrue("rel_emb" in self.model.model_weights)
        self.assertTrue("rel_hyper" in self.model.model_weights)


class TestTransR(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"ent_embedding_size": 16, "rel_embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = TransR(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
        self.assertTrue("ent_emb" in self.model.model_weights)
        self.assertTrue("rel_emb" in self.model.model_weights)
        self.assertTrue("rel_proj" in self.model.model_weights)


class TestUM(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = UM(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
        self.assertTrue("ent_emb" in self.model.model_weights)

class TestDistMult(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = DistMult(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
        self.assertTrue("ent_emb" in self.model.model_weights)
        self.assertTrue("rel_inter" in self.model.model_weights)


class TestRESCAL(BaseTestModels, unittest.TestCase):

    def setUp(self):
        self.embedding_params = {"embedding_size": 16}
        self.negative_ratio = 2
        self.corrupt_size = "h+t"
        self.model = RESCAL(
            embedding_params=self.embedding_params,
            negative_ratio=self.negative_ratio,
            corrupt_side=self.corrupt_size
        )
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
    
    def test_init_embeddings(self):
        self.model._model_weights_initial = None
        self.model.metadata = metadata
        self.model._init_embeddings(seed=None)
        self.assertTrue("ent_emb" in self.model.model_weights)
        self.assertTrue("rel_inter" in self.model.model_weights)


if __name__ == "__main__":
    unittest.main()



