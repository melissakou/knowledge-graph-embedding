import unittest
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from data import train, val, metadata
from KGE import score
from KGE import loss
from KGE import ns_strategy
from KGE.utils import rmtree
from KGE.models.translating_based.RotatE import RotatE
from KGE.models.translating_based.SE import SE
from KGE.models.translating_based.TransD import TransD
from KGE.models.translating_based.TransE import TransE
from KGE.models.translating_based.TransH import TransH
from KGE.models.translating_based.TransR import TransR
from KGE.models.translating_based.UM import UM
from KGE.models.semantic_based.DistMult import DistMult
from KGE.models.semantic_based.RESCAL import RESCAL

class TestIntegration(unittest.TestCase):

    def setUp(self):
        pool = mp.Pool(2)
        metadata["ind2type"] = ['A'] * (len(metadata["ind2ent"]) // 2) \
                                + ['B'] * (len(metadata["ind2ent"]) - len(metadata["ind2ent"]) // 2)
        metadata["type2inds"] = {}
        all_type = np.unique(metadata["ind2type"])
        for t in all_type:
            indices = [i for (i, ti) in enumerate(metadata["ind2type"]) if ti == t]
            metadata["type2inds"][t] = np.array(indices)
        self.all_scores = [score.LpDistance(p=2), score.LpDistancePow(p=2), score.Dot()]
        self.all_losses = [loss.PairwiseHingeLoss(margin=1.0), loss.PairwiseLogisticLoss(),
                           loss.BinaryCrossEntropyLoss(), loss.SquareErrorLoss(),
                           loss.SelfAdversarialNegativeSamplingLoss(margin=1.0, temperature=1.0)]
        self.all_ns_strategy = [ns_strategy.UniformStrategy(tf.range(len(metadata["ind2ent"]))),
                                ns_strategy.TypedStrategy(pool=pool, metadata=metadata)]

        self.ent_emb_dim, self.rel_emb_dim = 16, 16
        self.negative_ratio = 2
        self.corrupt_side = "h+t"
        self.epochs = 1
        self.batch_size = 4
    
    def test_RotatE(self):
        for score in self.all_scores[:2]:
            for loss in self.all_losses:
                for ns_strategy in self.all_ns_strategy:
                    model = RotatE(
                        embedding_params={"embedding_size": self.ent_emb_dim},
                        negative_ratio=self.negative_ratio,
                        corrupt_side=self.corrupt_side,
                        score_fn=score,
                        loss_fn=loss,
                        ns_strategy=ns_strategy
                    )
                    model.train(train_X=train, val_X=val, metadata=metadata,
                                epochs=self.epochs, batch_size=self.batch_size,
                                early_stopping_rounds=None, restore_best_weight=False,
                                optimizer=tf.optimizers.Adam(),
                                seed=12345, log_path="./tmp", log_projector=True)
                    model.evaluate(eval_X=val, corrupt_side="h")

    def test_SE(self):
        for score in self.all_scores:
            for loss in self.all_losses:
                for ns_strategy in self.all_ns_strategy:
                    model = SE(
                        embedding_params={"embedding_size": self.ent_emb_dim},
                        negative_ratio=self.negative_ratio,
                        corrupt_side=self.corrupt_side,
                        score_fn=score,
                        loss_fn=loss,
                        ns_strategy=ns_strategy
                    )
                    model.train(train_X=train, val_X=val, metadata=metadata,
                                epochs=self.epochs, batch_size=self.batch_size,
                                early_stopping_rounds=None, restore_best_weight=False,
                                optimizer=tf.optimizers.Adam(),
                                seed=12345, log_path="./tmp", log_projector=True)
                    model.evaluate(eval_X=val, corrupt_side="h")

    def test_TransD(self):
        for score in self.all_scores:
            for loss in self.all_losses:
                for ns_strategy in self.all_ns_strategy:
                    model = TransD(
                        embedding_params={"ent_embedding_size": self.ent_emb_dim,
                                          "rel_embedding_size": self.rel_emb_dim},
                        negative_ratio=self.negative_ratio,
                        corrupt_side=self.corrupt_side,
                        score_fn=score,
                        loss_fn=loss,
                        ns_strategy=ns_strategy
                    )
                    model.train(train_X=train, val_X=val, metadata=metadata,
                                epochs=self.epochs, batch_size=self.batch_size,
                                early_stopping_rounds=None, restore_best_weight=False,
                                optimizer=tf.optimizers.Adam(),
                                seed=12345, log_path="./tmp", log_projector=True)
                    model.evaluate(eval_X=val, corrupt_side="h")

    def test_TransE(self):
        for score in self.all_scores:
            for loss in self.all_losses:
                for ns_strategy in self.all_ns_strategy:
                    model = TransE(
                        embedding_params={"embedding_size": self.ent_emb_dim},
                        negative_ratio=self.negative_ratio,
                        corrupt_side=self.corrupt_side,
                        score_fn=score,
                        loss_fn=loss,
                        ns_strategy=ns_strategy
                    )
                    model.train(train_X=train, val_X=val, metadata=metadata,
                                epochs=self.epochs, batch_size=self.batch_size,
                                early_stopping_rounds=None, restore_best_weight=False,
                                optimizer=tf.optimizers.Adam(),
                                seed=12345, log_path="./tmp", log_projector=True)
                    model.evaluate(eval_X=val, corrupt_side="h")

    def test_TransH(self):
        for score in self.all_scores:
            for loss in self.all_losses:
                for ns_strategy in self.all_ns_strategy:
                    model = TransH(
                        embedding_params={"embedding_size": self.ent_emb_dim},
                        negative_ratio=self.negative_ratio,
                        corrupt_side=self.corrupt_side,
                        score_fn=score,
                        loss_fn=loss,
                        ns_strategy=ns_strategy
                    )
                    model.train(train_X=train, val_X=val, metadata=metadata,
                                epochs=self.epochs, batch_size=self.batch_size,
                                early_stopping_rounds=None, restore_best_weight=False,
                                optimizer=tf.optimizers.Adam(),
                                seed=12345, log_path="./tmp", log_projector=True)
                    model.evaluate(eval_X=val, corrupt_side="h")

    def test_TransR(self):
        for score in self.all_scores:
            for loss in self.all_losses:
                for ns_strategy in self.all_ns_strategy:
                    model = TransR(
                        embedding_params={"ent_embedding_size": self.ent_emb_dim,
                                          "rel_embedding_size": self.rel_emb_dim},
                        negative_ratio=self.negative_ratio,
                        corrupt_side=self.corrupt_side,
                        score_fn=score,
                        loss_fn=loss,
                        ns_strategy=ns_strategy
                    )
                    model.train(train_X=train, val_X=val, metadata=metadata,
                                epochs=self.epochs, batch_size=self.batch_size,
                                early_stopping_rounds=None, restore_best_weight=False,
                                optimizer=tf.optimizers.Adam(),
                                seed=12345, log_path="./tmp", log_projector=True)
                    model.evaluate(eval_X=val, corrupt_side="h")

    def test_UM(self):
        for score in self.all_scores:
            for loss in self.all_losses:
                for ns_strategy in self.all_ns_strategy:
                    model = UM(
                        embedding_params={"embedding_size": self.ent_emb_dim},
                        negative_ratio=self.negative_ratio,
                        corrupt_side=self.corrupt_side,
                        score_fn=score,
                        loss_fn=loss,
                        ns_strategy=ns_strategy
                    )
                    model.train(train_X=train, val_X=val, metadata=metadata,
                                epochs=self.epochs, batch_size=self.batch_size,
                                early_stopping_rounds=None, restore_best_weight=False,
                                optimizer=tf.optimizers.Adam(),
                                seed=12345, log_path="./tmp", log_projector=True)
                    model.evaluate(eval_X=val, corrupt_side="h")

    def test_DistMult(self):
        for loss in self.all_losses:
            for ns_strategy in self.all_ns_strategy:
                model = DistMult(
                    embedding_params={"embedding_size": self.ent_emb_dim},
                    negative_ratio=self.negative_ratio,
                    corrupt_side=self.corrupt_side,
                    loss_fn=loss,
                    ns_strategy=ns_strategy
                )
                model.train(train_X=train, val_X=val, metadata=metadata,
                            epochs=self.epochs, batch_size=self.batch_size,
                            early_stopping_rounds=None, restore_best_weight=False,
                            optimizer=tf.optimizers.Adam(),
                            seed=12345, log_path="./tmp", log_projector=True)
                model.evaluate(eval_X=val, corrupt_side="h")

    def test_RESCAL(self):
        for loss in self.all_losses:
            for ns_strategy in self.all_ns_strategy:
                model = RESCAL(
                    embedding_params={"embedding_size": self.ent_emb_dim},
                    negative_ratio=self.negative_ratio,
                    corrupt_side=self.corrupt_side,
                    loss_fn=loss,
                    ns_strategy=ns_strategy
                )
                model.train(train_X=train, val_X=val, metadata=metadata,
                            epochs=self.epochs, batch_size=self.batch_size,
                            early_stopping_rounds=None, restore_best_weight=False,
                            optimizer=tf.optimizers.Adam(),
                            seed=12345, log_path="./tmp", log_projector=True)
                model.evaluate(eval_X=val, corrupt_side="h")

    def tearDown(self):
        rmtree("./tmp")


if __name__ == "__main__":
    unittest.main()
