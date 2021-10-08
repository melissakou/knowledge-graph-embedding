import random
import numpy as np
import tensorflow as tf
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.models.TransH import TransH

from KGE.score import p_norm, dot
from KGE.loss import pairwise_hinge_loss, binary_cross_entropy_loss
from KGE.ns_strategy import uniform_strategy, typed_strategy

if __name__ == "__main__":
    train = "./data/fb15k_237/train"
    valid = "./data/fb15k_237/valid"

    meta_data = index_kg(train)
    meta_data["ind2type"] = random.choices(["A", "B", "C"], k=len(meta_data["ind2ent"]))


    convert_kg_to_index(train, meta_data["ent2ind"], meta_data["rel2ind"])
    convert_kg_to_index(valid, meta_data["ent2ind"], meta_data["rel2ind"])
    train = train + "_indexed"

    valid = valid + "_indexed"

    model = TransH(
        embedding_params={"embedding_size": 50},
        negative_ratio=20,
        corrupt_side="h+t",
        loss_param={"margin": 0.5},
        constraint_weight=0.015625)

    model.train(train_X=train, val_X=valid, meta_data=meta_data, epochs=500, batch_size=1200,
                early_stopping_rounds=None, restore_best_weight=True,
                opt=tf.optimizers.SGD(learning_rate=0.005),
                seed=12345, log_path="./tensorboard_logs_TransH", log_projector=True)