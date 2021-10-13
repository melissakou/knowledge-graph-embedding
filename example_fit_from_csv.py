import random
import numpy as np
import tensorflow as tf
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.models.translating_based.RotatE import RotatE
from KGE.loss import BinaryCrossEntropyLoss

if __name__ == "__main__":
    train = "./data/fb15k/train"
    valid = "./data/fb15k/valid"

    metadata = index_kg(train)
    metadata["ind2type"] = random.choices(["A", "B", "C"], k=len(metadata["ind2ent"]))


    convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])
    convert_kg_to_index(valid, metadata["ent2ind"], metadata["rel2ind"])
    train = train + "_indexed"
    valid = valid + "_indexed"

    model = RotatE(
        embedding_params={"embedding_size": 2},
        negative_ratio=1,
        corrupt_side="h+t",
        score_params={"p": 1},
        loss_fn=BinaryCrossEntropyLoss())
    model.train(train_X=train, val_X=valid, metadata=metadata, epochs=1, batch_size=256,
                early_stopping_rounds=None, restore_best_weight=False,
                optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                seed=12345, log_path="./tensorboard_logs", log_projector=True)