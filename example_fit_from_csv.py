import random
import numpy as np
import tensorflow as tf
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.models.translating_based.RotatE import RotatE

if __name__ == "__main__":
    train = "./data/fb15k_237/train"
    valid = "./data/fb15k_237/valid"

    meta_data = index_kg(train)
    meta_data["ind2type"] = random.choices(["A", "B", "C"], k=len(meta_data["ind2ent"]))


    convert_kg_to_index(train, meta_data["ent2ind"], meta_data["rel2ind"])
    convert_kg_to_index(valid, meta_data["ent2ind"], meta_data["rel2ind"])
    train = train + "_indexed"
    valid = valid + "_indexed"

    model = RotatE(
        embedding_params={"embedding_size": 1000},
        negative_ratio=128,
        corrupt_side="h+t",
        score_params={"p": 1},
        loss_param={"margin":24, "temperature": 1})
    model.train(train_X=train, val_X=valid, metadata=metadata, epochs=1000, batch_size=512,
                early_stopping_rounds=None, restore_best_weight=False,
                optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                seed=12345, log_path="./tensorboard_logs", log_projector=True)