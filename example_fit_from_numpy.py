import json
import random
import numpy as np
import tensorflow as tf
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.models.translating_based.TransE import TransE


if __name__ == "__main__":
    train = np.loadtxt("./data/fb15k/train/train.csv", dtype=str, delimiter=',')
    valid = np.loadtxt("./data/fb15k/valid/valid.csv", dtype=str, delimiter=',')
    test = np.loadtxt("./data/fb15k/test/test.csv", dtype=str, delimiter=',')

    metadata = index_kg(train)
    metadata["ind2type"] = random.choices(["A", "B", "C"], k=len(metadata["ind2ent"]))
    
    train = convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])
    valid = convert_kg_to_index(valid, metadata["ent2ind"], metadata["rel2ind"])
    test = convert_kg_to_index(test, metadata["ent2ind"], metadata["rel2ind"])


    model = TransE(
        embedding_params={"embedding_size": 10},
        negative_ratio=4,
        corrupt_side="h+t"
    )
    model.train(train_X=train, val_X=valid, metadata=metadata, epochs=2, batch_size=512,
                early_stopping_rounds=None, restore_best_weight=False,
                optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                seed=12345, log_path="./tensorboard_logs", log_projector=True)

    eval_result_filtered = model.evaluate(eval_X=test, corrupt_side="h", positive_X=np.concatenate((train, valid, test), axis=0))
    print(eval_result_filtered)