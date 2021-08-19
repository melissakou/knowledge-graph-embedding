import random
import numpy as np
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.models.UM import UM
from KGE.models.SE import SE
from KGE.models.TransE import TransE
from KGE.models.TransH import TransH
from KGE.models.TransR import TransR
from KGE.models.TransD import TransD
from KGE.models.RotatE import RotatE

from KGE.score import p_norm, dot
from KGE.loss import pairwise_hinge_loss, binary_cross_entropy_loss
from KGE.ns_strategy import uniform_strategy, typed_strategy

train = np.loadtxt("./data/fb15k_237/train/train.csv", dtype=str, delimiter=',')
valid = np.loadtxt("./data/fb15k_237/valid/valid.csv", dtype=str, delimiter=',')
test = np.loadtxt("./data/fb15k_237/test/test.csv", dtype=str, delimiter=',')

meta_data = index_kg(train)
train = convert_kg_to_index(train, meta_data["ent2ind"], meta_data["rel2ind"])
valid = convert_kg_to_index(valid, meta_data["ent2ind"], meta_data["rel2ind"])
test = convert_kg_to_index(test, meta_data["ent2ind"], meta_data["rel2ind"])

meta_data["ind2type"] = random.choices(["A", "B", "C"], k=len(meta_data["ind2ent"]))

model = UM(
    embedding_params={"embedding_size": 128},
    negative_ratio=2,
    corrupt_side="h+t")

model.fit(train_X=train, val_X=valid, meta_data=meta_data, epochs=2, batch_size=512,
          early_stopping_rounds=None, restore_best_weight=True, opt="Adam", opt_params=None,
          seed=None, log_path="./tensorboard_logs", log_projector=True)

model.evaluate(eval_X=test, corrupt_side="h", metrics=None, k=10,
               filter_pos=True, positive_X=np.concatenate((train, valid, test), axis=0), n_workers=4)