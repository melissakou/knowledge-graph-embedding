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

train = "./data/fb15k_237/train"
valid = "./data/fb15k_237/valid"

meta_data = index_kg(train)
meta_data["ind2type"] = random.choices(["A", "B", "C"], k=len(meta_data["ind2ent"]))


convert_kg_to_index(train, meta_data["ent2ind"], meta_data["rel2ind"])
convert_kg_to_index(valid, meta_data["ent2ind"], meta_data["rel2ind"])
train = train + "_indexed"

valid = valid + "_indexed"

model = UM(
    embedding_params={"embedding_size": 128},
    negative_ratio=2,
    corrupt_side="h+t")

model.fit(train_X=train, val_X=valid, meta_data=meta_data, epochs=100, batch_size=512,
          early_stopping_rounds=10, restore_best_weight=True, opt="Adam", opt_params=None,
          seed=None, log_path="./tensorboard_logs", log_projector=True)