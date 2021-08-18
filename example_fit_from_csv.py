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
from KGE.loss import pairwise_loss, neg_log_likelihood
from KGE.ns_strategy import uniform_strategy, typed_strategy

train = "./data/fb15k_237_train"
valid = "./data/fb15k_237_valid"

meta_data = index_kg(train)
meta_data["ind2type"] = random.choices(["A", "B", "C"], k=len(meta_data["ind2ent"]))


convert_kg_to_index(train, meta_data["ent2ind"], meta_data["rel2ind"])
convert_kg_to_index(valid, meta_data["ent2ind"], meta_data["rel2ind"])
train = train + "_indexed"

valid = valid + "_indexed"

model = RotatE(
    embedding_params={"embedding_size": 128},
    negative_ratio=2,
    corrupt_side="h+t",
    loss_fn=pairwise_loss,
    loss_params={"margin": 0.5},
    score_fn=p_norm,
    score_params={"p": 2},

    norm_emb = False,
    ns_strategy=uniform_strategy,
    n_workers=1
)

model.fit(train_X=train, val_X=valid, meta_data=meta_data, epochs=100, batch_size=512,
          early_stopping_rounds=10, restore_best_weight=True, opt="Adam", opt_params=None,
          seed=None, log_path="./tensorboard_logs", log_projector=True)