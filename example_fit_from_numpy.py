import numpy as np
from tensorflow.python.framework.indexed_slices import internal_convert_n_to_tensor_or_indexed_slices
from tensorflow.python.ops.variable_scope import _maybe_wrap_custom_getter
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.TransE import TransE
from KGE.score import p_norm, dot
from KGE.loss import pairwise_loss, neg_log_likelihood

train = np.loadtxt("./data/fb15k_237_train/fb15k_237_train.csv", dtype = str, delimiter = ',')
valid = np.loadtxt("./data/fb15k_237_valid/fb15k_237_valid.csv", dtype = str, delimiter = ',')

meta_data = index_kg(train)

train = convert_kg_to_index(train, meta_data["ent2ind"], meta_data["rel2ind"])
valid = convert_kg_to_index(valid, meta_data["ent2ind"], meta_data["rel2ind"])

model = TransE(
    embedding_params={"embedding_size": 10},
    negative_ratio=2,
    corrupt_side="t",
    loss_fn=pairwise_loss,
    loss_params={"margin": 0.5},
    score_fn=p_norm,
    score_params={"p": 2},
    norm_emb = False,
    ns_strategy="uniform"
)

model.fit(train_X=train, val_X=valid, meta_data=meta_data, epochs=100, batch_size=256,
          early_stopping_rounds=1, restore_best_weight=True, opt="Adam", opt_params=None,
          seed=None, log_path="./tensorboard_logs", log_projector=True)