import random
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.TransE import TransE
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

model = TransE(
    embedding_params={"embedding_size": 10},
    negative_ratio=2,
    corrupt_side="h+t",
    loss_fn=neg_log_likelihood,
    loss_params=None, #{"margin": 0.5},
    score_fn=dot,
    score_params=None, #{"p": 2},
    norm_emb = False,
    ns_strategy=typed_strategy,
    n_workers=1
)

model.fit(train_X=train, val_X=valid, meta_data=meta_data, epochs=100, batch_size=256,
          early_stopping_rounds=1, restore_best_weight=True, opt="Adam", opt_params=None,
          seed=None, log_path="./tensorboard_logs", log_projector=True)