import os
import datetime
import logging
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from tqdm import tqdm, trange
from tensorboard.plugins import projector
from KGE.data_utils import calculate_data_size, set_tf_iterator
from KGE.ns_strategy import typed_strategy

logging.getLogger().setLevel(logging.INFO)

class KGE:

    def __init__(self, embedding_params, negative_ratio, corrupt_side, loss_fn, loss_params,
                 score_fn, score_params, norm_emb = False, ns_strategy="uniform", n_workers=2):
        if corrupt_side not in ['h+t', 'h', 't']:
            logging.error("Invalid corrupt_side, valid options: 'h+t', 'h', 't'")
            return
        
        self.embedding_params = embedding_params
        self.negative_ratio = negative_ratio
        self.corrupt_side = corrupt_side
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.score_fn = score_fn
        self.score_params = score_params
        self.norm_emb = norm_emb
        self.ns_strategy = ns_strategy
        self.n_workers = n_workers

    def fit(self, train_X, val_X, meta_data, epochs, batch_size, early_stopping_rounds,
            restore_best_weight=True, opt="Adam", opt_params=None, seed=None,
            log_path=None, log_projector=False):
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.opt = opt
        self.opt_params = {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07}
        if opt_params is not None:
            for param in list(opt_params.keys()):
                self.opt_params[param] = opt_params[param]
        self.seed = seed
        self.log_path = log_path
        
        # Create a Summary Writer to log the metrics to Tensorboard
        summary_writer = tf.summary.create_file_writer(log_path)
        train_logger = tf.summary.create_file_writer(log_path + '/scalar/train')
        if val_X is not None:
            val_logger = tf.summary.create_file_writer(log_path + '/scalar/validation')

        logging.info("[%s] Preparing for training..." % str(datetime.datetime.now()))
        train_iter, val_iter = self._prepare_for_train(train_X=train_X, val_X=val_X)
        train_loss_history = []
        val_loss_history = []
        patience_count = 0
        
        # Start Training
        logging.info("[%s] Start Training..." % str(datetime.datetime.now()))
        epoch_bar = trange(epochs, desc="Epoch", leave=True)
        for i in epoch_bar:
            train_loss = 0
            val_loss = 0
            batch_bar = trange(self.batch_count_train, desc="  Batch", leave=False)
            for b in batch_bar:
                train_batch_X = next(train_iter)
                train_batch_loss = self._run_single_batch(batch_data=train_batch_X, is_train=True)
                train_loss += train_batch_loss
                if val_iter is not None:
                    if b < self.batch_count_val:
                        val_batch_X = next(val_iter)
                        val_batch_loss = self._run_single_batch(batch_data=val_batch_X, is_train=False)
                        val_loss += val_batch_loss
                
            train_loss_history = self._normalize_loss_and_log(
                origin_loss = train_loss, normalize_factor=self.batch_size*self.batch_count_train,
                loss_history=train_loss_history, summary_writer=train_logger, step=i
            )
            if val_X is not None:
                val_loss_history = self._normalize_loss_and_log(
                    origin_loss = val_loss, normalize_factor=self.batch_size*self.batch_count_val,
                    loss_history=val_loss_history, summary_writer=val_logger, step=i
                )
                epoch_bar.set_description("epoch: %i, train loss: %f, valid loss: %f" % (i, train_loss_history[i], val_loss_history[i]))
            else:
                epoch_bar.set_description("epoch: %i, train loss: %f" % (i, train_loss_history[i]))

            epoch_bar.refresh()

            self._log_embeddings_histogram(summary_writer=summary_writer, step=i)
            
            if early_stopping_rounds is not None:
                assert val_X is not None, "val_X should be given if want to check early stopping."
                early_stop, patience_count = self._check_early_stopping(
                    metric_history=val_loss_history,
                    magnitude="larger",
                    patience_now=patience_count,
                    patience_max=early_stopping_rounds,
                    step=i,
                    restore_best_weight=restore_best_weight
                )
                if early_stop:
                    logging.info("[%s] Val loss does not improve within %i iterations, trigger early stopping." % (str(datetime.datetime.now()), early_stop))
                    if restore_best_weight:
                        logging.info("[%s] Restore best weight from %i to %i step." % (str(datetime.datetime.now()), i, self.best_step))
                    break
            else:
                self.ckpt_manager.save()                
        
        if log_projector:
            logging.info("[%s] Logging final embeddings into tensorboard projector..." % str(datetime.datetime.now()))
            self._log_embeddings_projector(log_path=log_path)

        logging.info("[%s] Finished training!" % str(datetime.datetime.now()))
        
    def _prepare_for_train(self, train_X, val_X):

        # calculate number of batch
        logging.info("[%s] - Calculating number of batch..." % str(datetime.datetime.now()))
        n_train = calculate_data_size(train_X)
        self.batch_count_train = int(np.ceil(n_train / self.batch_size))
        if val_X is not None:
            n_val = calculate_data_size(val_X)
            self.batch_count_val = int(np.ceil(n_val / self.batch_size))

        # set tensorflow dataset iterator
        logging.info("[%s] - Setting data iterator..." % str(datetime.datetime.now()))
        shuffle = self.seed is not None
        train_iter = set_tf_iterator(data=train_X, batch_size=self.batch_size, shuffle=shuffle, buffer_size=n_train, seed=self.seed)
        if val_X is not None:
            val_iter = set_tf_iterator(data=val_X, batch_size=self.batch_size, shuffle=shuffle, buffer_size=n_val, seed=self.seed)
        else:
            val_iter = None

        logging.info("[%s] - Initialized embedding..." % str(datetime.datetime.now()))
        self._init_embeddings(seed=self.seed)

        logging.info("[%s] - Initialized optimizer..." % str(datetime.datetime.now()))
        if self.opt == "Adam":
            self.optimizer = tf.optimizers.Adam(learning_rate=self.opt_params["learning_rate"],
                                                beta_1=self.opt_params["beta_1"],
                                                beta_2=self.opt_params["beta_2"],
                                                epsilon=self.opt_params["epsilon"])
        else:
            self.optimizer = self.opt

        logging.info("[%s] - Initialized checkpoint manager..." % str(datetime.datetime.now()))
        self.best_ckpt = tf.train.Checkpoint()
        self.best_ckpt.listed = []
        for w in list(self.model_weights.values()):
            self.best_ckpt.listed.append(w)
        self.best_ckpt.mapped = {k: v for k, v in zip(list(self.model_weights.keys()), list(self.model_weights.values()))}
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.best_ckpt, directory=self.log_path, max_to_keep=1)

        # Create type2inds metadata if using typed strategy
        if self.ns_strategy == typed_strategy:
            self.meta_data["type2inds"] = {}
            all_type = np.unique(self.meta_data["ind2type"])
            for t in all_type:
                indices = [i for (i, ti) in enumerate(self.meta_data["ind2type"]) if ti == t]
                self.meta_data["type2inds"][t] = np.array(indices)
        
        # Create pool for multiprocessing negative sampling
        if self.n_workers > 1:
            self.pool = mp.Pool(self.n_workers)
        else:
            self.pool = None
        
        return train_iter, val_iter
        
    def _init_embeddings(self, seed):
        raise NotImplementedError("subclass of KGE should implement _init_embeddings()")
    
    def _run_single_batch(self, batch_data, is_train):
        neg_triplet = self._negative_sampling(batch_data, strategy=self.ns_strategy)
        if self.norm_emb & is_train:
            self.model_weights["ent_emb"].assign(_normalized_embeddings(X=self.model_weights["ent_emb"], p=2))
        
        with tf.GradientTape() as g:
            pos_score = self._score_fn(batch_data)
            neg_score = self._score_fn(neg_triplet)
            batch_loss = self.loss_fn(pos_score, neg_score, self.loss_params)
        
        if is_train:
            gradients = g.gradient(batch_loss, list(self.model_weights.values()))
            self.optimizer.apply_gradients(zip(gradients, list(self.model_weights.values())))
        
        return batch_loss.numpy()

    def _negative_sampling(self, X, strategy):
        entities_pool = tf.range(len(self.meta_data["ind2ent"]), dtype=X.dtype)
        
        # combine hrt:
        if self.corrupt_side == 'h':
            neg_triplet = self._corrupt_h(X, entities_pool, self.negative_ratio, strategy)
        elif self.corrupt_side == "t":
            neg_triplet = self._corrupt_t(X, entities_pool, self.negative_ratio, strategy)
        elif self.corrupt_side == "h+t":
            neg_triplet_h = self._corrupt_h(X[:len(X)//2], entities_pool, self.negative_ratio, strategy)
            neg_triplet_t = self._corrupt_t(X[len(X)//2:], entities_pool, self.negative_ratio, strategy)
            neg_triplet = tf.concat([neg_triplet_h, neg_triplet_t], axis=0)

        return neg_triplet

    def _corrupt_h(self, X, sample_pool, negative_ratio, strategy):
        params = {"side": "h", "meta_data": self.meta_data, "n_workers": self.n_workers}
        sample_entities = strategy(X, sample_pool, negative_ratio, self.pool, params)
        h = sample_entities
        r = tf.repeat(X[:, 1], negative_ratio)
        t = tf.repeat(X[:, 2], negative_ratio)

        return tf.stack([h, r, t], axis = 1)

    def _corrupt_t(self, X, sample_pool, negative_ratio, strategy):
        params = {"side": "t", "meta_data": self.meta_data, "n_workers": self.n_workers}
        sample_entities = strategy(X, sample_pool, negative_ratio, self.pool, params)
        h = tf.repeat(X[:, 0], negative_ratio)
        r = tf.repeat(X[:, 1], negative_ratio)
        t = sample_entities

        return tf.stack([h, r, t], axis = 1)


    def translate(self, X):
        raise NotImplementedError("subclass of KGE should implement translate()")
    
    def _score_fn(self, triplets):
        trans, predicate = self.translate(X=triplets)
        return self.score_fn(trans, predicate, self.score_params)

    def _normalize_loss_and_log(self, origin_loss, normalize_factor, loss_history, summary_writer, step):
        loss = origin_loss / normalize_factor
        loss_history.append(loss)
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=step)
        
        return loss_history

    def _log_embeddings_histogram(self, summary_writer, step):
        with summary_writer.as_default():
            for w in list(self.model_weights.keys()):
                tf.summary.histogram(w, self.model_weights[w], step=step)

    def _check_early_stopping(self, metric_history, magnitude, patience_now,
                              patience_max, step, restore_best_weight=True):
        
        if step == 0:
            self.ckpt_manager.save()
            self.best_step = step
            return False, patience_now
        
        assert magnitude in ["larger", "smaller"], "magnitude must be 'larger' or 'smaller'"
        if self.best_step is None:
            self.best_step = step
        
        if magnitude == "larger":
            flag = metric_history[step] >= metric_history[self.best_step]
        elif magnitude == "smaller":
            flag = metric_history[step] <= metric_history[self.best_step]

        if flag:
            patience_now += 1
        else:
            patience_now = 0
            self.best_step = step
            self.ckpt_manager.save()

        if patience_now == patience_max:
            if restore_best_weight:
                self.best_ckpt.restore(self.ckpt_manager.latest_checkpoint)
            return True, patience_now

        return False, patience_now

    def _log_embeddings_projector(self, log_path):

        def write_metadata_file(path, obj):
            with open(path, "w") as f:
                for x in obj:
                    f.write("{}\n".format(x))
        
        write_metadata_file(path=os.path.join(log_path, "ent_metadata.tsv"), obj=self.meta_data["ind2ent"])
        write_metadata_file(path=os.path.join(log_path, "rel_metadata.tsv"), obj=self.meta_data["ind2rel"])

        ckpt = tf.train.Checkpoint(ent_emb=self.model_weights["ent_emb"], rel_emb=self.model_weights["rel_emb"])
        ckpt.save(os.path.join(log_path, "embedding.ckpt"))

        config = projector.ProjectorConfig()
        ent_embedding = config.embeddings.add()
        ent_embedding.tensor_name = "ent_emb/.ATTRIBUTES/VARIABLE_VALUE"
        ent_embedding.metadata_path = "ent_metadata.tsv"

        rel_embedding = config.embeddings.add()
        rel_embedding.tensor_name = "rel_emb/.ATTRIBUTES/VARIABLE_VALUE"
        rel_embedding.metadata_path = "rel_metadata.tsv"

        projector.visualize_embeddings(log_path, config)

    def get_rank(self, X, filter):
        X = self.convert_to_index(X)
        filter = self.convert_to_index(filter)
        filter_string = tf.strings.reduce_join(tf.strings.as_string(filter), axis = -1, separator = ',')
        filter_hash_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(filter_string, [1] * filter_string.shape[0]), 0)

        score = self._score_fn(X)

        ranks = []
        for i in tqdm(range(len(X))):
            neg_triplet = self._generate_corrupt_for_eval(X[i])
            neg_score = self._score_fn(neg_triplet)
            pos_score = score[i]
            test_triplet = tf.boolean_mask(neg_triplet, tf.less(neg_score, pos_score))
            if test_triplet.shape[0] == 0:
                rank = 1
            else:
                test_string = tf.strings.reduce_join(tf.strings.as_string(test_triplet), axis = -1, separator = ',')
                false_neg = tf.reduce_sum(filter_hash_table.lookup(test_string))
                rank = test_triplet.shape[0] - false_neg.numpy() + 1

            ranks.append(rank)

        return ranks
    
    def _summarize_embeddings(self, step):
        tf.summary.histogram('Entitiy Embeddings', self.ent_emb, step = step)
        tf.summary.histogram('Relation Embeddings', self.rel_emb, step = step)


def _normalized_embeddings(X, p):
    return X / tf.norm(X, ord=p, axis=1, keepdims=True)