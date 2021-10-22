"""
base module for Knowledge Graph Embedding Model
"""

import os
import datetime
import logging
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from tqdm import tqdm, trange
from tensorboard.plugins import projector
from KGE.ns_strategy import TypedStrategy, UniformStrategy
from KGE.data_utils import calculate_data_size, set_tf_iterator
from KGE.metrics import mean_reciprocal_rank, mean_rank, hits_at_k, median_rank, geometric_mean_rank, harmonic_mean_rank, std_rank

logging.getLogger().setLevel(logging.INFO)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

class KGEModel:
    """A base module for Knowledge Graph Embedding Model.

    Subclass of :class:`KGEModel` can have thier own translation and interation model.
    """

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 loss_fn, ns_strategy, n_workers):
        """Initialize KGEModel.

        Parameters
        ----------
        embedding_params : dict
            embedding dimension parameters
        negative_ratio : int
            number of negative sample
        corrupt_side : str
            corrupt from which side while trainging, can be :code:`'h'`, :code:`'t'`, or :code:`'h+t'`
        loss_fn : class
            loss function class :py:mod:`KGE.loss.Loss`
        ns_strategy : function
            negative sampling strategy
        n_workers : int
            number of workers for negative sampling
        """
        
        assert corrupt_side in ['h+t', 'h', 't'], "Invalid corrupt_side, valid options: 'h+t', 'h', 't'"
        
        self.embedding_params = embedding_params
        self.negative_ratio = negative_ratio
        self.corrupt_side = corrupt_side
        self.loss_fn = loss_fn
        self.ns_strategy = ns_strategy
        self.__n_workers = n_workers

    def train(self, train_X, val_X, metadata, epochs, batch_size,
              early_stopping_rounds=None, model_weights_initial=None,
              restore_best_weight=True, optimizer="Adam", seed=None,
              log_path="./logs", log_projector=False):
        """Train the Knowledge Graph Embedding Model.

        Parameters
        ----------
        train_X : np.ndarray or str
            training triplets. \n
            If :code:`np.ndarray`, shape should be :code:`(n,3)` for :math:`(h,r,t)` respectively. \n
            If :code:`str`, training triplets should be save under this folder path
            with csv format, every csv files should have 3 columns without
            header for :math:`(h,r,t)` respectively.
        val_X : np.ndarray or str
            validation triplets. \n
            If :code:`np.ndarray`, shape should be :code:`(n,3)` for :math:`(h,r,t)` respectively. \n
            If :code:`str`, training triplets should be save under this folder path
            with csv format, every csv files should have 3 columns without
            header for :math:`(h,r,t)` respectively.
        metadata : dict
            metadata for kg data. should have following keys: \n
            :code:`'ent2ind'`: dict, dictionay that mapping entity to index. \n
            :code:`'ind2ent'`: list, list that mapping index to entity. \n 
            :code:`'rel2ind'`: dict, dictionay that mapping relation to index. \n
            :code:`'ind2rel'`: list, list that mapping index to relation. \n
            can use KGE.data_utils.index_kg to index and get metadata.
        epochs : int
            number of epochs
        batch_size : int
            batch_size
        early_stopping_rounds : int, optional
            number of rounds that trigger early stopping,
            by default None (no early stopping)
        model_weights_initial : dict, optional
            initial model wieghts with specific value, by default None
        restore_best_weight : bool, optional
            restore weight to the best iteration if early stopping rounds
            is not None, by default True
        optimizer : str or tensorflow.keras.optimizers, optional
            optimizer that apply in training, by default :code:`'Adam'`,
            use the default setting of `tf.keras.optimizers.Adam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>`_
        seed : int, optional
            random seed for shuffling data & embedding initialzation, by default None
        log_path : str, optional
            path for tensorboard logging, by default "./logs"
        log_projector : bool, optional
            project the embbedings in the tensorboard projector tab, 
            setting this True will write the metadata and embedding tsv files
            in :code:`log_path` and project this data on tensorboard projector tab,
            by default False
        """

        self.metadata = metadata
        self.batch_size = batch_size
        self._model_weights_initial = model_weights_initial
        self.__optimizer = optimizer
        self.seed = seed
        self.log_path = log_path
        
        # Create a Summary Writer to log the metrics to Tensorboard
        summary_writer = tf.summary.create_file_writer(log_path)
        train_logger = tf.summary.create_file_writer(log_path + '/scalar/train')
        if val_X is not None:
            val_logger = tf.summary.create_file_writer(log_path + '/scalar/validation')

        logging.info("[%s] Preparing for training..." % str(datetime.datetime.now()))
        train_iter, val_iter = self.__prepare_for_train(train_X=train_X, val_X=val_X)
        train_loss_history = []
        val_loss_history = []
        patience_count = 0
        
        # Start Training
        logging.info("[%s] Start Training..." % str(datetime.datetime.now()))
        epoch_bar = trange(epochs, desc="Epoch", leave=True)
        for i in epoch_bar:
            train_loss = 0
            val_loss = 0
            batch_bar = trange(self.__batch_count_train, desc="  Batch", leave=False)
            for b in batch_bar:
                train_batch_X = next(train_iter)
                train_batch_loss = self.__run_single_batch(batch_data=train_batch_X, is_train=True)
                train_loss += train_batch_loss
                if val_iter is not None:
                    if b < self.__batch_count_val:
                        val_batch_X = next(val_iter)
                        val_batch_loss = self.__run_single_batch(batch_data=val_batch_X, is_train=False)
                        val_loss += val_batch_loss
                
            train_loss /= self.__batch_count_train
            val_loss /= self.__batch_count_val
            train_loss_history = self.__append_history_and_log(
                loss = train_loss, loss_history=train_loss_history, summary_writer=train_logger, step=i
            )
            if val_X is not None:
                val_loss_history = self.__append_history_and_log(
                    loss = val_loss, loss_history=val_loss_history, summary_writer=val_logger, step=i
                )
                epoch_bar.set_description("epoch: %i, train loss: %f, valid loss: %f" % (i, train_loss_history[i], val_loss_history[i]))
            else:
                epoch_bar.set_description("epoch: %i, train loss: %f" % (i, train_loss_history[i]))

            epoch_bar.refresh()

            self.__log_embeddings_histogram(summary_writer=summary_writer, step=i)
            
            if early_stopping_rounds is not None:
                assert val_X is not None, "val_X should be given if want to check early stopping."
                early_stop, patience_count = self.__check_early_stopping(
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
            self.__log_embeddings_projector(log_path=log_path)

        logging.info("[%s] Finished training!" % str(datetime.datetime.now()))
        # if hasattr(self.ns_strategy, "pool"):
        #     if self.ns_strategy.pool is not None:
        #         self.ns_strategy.pool.close()
        #         self.ns_strategy.pool.join()
        
    def __prepare_for_train(self, train_X, val_X):
        """Prepartion before training.

        Do the following steps:
        - calculate number of batch
        - set tensorflow dataset iterator
        - initilized embedding, optimizer & checkpoint manager
        - create "type2inds" metadata if using typed_strategy
        - create pool for multiprocessing negative sampling if n_workers > 1

        Parameters
        ----------
        train_X : np.ndarray or str
            training triplets.
            If `np.ndarray`, shape should be (n,3) for (h,r,t) respectively.
            If `str`, training triplets should be save under this folder path
            with csv format, every csv files should have 3 columns without
            header for (h,r,t) respectively.
        val_X : np.ndarray or str
            validation triplets.
            If `np.ndarray`, shape should be (n,3) for (h,r,t) respectively.
            If `str`, training triplets should be save under this folder path
            with csv format, every csv files should have 3 columns without
            header for (h,r,t) respectively.

        Returns
        -------
        iterator, iterator
            training and validation data iterator.
        """
        
        # calculate number of batch
        logging.info("[%s] - Calculating number of batch..." % str(datetime.datetime.now()))
        n_train = calculate_data_size(train_X)
        self.__batch_count_train = int(np.ceil(n_train / self.batch_size))
        if val_X is not None:
            n_val = calculate_data_size(val_X)
            self.__batch_count_val = int(np.ceil(n_val / self.batch_size))

        # set tensorflow dataset iterator
        logging.info("[%s] - Setting data iterator..." % str(datetime.datetime.now()))
        train_iter = set_tf_iterator(data=train_X, batch_size=self.batch_size, shuffle=True, buffer_size=n_train, seed=self.seed)
        if val_X is not None:
            val_iter = set_tf_iterator(data=val_X, batch_size=self.batch_size, shuffle=False, buffer_size=None, seed=None)
        else:
            val_iter = None

        # initilized embedding, optimizer & checkpoint manager
        logging.info("[%s] - Initialized embedding..." % str(datetime.datetime.now()))
        self._init_embeddings(seed=self.seed)
        logging.info("[%s] - Initialized optimizer..." % str(datetime.datetime.now()))
        if self.__optimizer == "Adam":
            self.__optimizer = tf.optimizers.Adam()
        else:
            self.__optimizer = self.__optimizer
        logging.info("[%s] - Initialized checkpoint manager..." % str(datetime.datetime.now()))
        self.best_ckpt = tf.train.Checkpoint()
        self.best_ckpt.listed = []
        for w in list(self.model_weights.values()):
            self.best_ckpt.listed.append(w)
        self.best_ckpt.mapped = {k: v for k, v in zip(list(self.model_weights.keys()), list(self.model_weights.values()))}
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.best_ckpt, directory=self.log_path, max_to_keep=1)

        # Create type2inds metadata if using typed strategy
        if self.ns_strategy == UniformStrategy:
            self.ns_strategy = UniformStrategy(sample_pool=tf.range(len(self.metadata["ind2ent"])))
        elif self.ns_strategy == TypedStrategy:
            self.metadata["type2inds"] = {}
            all_type = np.unique(self.metadata["ind2type"])
            for t in all_type:
                indices = [i for (i, ti) in enumerate(self.metadata["ind2type"]) if ti == t]
                self.metadata["type2inds"][t] = np.array(indices)

            # Create pool for multiprocessing negative sampling
            if self.__n_workers > 1:
                pool = mp.Pool(self.__n_workers)
            else:
                pool = None

            self.ns_strategy = TypedStrategy(
                pool=pool, metadata={
                    "type2inds": self.metadata["type2inds"],
                    "ind2type": self.metadata["ind2type"]
                }
            )

        return train_iter, val_iter
        
    def _init_embeddings(self):
        """Initialized embeddings.
        
        Should be implemented in subclass for their own embedding parameters.

        Raises
        ------
        NotImplementedError
            subclass doesnt not implement _init_embeddings().
        """

        raise NotImplementedError("subclass of KGEModel should implement _init_embeddings()")
    
    def __run_single_batch(self, batch_data, is_train):
        """Run training procedure on one single batch.

        whole training procedure:
        - perform negative sampling
        - calculate contraint term or contraint embedding
        - calculate positive & negative score
        - calculate loss
        - backpropagation & update model weights (if is_train)

        Parameters
        ----------
        batch_data : tf.Tensor
            batch data to be processed, shape should be (n,3)
        is_train : bool
            whether to calculate gradients and update model weights

        Returns
        -------
        float
            loss of this batch.
        """

        neg_triplet = self.__negative_sampling(batch_data, strategy=self.ns_strategy)

        with tf.GradientTape() as g:
            constraint_term = self._constraint_loss(batch_data)
            pos_score = self.score_hrt(batch_data[:, 0], batch_data[:, 1], batch_data[:, 2])
            neg_score = self.score_hrt(neg_triplet[:, 0], neg_triplet[:, 1], neg_triplet[:, 2])
            batch_loss = self.loss_fn(pos_score, neg_score)
            batch_loss += constraint_term

        if is_train:
            gradients = g.gradient(batch_loss, list(self.model_weights.values()))
            gradients = [tf.clip_by_norm(grad, clip_norm=5.0) for grad in gradients]
            self.__optimizer.apply_gradients(zip(gradients, list(self.model_weights.values())))
        
        return batch_loss.numpy()

    def __negative_sampling(self, X, strategy):
        """Perfoem negative sampling

        Parameters
        ----------
        X : tf.Tensor
            triplets to be corrupt with shape (n,3)
        strategy : function
            negative sampling strategy function in KGE.ns_strategy

        Returns
        -------
        tf.Tensor
            corrupted triplets with shaep (n*self.negative_ratio, 3)
        """
        
        # combine hrt:
        if self.corrupt_side == 'h':
            neg_triplet = self.__corrupt_h(X, self.negative_ratio, strategy)
        elif self.corrupt_side == "t":
            neg_triplet = self.__corrupt_t(X, self.negative_ratio, strategy)
        elif self.corrupt_side == "h+t":
            neg_triplet_h = self.__corrupt_h(X, self.negative_ratio // 2, strategy)
            neg_triplet_t = self.__corrupt_t(X, self.negative_ratio // 2, strategy)
            neg_triplet = tf.reshape(tf.concat([neg_triplet_h, neg_triplet_t], axis=-1), [-1, 3])

        return neg_triplet

    def __corrupt_h(self, X, negative_ratio, strategy):
        """Corrupt triplets from head side

        Parameters
        ----------
        X : tf.Tensor
            triplets to be corrupt with shape (n,3)
        negative_ratio : int
            number of negative triplets to be generated for each triplet
        strategy : function
            negative sampling strategy function in KGE.ns_strategy

        Returns
        -------
        tf.Tensor
            corrupted triplets with shaep (n*negative_ratio, 3)
        """       
        
        sample_entities = strategy(X, negative_ratio=negative_ratio, side="h")
        h = sample_entities
        r = tf.repeat(X[:, 1], negative_ratio)
        t = tf.repeat(X[:, 2], negative_ratio)

        return tf.stack([h, r, t], axis = 1)

    def __corrupt_t(self, X, negative_ratio, strategy):
        """Corrupt triplets from tail side

        Parameters
        ----------
        X : tf.Tensor
            triplets to be corrupt with shape (n,3)
        negative_ratio : int
            number of negative triplets to be generated for each triplet
        strategy : function
            negative sampling strategy function in KGE.ns_strategy

        Returns
        -------
        tf.Tensor
            corrupted triplets with shaep (n*negative_ratio, 3)
        """

        sample_entities = strategy(X, negative_ratio=negative_ratio, side="t")
        h = tf.repeat(X[:, 0], negative_ratio)
        r = tf.repeat(X[:, 1], negative_ratio)
        t = sample_entities

        return tf.stack([h, r, t], axis = 1)
    
    def score_hrt(self, h, r, t):
        """Scoring the triplets.

        Should be implemented in subclass for their own scoring function.

        Raises
        ------
        NotImplementedError
            subclass doesnt not implement score_hrt().
        """
        assert ~(h is None and t is None), "h and t should not be None simultaneously"
        if h is None:
            assert len(r.shape) == 0
            assert len(t.shape) == 0
            h = np.arange(len(self.metadata["ind2ent"]))
        if t is None:
            assert len(h.shape) == 0
            assert len(r.shape) == 0
            t = np.arange(len(self.metadata["ind2ent"]))

        return h, r, t      
    
    def _constraint_loss(self):
        """Perform penalty on loss or constraint on model weights.

        Should be implemented in subclass for their own constraint.

        Raises
        ------
        NotImplementedError
            subclass doesnt not implement _constraint_loss().
        """
        raise NotImplementedError("subclass of KGEModel should implement _constraint_loss()")

    def __append_history_and_log(self, loss, loss_history, summary_writer, step):
        """Append current loss history and log into tensorboard.

        Parameters
        ----------
        loss : float
            current loss should be appended loss_history
        loss_history : list of flost
            loss history to be appended
        summary_writer : tensorflow.python.ops.summary_ops_v2.ResourceSummaryWriter
            tensorboard summary writer
        step : int
            current step

        Returns
        -------
        list of float
            appended loss history
        """

        loss_history.append(loss)
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=step)
        
        return loss_history

    def __log_embeddings_histogram(self, summary_writer, step):
        """Logging all model weights to tensorboard histogram.

        Parameters
        ----------
        summary_writer : tensorflow.python.ops.summary_ops_v2.ResourceSummaryWriter
            tensorboard summary writer
        step : int
            current step
        """

        with summary_writer.as_default():
            for w in list(self.model_weights.keys()):
                tf.summary.histogram(w, self.model_weights[w], step=step)

    def __check_early_stopping(self, metric_history, magnitude, patience_now,
                               patience_max, step, restore_best_weight=True):
        """Check early stopping and restore the weights to the best step
           if early stopping criteria is match.

        Parameters
        ----------
        metric_history : list of float
            metric history to be check for early stopping.
        magnitude : str
           overfitting metric magnitude, can be 'larger' or 'smaller'
           for example, if metric is loss,
           the loss becomes larger and larger when overfitting occur.
        patience_now : int
            how many times that metric does not improve.
        patience_max : int
            maximum patience that metrics does not improve.
            when patience_now == patience_max, trigger early stopping.
        step : int
            current step
        restore_best_weight : bool, optional
            whether to restore model weights to the best step, by default True

        Returns
        -------
        bool, int
            whther trigger early stopping, updated patience_now
        """
        
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

    def __log_embeddings_projector(self, log_path):
        """Log embedding to TensorBoard projector tab.

        Parameters
        ----------
        log_path : str
            path for tensorboard logging
        """

        def write_metadata_file(path, obj):
            with open(path, "w") as f:
                for x in obj:
                    f.write("{}\n".format(x))
        
        write_metadata_file(path=os.path.join(log_path, "ent_metadata.tsv"), obj=self.metadata["ind2ent"])

        if self.model_weights.get("rel_emb") is not None:
            write_metadata_file(path=os.path.join(log_path, "rel_metadata.tsv"), obj=self.metadata["ind2rel"])
            ckpt = tf.train.Checkpoint(ent_emb=self.model_weights["ent_emb"], rel_emb=self.model_weights["rel_emb"])
        else:
            ckpt = tf.train.Checkpoint(ent_emb=self.model_weights["ent_emb"])

        ckpt.save(os.path.join(log_path, "embedding.ckpt"))

        config = projector.ProjectorConfig()
        ent_embedding = config.embeddings.add()
        ent_embedding.tensor_name = "ent_emb/.ATTRIBUTES/VARIABLE_VALUE"
        ent_embedding.metadata_path = "ent_metadata.tsv"

        if self.model_weights.get("rel_emb") is not None:
            rel_embedding = config.embeddings.add()
            rel_embedding.tensor_name = "rel_emb/.ATTRIBUTES/VARIABLE_VALUE"
            rel_embedding.metadata_path = "rel_metadata.tsv"

        projector.visualize_embeddings(log_path, config)

    def evaluate(self, eval_X, corrupt_side, positive_X=None):
        """Evaluate triplets.

        Parameters
        ----------
        eval_X : tf.Tensor or np.array
            triplets to be evaluated
        corrupt_side : str
            corrupt triplets from which side, can be :code:`'h'` and :code:`'t'`
        positive_X : tf.Tensor or np.array, optional
            positive triplets that should be filtered while generating
            corrupted triplets, by default None (no filter applied)

        Returns
        -------
        dict
            evaluation result
        """

        n_eval = calculate_data_size(eval_X)
        eval_iter = set_tf_iterator(data=eval_X, batch_size=1, shuffle=False)

        ranks = []

        for _ in tqdm(range(n_eval)):
            eval_x = next(eval_iter)
            ranks.append(self.get_rank(eval_x, positive_X, corrupt_side))

        eval_result = {
            "mean_rank": mean_rank(ranks),
            "mean_reciprocal_rank": mean_reciprocal_rank(ranks),
            "median_rank": median_rank(ranks),
            "geometric_mean_rank": geometric_mean_rank(ranks),
            "harmonic_mean_rank": harmonic_mean_rank(ranks),
            "std_rank": std_rank(ranks),
            "hit@1": hits_at_k(ranks, k=1),
            "hit@3": hits_at_k(ranks, k=3),
            "hit@10": hits_at_k(ranks, k=10)
        }
                
        return eval_result

    def get_rank(self, x, positive_X, corrupt_side):
        """Get rank for specific one triplet.

        Parameters
        ----------
        x : tf.Tensor or np.array
            rank this triplet
        positive_X : tf.Tensor or np.array, optional
            positive triplets that should bt filtered while generating
            corrupted triplets, if :code:`None`, no filter applied
        corrupt_side : str
            corrupt triplets from which side, can be :code:`'h'` and :code:`'t'`

        Returns
        -------
        int
           ranking result 
        """
        x = tf.squeeze(x)
        if corrupt_side == "h":
            filter_side, corrupt_side = 2, 0
            scores = self.score_hrt(h=None, r=x[1], t=x[2])
        elif corrupt_side == "t":
            filter_side, corrupt_side = 0, 2
            scores = self.score_hrt(h=x[0], r=x[1], t=None)
        
        if positive_X is not None:
            r_mask = positive_X[:, 1] == x[1]
            e_mask = positive_X[:, filter_side] == x[filter_side]
            positive_e = positive_X[r_mask & e_mask, corrupt_side]
            scores = tf.tensor_scatter_nd_update(scores, tf.expand_dims(positive_e, -1), [-np.inf] * len(positive_e))

        pos_score = self.score_hrt(x[0], x[1], x[2])

        return tf.reduce_sum(tf.cast(scores > pos_score, tf.int16)).numpy() + 1

    def restore_model_weights(self, model_weights):
        """Restore the model weights.

        Parameters
        ----------
        model_weights : dict
            dictionary of model weights to be restored
        """

        self._check_model_weights()
        self.model_weights = model_weights

    def _check_model_weights(self):
        '''Check modle weights have required keys.

        Should be implemented in subclass for their own key checking.

        Raises
        ------
        NotImplementedError
            subclass doesnt not implement _check_model_weights().
        '''

        raise NotImplementedError("subclass of KGEModel should implement _check_model_weights()")