import os
import csv
import datetime
import subprocess
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, trange
from tensorboard.plugins import projector

logging.getLogger().setLevel(logging.INFO)

class TransE():

    def __init__(self, embedding_size, negative_ratio, corrupt_side, margin, norm = 2, batch_corrupt = False, norm_emb = False):
        if corrupt_side not in ['h+t', 'h', 't']:
            logging.error("Invalid corrupt_side, valid options: 'h+t', 'h', 't'")
            return
        
        self.embedding_size = embedding_size
        self.negative_ratio = negative_ratio
        self.corrupt_side = corrupt_side
        self.margin = margin
        self.norm = norm
        self.batch_corrupt = batch_corrupt
        self.norm_emb = norm_emb


    def fit(self, train_X, val_X, epochs, learning_rate, batch_count, early_stopping, log_step = 1, log_path = None):
        # Create a Summary Writer to log the metrics to Tensorboard
        self.summary_writer = tf.summary.create_file_writer(log_path)
        train_logger = tf.summary.create_file_writer(log_path + '/scalar/train')
        if val_X is not None:
            val_logger = tf.summary.create_file_writer(log_path + '/scalar/validation')

        logging.info("[%s] Preprocessing training triplet..." % str(datetime.datetime.now()))
        if type(train_X) == str:
            filenames = os.listdir(train_X)
            filenames = [train_X + "/" + f for f in filenames]
            entities = []
            relations = []
            for f in filenames:
                tmp = pd.read_csv(f, header = None, dtype = str)
                entities.extend(list(tmp.iloc[:, 0]))
                entities.extend(list(tmp.iloc[:, 2]))
                relations.extend(list(tmp.iloc[:, 1]))
            entities = list(set(entities))
            relations = list(set(relations))            
            
            dict_ent2id = {}
            dict_id2ent = []
            for i, x in enumerate(entities):
                dict_ent2id[x] = i
                dict_id2ent.append(x)
            dict_rel2id = {}
            dict_id2rel = []
            for i, x in enumerate(relations):
                dict_rel2id[x] = i
                dict_id2rel.append(x)

            entitiy_to_index = np.vectorize(lambda x: dict_ent2id[x.decode()] if type(x) == bytes else dict_ent2id[x])
            rel_to_index = np.vectorize(lambda x: dict_rel2id[x.decode()] if type(x) == bytes else dict_rel2id[x])
                
            self.entities = list(set(entities))
            self.relations = list(set(relations))
            self.dict_ent2id = dict_ent2id
            self.dict_id2ent = dict_id2ent
            self.dict_rel2id = dict_rel2id
            self.dict_id2rel = dict_id2rel
            self.entitiy_to_index = entitiy_to_index
            self.rel_to_index = rel_to_index
            
            n_train = [int(subprocess.getoutput("wc -l " + f).split()[0]) for f in filenames]
            batch_size_train = int(sum(n_train) / batch_count)
                
            
            filenames = tf.data.Dataset.list_files(filenames, shuffle = False).repeat()
            train_dataset = filenames.interleave(
                lambda x: tf.data.TextLineDataset(x).shuffle(max(n_train)).batch(batch_size_train),
                cycle_length = 1)
            train_iter = iter(train_dataset)
            
            if val_X is not None:
                val_filenames = os.listdir(val_X)
                val_filenames = [val_X + "/" + f for f in val_filenames]
                n_val = sum([int(subprocess.getoutput("wc -l " + f).split()[0]) for f in val_filenames])
                batch_size_val = int(n_val / batch_count)
                val_filenames = tf.data.Dataset.list_files(val_filenames, shuffle = False).repeat()
                val_dataset = val_filenames.interleave(
                    lambda x: tf.data.TextLineDataset(x).batch(batch_size_val),
                    cycle_length = 1)
                val_iter = iter(val_dataset)
            
        
        if type(train_X) == np.ndarray:
            self.train_X = train_X
            self._initialize_train_kg(self.train_X)
            self.train_X = self.convert_to_index(self.train_X)
            batch_size_train = int(len(self.train_X) / batch_count)
            if val_X is not None:
                self.val_X = val_X
                self.val_X = self.convert_to_index(self.val_X)
                batch_size_val = int(len(self.val_X) / batch_count)
            
            train_dataset = tf.data.Dataset.from_tensor_slices(self.train_X)
            train_dataset = train_dataset.repeat().batch(batch_size_train).prefetch(1)
            train_iter = iter(train_dataset)
            if val_X is not None:
                val_dataset = tf.data.Dataset.from_tensor_slices(self.val_X)
                val_dataset = val_dataset.repeat().batch(batch_size_val).prefetch(1)
                val_iter = iter(val_dataset)
            
            
        logging.info("[%s] Initialize Embeddings..." % str(datetime.datetime.now()))
        xavier = tf.initializers.GlorotUniform()
        self.ent_emb = tf.Variable(xavier([len(self.entities), self.embedding_size]), name = 'entities_embedding')
        self.rel_emb = tf.Variable(xavier([len(self.relations), self.embedding_size]), name = 'relations_embedding')
        if self.norm_emb:
            self.rel_emb.assign(tf.clip_by_norm(self.rel_emb, clip_norm = 1, axes = 1), read_value = False)

        
        
        optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        loss_history = []
        loss_history_val = []
        patience_count = 0
        logging.info("[%s] Start Training..." % str(datetime.datetime.now()))
        pbar = trange(epochs, desc = "Epoch", leave = True)
        for i in pbar:
            if early_stopping is not None:
                if patience_count == early_stopping['patience']:
                    logging.info("val_loss dose not improve within %i iterations, trigger early stopping." % early_stopping['patience'])
                    break
            loss = 0
            loss_val = 0
            for b in range(batch_count):
                train_batch_X = next(train_iter)
                if len(train_batch_X.shape) == 1:
                    train_batch_X = tf.strings.split(train_batch_X, sep = ",").to_tensor()
                    train_batch_X = self.convert_to_index(train_batch_X)
                pos_triplet, neg_triplet = self._generate_corrupt_for_fit(train_batch_X, batch_corrupt = self.batch_corrupt)
                if val_X is not None:
                    val_batch_X = next(val_iter)
                    if len(val_batch_X.shape) == 1:
                        val_batch_X = tf.strings.split(val_batch_X, sep = ",").to_tensor()
                        val_batch_X = self.convert_to_index(val_batch_X)
                    pos_triplet_val, neg_triplet_val = self._generate_corrupt_for_fit(val_batch_X, batch_corrupt = False)

                if self.norm_emb:
                    self.ent_emb.assign(tf.clip_by_norm(self.ent_emb, clip_norm = 1, axes = 1), read_value = False)
        
                with tf.GradientTape() as g:
                    pos_score = self._score_fn(pos_triplet)
                    neg_score = self._score_fn(neg_triplet)
                    batch_loss = self._pairwise_loss(pos_score, neg_score)
                    
                gradients = g.gradient(batch_loss, [self.ent_emb, self.rel_emb])
                optimizer.apply_gradients(zip(gradients, [self.ent_emb, self.rel_emb]))
                loss += batch_loss

                if val_X is not None:
                    pos_score_val = self._score_fn(pos_triplet_val)
                    neg_score_val = self._score_fn(neg_triplet_val)
                    batch_loss_val = self._pairwise_loss(pos_score_val, neg_score_val)  
                    loss_val += batch_loss_val
                    
                    
            
            
            loss = loss.numpy() / batch_count
            loss_history.append(loss)
            
            if i % log_step == 0:
                with train_logger.as_default():
                    tf.summary.scalar('loss', loss, step = i)
                with self.summary_writer.as_default():
                    self._summarize_embeddings(i)
            
            if val_X is not None:
                loss_val = loss_val.numpy() / batch_count
                loss_history_val.append(loss_val)
                if early_stopping is not None:
                    if loss_history_val[i] - early_stopping['min_delta'] >= loss_history_val[i-1]:
                        patience_count += 1
                    else:
                        patience_count = 0
                
                if i % log_step == 0:
                    with val_logger.as_default():
                        tf.summary.scalar('loss', loss_val, step = i)
            
            pbar.set_description("epoch: %i, train loss: %f, valid loss: %f" % (i, loss, loss_val))
            pbar.refresh()
        
        self.loss_history = loss_history
        self.loss_history_val = loss_history_val
        
        with open(os.path.join(log_path, 'ent_metadata.tsv'), 'w') as f:
            for ent in self.dict_id2ent:
                f.write("{}\n".format(ent))
        
        with open(os.path.join(log_path, 'rel_metadata.tsv'), 'w') as f:
            for rel in self.dict_id2rel:
                f.write("{}\n".format(rel))
        
            
        
        ckpt = tf.train.Checkpoint(ent_embedding = self.ent_emb, rel_embedding = self.rel_emb)
        ckpt.save(os.path.join(log_path, "embedding.ckpt"))
        
        config = projector.ProjectorConfig()
        ent_embedding = config.embeddings.add()
        ent_embedding.tensor_name = "ent_embedding/.ATTRIBUTES/VARIABLE_VALUE"
        ent_embedding.metadata_path = 'ent_metadata.tsv'
        
        rel_embedding = config.embeddings.add()
        rel_embedding.tensor_name = "rel_embedding/.ATTRIBUTES/VARIABLE_VALUE"
        rel_embedding.metadata_path = 'rel_metadata.tsv'
            
        projector.visualize_embeddings(log_path, config)
        
    def get_embeddings(self, x, embedding_type):
        if embedding_type == 'entity':
            ent_id = self.entitiy_to_index(x)
            return tf.nn.embedding_lookup(self.ent_emb, ent_id).numpy()
        
        if embedding_type == 'relation':
            rel_id = self.rel_to_index(x)
            return tf.nn.embedding_lookup(self.rel_emb, rel_id).numpy()
            
        if embedding_type not in ['entity', 'relation']:
            logging.error("Invalid embedding_type, valid options: 'entity', 'relation'")
            return
        
        


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
    

    def _initialize_train_kg(self, kg_data):
        entities = list(np.unique(np.append(kg_data[:, 0], kg_data[:, 2])))
        relations = list(np.unique(kg_data[:, 1]))
        dict_ent2id = {}
        dict_id2ent = []
        for i, x in enumerate(entities):
            dict_ent2id[x] = i
            dict_id2ent.append(x)
        dict_rel2id = {}
        dict_id2rel = []
        for i, x in enumerate(relations):
            dict_rel2id[x] = i
            dict_id2rel.append(x)

        entitiy_to_index = np.vectorize(lambda x: dict_ent2id[x.decode()] if type(x) == bytes else dict_ent2id[x])
        rel_to_index = np.vectorize(lambda x: dict_rel2id[x.decode()] if type(x) == bytes else dict_rel2id[x])
        
        self.entities = entities
        self.relations = relations
        self.dict_ent2id = dict_ent2id
        self.dict_id2ent = dict_id2ent
        self.dict_rel2id = dict_rel2id
        self.dict_id2rel = dict_id2rel
        self.entitiy_to_index = entitiy_to_index
        self.rel_to_index = rel_to_index

    def convert_to_index(self, kg_data):
        index_kg_data = np.ones([kg_data.shape[0], kg_data.shape[1]], dtype = np.int32)
        index_kg_data[:, 0] = self.entitiy_to_index(kg_data[:, 0])
        index_kg_data[:, 2] = self.entitiy_to_index(kg_data[:, 2])
        index_kg_data[:, 1] = self.rel_to_index(kg_data[:, 1])

        return index_kg_data

    
    def _generate_corrupt_for_fit(self, X, batch_corrupt):
        if batch_corrupt:
            corrupt_entities = tf.unique(tf.concat((X[:, 0], X[:, 2]), axis = 0))[0]
        else:
            corrupt_entities = tf.range(len(self.entities))
            
        if self.corrupt_side == 'h+t':
            corrupt_h_mask = tf.random.uniform([X.shape[0] * self.negative_ratio], 0, 2, dtype = tf.int32)
            corrupt_t_mask = 1 - corrupt_h_mask
            
        if self.corrupt_side == 'h':
            corrupt_h_mask = tf.ones(X.shape[0] * self.negative_ratio, dtype = np.int32)
            corrupt_t_mask = tf.zeros(X.shape[0] * self.negative_ratio, dtype = np.int32)
    
        if self.corrupt_side == 't':
            corrupt_h_mask = tf.zeros(X.shape[0] * self.negative_ratio, dtype = np.int32)
            corrupt_t_mask = tf.ones(X.shape[0] * self.negative_ratio, dtype = np.int32)    

        pos_triplet = tf.tile(X, [self.negative_ratio, 1])
    
        sample_index = tf.expand_dims(tf.random.uniform([X.shape[0] * self.negative_ratio], 0, len(corrupt_entities), dtype = tf.int32), 1)
        sample_entities = tf.gather_nd(corrupt_entities, sample_index)
  
        h = corrupt_h_mask * sample_entities + (1-corrupt_h_mask) * pos_triplet[:, 0]
        r = pos_triplet[:, 1]
        t = corrupt_t_mask * sample_entities + (1-corrupt_t_mask) * pos_triplet[:, 2]
   
        neg_triplet = tf.stack([h, r, t], axis = 1)
    
        return pos_triplet, neg_triplet

    def _generate_corrupt_for_eval(self, X):
        repeat_triplet = tf.tile(tf.expand_dims(X, 0), [len(self.entities),1])
    
        corrupt_entities = tf.range(len(self.entities))
        corrupt_entities = tf.expand_dims(corrupt_entities, 1)

        if self.corrupt_side == 'h+t':
            corrupt_h = tf.concat((corrupt_entities, repeat_triplet[:, 1:]), axis = 1)
            corrupt_t = tf.concat((repeat_triplet[:, :2], corrupt_entities), axis = 1)

            return tf.concat((corrupt_h, corrupt_t), axis = 0)

        if self.corrupt_side == 'h':
            return tf.concat((corrupt_entities, repeat_triplet[:, 1:]), axis = 1)

        if self.corrupt_side == 't':
            return tf.concat((repeat_triplet[:, :2], corrupt_entities), axis = 1)

    
    def _score_fn(self, triplets):
        h_emb = tf.nn.embedding_lookup(self.ent_emb, triplets[:, 0])
        r_emb = tf.nn.embedding_lookup(self.rel_emb, triplets[:, 1])
        t_emb = tf.nn.embedding_lookup(self.ent_emb, triplets[:, 2])
               
        return tf.sqrt(tf.reduce_sum(tf.pow(h_emb + r_emb - t_emb, self.norm), axis = 1))


    def _pairwise_loss(self, pos_score, neg_score):
        return tf.reduce_mean(tf.clip_by_value(self.margin + pos_score - neg_score, 0, np.inf))
    
    
    def _summarize_embeddings(self, step):
        tf.summary.histogram('Entitiy Embeddings', self.ent_emb, step = step)
        tf.summary.histogram('Relation Embeddings', self.rel_emb, step = step)