import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from .utils import check_path_exist_and_create


class myIter:
    def __init__(self, iter_obj):
        self.iter_obj = iter_obj
    
    def __iter__(self):
        self.iter_obj.__iter__
    
    def __next__(self):
        return tf.stack(next(self.iter_obj), axis=1)


def index_kg(kg_data):
    """ Index the Knowledge Graph data.

    Parameters
    ----------
    kg_data : np.array
        KG data to be indexed

    Returns
    -------
    dict
        metadata of KG <br>
        :code:`'ent2ind'`: dictionary that map entity to index
        :code:`'ind2ent'`: list that map index to entity
        :code:`'rel2ind'`: dictionary that map relation to index
        :code:`'ind2rel'`: list that map index to relation
    """

    if isinstance(kg_data, np.ndarray):
        entities = list(np.unique(np.append(kg_data[:, 0], kg_data[:, 2])))
        relations = list(np.unique(kg_data[:, 1]))
    else:
        entities = pd.Series([])
        relations = pd.Series([])
        filenames = os.listdir(kg_data)
        filenames = [kg_data + "/" + f for f in filenames]
        for f in filenames:
            tmp = pd.read_csv(f, header = None, dtype = str)
            entities = entities.append(tmp.iloc[:, 0])
            entities = entities.append(tmp.iloc[:, 2])
            relations = relations.append(tmp.iloc[:, 1])
        entities = list(pd.unique(entities))
        relations = list(pd.unique(relations))            
        
    ent2ind = {e: i for i, e in enumerate(entities)}
    ind2ent = [e for i, e in enumerate(entities)]
    rel2ind = {r: i for i, r in enumerate(relations)}
    ind2rel = [r for i, r in enumerate(relations)]

    return {"ent2ind": ent2ind, "ind2ent": ind2ent, "rel2ind": rel2ind, "ind2rel": ind2rel}


def convert_kg_to_index(kg_data, ent2ind, rel2ind):
    """ Convert the KG data into index

    Parameters
    ----------
    kg_data : np.array
        KG data to be converted
    ent2ind : dict
        dictionary that map entity to index
    rel2ind : dict
        dictionary that map relation to index

    Returns
    -------
    np.array
        indexed KG data
    """

    if isinstance(kg_data, np.ndarray):
        h = list(map(ent2ind.get, list(kg_data[:, 0])))
        r = list(map(rel2ind.get, list(kg_data[:, 1])))
        t = list(map(ent2ind.get, list(kg_data[:, 2])))

        return np.array([h,r,t]).T

    else:
        filenames = os.listdir(kg_data)
        check_path_exist_and_create(kg_data + "_indexed")
        for f in filenames:
            tmp = pd.read_csv(kg_data + "/" + f, header = None, dtype = str)
            tmp.iloc[:, 0] = tmp.iloc[:, 0].map(ent2ind)
            tmp.iloc[:, 1] = tmp.iloc[:, 1].map(rel2ind)
            tmp.iloc[:, 2] = tmp.iloc[:, 2].map(ent2ind)
            tmp.to_csv(kg_data + "_indexed/" + f, index=False, header=False)
        logging.info("indexed_kg has been save to %s" % kg_data+"_indexed")


def train_test_split_no_unseen(X, test_size, seed):
    """Split KG data into train and test

    Split KG data into train and test, this function guarantees that the
    entities in test data are also present in the train data.

    Parameters
    ----------
    X : np.array
        KG data to be splitted
    test_size : int or float
        desired test size, if :code:`int`, represents the absolute
        test size, if :code:`float`, represents the relative proportion.
    seed : int
        random seed

    Returns
    -------
    np.array, np.array
        splitted train, test KG data
    """
    
    if isinstance(test_size, float):
        test_size = int(len(X) * test_size)
    
    e, e_cnt = np.unique(np.append(X[:, 0], X[:, 2]), return_counts = True)
    r, r_cnt = np.unique(X[:, 1], return_counts = True)
    e_dict = dict(zip(e, e_cnt))
    r_dict = dict(zip(r, r_cnt))
    
    test_id = np.array([], dtype=int)
    train_id = np.arange(len(X))
    loop_count = 0
    max_loop = len(X) * 10
    rnd = np.random.RandomState(seed)
    
    pbar = tqdm(total=test_size, desc="test size", leave=True)
    while len(test_id) < test_size:
        i = rnd.choice(train_id)
        if e_dict[X[i, 0]] > 1 and r_dict[X[i, 1]] > 1 and e_dict[X[i, 2]] > 1:
            e_dict[X[i, 0]] -= 1
            r_dict[X[i, 1]] -= 1
            e_dict[X[i, 2]] -= 1
    
            test_id = np.unique(np.append(test_id, i))
            pbar.update(1)
            pbar.refresh()
        
        loop_count += 1
    
        if loop_count == max_loop:
            logging.error("Cannot split a test set with desired size, please reduce the test size")
            return
    pbar.close()
    
    train_id = np.setdiff1d(train_id, test_id)
    
    return X[train_id], X[test_id]


def calculate_data_size(X):
    
    if isinstance(X, str):
        filenames = os.listdir(X)
        filenames = [X + "/" + f for f in filenames]
        total_size = 0
        for f in filenames:
            partition = pd.read_csv(f, header=None)
            total_size += len(partition)
        return total_size
    else:
        return len(X)


def set_tf_iterator(data, batch_size, shuffle, buffer_size=None, seed=None):
    
    if isinstance(data, str):
        filenames = os.listdir(data)
        filenames = [data + "/" + f for f in filenames]
        tf_dataset = tf.data.Dataset.list_files(filenames) \
            .interleave(lambda x: tf.data.experimental.CsvDataset(x, record_defaults=[tf.int32]*3),
                        cycle_length=tf.data.experimental.AUTOTUNE)
    else:
        tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        assert buffer_size is not None, "buffer_size must be given when shuffle is True"
        tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    tf_dataset = tf_dataset.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    iterator = iter(tf_dataset)

    if isinstance(data, str):
        iterator = myIter(iterator)

    return iterator
