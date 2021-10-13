import os
import shutil
import numpy as np
import tensorflow as tf

def check_path_exist_and_create(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def ns_with_same_type(x, metadata, negative_ratio):
    sample_pool = metadata["type2inds"][metadata["ind2type"][x]]
    sample_pool = np.delete(sample_pool, np.where(sample_pool == x), axis=0)
    sample_entities = np.random.choice(sample_pool, size=negative_ratio)

    return sample_entities