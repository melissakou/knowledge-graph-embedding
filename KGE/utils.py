import os
import stat
import numpy as np
import tensorflow as tf

def check_path_exist_and_create(path):
    if os.path.exists(path):
        rmtree(path)
    os.makedirs(path)

def ns_with_same_type(x, metadata, negative_ratio):
    sample_pool = metadata["type2inds"][metadata["ind2type"][x]]
    sample_pool = np.delete(sample_pool, np.where(sample_pool == x), axis=0)
    sample_entities = np.random.choice(sample_pool, size=negative_ratio)

    return sample_entities

def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)      