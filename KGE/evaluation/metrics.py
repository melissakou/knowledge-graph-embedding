import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import hmean
from scipy.stats.mstats import gmean

def train_test_split_no_unseen(X, test_size, seed):
    
    if type(test_size) is float:
        test_size = int(len(X) * test_size)
    
    h, h_cnt = np.unique(X[:, 0], return_counts = True)
    r, r_cnt = np.unique(X[:, 1], return_counts = True)
    t, t_cnt = np.unique(X[:, 2], return_counts = True)
    h_dict = dict(zip(h, h_cnt))
    r_dict = dict(zip(r, r_cnt))
    t_dict = dict(zip(t, t_cnt))
    
    test_id = np.array([], dtype=int)
    train_id = np.arange(len(X))
    loop_count = 0
    max_loop = len(X) * 10
    rnd = np.random.RandomState(seed)
    
    pbar = tqdm(total = test_size, desc = 'test size', leave = True)
    while len(test_id) < test_size:
        i = rnd.choice(train_id)
        if h_dict[X[i, 0]] > 1 and r_dict[X[i, 1]] > 1 and t_dict[X[i, 2]] > 1:
            h_dict[X[i, 0]] -= 1
            r_dict[X[i, 1]] -= 1
            t_dict[X[i, 2]] -= 1
    
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

def mean_reciprocal_rank(ranks):
    return np.mean(1 / np.array(ranks))

def mean_rank(ranks):
    return np.mean(ranks)

def median_rank(ranks):
    return np.median(ranks)

def geometric_mean_rank(ranks):
    return gmean(ranks)

def harmonic_mean_rank(ranks):
    return hmean(ranks)

def std_rank(ranks):
    return np.std(ranks)

def hits_at_k(ranks, k):
    return np.mean(np.array(ranks) <= k)