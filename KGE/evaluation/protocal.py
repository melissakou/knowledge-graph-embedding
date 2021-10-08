import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


def generate_negative_lcwa(X, entity_pool, corrupt_side, positive_X, n_workers=1):

    assert corrupt_side in ['h', 't'], "Invalid corrupt_side, valid options: 'h', 't'"

    X = np.split(X, len(X))

    if n_workers == 1:  
        corrupt_entities = [generate_corrupt_entities for x in tqdm(X)]
    else:
        with mp.Pool(n_workers) as pool:
            corrupt_entities = list(tqdm(pool.imap(
                partial(generate_corrupt_entities, entity_pool=entity_pool, corrupt_side=corrupt_side, positive_X=positive_X), X
            ), total=len(X)))
        pool.close()
        pool.join()

    return corrupt_entities


def generate_corrupt_entities(x, entity_pool, corrupt_side, positive_X):
    if corrupt_side == "h":
        filter_side, corrupt_side = 2, 0
    elif corrupt_side == "t":
        filter_side, corrupt_side = 0, 2

    x = np.squeeze(x)
    r_mask = positive_X[:, 1] == x[1]
    e_mask = positive_X[:, filter_side] == x[filter_side]
    positive_e = positive_X[r_mask & e_mask][:, corrupt_side]
    corrupt_entities = np.setdiff1d(entity_pool, positive_e)

    return corrupt_entities
    

def array_diff(a, b):
    a_rows = a.view([('', a.dtype)] * a.shape[1])
    b_rows = b.view([('', b.dtype)] * b.shape[1])
    return np.setdiff1d(a_rows, b_rows, assume_unique=True).view(a.dtype).reshape(-1, a.shape[1])