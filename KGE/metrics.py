import numpy as np
from scipy.stats import hmean
from scipy.stats.mstats import gmean

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
    assert k >= 1, "k needs >= 1"
    return np.mean(np.array(ranks) <= k)