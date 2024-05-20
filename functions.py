import numpy as np
from scipy.special import binom
from itertools import combinations_with_replacement


def n_dim(d, order):
    return int(binom(d + order, d))


def return_dict(d, order):
    comb = [list(combinations_with_replacement(np.arange(1, d + 1), i)) for i in range(1, order + 1)]
    comb = sum(comb, [])
    comb = np.array(list(map(lambda arr: np.pad(arr, (0, order - len(arr))), comb)))
    comb = np.vstack((np.zeros(order), comb)).astype('int')
    return np.vstack([np.bincount(comb[i], minlength=d+1)[1:] for i in range(comb.shape[0])])


def ind_to_mult(ind, dicts):
    return dicts[ind]


# def mult_to_ind(ind, dicts):
#     dims = dicts.max(0) + 1
#     return np.where(np.in1d(np.ravel_multi_index(dicts.T, dims), np.ravel_multi_index(np.array(ind).T, dims)))[0]


def mult_to_ind(ind, dicts):
    return np.where((dicts == np.atleast_2d(ind)[:, None]).all(-1))[1]


def multi_binom(mult1, mult2):
    return binom(mult1, mult2).prod(axis=-1)


def multinom(mult):
    return multi_binom(np.cumsum(mult, axis=-1)[..., 1:], mult[..., 1:])


def mult_pow(arr, mult):
    return np.prod(arr ** mult, axis=-1)


def mask(mult, dicts, typ):
    if typ == 'leq':
        return np.all(dicts <= mult, axis=1)
    elif typ == 'eq_abs':
        return dicts.sum(axis=1) == mult.sum()
    elif typ == 'leq_abs':
        return dicts.sum(axis=1) <= mult.sum()


def unit_vec(k, j, as_column=False):
    vec = np.zeros(k)
    vec[j - 1] = 1
    if as_column:
        return vec.reshape(-1, 1)
    else:
        return vec
