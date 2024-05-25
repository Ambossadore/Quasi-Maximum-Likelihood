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
        return dicts.sum(axis=1) == np.sum(mult)
    elif typ == 'leq_abs':
        return dicts.sum(axis=1) <= np.sum(mult)


def unit_vec(k, j, as_column=False):
    vec = np.zeros(k)
    vec[j - 1] = 1
    if as_column:
        return vec.reshape(-1, 1)
    else:
        return vec


def tracy_singh(A, B, A_partition, B_partition):
    num_blocks_A = (A.shape[0] // A_partition[0], A.shape[1] // A_partition[1])
    num_blocks_B = (B.shape[0] // B_partition[0], B.shape[1] // B_partition[1])
    partition_A = [np.split(arr, num_blocks_A[1], axis=1) for arr in np.split(A, num_blocks_A[0])]
    partition_B = [np.split(arr, num_blocks_B[1], axis=1) for arr in np.split(B, num_blocks_B[0])]
    result = np.zeros((A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]))
    size_blocks = (A_partition[0] * B_partition[0], A_partition[1] * B_partition[1])
    for i in range(num_blocks_A[0] * num_blocks_B[0]):
        for j in range(num_blocks_A[1] * num_blocks_B[1]):
            A_block_num = (i // num_blocks_B[0], j // num_blocks_B[1])
            B_block_num = (i % num_blocks_B[0], j % num_blocks_B[1])
            A_block = partition_A[A_block_num[0]][A_block_num[1]]
            B_block = partition_B[B_block_num[0]][B_block_num[1]]
            result[i * size_blocks[0]:(i + 1) * size_blocks[0], j * size_blocks[1]:(j + 1) * size_blocks[1]] = np.kron(A_block, B_block)
    return result


def cummean(arr, axis=None):
    if axis is not None:
        target_shape = np.ones(np.ndim(arr))
        target_shape[0] = arr.shape[0]
        return np.cumsum(arr, axis=axis) / np.arange(1, arr.shape[0] + 1).reshape(target_shape.astype('int'))
    else:
        return np.cumsum(arr) / np.arange(1, arr.shape[0] + 1)


def sym(arr):
    if np.ndim(arr) == 2:
        assert arr.shape[0] == arr.shape[1]
        return arr + arr.T
    else:
        assert arr.shape[1] == arr.shape[2]
        return arr + np.transpose(arr, (0, 2, 1))

