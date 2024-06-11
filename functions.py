from itertools import combinations_with_replacement

import numpy as np
from scipy.special import binom
from scipy.stats import invgauss
from z3 import Int, Solver, And, Or, Sum, sat

it = 1


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


def solve_int(A, b, R=None, maxnumsol=24):
    """Solves a constrained linear system of equations Ax = b, Rx >= 0, over the integers"""

    m, n = np.shape(A)
    r = np.shape(R)[0]
    X = [Int('x%d' % i) for i in range(n)]
    s = Solver()

    s.add(And([Sum([A[i][j] * X[j] for j in range(n)]) == b[i] for i in range(m)]))
    s.add(And([Sum([R[i][j] * X[j] for j in range(n)]) >= 0 for i in range(r)]))

    sol = []
    while (s.check() == sat) & (len(sol) < maxnumsol):
        part = [s.model().evaluate(X[i]).as_long() for i in range(n)]
        forbid = Or([X[i] != part[i] for i in range(n)])
        sol.append(part)
        s.add(forbid)

    return np.array(sol)


def unit_vec(k, j, as_column=False):
    vec = np.zeros(k)
    vec[j - 1] = 1
    if as_column:
        return vec.reshape(-1, 1)
    else:
        return vec



def format_time(secs):
    if secs < 60:
        return '{:.2f}s'.format(secs)
    else:
        mins = secs // 60
        secs -= mins * 60
        if mins < 60:
            return '{:.0f}min {:.0f}s'.format(mins, secs)
        else:
            hrs = mins // 60
            mins -= hrs * 60
            if hrs < 24:
                return '{:.0f}h {:.0f}min {:.0f}s'.format(hrs, mins, secs)
            else:
                days = hrs // 24
                hrs -= days * 24
                return '{:.0f}d {:.0f}h {:.0f}min {:.0f}s'.format(days, hrs, mins, secs)


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


def kron_sum(A, B):
    return np.kron(np.eye(B.shape[0]), A) + np.kron(B, np.eye(A.shape[0]))


def kron_sym(A, B):
    return np.kron(A, B) + np.kron(B, A)


def invgauss_bn(xi, eta, size, seed=None):
    mu = xi / eta
    lamb = xi**2
    return invgauss.rvs(mu=mu / lamb, loc=0, scale=lamb, size=size, random_state=seed)


class InitialDistribution:
    def __init__(self, dist, hyper=None):
        self.dist = dist
        self.hyper = np.array(hyper) if hyper is not None else None

    def E_0(self, param=None, order=0, wrt=None):
        wrt = np.atleast_1d(wrt)
        if self.dist == 'Dirac':
            if order == 0:
                return self.hyper
            elif order == 1:
                deriv_array = np.zeros((np.size(param), np.size(self.hyper)))
                return deriv_array[wrt]
            elif order == 2:
                deriv_array = np.zeros((np.size(param), np.size(param), np.size(self.hyper)))
                deriv_array = deriv_array[np.ix_(wrt, wrt)]
                return deriv_array[np.triu_indices(len(wrt))]
        elif self.dist == 'Gamma_Dirac':
            kappa, theta, sigma, rho = param
            if order == 0:
                return np.append(theta, self.hyper)
            elif order == 1:
                deriv_array = np.zeros((4, 3))
                deriv_array[1, 0] = 1
                return deriv_array[wrt]
            elif order == 2:
                deriv_array = np.zeros((4, 4, 3))
                deriv_array = deriv_array[np.ix_(wrt, wrt)]
                return deriv_array[np.triu_indices(len(wrt))]

    def Cov_0(self, param=None, order=0, wrt=None):
        wrt = np.atleast_1d(wrt)
        if self.dist == 'Dirac':
            if order == 0:
                return np.zeros((np.size(self.hyper), np.size(self.hyper)))
            elif order == 1:
                deriv_array = np.zeros((np.size(param), np.size(self.hyper), np.size(self.hyper)))
                return deriv_array[wrt]
            elif order == 2:
                deriv_array = np.zeros((np.size(param), np.size(param), np.size(self.hyper), np.size(self.hyper)))
                deriv_array = deriv_array[np.ix_(wrt, wrt)]
                return deriv_array[np.triu_indices(len(wrt))]
        elif self.dist == 'Gamma_Dirac':
            kappa, theta, sigma, rho = param
            if order == 0:
                arr = np.zeros((3, 3))
                arr[0, 0] = theta * sigma**2 / (2 * kappa)
                return arr
            elif order == 1:
                deriv_array = np.zeros((4, 3, 3))
                deriv_array[0, 0, 0] = -theta * sigma**2 / (2 * kappa**2)
                deriv_array[1, 0, 0] = sigma**2 / (2 * kappa)
                deriv_array[2, 0, 0] = theta * sigma / kappa
                return deriv_array[wrt]
            elif order == 2:
                deriv_array = np.zeros((4, 4, 3, 3))
                deriv_array[0, 0, 0, 0] = theta * sigma**2 / kappa**3
                deriv_array[0, 1, 0, 0] = -sigma**2 / (2 * kappa**2)
                deriv_array[0, 2, 0, 0] = -theta * sigma**2 / kappa**2
                deriv_array[1, 2, 0, 0] = sigma / kappa
                deriv_array[2, 2, 0, 0] = theta / kappa
                deriv_array = deriv_array[np.ix_(wrt, wrt)]
                return deriv_array[np.triu_indices(len(wrt))]

    def sample(self, param=None, n=1):
        if self.dist == 'Dirac':
            sample = np.tile(self.hyper, (n, 1))
            return sample.squeeze()
        elif self.dist == 'Gamma_Dirac':
            kappa, theta, sigma, rho = param
            v = np.random.gamma(shape=2 * kappa * theta / sigma**2, scale=sigma**2 / (2 * kappa), size=(n, 1))
            sample = np.hstack((v, np.tile(self.hyper, (n, 1))))
            return sample.squeeze()
