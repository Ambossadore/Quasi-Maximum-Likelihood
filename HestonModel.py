import numpy as np
from functions import *
from scipy.linalg import expm
from tqdm import tqdm, trange
from z3 import Int, Solver, And, Or, Sum, sat
from itertools import product, compress


def solve_int(A, b, R=None, maxnumsol=20):
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


kappa = 1
nu = 0.4 ** 2
sigma = 0.3
rho = -0.5
mu = 0.
delta = -0.5

dim = 2
order = 8
dicts = return_dict(dim, order)

heston_A = np.zeros((n_dim(dim, order), n_dim(dim, order)))
heston_A[mult_to_ind([1, 0], dicts), 0] = kappa * nu
heston_A[mult_to_ind([1, 0], dicts), mult_to_ind([1, 0], dicts)] = -kappa
heston_A[mult_to_ind([0, 1], dicts), 0] = mu
heston_A[mult_to_ind([0, 1], dicts), mult_to_ind([1, 0], dicts)] = delta
heston_A[mult_to_ind([2, 0], dicts), mult_to_ind([1, 0], dicts)] = sigma ** 2
heston_A[mult_to_ind([1, 1], dicts), mult_to_ind([1, 0], dicts)] = sigma * rho
heston_A[mult_to_ind([0, 2], dicts), mult_to_ind([1, 0], dicts)] = 1


def get_continuous_B(A, dicts):
    B = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            masks = mask(dicts[i], dicts, typ='leq') & mask(dicts[j], dicts, typ='leq')
            lamb_ell = (ind_to_mult(i, dicts) - dicts)[masks]
            mu_ell = (ind_to_mult(j, dicts) - dicts)[masks]
            B[i, j] = (multi_binom(dicts[i], dicts[masks]) * A[mult_to_ind(lamb_ell, dicts), mult_to_ind(mu_ell, dicts)]).sum()
    return B


heston_Bc = get_continuous_B(heston_A, dicts)
heston_B = expm(heston_Bc)
heston_B_diff = np.zeros(heston_B.shape)
heston_B_diff[:, dicts[:, 1] == 0] = heston_B[:, dicts[:, 1] == 0]

heston_B_diff_sq = np.zeros((n_dim(dim + 1, order // 2), n_dim(dim + 1, order // 2)))
dicts_sq = return_dict(dim + 1, order // 2)
cols = np.where((dicts_sq[:, 1] == 0) & (dicts_sq[:, 2] == 0))[0]
for i in range(heston_B_diff_sq.shape[0]):
    lamb = ind_to_mult(i, dicts_sq)
    lamb_tilde = np.array([lamb[0], lamb[1] + 2 * lamb[2]])
    ind = mult_to_ind(lamb_tilde, dicts)
    heston_B_diff_sq[i, cols] = heston_B[ind, cols]

k = 6
d = 3
np.random.seed(0)
Y = np.random.normal(size=(k, k))
C = np.random.normal(size=(k, d))


def calc_filter_B(B, Y, C, order):
    k, d = C.shape
    B_large = np.zeros((n_dim(d + k, order), n_dim(d + k, order)))
    dictd = return_dict(d, order)
    dictk = return_dict(k, order)
    dict_large = return_dict(d + k, order)

    def S_func(trans, order):
        dim1, dim2 = trans.shape
        dict1, dict2 = return_dict(dim1, order), return_dict(dim2, order)
        raw_collections1 = np.array([np.sort(dict1[i][dict1[i] != 0]).tolist() for i in range(n_dim(dim1, order))], dtype=object)
        raw_collections2 = np.array([np.sort(dict2[i][dict2[i] != 0]).tolist() for i in range(n_dim(dim2, order))], dtype=object)
        locations1 = [np.where(np.in1d(dict1[i], raw_collections1[i]))[0][np.argsort(dict1[i][dict1[i] != 0])] for i in range(n_dim(dim1, order))]
        locations2 = [np.where(np.in1d(dict2[i], raw_collections2[i]))[0][np.argsort(dict2[i][dict2[i] != 0])] for i in range(n_dim(dim2, order))]
        collections1 = np.array([x for i, x in enumerate(raw_collections1.tolist()) if x not in raw_collections1.tolist()[:i]], dtype=object)
        collections2 = np.array([x for i, x in enumerate(raw_collections2.tolist()) if x not in raw_collections2.tolist()[:i]], dtype=object)
        coll_locator1 = np.array([collections1.tolist().index(x) for x in raw_collections1])
        coll_locator2 = np.array([collections2.tolist().index(x) for x in raw_collections2])

        raw_combinations = np.array(list(product(collections1, collections2)), dtype=object)
        coll_combinations = np.array(sum([list(product(list(compress(collections1, [np.sum(coll) == i for coll in collections1])), list(compress(collections2, [np.sum(coll) == i for coll in collections2])))) for i in range(1, order + 1)], []), dtype=object)
        raw_comb_locator = []
        for comb in raw_combinations:
            try:
                index = np.where(np.all(coll_combinations == comb, axis=1))[0][0]
            except IndexError:
                index = -1
            raw_comb_locator.append(index)
        raw_comb_locator = np.array(raw_comb_locator)
        solutions = []

        for comb in coll_combinations:
            dim_mat = len(comb[1]), len(comb[0])
            A = np.zeros((np.sum(dim_mat), np.prod(dim_mat)))
            A[np.repeat(np.arange(dim_mat[0]), dim_mat[1]), np.arange(A.shape[1])] = 1
            A[np.tile(np.arange(dim_mat[1]), dim_mat[0]) + dim_mat[0], np.arange(A.shape[1])] = 1
            b = np.hstack((comb[1], comb[0]))
            R = np.eye(A.shape[1])
            sol = solve_int(A, b, R)
            sol = sol.reshape((sol.shape[0], dim_mat[0], dim_mat[1]))
            solutions.append(np.transpose(sol, axes=(0, 2, 1)))

        solutions = np.array(solutions, dtype=object)

        I, J = np.meshgrid(np.arange(n_dim(dim1, order)), np.arange(n_dim(dim2, order)))
        S = np.zeros((n_dim(dim1, order), n_dim(dim2, order)))
        for i in range(n_dim(dim1, order)):
            t.update(1)
            index_in_raw_comb = np.where((raw_combinations == np.expand_dims(np.array([collections1[coll_locator1[I[:, i]]], collections2[coll_locator2[J[:, i]]]]).T, -2)).all(-1))[-1]
            sol_locator = raw_comb_locator[index_in_raw_comb]

            sol = solutions[sol_locator]
            for j in range(n_dim(dim2, order=4)):
                if sol_locator[j] == -1:
                    S[i, j] = 0
                    continue
                large_solution = np.zeros((sol[j].shape[0], dim1, dim2))
                large_solution[np.ix_(np.arange(sol[j].shape[0]), locations1[i], locations2[j])] = sol[j]
                S[i, j] = np.prod(multinom(large_solution) * mult_pow(trans, large_solution), axis=1).sum()

        S[0, 0] = 1
        return S

    t = tqdm(total=2 * n_dim(k, order) + n_dim(d + k, order), desc='Calculating S')
    S_mat = S_func(Y, order)
    S_mat_d = S_func(C, order)

    t.set_description('Calculating B')
    for i in range(dict_large.shape[0]):
        t.update(1)
        large_ind = dict_large[i]
        lamb, lamb_tilde = np.split(large_ind, [d])
        lamb_ind = mult_to_ind(lamb, dictd)
        mu_inds, mu_locs = dict_large[mask(large_ind, dict_large, typ='leq_abs')], np.where(mask(large_ind, dict_large, typ='leq_abs'))[0]
        for mu_ind, mu_loc in zip(mu_inds, mu_locs):
            mu, mu_tild = np.split(mu_ind, [d])
            mu_tild_ind = mult_to_ind(mu_tild, dictk)
            nus, nus_ind = dictd[mask(mu, dictd, typ='leq') & mask(lamb_tilde - mu_tild, dictd, typ='eq_abs')], np.where(mask(mu, dictd, typ='leq') & mask(lamb_tilde - mu_tild, dictd, typ='eq_abs'))[0]
            etas, etas_ind = dictk[mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs')], np.where(mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs'))[0]
            lamb_eta_ind = mult_to_ind(lamb_tilde - etas, dictk)
            mu_nu_ind = mult_to_ind(mu - nus, dictd)
            prods, prods_int, prods_int2 = product(nus, etas), product(nus_ind, etas_ind), product(mu_nu_ind, lamb_eta_ind)
            B_large[i, mu_loc] = np.sum([multi_binom(lamb_tilde, prod[1]) * S_mat[prod_int2[1], mu_tild_ind] * S_mat_d[prod_int[1], prod_int[0]] * B[lamb_ind, prod_int2[0]] for prod, prod_int, prod_int2 in zip(prods, prods_int, prods_int2)])

    return B_large


B_large = calc_filter_B(heston_B_diff_sq, Y, C, order=4)
