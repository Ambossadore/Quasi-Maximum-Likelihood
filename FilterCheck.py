import numpy as np
from scipy.special import factorial
from functions import *
from itertools import product, compress
from z3 import Int, Solver, And, Or, Sum, sat
from tqdm import tqdm, trange
from time import time


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


# Example Case
lamb_tilde = np.array([0, 2, 1, 1, 0, 0])
mu_tilde = np.array([0, 0, 0, 0, 1, 1])
eta = np.array([0, 1, 1, 0, 0, 0])

lamb_eta = lamb_tilde - eta
dim_mat = (mu_tilde > 0).sum(), (lamb_eta > 0).sum()
A = np.zeros((np.sum(dim_mat), np.prod(dim_mat)))
A[np.repeat(np.arange(dim_mat[0]), dim_mat[1]), np.arange(A.shape[1])] = 1
A[np.tile(np.arange(dim_mat[1]), dim_mat[0]) + dim_mat[0], np.arange(A.shape[1])] = 1
b = np.hstack((mu_tilde[mu_tilde > 0], lamb_eta[lamb_eta > 0]))
R = np.eye(A.shape[1])
sol = solve_int(A, b, R)
sol = sol.reshape((sol.shape[0], dim_mat[0], dim_mat[1]))
sol = np.transpose(sol, axes=(0, 2, 1))

sol_vec = np.zeros((sol.shape[0], lamb_eta.shape[0], mu_tilde.shape[0]))
sol_vec[np.ix_(np.arange(sol.shape[0]), lamb_eta > 0, mu_tilde > 0)] = sol


# Offline Computation of S_tilde and S
d = 3
k = 6

np.random.seed(0)
Y = np.random.normal(size=(k, k))
C = np.random.normal(size=(k, d))
Xs = np.random.normal(size=d)
Xt = np.random.normal(size=d)
Zs = np.random.normal(size=k)
Zt = Y @ Zs + C @ Xs
dictd = return_dict(d, order=4)
dictk = return_dict(k, order=4)


def S_func(trans, order):
    dim1, dim2 = trans.shape
    dict1, dict2 = return_dict(dim1, order), return_dict(dim2, order)
    raw_collections1 = np.array([np.sort(dict1[i][dict1[i] != 0]).tolist() for i in range(n_dim(dim1, order=4))], dtype=object)
    raw_collections2 = np.array([np.sort(dict2[i][dict2[i] != 0]).tolist() for i in range(n_dim(dim2, order=4))], dtype=object)
    locations1 = [np.where(np.in1d(dict1[i], raw_collections1[i]))[0][np.argsort(dict1[i][dict1[i] != 0])] for i in range(n_dim(dim1, order=4))]
    locations2 = [np.where(np.in1d(dict2[i], raw_collections2[i]))[0][np.argsort(dict2[i][dict2[i] != 0])] for i in range(n_dim(dim2, order=4))]
    collections1 = np.array([x for i, x in enumerate(raw_collections1.tolist()) if x not in raw_collections1.tolist()[:i]], dtype=object)
    collections2 = np.array([x for i, x in enumerate(raw_collections2.tolist()) if x not in raw_collections2.tolist()[:i]], dtype=object)
    coll_locator1 = np.array([collections1.tolist().index(x) for x in raw_collections1])
    coll_locator2 = np.array([collections2.tolist().index(x) for x in raw_collections2])

    raw_combinations = np.array(list(product(collections1, collections2)), dtype=object)
    coll_combinations = np.array(sum([list(product(list(compress(collections1, [np.sum(coll) == i for coll in collections1])), list(compress(collections2, [np.sum(coll) == i for coll in collections2])))) for i in range(1, 5)], []), dtype=object)
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

    I, J = np.meshgrid(np.arange(n_dim(dim1, order=4)), np.arange(n_dim(dim2, order=4)))
    S = np.zeros((n_dim(dim1, order=4), n_dim(dim2, order=4)))
    for i in trange(n_dim(dim1, order=4)):
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


S_mat = S_func(Y, order=4)
S_mat_d = S_func(C, order=4)


def S_tilde_func(mu_tilde, eta, Y):
    dim_mat = (mu_tilde > 0).sum(), (eta > 0).sum()
    A = np.zeros((np.sum(dim_mat), np.prod(dim_mat)))
    A[np.repeat(np.arange(dim_mat[0]), dim_mat[1]), np.arange(A.shape[1])] = 1
    A[np.tile(np.arange(dim_mat[1]), dim_mat[0]) + dim_mat[0], np.arange(A.shape[1])] = 1
    b = np.hstack((mu_tilde[mu_tilde > 0], eta[eta > 0]))
    R = np.eye(A.shape[1])
    sol = solve_int(A, b, R)
    sol = sol.reshape((sol.shape[0], dim_mat[0], dim_mat[1]))
    sol = np.transpose(sol, axes=(0, 2, 1))

    sol_vec = np.zeros((sol.shape[0], eta.shape[0], mu_tilde.shape[0]))
    sol_vec[np.ix_(np.arange(sol.shape[0]), eta > 0, mu_tilde > 0)] = sol

    return np.prod(multinom(sol_vec) * mult_pow(Y, sol_vec), axis=1).sum()


def S_d_func(mu, eta, C):
    dim_mat = (mu > 0).sum(), (eta > 0).sum()
    A = np.zeros((np.sum(dim_mat), np.prod(dim_mat)))
    A[np.repeat(np.arange(dim_mat[0]), dim_mat[1]), np.arange(A.shape[1])] = 1
    A[np.tile(np.arange(dim_mat[1]), dim_mat[0]) + dim_mat[0], np.arange(A.shape[1])] = 1
    b = np.hstack((mu[mu > 0], eta[eta > 0]))
    R = np.eye(A.shape[1])
    sol = solve_int(A, b, R)
    sol = sol.reshape((sol.shape[0], dim_mat[0], dim_mat[1]))
    sol = np.transpose(sol, axes=(0, 2, 1))

    sol_vec = np.zeros((sol.shape[0], eta.shape[0], mu.shape[0]))
    sol_vec[np.ix_(np.arange(sol.shape[0]), eta > 0, mu > 0)] = sol

    return np.prod(multinom(sol_vec) * mult_pow(C, sol_vec), axis=1).sum()


# Check equations
d = 3
k = 6

np.random.seed(0)
Y = np.random.normal(size=(k, k))
C = np.random.normal(size=(k, d))
Xs = np.random.normal(size=d)
Xt = np.random.normal(size=d)
Zs = np.random.normal(size=k)
Zt = Y @ Zs + C @ Xs

lamb = np.array([1, 0, 0])
lamb_tilde = np.array([0, 2, 1, 0, 0, 0])
dictd = return_dict(d, order=4)
dictk = return_dict(k, order=4)

result = mult_pow(Xt, lamb) * mult_pow(Zt, lamb_tilde)
result2 = mult_pow(Xt, lamb) * mult_pow(Y @ Zs + C @ Xs, lamb_tilde)

eta = dictk[mask(lamb_tilde, dictk, typ='leq')]
result3 = mult_pow(Xt, lamb) * np.sum(multi_binom(lamb_tilde, eta) * mult_pow(C @ Xs, lamb_tilde - eta) * mult_pow(Y @ Zs, eta))


def S_tilde(mu_tilde, eta, Y):
    dim_mat = (mu_tilde > 0).sum(), (eta > 0).sum()
    A = np.zeros((np.sum(dim_mat), np.prod(dim_mat)))
    A[np.repeat(np.arange(dim_mat[0]), dim_mat[1]), np.arange(A.shape[1])] = 1
    A[np.tile(np.arange(dim_mat[1]), dim_mat[0]) + dim_mat[0], np.arange(A.shape[1])] = 1
    b = np.hstack((mu_tilde[mu_tilde > 0], eta[eta > 0]))
    R = np.eye(A.shape[1])
    sol = solve_int(A, b, R)
    sol = sol.reshape((sol.shape[0], dim_mat[0], dim_mat[1]))
    sol = np.transpose(sol, axes=(0, 2, 1))

    sol_vec = np.zeros((sol.shape[0], eta.shape[0], mu_tilde.shape[0]))
    sol_vec[np.ix_(np.arange(sol.shape[0]), eta > 0, mu_tilde > 0)] = sol

    return np.prod(multinom(sol_vec) * mult_pow(Y, sol_vec), axis=1).sum()


eta = dictk[mask(lamb_tilde, dictk, typ='leq')]
temp = []
for et in eta:
    mu_tilde = dictk[mask(et, dictk, typ='eq_abs')]
    temp.append(multi_binom(lamb_tilde, et) * mult_pow(C @ Xs, lamb_tilde - et) * np.sum([S_tilde(mu_tild, et, Y) * mult_pow(Zs, mu_tild) for mu_tild in mu_tilde]))
result4 = mult_pow(Xt, lamb) * np.sum(temp)

mu_tilde = dictk[mask(lamb_tilde, dictk, typ='leq_abs')]
temp = []
for mu_tild in tqdm(mu_tilde):
    eta = dictk[mask(lamb_tilde, dictk, typ='leq') & mask(mu_tild, dictk, typ='eq_abs')]
    temp.append(mult_pow(Zs, mu_tild) * np.sum([multi_binom(lamb_tilde, et) * mult_pow(C @ Xs, lamb_tilde - et) * S_tilde(mu_tild, et, Y) for et in eta]))
result5 = mult_pow(Xt, lamb) * np.sum(temp)


def S(mu, eta, C):
    dim_mat = (mu > 0).sum(), (eta > 0).sum()
    A = np.zeros((np.sum(dim_mat), np.prod(dim_mat)))
    A[np.repeat(np.arange(dim_mat[0]), dim_mat[1]), np.arange(A.shape[1])] = 1
    A[np.tile(np.arange(dim_mat[1]), dim_mat[0]) + dim_mat[0], np.arange(A.shape[1])] = 1
    b = np.hstack((mu[mu > 0], eta[eta > 0]))
    R = np.eye(A.shape[1])
    sol = solve_int(A, b, R)
    sol = sol.reshape((sol.shape[0], dim_mat[0], dim_mat[1]))
    sol = np.transpose(sol, axes=(0, 2, 1))

    sol_vec = np.zeros((sol.shape[0], eta.shape[0], mu.shape[0]))
    sol_vec[np.ix_(np.arange(sol.shape[0]), eta > 0, mu > 0)] = sol

    return np.prod(multinom(sol_vec) * mult_pow(C, sol_vec), axis=1).sum()


mu_tilde = dictk[mask(lamb_tilde, dictk, typ='leq_abs')]
temp = []
for mu_tild in tqdm(mu_tilde):
    nu = dictd[mask(lamb_tilde - mu_tild, dictd, typ='eq_abs')]
    temp_inner = []
    for n in nu:
        eta = dictk[mask(lamb_tilde, dictk, typ='leq') & mask(n, dictk, typ='eq_abs')]
        temp_inner.append(np.sum([multi_binom(lamb_tilde, et) * S_tilde(mu_tild, lamb_tilde - et, Y) * S(n, et, C) for et in eta]) * mult_pow(Xs, n))
    c_lambt_mut = np.sum(temp_inner)
    temp.append(mult_pow(Zs, mu_tild) * c_lambt_mut)
result6 = mult_pow(Xt, lamb) * np.sum(temp)


def S_func(trans, order):
    dim1, dim2 = trans.shape
    dict1, dict2 = return_dict(dim1, order), return_dict(dim2, order)
    raw_collections1 = np.array([np.sort(dict1[i][dict1[i] != 0]).tolist() for i in range(n_dim(dim1, order=4))], dtype=object)
    raw_collections2 = np.array([np.sort(dict2[i][dict2[i] != 0]).tolist() for i in range(n_dim(dim2, order=4))], dtype=object)
    locations1 = [np.where(np.in1d(dict1[i], raw_collections1[i]))[0][np.argsort(dict1[i][dict1[i] != 0])] for i in range(n_dim(dim1, order=4))]
    locations2 = [np.where(np.in1d(dict2[i], raw_collections2[i]))[0][np.argsort(dict2[i][dict2[i] != 0])] for i in range(n_dim(dim2, order=4))]
    collections1 = np.array([x for i, x in enumerate(raw_collections1.tolist()) if x not in raw_collections1.tolist()[:i]], dtype=object)
    collections2 = np.array([x for i, x in enumerate(raw_collections2.tolist()) if x not in raw_collections2.tolist()[:i]], dtype=object)
    coll_locator1 = np.array([collections1.tolist().index(x) for x in raw_collections1])
    coll_locator2 = np.array([collections2.tolist().index(x) for x in raw_collections2])

    raw_combinations = np.array(list(product(collections1, collections2)), dtype=object)
    coll_combinations = np.array(sum([list(product(list(compress(collections1, [np.sum(coll) == i for coll in collections1])), list(compress(collections2, [np.sum(coll) == i for coll in collections2])))) for i in range(1, 5)], []), dtype=object)
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

    I, J = np.meshgrid(np.arange(n_dim(dim1, order=4)), np.arange(n_dim(dim2, order=4)))
    S = np.zeros((n_dim(dim1, order=4), n_dim(dim2, order=4)))
    for i in trange(n_dim(dim1, order=4)):
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


mu_tilde = dictk[mask(lamb_tilde, dictk, typ='leq_abs')]
temp = []
for mu_tild in tqdm(mu_tilde):
    nus = dictd[mask(lamb_tilde - mu_tild, dictd, typ='eq_abs')]
    etas = dictk[mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs')]
    c_lambt_mut = np.sum([multi_binom(lamb_tilde, eta) * S_tilde(mu_tild, lamb_tilde - eta, Y) * S(nu, eta, C) * mult_pow(Xs, nu) for nu, eta in product(nus, etas)])
    temp.append(mult_pow(Zs, mu_tild) * c_lambt_mut)
result7 = mult_pow(Xt, lamb) * np.sum(temp)


S_mat = S_func(Y, order=4)
S_mat_d = S_func(C, order=4)
mu_tilde, mu_tilde_ind = dictk[mask(lamb_tilde, dictk, typ='leq_abs')], np.where(mask(lamb_tilde, dictk, typ='leq_abs'))[0]
temp = []
for mu_tild, mu_tild_ind in zip(mu_tilde, mu_tilde_ind):
    nus, nus_ind = dictd[mask(lamb_tilde - mu_tild, dictd, typ='eq_abs')], np.where(mask(lamb_tilde - mu_tild, dictd, typ='eq_abs'))[0]
    etas, etas_ind = dictk[mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs')], np.where(mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs'))[0]
    lamb_eta_ind = mult_to_ind(lamb_tilde - etas, dictk)
    prods, prods_int, prods_int2 = product(nus, etas), product(nus_ind, etas_ind), product(nus_ind, lamb_eta_ind)
    c_lambt_mut = np.sum([multi_binom(lamb_tilde, prod[1]) * S_mat[prod_int2[1], mu_tild_ind] * S_mat_d[prod_int[1], prod_int[0]] * mult_pow(Xs, prod[0]) for prod, prod_int, prod_int2 in zip(prods, prods_int, prods_int2)])
    temp.append(mult_pow(Zs, mu_tild) * c_lambt_mut)
result8 = mult_pow(Xt, lamb) * np.sum(temp)


# What happens if one component is 1?
d = 3
k = 6

np.random.seed(0)
Y = np.random.normal(size=(k, k))
C = np.random.normal(size=(k, d))
Xs = np.random.normal(size=d)
Xt = np.random.normal(size=d)
Zs = np.random.normal(size=k)

C[-1] = 0
Y[-1] = [0, 0, 0, 0, 0, 1]
Zs[-1] = 1

Zt = Y @ Zs + C @ Xs

lamb = np.array([1, 0, 0])
lamb_tilde = np.array([0, 2, 1, 0, 0, 0])
dictd = return_dict(d, order=4)
dictk = return_dict(k, order=4)

result = mult_pow(Xt, lamb) * mult_pow(Zt, lamb_tilde)


def S_func(trans, order):
    dim1, dim2 = trans.shape
    dict1, dict2 = return_dict(dim1, order), return_dict(dim2, order)
    raw_collections1 = np.array([np.sort(dict1[i][dict1[i] != 0]).tolist() for i in range(n_dim(dim1, order=4))], dtype=object)
    raw_collections2 = np.array([np.sort(dict2[i][dict2[i] != 0]).tolist() for i in range(n_dim(dim2, order=4))], dtype=object)
    locations1 = [np.where(np.in1d(dict1[i], raw_collections1[i]))[0][np.argsort(dict1[i][dict1[i] != 0])] for i in range(n_dim(dim1, order=4))]
    locations2 = [np.where(np.in1d(dict2[i], raw_collections2[i]))[0][np.argsort(dict2[i][dict2[i] != 0])] for i in range(n_dim(dim2, order=4))]
    collections1 = np.array([x for i, x in enumerate(raw_collections1.tolist()) if x not in raw_collections1.tolist()[:i]], dtype=object)
    collections2 = np.array([x for i, x in enumerate(raw_collections2.tolist()) if x not in raw_collections2.tolist()[:i]], dtype=object)
    coll_locator1 = np.array([collections1.tolist().index(x) for x in raw_collections1])
    coll_locator2 = np.array([collections2.tolist().index(x) for x in raw_collections2])

    raw_combinations = np.array(list(product(collections1, collections2)), dtype=object)
    coll_combinations = np.array(sum([list(product(list(compress(collections1, [np.sum(coll) == i for coll in collections1])), list(compress(collections2, [np.sum(coll) == i for coll in collections2])))) for i in range(1, 5)], []), dtype=object)
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

    I, J = np.meshgrid(np.arange(n_dim(dim1, order=4)), np.arange(n_dim(dim2, order=4)))
    S = np.zeros((n_dim(dim1, order=4), n_dim(dim2, order=4)))
    for i in trange(n_dim(dim1, order=4)):
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


S_mat = S_func(Y, order=4)
S_mat_d = S_func(C, order=4)
mu_tilde, mu_tilde_ind = dictk[mask(lamb_tilde, dictk, typ='leq_abs')], np.where(mask(lamb_tilde, dictk, typ='leq_abs'))[0]
temp = []
for mu_tild, mu_tild_ind in zip(mu_tilde, mu_tilde_ind):
    nus, nus_ind = dictd[mask(lamb_tilde - mu_tild, dictd, typ='eq_abs')], np.where(mask(lamb_tilde - mu_tild, dictd, typ='eq_abs'))[0]
    etas, etas_ind = dictk[mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs')], np.where(mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs'))[0]
    lamb_eta_ind = mult_to_ind(lamb_tilde - etas, dictk)
    prods, prods_int, prods_int2 = product(nus, etas), product(nus_ind, etas_ind), product(nus_ind, lamb_eta_ind)
    c_lambt_mut = np.sum([multi_binom(lamb_tilde, prod[1]) * S_mat[prod_int2[1], mu_tild_ind] * S_mat_d[prod_int[1], prod_int[0]] * mult_pow(Xs, prod[0]) for prod, prod_int, prod_int2 in zip(prods, prods_int, prods_int2)])
    print(c_lambt_mut)
    temp.append(mult_pow(Zs, mu_tild) * c_lambt_mut)
result8 = mult_pow(Xt, lamb) * np.sum(temp)


# Check expectations
d = 3
k = 6

np.random.seed(0)
Y = np.random.normal(size=(k, k))
C = np.random.normal(size=(k, d))
Xs = np.random.normal(size=d)
Xt = np.random.normal(size=d)
Zs = np.random.normal(size=k)
B = np.random.normal(size=(n_dim(d, order=4), n_dim(d, order=4)))
c = np.random.normal(size=k)
Zt = c + Y @ Zs + C @ Xs

dictd = return_dict(d, order=4)
dictk = return_dict(k, order=4)


def calc_filter_B(B, Y, C, c=None, order=4):
    k, d = C.shape
    if c is not None:
        C = np.vstack((C, np.repeat(0, d)))
        Y = np.vstack((Y, np.repeat(0, k)))
        c = np.append(c, 1)[:, None]
        Y = np.hstack((Y, c))
        k += 1
        t = tqdm(total=2 * n_dim(k, order) + n_dim(d + k, order) + n_dim(d + k - 1, order), desc='Calculating S')
    else:
        t = tqdm(total=2 * n_dim(k, order) + n_dim(d + k, order), desc='Calculating S')

    dictd = return_dict(d, order)
    dictk = return_dict(k, order)

    B_large = np.zeros((n_dim(d + k, order), n_dim(d + k, order)))
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
            sol = solve_int(A, b, R, maxnumsol=factorial(order))
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

    if c is not None:
        B_final = np.zeros((n_dim(d + k - 1, order), n_dim(d + k - 1, order)))
        dict_final = return_dict(d + k - 1, order)
        t.set_description('Incorporating c')
        for i in range(n_dim(d + k - 1, order)):
            t.update(1)
            lamb_ind = dict_final[i]
            mu_inds, mu_locs = dict_final[mask(lamb_ind, dict_final, typ='leq_abs')], np.where(mask(lamb_ind, dict_final, typ='leq_abs'))[0]
            for mu_ind, mu_loc in zip(mu_inds, mu_locs):
                B_final[i, mu_loc] = B_large[mult_to_ind(np.append(lamb_ind, 0), dict_large), np.where((dict_large[:, :-1] == np.atleast_2d(mu_ind)[:, None]).all(-1))[1]].sum()
    else:
        B_final = B_large

    return B_final


B_large = calc_filter_B(B, Y, C, c, order=4)
dict_large = return_dict(d + k, order=4)

lamb = np.array([0, 0, 0])
lamb_tilde = np.array([1, 0, 0, 0, 0, 0])

mu_indices = dictd[mask(lamb, dictd, typ='leq_abs')]
result = B[mult_to_ind(lamb, dictd), mult_to_ind(mu_indices, dictd)] @ mult_pow(Xs, mu_indices) * mult_pow(Zt, lamb_tilde)

lamb_large = np.append(lamb, lamb_tilde)
mu_large = dict_large[mask(lamb_large, dict_large, typ='leq_abs')]
result_test = B_large[mult_to_ind(lamb_large, dict_large), mult_to_ind(mu_large, dict_large)] @ mult_pow(np.append(Xs, Zs), mu_large)

lamb = np.array([0, 0])
lamb_tilde = np.array([0, 1, 1, 1, 1])
mu_indices = dictd[mask(lamb, dictd, typ='leq_abs')]
result = B[mult_to_ind(lamb, dictd), mult_to_ind(mu_indices, dictd)] @ mult_pow(Xs, mu_indices) * mult_pow(Zt, lamb_tilde)

lamb_large = np.append(lamb, lamb_tilde)
mu_large = dict_final[mask(lamb_large, dict_final, typ='leq_abs')]
result_test1 = B_large_check[mult_to_ind(lamb_large, dict_final), mult_to_ind(mu_large, dict_final)] @ mult_pow(np.append(Xs, Zs), mu_large)
result_test2 = B_final[mult_to_ind(lamb_large, dict_final), mult_to_ind(mu_large, dict_final)] @ mult_pow(np.append(Xs, Zs), mu_large)
