import numpy as np
from functions import *
from scipy.linalg import expm

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
            mask = mask_less_than(dicts[i], dicts) & mask_less_than(dicts[j], dicts)
            lamb_ell = (ind_to_mult(i, dicts) - dicts)[mask]
            mu_ell = (ind_to_mult(j, dicts) - dicts)[mask]
            B[i, j] = (multi_binom(dicts[i], dicts[mask]) * A[mult_to_ind(lamb_ell, dicts), mult_to_ind(mu_ell, dicts)]).sum()
    return B


heston_Bc = get_continuous_B(heston_A, dicts)
heston_B = expm(heston_Bc)
heston_B_diff = np.zeros(heston_B.shape)
heston_B_diff[:, dicts[:, 1] == 0] = heston_B[:, dicts[:, 1] == 0]

heston_B_diff_sq = np.zeros((n_dim(dim + 1, order // 2), n_dim(dim + 1, order // 2)))
dicts_sq = return_dict(dim + 1, order // 2)
cols = (dicts_sq[:, 1] == 0) & (dicts_sq[:, 2] == 0)
for i in range(heston_B_diff_sq.shape[0]):
    lamb = ind_to_mult(i, dicts_sq)
    lamb_tilde = np.array([lamb[0], lamb[1] + 2 * lamb[2]])
    ind = mult_to_ind(lamb_tilde, dicts)
    heston_B_diff_sq[i, cols] = heston_B[ind, cols]