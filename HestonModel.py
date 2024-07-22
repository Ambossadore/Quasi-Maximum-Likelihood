import os
import pickle as pkl
import warnings
from pathlib import Path
from functools import partial
from itertools import product, compress
from time import time

from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import factorial
from tqdm import tqdm, trange

from functions import *


class KalmanFilter:
    def __init__(self, dim, a, A, C, E_0, Cov_0, C_lim=None, first_observed=0):
        self.dim = dim
        self.a = a
        self.A = A
        self.C = C
        self.C_lim = C_lim
        self.E_0 = E_0
        self.Cov_0 = Cov_0
        self.first_observed = first_observed
        self.Sig_tp1_t_list = None
        self.Sig_tt_list = None
        self.Sig_tp1_t_lim = None
        self.Sig_tt_lim = None
        self.H = np.pad(np.eye(dim - first_observed), ((0, 0), (first_observed, 0)))
        self.K_lim = None
        self.F_lim = None
        self.X_hat_tp1_t_list = None
        self.X_hat_tt_list = None
        self.X_hat_tp1_t_list_hom = None
        self.X_hat_tt_list_hom = None

    def build_covariance(self, t_max=100000):
        Cov_0 = self.Cov_0()
        Sig_tp1_t_list = [Cov_0]
        Sig_tt_list = [Cov_0 - Cov_0[:, self.first_observed:] @ np.linalg.pinv(Cov_0[self.first_observed:, self.first_observed:]) @ Cov_0[:, self.first_observed:].T]
        Sig_tp1_t_list.append(self.A() @ Sig_tt_list[0] @ self.A().T + self.C(t=1))
        Sig_tp1_t = Sig_tp1_t_list[-1]
        for t in range(1, t_max):
            Sig_tt = Sig_tp1_t - Sig_tp1_t[:, self.first_observed:] @ np.linalg.inv(Sig_tp1_t[self.first_observed:, self.first_observed:]) @ Sig_tp1_t[:, self.first_observed:].T
            Sig_tp1_t = self.A() @ Sig_tt @ self.A().T + self.C(t=t + 1)
            if np.isclose(Sig_tp1_t, Sig_tp1_t_list[-1], rtol=1e-7, atol=1e-7).all():
                break
            Sig_tp1_t_list.append(Sig_tp1_t)
            Sig_tt_list.append(Sig_tt)
        if len(Sig_tt_list) == t_max:
            warnings.warn('Kalman filter covariance matrices have not converged.')
        self.Sig_tp1_t_list = np.stack(Sig_tp1_t_list)
        self.Sig_tt_list = np.stack(Sig_tt_list)
        self.Sig_tp1_t_lim = Sig_tp1_t_list[-1]
        self.Sig_tt_lim = Sig_tt_list[-1]
        self.K_lim = self.A() @ self.Sig_tp1_t_lim[:, self.first_observed:] @ np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
        self.F_lim = self.A() - self.K_lim @ self.H

    def S_star(self, wrt):
        BB_lim_partial = self.C_lim(wrt=wrt, order=1)
        right_side = np.einsum('jk, kl, lmi -> ijm', self.A(), self.Sig_tt_lim, self.A(wrt=wrt, order=1).T) + np.einsum('ijk, kl, lm -> ijm', self.A(wrt=wrt, order=1), self.Sig_tt_lim, self.A().T) + BB_lim_partial
        S_star_vectorized = (np.linalg.inv(np.eye(self.dim ** 2) - np.kron(self.F_lim, self.F_lim)) @ right_side.reshape(np.size(wrt), self.dim**2, order='F').T).T
        return S_star_vectorized.reshape(np.size(wrt), self.dim, self.dim, order='F')

    def R_star(self, wrt):
        BB_lim_partial2 = self.C_lim(wrt=wrt, order=2)
        partial_A_i = self.A(wrt=wrt, order=1)[np.triu_indices(np.size(wrt))[0]]
        partial_A_j = self.A(wrt=wrt, order=1)[np.triu_indices(np.size(wrt))[1]]
        s_star_i = self.S_star(wrt=wrt)[np.triu_indices(np.size(wrt))[0]]
        s_star_j = self.S_star(wrt=wrt)[np.triu_indices(np.size(wrt))[1]]
        Sig_inv = np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])

        I_minus_kh = np.eye(self.dim) - self.Sig_tp1_t_lim[:, self.first_observed:] @ Sig_inv @ self.H
        s_star_i_tt = np.einsum('jk, ikl, lm -> ijm', I_minus_kh, s_star_i, I_minus_kh.T)
        s_star_j_tt = np.einsum('jk, ikl, lm -> ijm', I_minus_kh, s_star_j, I_minus_kh.T)

        s_star_i_tilde = np.einsum('jk, ikl, lm -> ijm', Sig_inv, s_star_i[:, self.first_observed:, self.first_observed:], Sig_inv)
        s_star_j_tilde = np.einsum('jk, ikl, lm -> ijm', Sig_inv, s_star_j[:, self.first_observed:, self.first_observed:], Sig_inv)
        s_hat = np.einsum('ijk, ikl -> ijl', s_star_j[:, :, self.first_observed:], s_star_i_tilde) + np.einsum('ijk, ikl -> ijl', s_star_i[:, :, self.first_observed:], s_star_j_tilde)
        s_hat_o = np.einsum('ijk, ikl -> ijl', s_star_j[:, self.first_observed:, self.first_observed:], s_star_i_tilde) + np.einsum('ijk, ikl -> ijl', s_star_i[:, self.first_observed:, self.first_observed:], s_star_j_tilde)
        r_tilde = np.einsum('ijk, kl, lmi -> ijm', partial_A_i, self.Sig_tt_lim, partial_A_j.T) + np.einsum('jk, ikl, lmi -> ijm', self.A(), s_star_i_tt, partial_A_j.T) + np.einsum('jk, ikl, lmi -> ijm', self.A(), s_star_j_tt, partial_A_i.T) + np.einsum('jk, kl, ilm -> ijm', self.A(), self.Sig_tt_lim, self.A(wrt=wrt, order=2))
        r_bar = np.einsum('ijk, kl -> ijl', s_hat, self.Sig_tp1_t_lim[:, self.first_observed:].T) - np.einsum('ijk, kl, lmi -> ijm', s_star_j[:, :, self.first_observed:], Sig_inv, s_star_i[:, :, self.first_observed:].T)
        right_side = sym(r_tilde) - np.einsum('jk, ikl, lm, mn -> ijn', self.K_lim, s_hat_o, self.Sig_tp1_t_lim[:, self.first_observed:].T, self.A().T) + np.einsum('jk, ikl, lm -> ijm', self.A(), sym(r_bar), self.A().T) + BB_lim_partial2
        R_star_vectorized = (np.linalg.inv(np.eye(self.dim ** 2) - np.kron(self.F_lim, self.F_lim)) @ right_side.reshape(BB_lim_partial2.shape[0], self.dim**2, order='F').T).T
        return R_star_vectorized.reshape(BB_lim_partial2.shape[0], self.dim, self.dim, order='F')

    def build_kalman_filter(self, observations, t_max=None, verbose=0, close_pb=True):
        if t_max is None:
            t_max = observations.shape[0]
        if self.Sig_tp1_t_list is None:
            raise Exception('Method build_covariance needs to be called before method kalman_filter.')
        observations = observations[:, self.first_observed:]
        X_hat_tp1_t_list = [self.E_0()]
        X_hat_tt_list = [self.E_0() + self.Cov_0()[:, self.first_observed:] @ np.linalg.pinv(self.Cov_0()[self.first_observed:, self.first_observed:]) @ (observations[0] - self.E_0()[self.first_observed:])]
        X_hat_tp1_t_list.append(self.a() + self.A() @ X_hat_tt_list[0])
        X_hat_tp1_t = X_hat_tp1_t_list[-1]
        t_conv = self.Sig_tp1_t_list.shape[0]
        Sig_inv = np.linalg.inv(self.Sig_tp1_t_list[1:, self.first_observed:, self.first_observed:])
        Sig_tp1_t_list_o = self.Sig_tp1_t_list[:, :, self.first_observed:]
        if isinstance(verbose, tqdm):
            tr = verbose
        elif verbose == 1:
            tr = tqdm(total=t_max)
        else:
            tr = None
        if verbose:
            tr.update(1)
        for t in range(1, t_conv):
            X_hat_tt = X_hat_tp1_t + Sig_tp1_t_list_o[t, :, :] @ Sig_inv[t - 1] @ (observations[t] - X_hat_tp1_t[self.first_observed:])
            X_hat_tp1_t = self.a() + self.A() @ X_hat_tt
            X_hat_tp1_t_list.append(X_hat_tp1_t)
            X_hat_tt_list.append(X_hat_tt)
            if verbose:
                tr.update(1)
        for t in range(t_conv, t_max):
            X_hat_tt = X_hat_tp1_t + self.Sig_tp1_t_lim[:, self.first_observed:] @ Sig_inv[-1] @ (observations[t] - X_hat_tp1_t[self.first_observed:])
            X_hat_tp1_t = self.a() + self.A() @ X_hat_tt
            X_hat_tp1_t_list.append(X_hat_tp1_t)
            X_hat_tt_list.append(X_hat_tt)
            if verbose:
                tr.update(1)
        if verbose and close_pb:
            tr.close()
        self.X_hat_tp1_t_list = np.stack(X_hat_tp1_t_list)
        self.X_hat_tt_list = np.stack(X_hat_tt_list)

    def build_kalman_filter_hom(self, observations, t_max=None, verbose=0, close_pb=True):
        if t_max is None:
            t_max = observations.shape[0]
        if self.Sig_tp1_t_lim is None:
            raise Exception('Method build_covariance needs to be called before method kalman_filter.')
        observations = observations[:, self.first_observed:]
        X_hat_tp1_t_list_hom = [self.E_0()]
        X_hat_tt_list_hom = [self.E_0() + self.Sig_tp1_t_lim[:, self.first_observed:] @ np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:]) @ (observations[0] - self.E_0()[self.first_observed:])]
        X_hat_tp1_t_list_hom.append(self.a() + self.A() @ X_hat_tt_list_hom[0])
        X_hat_tp1_t = X_hat_tp1_t_list_hom[-1]
        Sig_inv = np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
        Sig_tp1_t_lim_o = self.Sig_tp1_t_lim[:, self.first_observed:]
        if isinstance(verbose, tqdm):
            tr = verbose
        elif verbose == 1:
            tr = tqdm(total=t_max)
        else:
            tr = None
        if verbose:
            tr.update(1)
        for t in range(1, t_max):
            X_hat_tt = X_hat_tp1_t + Sig_tp1_t_lim_o @ Sig_inv @ (observations[t] - X_hat_tp1_t[self.first_observed:])
            X_hat_tp1_t = self.a() + self.A() @ X_hat_tt
            X_hat_tp1_t_list_hom.append(X_hat_tp1_t)
            X_hat_tt_list_hom.append(X_hat_tt)
            if verbose:
                tr.update(1)
        if verbose and close_pb:
            tr.close()
        self.X_hat_tp1_t_list_hom = np.stack(X_hat_tp1_t_list_hom)
        self.X_hat_tt_list_hom = np.stack(X_hat_tt_list_hom)

    def deriv_filter_hom(self, observations, wrt, t_max=None, verbose=0, close_pb=True):
        if t_max is None:
            t_max = observations.shape[0]
        s_star = self.S_star(wrt=wrt)
        partial_a = self.a(wrt=wrt, order=1)
        partial_A = self.A(wrt=wrt, order=1)
        observations = observations[:, self.first_observed:]
        Sig_inv = np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
        s_tilde = np.einsum('jk, ikl, lm -> ijm', Sig_inv, s_star[:, self.first_observed:, self.first_observed:], Sig_inv)
        k_tilde = self.Sig_tp1_t_lim[:, self.first_observed:] @ Sig_inv
        V_tp1_t_list = [self.E_0(order=1, wrt=wrt)]
        V_tt_list = [self.E_0(order=1, wrt=wrt) + (s_star[:, :, self.first_observed:] @ Sig_inv - np.einsum('jk, ikl -> ijl', self.Sig_tp1_t_lim[:, self.first_observed:], s_tilde)) @ (observations[0] - self.X_hat_tp1_t_list_hom[0, self.first_observed:]) - np.einsum('jk, ik -> ij', k_tilde, self.E_0(order=1, wrt=wrt)[:, self.first_observed:])]
        V_tp1_t_list.append(partial_a + partial_A @ self.X_hat_tt_list_hom[0] + np.einsum('jk, ik -> ij', self.A(), V_tt_list[0]))
        V_tp1_t = V_tp1_t_list[-1]
        if isinstance(verbose, tqdm):
            tr = verbose
        elif verbose == 1:
            tr = tqdm(total=t_max)
        else:
            tr = None
        if verbose:
            tr.update(1)
        for t in range(1, t_max):
            V_tt = V_tp1_t + (s_star[:, :, self.first_observed:] @ Sig_inv - np.einsum('jk, ikl -> ijl', self.Sig_tp1_t_lim[:, self.first_observed:], s_tilde)) @ (observations[t] - self.X_hat_tp1_t_list_hom[t, self.first_observed:]) - np.einsum('jk, ik -> ij', k_tilde, V_tp1_t[:, self.first_observed:])
            V_tp1_t = partial_a + partial_A @ self.X_hat_tt_list_hom[t] + np.einsum('jk, ik -> ij', self.A(), V_tt)
            V_tp1_t_list.append(V_tp1_t)
            V_tt_list.append(V_tt)
            if verbose:
                tr.update(1)
        if verbose and close_pb:
            tr.close()
        return np.stack(V_tp1_t_list), np.stack(V_tt_list)

    def deriv2_filter_hom(self, observations, wrt, t_max=None, verbose=0, close_pb=True, deriv_filters=None):
        if t_max is None:
            t_max = observations.shape[0]
        Sig_inv = np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
        s_star = self.S_star(wrt=wrt)
        s_star_i = s_star[np.triu_indices(np.size(wrt))[0]]
        s_star_j = s_star[np.triu_indices(np.size(wrt))[1]]
        s_star_i_tilde = np.einsum('jk, ikl, lm -> ijm', Sig_inv, s_star_i[:, self.first_observed:, self.first_observed:], Sig_inv)
        s_star_j_tilde = np.einsum('jk, ikl, lm -> ijm', Sig_inv, s_star_j[:, self.first_observed:, self.first_observed:], Sig_inv)
        s_hat = np.einsum('ijk, ikl -> ijl', s_star_j[:, :, self.first_observed:], s_star_i_tilde) + np.einsum('ijk, ikl -> ijl', s_star_i[:, :, self.first_observed:], s_star_j_tilde)
        s_hat_o = np.einsum('ijk, ikl -> ijl', s_star_j[:, self.first_observed:, self.first_observed:], s_star_i_tilde) + np.einsum('ijk, ikl -> ijl', s_star_i[:, self.first_observed:, self.first_observed:], s_star_j_tilde)

        r_star = self.R_star(wrt=wrt)
        partial_a = self.a(wrt=wrt, order=2)
        partial_A = self.A(wrt=wrt, order=2)
        partial_A_i = self.A(wrt=wrt, order=1)[np.triu_indices(np.size(wrt))[0]]
        partial_A_j = self.A(wrt=wrt, order=1)[np.triu_indices(np.size(wrt))[1]]
        if deriv_filters is not None:
            V_tp1_t, V_tt = deriv_filters
        else:
            V_tp1_t, V_tt = self.deriv_filter_hom(observations, wrt=wrt, t_max=t_max, verbose=verbose, close_pb=close_pb)
        V_tp1_t_i, V_tt_i = V_tp1_t[:, np.triu_indices(np.size(wrt))[0], :], V_tt[:, np.triu_indices(np.size(wrt))[0], :]
        V_tp1_t_j, V_tt_j = V_tp1_t[:, np.triu_indices(np.size(wrt))[1], :], V_tt[:, np.triu_indices(np.size(wrt))[1], :]
        observations = observations[:, self.first_observed:]

        k_tilde = self.Sig_tp1_t_lim[:, self.first_observed:] @ Sig_inv
        M = r_star[:, :, self.first_observed:] @ Sig_inv - s_hat + np.einsum('jk, ikl -> ijl', k_tilde, s_hat_o) - np.einsum('jk, ikl, lm -> ijm', k_tilde, r_star[:, self.first_observed:, self.first_observed:], Sig_inv)
        N_i = np.einsum('jk, ikl -> ijl', self.Sig_tp1_t_lim[:, self.first_observed:], s_star_i_tilde) - s_star_i[:, :, self.first_observed:] @ Sig_inv
        N_j = np.einsum('jk, ikl -> ijl', self.Sig_tp1_t_lim[:, self.first_observed:], s_star_j_tilde) - s_star_j[:, :, self.first_observed:] @ Sig_inv

        W_tp1_t_list = [self.E_0(order=2, wrt=wrt)]
        W_tt_list = [W_tp1_t_list[0] + M @ (observations[0] - self.X_hat_tp1_t_list_hom[0, self.first_observed:]) + np.einsum('ijk, ik -> ij', N_j, V_tp1_t_i[0, :, self.first_observed:]) + np.einsum('ijk, ik -> ij', N_i, V_tp1_t_j[0, :, self.first_observed:]) - np.einsum('jk, ik -> ij', k_tilde, W_tp1_t_list[0][:, self.first_observed:])]
        W_tp1_t_list.append(partial_a + partial_A @ self.X_hat_tt_list_hom[0] + np.einsum('ijk, ik -> ij', partial_A_j, V_tt_i[0]) + np.einsum('ijk, ik -> ij', partial_A_i, V_tt_j[0]) + np.einsum('jk, ik -> ij', self.A(), W_tt_list[0]))
        W_tp1_t = W_tp1_t_list[-1]
        if isinstance(verbose, tqdm):
            tr = verbose
        elif verbose == 1:
            tr = tqdm(total=t_max)
        else:
            tr = None
        if verbose:
            tr.update(1)
        for t in range(1, t_max):
            eps = observations[t] - self.X_hat_tp1_t_list_hom[t, self.first_observed:]
            W_tt = W_tp1_t + M @ eps + np.einsum('ijk, ik -> ij', N_j, V_tp1_t_i[t, :, self.first_observed:]) + np.einsum('ijk, ik -> ij', N_i, V_tp1_t_j[t, :, self.first_observed:]) - np.einsum('jk, ik -> ij', k_tilde, W_tp1_t[:, self.first_observed:])
            W_tp1_t = partial_a + partial_A @ self.X_hat_tt_list_hom[t] + np.einsum('ijk, ik -> ij', partial_A_j, V_tt_i[t]) + np.einsum('ijk, ik -> ij', partial_A_i, V_tt_j[t]) + np.einsum('jk, ik -> ij', self.A(), W_tt)
            W_tp1_t_list.append(W_tp1_t)
            W_tt_list.append(W_tt)
            if verbose:
                tr.update(1)
        if verbose and close_pb:
            tr.close()
        return np.stack(W_tp1_t_list), np.stack(W_tt_list)


class PolynomialModel:
    def __init__(self, first_observed, init, dt, true_param=None):
        self.first_observed = first_observed
        self.true_param = np.array(true_param) if true_param is not None else None
        self.init = init
        self.dt = dt
        self.dim = None
        self.params_names = ''
        self.params_bounds = None
        self.savestring = ''
        self.observations = None
        self.seed = None
        self.wrt = None

        self.filter_c4 = None
        self.filter_c2 = None
        self.filter_C4 = None
        self.filter_C2 = None
        self.filter_Y4 = None
        self.filter_Y2 = None
        self.filter_B4 = None
        self.lim_expec4 = None
        self.filter_B2 = None
        self.lim_expec2 = None
        self.U = None
        self.W = None
        self.kalman_filter = None

        init_path = './saves/' + self.__class__.__name__
        paths = [init_path + string for string in ['/Covariance/Estimated', '/Covariance/Explicit', '/Observations', '/Polynomial Matrices', '/QML Sequences']]

        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def a(self, param, order=None, wrt=None):
        pass

    def A(self, param, order=None, wrt=None):
        pass

    def B(self, param, order):
        pass

    def C(self, param, t):
        pass

    def C_lim(self, param, order=None, wrt=None):
        pass

    def Q(self, param, num):
        pass

    def calc_filter_B(self, order):
        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        if (self.filter_c4 is None) | (self.filter_C4 is None) | (self.filter_Y4 is None):
            raise Exception('Method setup_filter has to be called first.')

        if order == 4:
            C = self.filter_C4
            c = self.filter_c4
            Y = self.filter_Y4
            filepath = './saves/' + self.__class__.__name__ + '/Polynomial Matrices/B4{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        elif order == 2:
            C = self.filter_C2
            c = self.filter_c2
            Y = self.filter_Y2
            filepath = './saves/' + self.__class__.__name__ + '/Polynomial Matrices/B2{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        else:
            raise Exception('Argument order has to be 4 or 2.')

        k, d = C.shape
        B = self.B(param=self.true_param, order=order)
        if np.any(c):
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
                for j in range(n_dim(dim2, order=order)):
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

        if np.any(c):
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

        t.set_description('Calculating limiting power expectations')
        if order == 4:
            self.filter_B4 = B_final
            self.lim_expec4 = np.append(1, np.linalg.inv(np.eye(B_final.shape[0] - 1) - B_final[1:, 1:]) @ B_final[1:, 0])
            np.savetxt(filepath, self.filter_B4)
        elif order == 2:
            self.filter_B2 = B_final
            self.lim_expec2 = np.append(1, np.linalg.inv(np.eye(B_final.shape[0] - 1) - B_final[1:, 1:]) @ B_final[1:, 0])
            np.savetxt(filepath, self.filter_B2)

    def setup_filter(self, wrt):
        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        self.wrt = wrt

        filepath_U = './saves/' + self.__class__.__name__ + '/Covariance/Explicit/U{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        filepath_W = './saves/' + self.__class__.__name__ + '/Covariance/Explicit/W{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        self.U = np.atleast_2d(np.loadtxt(filepath_U)) if os.path.exists(filepath_U) else None
        self.W = np.atleast_2d(np.loadtxt(filepath_W)) if os.path.exists(filepath_W) else None

        self.dicts = return_dict(self.dim * (np.size(wrt) + 2), order=4)
        self.dicts2 = return_dict(self.dim * (int(np.size(wrt) * (np.size(wrt) + 1) / 2) + np.size(wrt) + 2), order=2)

        partial_a = partial(self.a, param=self.true_param)
        partial_A = partial(self.A, param=self.true_param)
        partial_C = partial(self.C, param=self.true_param)
        partial_E_0 = partial(self.init.E_0, param=self.true_param)
        partial_Cov_0 = partial(self.init.Cov_0, param=self.true_param)
        partial_C_lim = partial(self.C_lim, param=self.true_param)
        self.kalman_filter = KalmanFilter(dim=self.dim, a=partial_a, A=partial_A, C=partial_C, E_0=partial_E_0, Cov_0=partial_Cov_0, C_lim=partial_C_lim, first_observed=self.first_observed)
        self.kalman_filter.build_covariance()

        k = np.size(self.wrt)
        a_0 = self.a(self.true_param, order=0)
        a_1 = self.a(self.true_param, order=1, wrt=self.wrt)
        a_2 = self.a(self.true_param, order=2, wrt=self.wrt)
        self.filter_c4 = np.append(a_0, a_1.flatten())
        self.filter_c2 = np.hstack((a_0, a_1.flatten(), a_2.flatten()))

        A_0 = self.kalman_filter.K_lim @ self.kalman_filter.H
        A_00 = self.kalman_filter.F_lim

        S = self.kalman_filter.S_star(wrt=self.wrt)
        R = self.kalman_filter.R_star(wrt=self.wrt)
        Sig = self.kalman_filter.Sig_tp1_t_lim
        Sig_inv = np.linalg.inv(Sig[self.first_observed:, self.first_observed:])
        S_tilde = Sig_inv @ S[:, self.first_observed:, self.first_observed:] @ Sig_inv
        K_tilde = Sig[:, self.first_observed:] @ Sig_inv

        S_i = S[np.triu_indices(k)[0]]
        S_j = S[np.triu_indices(k)[1]]
        S_i_tilde = np.einsum('jk, ikl, lm -> ijm', Sig_inv, S_i[:, self.first_observed:, self.first_observed:], Sig_inv)
        S_j_tilde = np.einsum('jk, ikl, lm -> ijm', Sig_inv, S_j[:, self.first_observed:, self.first_observed:], Sig_inv)
        S_hat = np.einsum('ijk, ikl -> ijl', S_j[:, :, self.first_observed:], S_i_tilde) + np.einsum('ijk, ikl -> ijl', S_i[:, :, self.first_observed:], S_j_tilde)
        S_hat_o = np.einsum('ijk, ikl -> ijl', S_j[:, self.first_observed:, self.first_observed:], S_i_tilde) + np.einsum('ijk, ikl -> ijl', S_i[:, self.first_observed:, self.first_observed:], S_j_tilde)
        SS = (S[:, :, self.first_observed:] @ Sig_inv - Sig[:, self.first_observed:] @ S_tilde)

        A_1 = (self.A(self.true_param, order=1, wrt=self.wrt) @ K_tilde + self.A(self.true_param) @ SS) @ self.kalman_filter.H
        A_10 = self.A(self.true_param, order=1, wrt=self.wrt) @ (np.eye(self.dim) - K_tilde @ self.kalman_filter.H) - self.A(self.true_param) @ SS @ self.kalman_filter.H
        A_11 = self.kalman_filter.F_lim

        prod_i = np.einsum('ijk, lkm -> iljm', self.A(self.true_param, order=1, wrt=self.wrt), SS) @ self.kalman_filter.H
        prod_ij = (prod_i + np.transpose(prod_i, (1, 0, 2, 3)))[np.triu_indices(k)]
        M = R[:, :, self.first_observed:] @ Sig_inv - S_hat + np.einsum('jk, ikl -> ijl', K_tilde, S_hat_o) - np.einsum('jk, ikl, lm -> ijm', K_tilde, R[:, self.first_observed:, self.first_observed:], Sig_inv)
        N_i = np.einsum('jk, ikl -> ijl', Sig[:, self.first_observed:], S_i_tilde) - S_i[:, :, self.first_observed:] @ Sig_inv
        N_j = np.einsum('jk, ikl -> ijl', Sig[:, self.first_observed:], S_j_tilde) - S_j[:, :, self.first_observed:] @ Sig_inv

        A_2 = self.A(self.true_param, order=2, wrt=self.wrt) @ K_tilde @ self.kalman_filter.H + prod_ij + self.A(self.true_param) @ M @ self.kalman_filter.H
        A_20 = self.A(self.true_param, order=2, wrt=self.wrt) @ (np.eye(self.dim) - K_tilde @ self.kalman_filter.H) - prod_ij - self.A(self.true_param) @ M @ self.kalman_filter.H
        A_21_i = self.A(self.true_param, order=1, wrt=self.wrt)[np.triu_indices(k)[0]] @ (np.eye(self.dim) - K_tilde @ self.kalman_filter.H) + self.A(self.true_param) @ N_i @ self.kalman_filter.H
        A_21_j = self.A(self.true_param, order=1, wrt=self.wrt)[np.triu_indices(k)[1]] @ (np.eye(self.dim) - K_tilde @ self.kalman_filter.H) + self.A(self.true_param) @ N_j @ self.kalman_filter.H
        A_22 = self.kalman_filter.F_lim

        self.filter_C4 = np.block([[A_0], [np.vstack(A_1)]])
        self.filter_C2 = np.block([[A_0], [np.vstack(A_1)], [np.vstack(A_2)]])

        self.filter_Y4 = np.block([[A_00, np.zeros((self.dim, self.dim * k))], [np.vstack(A_10), np.kron(np.eye(k), A_11)]])
        first_row_Y2 = [A_00, np.zeros((self.dim, self.dim * int(k + k * (k + 1) / 2)))]
        second_row_Y2 = [np.vstack(A_10), np.kron(np.eye(k), A_11), np.zeros((self.dim * k, self.dim * int(k * (k + 1) / 2)))]

        block = np.zeros((self.dim * int(k * (k + 1) / 2), self.dim * k))
        for l in range(int(k * (k + 1) / 2)):
            i, j = np.triu_indices(k)[0][l], np.triu_indices(k)[1][l]
            block[(self.dim * l):(self.dim * (l + 1)), (self.dim * i):(self.dim * (i + 1))] += A_21_j[l]
            block[(self.dim * l):(self.dim * (l + 1)), (self.dim * j):(self.dim * (j + 1))] += A_21_i[l]

        third_row_Y2 = [np.vstack(A_20), block, np.kron(np.eye(int(k * (k + 1) / 2)), A_22)]
        self.filter_Y2 = np.block([first_row_Y2, second_row_Y2, third_row_Y2])

        filepath_B4 = './saves/' + self.__class__.__name__ + '/Polynomial Matrices/B4{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        if os.path.exists(filepath_B4):
            self.filter_B4 = np.loadtxt(filepath_B4)
            self.lim_expec4 = np.append(1, np.linalg.inv(np.eye(self.filter_B4.shape[0] - 1) - self.filter_B4[1:, 1:]) @ self.filter_B4[1:, 0])
        else:
            self.calc_filter_B(order=4)

        filepath_B2 = './saves/' + self.__class__.__name__ + '/Polynomial Matrices/B2{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        if os.path.exists(filepath_B2):
            self.filter_B2 = np.loadtxt(filepath_B2)
            self.lim_expec2 = np.append(1, np.linalg.inv(np.eye(self.filter_B2.shape[0] - 1) - self.filter_B2[1:, 1:]) @ self.filter_B2[1:, 0])
        else:
            self.calc_filter_B(order=2)

    def generate_observations(self, t_max, inter_steps, seed, verbose):
        pass

    @classmethod
    def from_observations(cls, first_observed, init, dt, obs, inter_steps=None, true_param=None, wrt=None, seed=None):
        if isinstance(obs, str):
            filename = './saves/' + cls.__name__ + '/Observations/' + obs
        else:
            if seed is not None:
                filename = './saves/' + cls.__name__ + '/Observations/observations_par=[' + ''.join('{:.3f}, '.format(item) for item in true_param[:-1]) + '{:.3f}]_dt={:.1e}_seed{}_{}obs.txt'.format(true_param[-1], dt, seed, obs)
            else:
                filename = './saves/' + cls.__name__ + '/Observations/observations_par=[' + ''.join('{:.3f}, '.format(item) for item in true_param[:-1]) + '{:.3f}]_dt={:.1e}_{}obs.txt'.format(true_param[-1], dt, obs)
        obj = cls(first_observed=first_observed, init=init, dt=dt, true_param=true_param, wrt=wrt)

        if os.path.exists(filename):
            observations = np.loadtxt(filename)
            obj.observations = observations
        else:
            obj.generate_observations(t_max=obs * dt, inter_steps=inter_steps, seed=seed, verbose=1)

        obj.seed = seed
        return obj

    def log_lik(self, param, t, verbose=0):
        kfilter = KalmanFilter(dim=self.dim, a=partial(self.a, param=param), A=partial(self.A, param=param), C=partial(self.C, param=param), E_0=partial(self.init.E_0, param=param), Cov_0=partial(self.init.Cov_0, param=param), first_observed=self.first_observed)
        kfilter.build_covariance(t_max=t)
        kfilter.build_kalman_filter(observations=self.observations, t_max=t, verbose=verbose)
        eps = (self.observations[1:t + 1, self.first_observed:] - kfilter.X_hat_tp1_t_list[1:t + 1, self.first_observed:])
        Sig_tp1_t_list_inv = np.linalg.inv(kfilter.Sig_tp1_t_list[1:, self.first_observed:, self.first_observed:])
        Sig_tp1_t_list_det = np.log(np.abs(np.linalg.det(kfilter.Sig_tp1_t_list[1:, self.first_observed:, self.first_observed:])))
        if t >= kfilter.Sig_tp1_t_list.shape[0]:
            final_inv = np.linalg.inv(kfilter.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
            final_det = np.log(np.abs(np.linalg.det(kfilter.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])))
            Sig_tp1_t_list_inv = np.vstack((Sig_tp1_t_list_inv, np.tile(final_inv, (t - kfilter.Sig_tp1_t_list.shape[0] + 1, 1, 1))))
            Sig_tp1_t_list_det = np.append(Sig_tp1_t_list_det, np.repeat(final_det, t - kfilter.Sig_tp1_t_list.shape[0] + 1))
        return np.sum(-0.5 * (Sig_tp1_t_list_det[:t] + np.einsum('ij, ijk, ik -> i', eps, Sig_tp1_t_list_inv[:t, :, :], eps)))

    def fit_qml(self, fit_parameter, initial, t=None, verbose=1, update_estimate=False):
        global it
        it = 1

        if t is None:
            t = self.observations.shape[0] - 1

        fix_keys = np.setdiff1d(np.arange(np.size(self.true_param)), fit_parameter)
        fix_indices = fix_keys - np.arange(len(fix_keys))
        params_names = np.atleast_1d(self.params_names[fit_parameter])

        if np.isnan(np.array(self.true_param)[fix_keys]).any():
            raise Exception('Cannot fix parameters ' + ', '.join(self.params_names[fix_keys]) + ' if they are not known in advance')
        if np.isnan(self.true_param).any():
            update_estimate = True

        neg_loglik = lambda param: -self.log_lik(np.insert(param, fix_indices, np.array(self.true_param)[fix_keys]), t)

        def callback(params):
            """ Simple callback function to be submitted to the scipy.optimize.minimize routine in the calibrate method"""
            global it
            print('It: {:4d},  '.format(it) + ''.join([name + ' = {:.7e}, '.format(param) for name, param in zip(params_names, params)]) + 'Average Log-Likelihood = {:.15f}'.format(-neg_loglik(params) / t))
            it += 1

        bounds = np.atleast_2d(self.params_bounds[fit_parameter]).tolist()
        start = time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if verbose == 1:
                res = minimize(neg_loglik, initial, bounds=bounds, callback=callback, method='L-BFGS-B')
            else:
                res = minimize(neg_loglik, initial, bounds=bounds, method='L-BFGS-B')
        end = time()
        if verbose == 1:
            print('Elapsed Time: ' + format_time(end - start))

        result = res.x[0] if np.size(res.x) == 1 else res.x

        if update_estimate:
            self.true_param[fit_parameter] = result
            self.savestring = 'par=[{:.3f}, {:.3f}, {:.3f}, {:.3f}]_dt={:.1e}'.format(self.true_param[0], self.true_param[1], self.true_param[2], self.true_param[3], self.dt)

        return result

    def fit_qml_sequence(self, fit_parameter, initial, t_max=None, every=50, verbose=1, update_estimate=False):
        start = time()
        fit_parameter = np.atleast_1d(fit_parameter).tolist()

        if np.isnan(self.true_param).any():
            update_estimate = True

        if t_max is None:
            t_max = self.observations.shape[0] - 1
        if self.seed is not None:
            filename = './saves/' + self.__class__.__name__ + '/QML Sequences/qml{}_m={}_{}_seed{}_{}obs_every{}th.txt'.format(fit_parameter, self.first_observed, self.savestring, self.seed, t_max, every)
        else:
            filename = './saves/' + self.__class__.__name__ + '/QML Sequences/qml{}_m={}_{}_{}obs_every{}th.txt'.format(fit_parameter, self.first_observed, self.savestring, t_max, every)
        if os.path.exists(filename):
            qml_list = np.loadtxt(filename)

            if update_estimate:
                self.true_param[fit_parameter] = qml_list[-1]
                self.savestring = 'par=[{:.3f}, {:.3f}, {:.3f}, {:.3f}]_dt={:.1e}'.format(self.true_param[0], self.true_param[1], self.true_param[2], self.true_param[3], self.dt)

        else:
            t_range = np.arange(t_max + every)[::every][1:]
            qml_list = []
            if verbose == 0:
                t_range = tqdm(t_range)
            for t in t_range:
                if verbose == 1:
                    current = time()
                    print('Fitting QML with t = {} observations. Total Elapsed Time: {}'.format(t, format_time(current - start)))
                qml_list.append(self.fit_qml(fit_parameter=fit_parameter, initial=initial, t=t, verbose=verbose))
                initial = qml_list[-1]
            qml_list = np.array(qml_list)

            if update_estimate:
                self.true_param[fit_parameter] = qml_list[-1]
                self.savestring = 'par=[{:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(self.true_param[0], self.true_param[1], self.true_param[2], self.true_param[3])

            np.savetxt(filename, qml_list)

        end = time()
        if verbose == 1:
            print('Total Elapsed Time: ' + format_time(end - start))
        return qml_list

    def compute_U(self, kind='explicit', wrt=None, t_max=None, verbose=0, filter_unobserved=True, kfilter=None, deriv_filters=None, close_pb=True, save_raw=False):
        if wrt is None:
            if self.wrt is None:
                raise Exception('Method setup_filter needs to be called first or argument wrt has to be specified.')
            wrt = self.wrt

        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        k = np.size(wrt)

        if kind == 'explicit':
            if self.wrt is None:
                raise Exception('Method setup_filter needs to be called first.')

            Sig_inv = np.linalg.inv(self.kalman_filter.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
            S = self.kalman_filter.S_star(wrt=wrt)

            Gamma1 = self.kalman_filter.H.T @ Sig_inv @ self.kalman_filter.H
            Gamma2 = np.einsum('jk, kl, ilm, mn, nr -> ijr', self.kalman_filter.H.T, Sig_inv, S[:, self.first_observed:, self.first_observed:], Sig_inv, self.kalman_filter.H)
            Gammaj = np.zeros((k, self.dim * (k + 2), self.dim * (k + 2)))

            for j in range(k):
                Gamma1_stretch = np.kron(unit_vec(k=k, j=j + 1, as_column=True), Gamma1)
                Gammaj[j, :self.dim, :] = np.hstack((Gamma2[j], -Gamma2[j], Gamma1_stretch.T))
                Gammaj[j, self.dim:2 * self.dim, :] = -Gammaj[j, :self.dim, :]
                Gammaj[j, 2 * self.dim:, :] = Gammaj[j, :, 2 * self.dim:].T
            Gammaj *= 1 / 2

            Gammaj_coefs = np.triu(Gammaj) + np.triu(Gammaj, 1)
            alpha_f = np.vstack([Gammaj_coefs[j][np.triu_indices_from(Gammaj[0])] for j in range(k)])
            alpha_f = np.hstack((np.zeros((k, self.dim * (k + 2))), alpha_f))

            A_kron2 = self.filter_B4[1:n_dim(self.dim * (k + 2), 2), 1:n_dim(self.dim * (k + 2), 2)]
            alpha_g = np.einsum('ij, kj -> ki', np.linalg.inv(A_kron2 - np.eye(n_dim(self.dim * (k + 2), 2) - 1)).T, alpha_f)
            alpha_g = np.pad(alpha_g, ((0, 0), (1, n_dim(self.dim * (k + 2), 4) - n_dim(self.dim * (k + 2), 2))), mode='constant')
            alpha_h = np.einsum('ij, jk -> ik', alpha_g, self.filter_B4)

            coefs_gg = np.zeros((k, k, n_dim(self.dim * (k + 2), 4)))
            coefs_hh = np.zeros((k, k, n_dim(self.dim * (k + 2), 4)))

            t = tqdm(total=int(k * (k + 1) / 2) * n_dim(self.dim * (k + 2), 4), desc='Calculating coefficients for U')
            for i in range(k):
                for j in range(i, k):
                    for l in range(n_dim(self.dim * (k + 2), 4)):
                        t.update(1)
                        lamb = self.dicts[l]
                        mus = self.dicts[mask(2, self.dicts, 'leq_abs') & mask(lamb, self.dicts, 'leq')]
                        coefs_g = np.sum(alpha_g[i, mult_to_ind(mus, self.dicts)] * alpha_g[j, mult_to_ind(lamb - mus, self.dicts)])
                        coefs_h = np.sum(alpha_h[i, mult_to_ind(mus, self.dicts)] * alpha_h[j, mult_to_ind(lamb - mus, self.dicts)])
                        coefs_gg[i, j, l] = coefs_g
                        coefs_gg[j, i, l] = coefs_g
                        coefs_hh[i, j, l] = coefs_h
                        coefs_hh[j, i, l] = coefs_h
            t.close()

            coefs_U = coefs_gg - coefs_hh
            U = coefs_U @ self.lim_expec4
            self.U = U

            filepath_U = './saves/' + self.__class__.__name__ + '/Covariance/Explicit/U{}_m={}_{}.txt'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring)
            np.savetxt(filepath_U, U)

            return U
        elif kind == 'estimate':
            param = self.true_param
            if t_max is None:
                t_max = self.observations.shape[0] - 1

            if self.seed is not None:
                filepath = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/U_raw{}_m={}_{}_seed{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max)
            else:
                filepath = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/U_raw{}_m={}_{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, t_max)
            if os.path.exists(filepath):
                return cummean(pkl.load(open(filepath, 'rb')), axis=0)

            if isinstance(verbose, tqdm) or verbose == 0:
                tr = verbose
            elif verbose == 1:
                tr = tqdm(total=2 * t_max, desc='Computing U')
            else:
                raise Exception('verbose has to be 0 or 1 or instance of class tqdm.')
            if kfilter is None:
                C = partial(self.C, param=param)
                a = partial(self.a, param=param)
                A = partial(self.A, param=param)
                E_0 = partial(self.init.E_0, param=param)
                Cov_0 = partial(self.init.Cov_0, param=param)
                C_lim = partial(self.C_lim, param=param)

                kfilter = KalmanFilter(dim=self.dim, a=a, A=A, C=C, E_0=E_0, Cov_0=Cov_0, C_lim=C_lim, first_observed=self.first_observed)
                kfilter.build_covariance(t_max=1000)
                kfilter.build_kalman_filter_hom(observations=self.observations, t_max=t_max, verbose=tr, close_pb=False)
            if deriv_filters is None:
                deriv_filter_hom, _ = kfilter.deriv_filter_hom(observations=self.observations, wrt=wrt, t_max=t_max, verbose=tr, close_pb=False)
            else:
                deriv_filter_hom, _ = deriv_filters
            hatX = np.hstack((self.observations[1:t_max + 1], kfilter.X_hat_tp1_t_list_hom[1:t_max + 1], deriv_filter_hom[1:t_max + 1].reshape(t_max, k * self.dim)))
            Sig_inv = np.linalg.inv(kfilter.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
            s_star = kfilter.S_star(wrt=wrt)

            if verbose and close_pb:
                tr.close()

            Gamma1 = kfilter.H.T @ Sig_inv @ kfilter.H
            Gamma2 = np.einsum('jk, kl, ilm, mn, nr -> ijr', kfilter.H.T, Sig_inv, s_star[:, self.first_observed:, self.first_observed:], Sig_inv, kfilter.H)
            Gammaj = np.zeros((np.size(wrt), self.dim * (k + 2), self.dim * (k + 2)))

            for j in range(k):
                Gamma1_stretch = np.kron(unit_vec(k=k, j=j + 1, as_column=True), Gamma1)
                Gammaj[j, :self.dim, :] = np.hstack((Gamma2[j], -Gamma2[j], Gamma1_stretch.T))
                Gammaj[j, self.dim:2 * self.dim, :] = -Gammaj[j, :self.dim, :]
                Gammaj[j, 2 * self.dim:, :] = Gammaj[j, :, 2 * self.dim:].T
            Gamma = 1 / 2 * Gammaj.reshape(k, Gammaj.shape[1]**2, order='F')

            K_prime = self.A(param, order=1, wrt=wrt) @ kfilter.Sig_tp1_t_lim[:, self.first_observed:] @ Sig_inv + np.einsum('jk, ikl, lm -> ijm', self.A(param), s_star[:, :, self.first_observed:], Sig_inv) - np.einsum('jk, kl, lm, imn, nr -> ijr', self.A(param), kfilter.Sig_tp1_t_lim[:, self.first_observed:], Sig_inv, s_star[:, self.first_observed:, self.first_observed:], Sig_inv)
            F_prime = self.A(param, order=1, wrt=wrt) - K_prime @ kfilter.H
            a_hat = np.hstack((self.a(param), self.a(param), self.a(param, order=1, wrt=wrt).flatten()))
            A_hat = np.zeros((self.dim * (k + 2), self.dim * (k + 2)))
            A_hat[:self.dim, :self.dim] = self.A(param)
            A_hat[self.dim:2 * self.dim, :2 * self.dim] = np.hstack((kfilter.K_lim @ kfilter.H, kfilter.F_lim))
            A_hat[2 * self.dim:, :] = np.hstack(((K_prime @ kfilter.H).reshape(k * self.dim, self.dim), F_prime.reshape(k * self.dim, self.dim), np.kron(np.eye(k), kfilter.F_lim)))

            Q2_hat = tracy_singh(np.diag(unit_vec(k + 2, 1)), np.kron(np.diag(unit_vec(k + 2, 1)), self.Q(param, 2)), (k + 2, k + 2), (self.dim, self.dim))
            Q_hat = tracy_singh(np.diag(unit_vec(k + 2, 1)), np.kron(unit_vec(k + 2, 1, as_column=True), self.Q(param, 1)), (k + 2, k + 2), (self.dim, self.dim))
            q_hat = tracy_singh(unit_vec(k + 2, 1, as_column=True), np.kron(unit_vec(k + 2, 1, as_column=True), self.Q(param, 0).reshape(-1, 1)), (k + 2, 1), (self.dim, 1)).squeeze()
            lamb = np.kron(a_hat, a_hat) + q_hat
            Pi = np.kron(a_hat.reshape(-1, 1), A_hat) + np.kron(A_hat, a_hat.reshape(-1, 1)) + Q_hat
            Lamb = np.kron(A_hat, A_hat) + Q2_hat
            a_hat2 = np.hstack((a_hat, lamb))
            A_hat2 = np.zeros((np.size(a_hat2), np.size(a_hat2)))
            A_hat2[:A_hat.shape[0], :A_hat.shape[1]] = A_hat
            A_hat2[A_hat.shape[0]:, :] = np.hstack((Pi, Lamb))

            alpha_g_x = - Gamma @ np.linalg.inv(Lamb - np.eye(lamb.shape[0])) @ Pi @ np.linalg.inv(A_hat - np.eye(A_hat.shape[0]))
            alpha_g_xox = Gamma @ np.linalg.inv(Lamb - np.eye(lamb.shape[0]))
            alpha_g = np.hstack((alpha_g_x, alpha_g_xox))
            alpha_h = alpha_g @ A_hat2
            beta_h = alpha_g @ a_hat2

            if filter_unobserved:
                hatX[:-1, :self.first_observed] = kfilter.X_hat_tt_list_hom[1:, :self.first_observed]
            vechatX = np.hstack((hatX, np.einsum('nk, nl -> nkl', hatX, hatX).reshape(hatX.shape[0], -1)))
            g = lambda x: alpha_g @ x
            h = lambda x: alpha_h @ x + beta_h.reshape(-1, 1)
            raw = np.einsum('ki, li -> ikl', g(vechatX.T), g(vechatX.T)) - np.einsum('ki, li -> ikl', h(vechatX.T), h(vechatX.T))
            if save_raw:
                file = open(filepath, 'wb')
                pkl.dump(raw, file)

            return cummean(raw, axis=0)
        else:
            raise Exception('Argument kind needs to be set to "explicit" or "estimate".')

    def compute_W(self, kind='explicit', wrt=None, t_max=None, verbose=0, kfilter=None, deriv_filters=None, close_pb=True, save_raw=False):
        if wrt is None:
            if self.wrt is None:
                raise Exception('Method setup_filter needs to be called first or argument wrt has to be specified.')
            wrt = self.wrt

        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        k = np.size(wrt)

        if kind == 'explicit':
            if self.wrt is None:
                raise Exception('Method setup_filter needs to be called first.')
            S = self.kalman_filter.S_star(wrt=self.wrt)
            S_o_i = S[np.triu_indices(np.size(self.wrt))[0], self.first_observed:, self.first_observed:]
            S_o_j = S[np.triu_indices(np.size(self.wrt))[1], self.first_observed:, self.first_observed:]
            Sig_inv = np.linalg.inv(self.kalman_filter.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
            S_tilde_i = np.einsum('jk, ikl, lm -> ijm', Sig_inv, S_o_i, Sig_inv)
            S_tilde_j = np.einsum('jk, ikl, lm -> ijm', Sig_inv, S_o_j, Sig_inv)
            S_hat_o = np.einsum('ijk, ikl -> ijl', S_o_j, S_tilde_i) + np.einsum('ijk, ikl -> ijl', S_o_i, S_tilde_j)
            R_o = self.kalman_filter.R_star(wrt=self.wrt)[:, self.first_observed:, self.first_observed:]
            R_tilde_o = np.einsum('jk, ikl, lm -> ijm', Sig_inv, R_o, Sig_inv)
            psi = self.kalman_filter.H.T @ (np.einsum('jk, ikl -> ijl', Sig_inv, S_hat_o) - R_tilde_o) @ self.kalman_filter.H
            mu = -1 / 2 * np.trace(np.einsum('jk, ikl -> ijl', Sig_inv, R_o) - np.einsum('ijk, ikl -> ijl', S_tilde_i, S_o_j), axis1=1, axis2=2).reshape(1, -1)

            S_tilde_i = self.kalman_filter.H.T @ S_tilde_i @ self.kalman_filter.H
            S_tilde_j = self.kalman_filter.H.T @ S_tilde_j @ self.kalman_filter.H
            Sig_inv = self.kalman_filter.H.T @ Sig_inv @ self.kalman_filter.H

            Gamma_ij = np.zeros((int(k * (k + 1) / 2), self.dim * (int(k * (k + 1) / 2) + k + 2), self.dim * (int(k * (k + 1) / 2) + k + 2)))

            for l in range(int(k * (k + 1) / 2)):
                i, j = np.triu_indices(k)[0][l], np.triu_indices(k)[1][l]
                Gamma_ij[l, :2 * self.dim, :2 * self.dim] = np.block([[psi[l], -psi[l]], [-psi[l], psi[l]]])

                Gamma_ij[l, :2 * self.dim, (2 + i) * self.dim:(3 + i) * self.dim] += np.block([[S_tilde_j[l]], [-S_tilde_j[l]]])
                Gamma_ij[l, (2 + i) * self.dim:(3 + i) * self.dim, :2 * self.dim] += np.block([S_tilde_j[l], -S_tilde_j[l]])

                Gamma_ij[l, :2 * self.dim, (2 + j) * self.dim:(3 + j) * self.dim] += np.block([[S_tilde_i[l]], [-S_tilde_i[l]]])
                Gamma_ij[l, (2 + j) * self.dim:(3 + j) * self.dim, :2 * self.dim] += np.block([S_tilde_i[l], -S_tilde_i[l]])

                Gamma_ij[l, (2 + i) * self.dim:(3 + i) * self.dim, (2 + j) * self.dim:(3 + j) * self.dim] += Sig_inv
                Gamma_ij[l, (2 + j) * self.dim:(3 + j) * self.dim, (2 + i) * self.dim:(3 + i) * self.dim] += Sig_inv

                Gamma_ij[l, :2 * self.dim, (k + 2 + l) * self.dim:(k + 3 + l) * self.dim] = np.block([[-Sig_inv], [Sig_inv]])
                Gamma_ij[l, (k + 2 + l) * self.dim:(k + 3 + l) * self.dim, :2 * self.dim] = np.block([-Sig_inv, Sig_inv])
            Gamma_ij *= -1 / 2

            Gamma_ij_coefs = np.triu(Gamma_ij) + np.triu(Gamma_ij, 1)
            coefs = np.vstack([Gamma_ij_coefs[j][np.triu_indices_from(Gamma_ij[0])] for j in range(int(k * (k + 1) / 2))])
            coefs = np.hstack((mu.T, np.zeros((int(k * (k + 1) / 2), self.dim * (int(k * (k + 1) / 2) + k + 2))), coefs))

            W_components = coefs @ self.lim_expec2
            W = np.zeros((k, k))
            W[np.triu_indices(k)[0], np.triu_indices(k)[1]] = W_components
            W[np.triu_indices(k)[1], np.triu_indices(k)[0]] = W_components
            self.W = W

            filepath_W = './saves/' + self.__class__.__name__ + '/Covariance/Explicit/W{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
            np.savetxt(filepath_W, W)

            return W
        elif kind == 'estimate':
            param = self.true_param
            if t_max is None:
                t_max = self.observations.shape[0] - 1

            if self.seed is not None:
                filepath = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/W_raw{}_m={}_{}_seed{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max)
            else:
                filepath = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/W_raw{}_m={}_{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max)
            if os.path.exists(filepath):
                return cummean(pkl.load(open(filepath, 'rb')), axis=0)

            if isinstance(verbose, tqdm) or verbose == 0:
                tr = verbose
            elif verbose == 1:
                tr = tqdm(total=3 * t_max, desc='Computing W')
            else:
                raise Exception('verbose has to be 0 or 1 or instance of class tqdm')

            if kfilter is None:
                C = partial(self.C, param=param)
                a = partial(self.a, param=param)
                A = partial(self.A, param=param)
                E_0 = partial(self.init.E_0, param=param)
                Cov_0 = partial(self.init.Cov_0, param=param)
                C_lim = partial(self.C_lim, param=param)

                kfilter = KalmanFilter(dim=self.dim, a=a, A=A, C=C, E_0=E_0, Cov_0=Cov_0, C_lim=C_lim, first_observed=self.first_observed)
                kfilter.build_covariance(t_max=1000)
                kfilter.build_kalman_filter_hom(observations=self.observations, t_max=t_max, verbose=tr, close_pb=False)
            s_star = kfilter.S_star(wrt=wrt)
            s_star_o_i = s_star[np.triu_indices(np.size(wrt))[0], self.first_observed:, self.first_observed:]
            s_star_o_j = s_star[np.triu_indices(np.size(wrt))[1], self.first_observed:, self.first_observed:]
            Sig_inv = np.linalg.inv(kfilter.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
            s_tilde_i = np.einsum('jk, ikl, lm -> ijm', Sig_inv, s_star_o_i, Sig_inv)
            s_tilde_j = np.einsum('jk, ikl, lm -> ijm', Sig_inv, s_star_o_j, Sig_inv)
            s_hat_o = np.einsum('ijk, ikl -> ijl', s_star_o_j, s_tilde_i) + np.einsum('ijk, ikl -> ijl', s_star_o_i, s_tilde_j)
            r_star_o = kfilter.R_star(wrt=wrt)[:, self.first_observed:, self.first_observed:]
            r_tilde_o = np.einsum('jk, ikl, lm -> ijm', Sig_inv, r_star_o, Sig_inv)

            if deriv_filters is None:
                deriv_filter_tp1_t, deriv_filter_tt = kfilter.deriv_filter_hom(observations=self.observations, wrt=wrt, t_max=t_max, verbose=tr, close_pb=False)
            else:
                deriv_filter_tp1_t, deriv_filter_tt = deriv_filters
            deriv_filter_tp1_t_i, deriv_filter_tt_i = deriv_filter_tp1_t[:, np.triu_indices(np.size(wrt))[0], :], deriv_filter_tt[:, np.triu_indices(np.size(wrt))[0], :]
            deriv_filter_tp1_t_j, deriv_filter_tt_j = deriv_filter_tp1_t[:, np.triu_indices(np.size(wrt))[1], :], deriv_filter_tt[:, np.triu_indices(np.size(wrt))[1], :]
            deriv_filters = [deriv_filter_tp1_t, deriv_filter_tt]
            deriv2_filter, _ = kfilter.deriv2_filter_hom(observations=self.observations, wrt=wrt, t_max=t_max, verbose=tr, close_pb=False, deriv_filters=deriv_filters)
            mu = np.trace(np.einsum('jk, ikl -> ijl', Sig_inv, r_star_o) - np.einsum('ijk, ikl -> ijl', s_tilde_i, s_star_o_j), axis1=1, axis2=2).reshape(1, -1)
            nu = np.einsum('ijk, jkl -> ijl', deriv_filter_tp1_t_i[1:, :, self.first_observed:], s_tilde_j) + np.einsum('ijk, jkl -> ijl', deriv_filter_tp1_t_j[1:, :, self.first_observed:], s_tilde_i) - deriv2_filter[1:, :,  self.first_observed:] @ Sig_inv
            psi = np.einsum('jk, ikl -> ijl', Sig_inv, s_hat_o) - r_tilde_o
            eps = (self.observations[1:t_max + 1, self.first_observed:] - kfilter.X_hat_tp1_t_list_hom[1:, self.first_observed:])

            if verbose and close_pb:
                tr.close()

            out = -0.5 * (mu + 2 * np.einsum('ijk, ik -> ij', nu, eps) + np.einsum('ij, ljk, ik -> il', eps, psi, eps) + 2 * np.einsum('ilj, jk, ilk -> il', deriv_filter_tp1_t_i[1:, :, self.first_observed:], Sig_inv, deriv_filter_tp1_t_j[1:, :, self.first_observed:]))
            result = np.zeros((t_max, np.size(wrt), np.size(wrt)))
            result[:, np.triu_indices(np.size(wrt))[0], np.triu_indices(np.size(wrt))[1]] = out
            result[:, np.triu_indices(np.size(wrt))[1], np.triu_indices(np.size(wrt))[0]] = out

            if save_raw:
                file = open(filepath, 'wb')
                pkl.dump(result, file)

            return cummean(result, axis=0)
        else:
            raise Exception('Argument kind needs to be set to "explicit" or "estimate".')

    def compute_V(self, kind='explicit', wrt=None, t_max=None, verbose=0, filter_unobserved=True, from_raw=None, every=None):
        if wrt is None:
            if self.wrt is None:
                raise Exception('Method setup_filter needs to be called first or argument wrt has to be specified.')
            wrt = self.wrt

        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        if kind == 'explicit':
            if self.U is None:
                self.U = self.compute_U(wrt=wrt)
            if self.W is None:
                self.W = self.compute_W(wrt=wrt)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                W_inv = np.linalg.inv(self.W)
                V = W_inv @ self.U @ W_inv
                Std = np.sqrt(np.diagonal(V))
                Std_inv = np.eye(np.size(wrt)) / Std
                Corr = Std_inv @ V @ Std_inv
                Corr[np.diag_indices(np.size(wrt))[0], np.diag_indices(np.size(wrt))[1]] = 1

            return V, Std, Corr
        elif kind == 'estimate':
            param = self.true_param

            if from_raw is not None:
                if self.seed is not None:
                    U_raw_path = ['./saves/' + self.__class__.__name__ + '/Covariance/U_raw{}_m={}_{}_seed'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring) + filepath + 'obs.pkl' for filepath in from_raw]
                    W_raw_path = ['./saves/' + self.__class__.__name__ + '/Covariance/W_raw{}_m={}_{}_seed'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring) + filepath + 'obs.pkl' for filepath in from_raw]
                else:
                    U_raw_path = ['./saves/' + self.__class__.__name__ + '/Covariance/U_raw{}_m={}_{}'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring) + filepath + 'obs.pkl' for filepath in from_raw]
                    W_raw_path = ['./saves/' + self.__class__.__name__ + '/Covariance/W_raw{}_m={}_{}'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring) + filepath + 'obs.pkl' for filepath in from_raw]

                U_raw = [pkl.load(open(filepath, 'rb')) for filepath in U_raw_path]
                W_raw = [pkl.load(open(filepath, 'rb')) for filepath in W_raw_path]

                t_max = np.sum([u_raw.shape[0] for u_raw in U_raw])

                if every is not None:
                    if self.seed is not None:
                        filepath = './saves/' + self.__class__.__name__ + '/Covariance/V_raw{}_m={}_{}_seed{}_{}obs_every{}th.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max, every)
                    else:
                        filepath = './saves/' + self.__class__.__name__ + '/Covariance/V_raw{}_m={}_{}_{}obs_every{}th.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, t_max, every)
                else:
                    if self.seed is not None:
                        filepath = './saves/' + self.__class__.__name__ + '/Covariance/V_raw{}_m={}_{}_seed{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max)
                    else:
                        filepath = './saves/' + self.__class__.__name__ + '/Covariance/V_raw{}_m={}_{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, t_max)

                U_cumsum = []
                W_cumsum = []
                for j in range(len(from_raw)):
                    U_cumsum.append(U_raw[j].cumsum(axis=0)[::every] + np.sum([U_raw[k].sum(axis=0) for k in range(j)], axis=0))
                    W_cumsum.append(W_raw[j].cumsum(axis=0)[::every] + np.sum([W_raw[k].sum(axis=0) for k in range(j)], axis=0))
                U_cumsum = np.vstack(U_cumsum)
                W_cumsum = np.vstack(W_cumsum)

                U = U_cumsum / (np.arange(0, t_max, every) + 1)[:, None, None]
                W = W_cumsum / (np.arange(0, t_max, every) + 1)[:, None, None]
            else:
                if t_max is None:
                    t_max = self.observations.shape[0] - 1

                if every is not None:
                    warnings.warn('Argument "every" was not used since argument "from_raw" was not specified.')
                if self.seed is not None:
                    filepath = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/V_raw{}_m={}_{}_seed{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max)
                    filepath_U = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/U_raw{}_m={}_{}_seed{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max)
                    filepath_W = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/W_raw{}_m={}_{}_seed{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max)
                else:
                    filepath = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/V_raw{}_m={}_{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, t_max)
                    filepath_U = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/U_raw{}_m={}_{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, t_max)
                    filepath_W = './saves/' + self.__class__.__name__ + '/Covariance/Estimated/W_raw{}_m={}_{}_{}obs.pkl'.format(np.atleast_1d(wrt).tolist(), self.first_observed, self.savestring, self.seed, t_max)

                if os.path.exists(filepath):
                    V, Std, Corr = pkl.load(open(filepath, 'rb'))
                    return V.squeeze(), Std.squeeze(), Corr.squeeze()
                else:
                    C = partial(self.C, param=param)
                    a = partial(self.a, param=param)
                    A = partial(self.A, param=param)
                    E_0 = partial(self.init.E_0, param=param)
                    Cov_0 = partial(self.init.Cov_0, param=param)
                    C_lim = partial(self.C_lim, param=param)

                    if t_max >= self.observations.shape[0]:
                        raise Exception('Not enough observations available for t_max = {}'.format(t_max))

                    U_missing = not os.path.exists(filepath_U)
                    W_missing = not os.path.exists(filepath_W)

                    if U_missing | W_missing:
                        if (verbose == 1) & (not W_missing):
                            verbose = tqdm(total=2 * t_max, desc='Computing filter')
                        elif (verbose == 1) & W_missing:
                            verbose = tqdm(total=3 * t_max, desc='Computing filter')
                        kfilter = KalmanFilter(dim=self.dim, a=a, A=A, C=C, E_0=E_0, Cov_0=Cov_0, C_lim=C_lim, first_observed=self.first_observed)
                        kfilter.build_covariance(t_max=10000)
                        kfilter.build_kalman_filter_hom(observations=self.observations, t_max=t_max, verbose=verbose, close_pb=False)
                        if verbose:
                            verbose.desc = 'Computing derivative filter'
                        deriv_filters = kfilter.deriv_filter_hom(observations=self.observations, wrt=wrt, t_max=t_max, verbose=verbose, close_pb=False)
                        U = self.compute_U(kind='estimate', t_max=t_max, wrt=wrt, verbose=verbose, filter_unobserved=filter_unobserved, kfilter=kfilter, deriv_filters=deriv_filters, close_pb=False, save_raw=True)
                        if verbose:
                            if W_missing:
                                verbose.desc = 'Computing 2nd derivative filter'
                        W = self.compute_W(kind='estimate', t_max=t_max, wrt=wrt, verbose=verbose, kfilter=kfilter, deriv_filters=deriv_filters, save_raw=True)
                        if verbose:
                            verbose.close()
                    else:
                        U = self.compute_U(kind='estimate', t_max=t_max, wrt=wrt, verbose=verbose, filter_unobserved=filter_unobserved)
                        W = self.compute_W(kind='estimate', t_max=t_max, wrt=wrt, verbose=verbose)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                W_inv = np.linalg.inv(W)
                V = np.einsum('ijk, ikl, ilm -> ijm', W_inv, U, W_inv)
                Std = np.sqrt(np.diagonal(V, axis1=1, axis2=2))
                Std_inv = np.eye(np.size(wrt)) / np.sqrt(np.diagonal(V, axis1=1, axis2=2))[:, np.newaxis, :]
                Corr = np.einsum('ijk, ikl, ilm -> ijm', Std_inv, V, Std_inv)
                Corr[:, np.diag_indices(np.size(wrt))[0], np.diag_indices(np.size(wrt))[1]] = 1

            file = open(filepath, 'wb')
            pkl.dump([V.squeeze(), Std.squeeze(), Corr.squeeze()], file)

            return V.squeeze(), Std.squeeze(), Corr.squeeze()
        else:
            raise Exception('Argument kind needs to be set to "explicit" or "estimate".')


class HestonModel(PolynomialModel):
    def __init__(self, first_observed, init, dt, true_param=None, wrt=None):
        if true_param is None:
            true_param = np.repeat(np.nan, 4)

        super().__init__(first_observed, init, dt, true_param)
        self.dim = 3
        self.params_names = np.array(['kappa', 'theta', 'sigma', 'rho'])
        self.params_bounds = np.array([[0.0001, 10], [0.0001 ** 2, 1], [0.0001, 1], [-1, 1]])

        if not np.isnan(self.true_param).any():
            self.savestring = 'par=[{:.3f}, {:.3f}, {:.3f}, {:.3f}]_dt={:.1e}'.format(self.true_param[0], self.true_param[1], self.true_param[2], self.true_param[3], self.dt)
            if wrt is not None:
                self.setup_filter(wrt)
        elif wrt is not None:
            warnings.warn('Argument wrt was not used since the whole parameter has not yet been estimated. Please use method "setup_filter" after this has been done.', Warning)

    def a(self, param, order=0, wrt=np.array([0, 1, 2, 3])):
        wrt = np.atleast_1d(wrt)
        kappa, theta, sigma, rho = param
        if order == 0:
            return np.array([theta * (1 - np.exp(-kappa * self.dt)), 0, theta * self.dt + theta / kappa * (np.exp(-kappa * self.dt) - 1)])
        elif order == 1:
            deriv_array = np.zeros((4, 3))
            deriv_array[0] = [theta * self.dt * np.exp(-kappa * self.dt), 0, theta / kappa ** 2 * (1 - np.exp(-kappa * self.dt)) - theta / kappa * self.dt * np.exp(-kappa * self.dt)]
            deriv_array[1] = [1 - np.exp(-kappa * self.dt), 0, self.dt - 1 / kappa * (1 - np.exp(-kappa * self.dt))]
            return deriv_array[wrt]
        elif order == 2:
            deriv_array = np.zeros((4, 4, 3))
            deriv_array[0, 0] = [-theta * self.dt ** 2 * np.exp(-kappa * self.dt), 0, theta / kappa * self.dt **2 * np.exp(-kappa * self.dt) + 2 * theta / kappa**2 * self.dt * np.exp(-kappa * self.dt) - 2 * theta / kappa**3 * (1 - np.exp(-kappa * self.dt))]
            deriv_array[0, 1] = [self.dt * np.exp(-kappa * self.dt), 0, 1 / kappa**2 * (1 - np.exp(-kappa * self.dt)) - 1 / kappa * self.dt * np.exp(-kappa * self.dt)]
            deriv_array = deriv_array[np.ix_(wrt, wrt)]
            return deriv_array[np.triu_indices(len(wrt))]

    def A(self, param, order=0, wrt=np.array([0, 1, 2, 3])):
        wrt = np.atleast_1d(wrt)
        kappa, theta, sigma, rho = param
        if order == 0:
            return np.array([[np.exp(-kappa * self.dt), 0, 0], [0, 0, 0], [1 / kappa * (1 - np.exp(-kappa * self.dt)), 0, 0]])
        elif order == 1:
            deriv_array = np.zeros((4, 3, 3))
            deriv_array[0, :, 0] = [-self.dt * np.exp(-kappa * self.dt), 0, 1 / kappa**2 * (np.exp(-kappa * self.dt) - 1) + 1 / kappa * self.dt * np.exp(-kappa * self.dt)]
            return deriv_array[wrt]
        elif order == 2:
            deriv_array = np.zeros((4, 4, 3, 3))
            deriv_array[0, 0, :, 0] = [self.dt **2 * np.exp(-kappa * self.dt), 0, 2 / kappa**3 * (1 - np.exp(-kappa * self.dt)) - 2 / kappa**2 * self.dt * np.exp(-kappa * self.dt) - 1 / kappa * self.dt ** 2 * np.exp(-kappa * self.dt)]
            deriv_array = deriv_array[np.ix_(wrt, wrt)]
            return deriv_array[np.triu_indices(len(wrt))]

    def B(self, param, order):
        kappa, theta, sigma, rho = param
        mu, delta = 0, 0
        dicts = return_dict(2, 2 * order)

        heston_A = np.zeros((n_dim(2, 2 * order), n_dim(2, 2 * order)))
        heston_A[mult_to_ind([1, 0], dicts), 0] = kappa * theta
        heston_A[mult_to_ind([1, 0], dicts), mult_to_ind([1, 0], dicts)] = -kappa
        heston_A[mult_to_ind([0, 1], dicts), 0] = mu
        heston_A[mult_to_ind([0, 1], dicts), mult_to_ind([1, 0], dicts)] = delta
        heston_A[mult_to_ind([2, 0], dicts), mult_to_ind([1, 0], dicts)] = sigma ** 2
        heston_A[mult_to_ind([1, 1], dicts), mult_to_ind([1, 0], dicts)] = sigma * rho
        heston_A[mult_to_ind([0, 2], dicts), mult_to_ind([1, 0], dicts)] = 1

        heston_Bc = np.zeros(heston_A.shape)
        for i in range(heston_A.shape[0]):
            for j in range(heston_Bc.shape[0]):
                masks = mask(dicts[i], dicts, typ='leq') & mask(dicts[j], dicts, typ='leq')
                lamb_ell = (ind_to_mult(i, dicts) - dicts)[masks]
                mu_ell = (ind_to_mult(j, dicts) - dicts)[masks]
                heston_Bc[i, j] = (multi_binom(dicts[i], dicts[masks]) * heston_A[mult_to_ind(lamb_ell, dicts), mult_to_ind(mu_ell, dicts)]).sum()

        heston_B = expm(heston_Bc * self.dt)
        heston_B_diff = np.zeros(heston_B.shape)
        heston_B_diff[:, dicts[:, 1] == 0] = heston_B[:, dicts[:, 1] == 0]

        heston_B_diff_sq = np.zeros((n_dim(3, order), n_dim(3, order)))
        dicts_sq = return_dict(3, order)
        cols = np.where((dicts_sq[:, 1] == 0) & (dicts_sq[:, 2] == 0))[0]
        cols2 = np.where((dicts[:, 1] == 0))[0][:cols.shape[0]]
        for i in range(heston_B_diff_sq.shape[0]):
            lamb = ind_to_mult(i, dicts_sq)
            lamb_tilde = np.array([lamb[0], lamb[1] + 2 * lamb[2]])
            ind = mult_to_ind(lamb_tilde, dicts)
            heston_B_diff_sq[i, cols] = heston_B[ind, cols2]
        return heston_B_diff_sq

    def C(self, param, t):
        kappa, theta, sigma, rho = param
        v0 = self.init.E_0(param)[0]
        v0_2 = self.init.Cov_0(param)[0, 0] + v0**2
        B11 = (1 - np.exp(-kappa * self.dt)) * sigma ** 2 / kappa * ((1 - np.exp(-kappa * t)) * theta + np.exp(-kappa * t) * v0 - (1 - np.exp(-kappa * self.dt)) * theta / 2)
        B13 = sigma ** 2 / (2 * kappa ** 2) * ((1 + 4 * rho ** 2 - np.exp(-kappa * self.dt)) * theta - 2 * (v0 - theta) * np.exp(-kappa * t)) * (1 - np.exp(-kappa * self.dt)) + 1 / kappa * (sigma ** 2 * (1 + kappa * rho ** 2 * self.dt) * (v0 - theta) * self.dt * np.exp(-kappa * t) - 2 * rho ** 2 * sigma ** 2 * theta * self.dt * np.exp(-kappa * self.dt))
        B33 = ((2 / kappa ** 2 * (v0_2 - 2 * theta * v0 + theta**2) - sigma ** 2 / kappa ** 3 * (2 * v0 - theta)) * np.exp(-2 * kappa * (t - self.dt)) - sigma ** 2 / kappa ** 3 * (v0 - theta) * np.exp(-kappa * (t - self.dt)) - sigma ** 2 / (2 * kappa ** 3) * theta) * (1 - np.exp(-kappa * self.dt)) ** 2 + 2 / kappa ** 3 * (3 * sigma ** 2 * (1 + 2 * rho ** 2) + 2 * kappa ** 2 * theta * self.dt) * (v0 - theta) * np.exp(-kappa * t) * (np.exp(kappa * self.dt) - 1) - 6 * sigma ** 2 / kappa ** 2 * (1 + 2 * rho ** 2 + self.dt * kappa * rho ** 2) * (v0 - theta) * self.dt * np.exp(-kappa * t) - 3 * sigma ** 2 / kappa ** 3 * theta * (1 + 8 * rho ** 2) * (1 - np.exp(-kappa * self.dt)) + 12 * sigma ** 2 / kappa ** 2 * rho ** 2 * theta * self.dt * (1 + np.exp(-kappa * self.dt)) + 3 * sigma ** 2 / kappa ** 2 * theta * self.dt + 2 * theta ** 2 * self.dt ** 2
        B22 = theta * self.dt + 1 / kappa * (v0 - theta) * np.exp(-kappa * t) * (np.exp(kappa * self.dt) - 1)
        B12 = theta / kappa * rho * sigma * (1 - np.exp(-kappa * self.dt)) + rho * sigma * (v0 - theta) * self.dt * np.exp(-kappa * t)
        B23 = 3 * rho * sigma / kappa ** 2 * ((v0 - theta) * np.exp(-kappa * t) - theta * np.exp(-kappa * self.dt)) * (np.exp(kappa * self.dt) - 1) - 3 * rho * sigma / kappa * (v0 - theta) * np.exp(-kappa * t) + 3 * rho * sigma / kappa * theta * self.dt
        return np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

    def C_lim(self, param, order=0, wrt=np.array([0, 1, 2, 3])):
        wrt = np.atleast_1d(wrt)
        kappa, theta, sigma, rho = param
        if order == 0:
            B11 = (1 - np.exp(-2 * kappa * self.dt)) * sigma ** 2 / (2 * kappa) * theta
            B13 = sigma ** 2 / (2 * kappa ** 2) * (1 + 4 * rho ** 2 - np.exp(-kappa * self.dt)) * theta * (1 - np.exp(-kappa * self.dt)) - 2 / kappa * rho ** 2 * sigma ** 2 * theta * self.dt * np.exp(-kappa * self.dt)
            B33 = -sigma ** 2 / (2 * kappa ** 3) * theta * (1 - np.exp(-kappa * self.dt)) ** 2 - 3 * sigma ** 2 / kappa ** 3 * theta * (1 + 8 * rho ** 2) * (1 - np.exp(-kappa * self.dt)) + 12 * sigma ** 2 / kappa ** 2 * rho ** 2 * theta * self.dt * (1 + np.exp(-kappa * self.dt)) + 3 * sigma ** 2 / kappa ** 2 * theta * self.dt + 2 * theta ** 2 * self.dt ** 2
            B22 = theta * self.dt
            B12 = theta / kappa * rho * sigma * (1 - np.exp(-kappa * self.dt))
            B23 = 3 * rho * sigma / kappa ** 2 * theta * (np.exp(-kappa * self.dt) - 1) + 3 * rho * sigma / kappa * theta * self.dt
            return np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])
        elif order == 1:
            deriv_array = np.zeros((4, 3, 3))
            B11 = ((1 + 2 * kappa * self.dt) * np.exp(-2 * kappa * self.dt) - 1) / (2 * kappa**2) * sigma**2 * theta
            B12 = theta / kappa * rho * sigma * self.dt * np.exp(-kappa * self.dt) - theta / kappa**2 * rho * sigma * (1 - np.exp(-kappa * self.dt))
            B13 = -sigma ** 2 / kappa ** 3 * (1 + 4 * rho ** 2 - np.exp(-kappa * self.dt)) * theta * (1 - np.exp(-kappa * self.dt)) + sigma ** 2 / kappa ** 2 * (1 + 2 * rho ** 2 - np.exp(-kappa * self.dt)) * theta * self.dt * np.exp(-kappa * self.dt) + 2 / kappa ** 2 * rho ** 2 * sigma ** 2 * theta * self.dt * np.exp(-kappa * self.dt) * (1 + kappa * self.dt)
            B22 = 0
            B23 = 6 * rho * sigma / kappa**3 * theta * (1 - np.exp(-kappa * self.dt)) - 3 * rho * sigma / kappa**2 * theta * self.dt * (1 + np.exp(-kappa * self.dt))
            B33 = 3 * sigma ** 2 / (2 * kappa ** 4) * theta * (1 - np.exp(-kappa * self.dt)) ** 2 - sigma ** 2 / kappa ** 3 * theta * (1 - np.exp(-kappa * self.dt)) * self.dt * np.exp(-kappa * self.dt) + 9 * sigma ** 2 / kappa ** 4 * theta * (1 + 8 * rho ** 2) * (1 - np.exp(-kappa * self.dt)) - 3 * sigma ** 2 / kappa ** 3 * theta * (1 + 8 * rho ** 2) * self.dt * np.exp(-kappa * self.dt) - 24 * sigma ** 2 / kappa ** 3 * rho ** 2 * theta * self.dt * (1 + np.exp(-kappa * self.dt)) - 12 * sigma ** 2 / kappa ** 2 * rho ** 2 * theta * self.dt ** 2 * np.exp(-kappa * self.dt) - 6 * sigma ** 2 / kappa ** 3 * theta * self.dt
            deriv_array[0] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = (1 - np.exp(-2 * kappa * self.dt)) * sigma**2 / (2 * kappa)
            B12 = rho * sigma / kappa * (1 - np.exp(-kappa * self.dt))
            B13 = sigma**2 / (2 * kappa**2) * (1 + 4 * rho**2 - np.exp(-kappa * self.dt)) * (1 - np.exp(-kappa * self.dt)) - 2 / kappa * rho**2 * sigma**2 * self.dt * np.exp(-kappa * self.dt)
            B22 = self.dt
            B23 = 3 * rho * sigma / kappa**2 * (np.exp(-kappa * self.dt) - 1) + 3 * rho * sigma / kappa * self.dt
            B33 = -sigma**2 / (2 * kappa**3) * (1 - np.exp(-kappa * self.dt))**2 - 3 * sigma**2 / kappa**3 * (1 + 8 * rho**2) * (1 - np.exp(-kappa * self.dt)) + 12 * sigma**2 / kappa**2 * rho**2 * self.dt * (1 + np.exp(-kappa * self.dt)) + 3 * sigma**2 / kappa**2 * self.dt + 4 * theta * self.dt ** 2
            deriv_array[1] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = (1 - np.exp(-2 * kappa * self.dt)) * sigma / kappa * theta
            B13 = sigma / kappa ** 2 * (1 + 4 * rho ** 2 - np.exp(-kappa * self.dt)) * theta * (1 - np.exp(-kappa * self.dt)) - 4 / kappa * rho ** 2 * sigma * theta * self.dt * np.exp(-kappa * self.dt)
            B33 = -sigma / kappa ** 3 * theta * (1 - np.exp(-kappa * self.dt)) ** 2 - 6 * sigma / kappa ** 3 * theta * (1 + 8 * rho ** 2) * (1 - np.exp(-kappa * self.dt)) + 24 * sigma / kappa ** 2 * rho ** 2 * theta * self.dt * (1 + np.exp(-kappa * self.dt)) + 6 * sigma / kappa ** 2 * theta * self.dt
            B22 = 0
            B12 = theta / kappa * rho * (1 - np.exp(-kappa * self.dt))
            B23 = 3 * rho / kappa ** 2 * theta * (np.exp(-kappa * self.dt) - 1) + 3 * rho / kappa * theta * self.dt
            deriv_array[2] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = 0
            B12 = theta / kappa * sigma * (1 - np.exp(-kappa * self.dt))
            B13 = 4 * sigma**2 / kappa**2 * rho * theta * (1 - np.exp(-kappa * self.dt)) - 4 / kappa * rho * sigma**2 * theta * self.dt * np.exp(-kappa * self.dt)
            B22 = 0
            B23 = 3 * sigma / kappa**2 * theta * (np.exp(-kappa * self.dt) - 1) + 3 * sigma / kappa * theta * self.dt
            B33 = -48 * sigma**2 / kappa**3 * rho * theta * (1 - np.exp(-kappa * self.dt)) + 24 * sigma**2 / kappa**2 * rho * theta * self.dt * (1 + np.exp(-kappa * self.dt))
            deriv_array[3] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])
            return deriv_array[wrt]
        elif order == 2:
            deriv_array = np.zeros((4, 4, 3, 3))

            B11 = (1 - (1 + 2 * kappa * self.dt + 2 * kappa**2 * self.dt ** 2) * np.exp(-2 * kappa * self.dt)) / kappa**3 * sigma**2 * theta
            B12 = 2 * theta / kappa**3 * rho * sigma * (1 - np.exp(-kappa * self.dt)) - 2 * theta / kappa**2 * rho * sigma * self.dt * np.exp(-kappa * self.dt) - theta / kappa * rho * sigma * self.dt **2 * np.exp(-kappa * self.dt)
            B13 = 3 * sigma**2 / kappa**4 * (1 + 4 * rho**2 - np.exp(-kappa * self.dt)) * theta * (1 - np.exp(-kappa * self.dt)) - 4 * sigma**2 / kappa**3 * theta * self.dt * np.exp(-kappa * self.dt) * (1 + (3 + kappa * self.dt) * rho**2 - np.exp(-kappa * self.dt)) - sigma**2 / kappa**2 * theta * self.dt ** 2 * np.exp(-kappa * self.dt) * (1 + 2 * (1 + kappa * self.dt) * rho**2 - 2 * np.exp(-kappa * self.dt))
            B22 = 0
            B23 = -18 * rho * sigma / kappa**4 * theta * (1 - np.exp(-kappa * self.dt)) + 6 * rho * sigma / kappa**3 * theta * self.dt * (1 + 2 * np.exp(-kappa * self.dt)) + 3 * rho * sigma / kappa**2 * theta * self.dt ** 2 * np.exp(-kappa * self.dt)
            B33 = -6 * sigma**2 / kappa**5 * theta * (1 - np.exp(-kappa * self.dt)) * (1 - np.exp(-kappa * self.dt) + 6 * (1 + 8 * rho**2)) + 6 * sigma**2 / kappa**4 * theta * self.dt * np.exp(-kappa * self.dt) * (1 - np.exp(-kappa * self.dt) + 3 * (1 + 8 * rho**2)) + 72 * sigma**2 / kappa**4 * rho**2 * theta * self.dt * (1 + np.exp(-kappa * self.dt)) + sigma**2 / kappa**3 * theta * self.dt ** 2 * np.exp(-kappa * self.dt) * (1 - 2 * np.exp(-kappa * self.dt) + 3 * (1 + 24 * rho**2)) + 12 * sigma**2 / kappa**2 * rho**2 * theta * self.dt**3 * np.exp(-kappa * self.dt) + 18 * sigma**2 / kappa**4 * theta * self.dt
            deriv_array[0, 0] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = ((1 + 2 * kappa * self.dt) * np.exp(-2 * kappa * self.dt) - 1) / (2 * kappa ** 2) * sigma ** 2
            B12 = 1 / kappa * rho * sigma * self.dt * np.exp(-kappa * self.dt) - 1 / kappa ** 2 * rho * sigma * (1 - np.exp(-kappa * self.dt))
            B13 = -sigma ** 2 / kappa ** 3 * (1 + 4 * rho ** 2 - np.exp(-kappa * self.dt)) * (1 - np.exp(-kappa * self.dt)) + sigma ** 2 / kappa ** 2 * (1 + 2 * rho ** 2 - np.exp(-kappa * self.dt)) * self.dt * np.exp(-kappa * self.dt) + 2 / kappa ** 2 * rho ** 2 * sigma ** 2 * self.dt * np.exp(-kappa * self.dt) * (1 + kappa * self.dt)
            B22 = 0
            B23 = 6 * rho * sigma / kappa ** 3 * (1 - np.exp(-kappa * self.dt)) - 3 * rho * sigma / kappa ** 2 * self.dt * (1 + np.exp(-kappa * self.dt))
            B33 = 3 * sigma ** 2 / (2 * kappa ** 4) * (1 - np.exp(-kappa * self.dt)) ** 2 - sigma ** 2 / kappa ** 3 * (1 - np.exp(-kappa * self.dt)) * self.dt * np.exp(-kappa * self.dt) + 9 * sigma ** 2 / kappa ** 4 * (1 + 8 * rho ** 2) * (1 - np.exp(-kappa * self.dt)) - 3 * sigma ** 2 / kappa ** 3 * (1 + 8 * rho ** 2) * self.dt * np.exp(-kappa * self.dt) - 24 * sigma ** 2 / kappa ** 3 * rho ** 2 * self.dt * (1 + np.exp(-kappa * self.dt)) - 12 * sigma ** 2 / kappa ** 2 * rho ** 2 * self.dt ** 2 * np.exp(-kappa * self.dt) - 6 * sigma ** 2 / kappa ** 3 * self.dt
            deriv_array[0, 1] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = ((1 + 2 * kappa * self.dt) * np.exp(-2 * kappa * self.dt) - 1) / kappa**2 * sigma * theta
            B12 = theta / kappa * rho * self.dt * np.exp(-kappa * self.dt) - theta / kappa**2 * rho * (1 - np.exp(-kappa * self.dt))
            B13 = -2 * sigma / kappa ** 3 * (1 + 4 * rho ** 2 - np.exp(-kappa * self.dt)) * theta * (1 - np.exp(-kappa * self.dt)) + 2 * sigma / kappa ** 2 * (1 + 2 * rho ** 2 - np.exp(-kappa * self.dt)) * theta * self.dt * np.exp(-kappa * self.dt) + 4 / kappa ** 2 * rho ** 2 * sigma * theta * self.dt * np.exp(-kappa * self.dt) * (1 + kappa * self.dt)
            B22 = 0
            B23 = 6 * rho / kappa**3 * theta * (1 - np.exp(-kappa * self.dt)) - 3 * rho / kappa**2 * theta * self.dt * (1 + np.exp(-kappa * self.dt))
            B33 = 3 * sigma / kappa ** 4 * theta * (1 - np.exp(-kappa * self.dt)) ** 2 - 2 * sigma / kappa ** 3 * theta * (1 - np.exp(-kappa * self.dt)) * np.exp(-kappa * self.dt) + 18 * sigma / kappa ** 4 * theta * (1 + 8 * rho ** 2) * (1 - np.exp(-kappa * self.dt)) - 6 * sigma / kappa ** 3 * theta * (1 + 8 * rho ** 2) * self.dt * np.exp(-kappa * self.dt) - 48 * sigma / kappa ** 3 * rho ** 2 * theta * self.dt * (1 + np.exp(-kappa * self.dt)) - 24 * sigma / kappa ** 2 * rho ** 2 * theta * self.dt**2 * np.exp(-kappa * self.dt) - 12 * sigma / kappa ** 3 * theta * self.dt
            deriv_array[0, 2] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = 0
            B12 = theta / kappa * sigma * self.dt * np.exp(-kappa * self.dt) - theta / kappa**2 * sigma * (1 - np.exp(-kappa * self.dt))
            B13 = -8 * sigma**2 / kappa**3 * rho * theta * (1 - np.exp(-kappa * self.dt)) + 4 * sigma**2 / kappa**2 * rho * theta * self.dt * np.exp(-kappa * self.dt) * (2 + kappa * self.dt)
            B22 = 0
            B23 = 6 * sigma / kappa**3 * theta * (1 - np.exp(-kappa * self.dt)) - 3 * sigma / kappa**2 * theta * self.dt * (1 + np.exp(-kappa * self.dt))
            B33 = 144 * sigma**2 / kappa**4 * rho * theta * (1 - np.exp(-kappa * self.dt)) - 48 * sigma**2 / kappa**3 * rho * theta * self.dt * (1 + 2 * np.exp(-kappa * self.dt)) - 24 * sigma**2 / kappa**2 * rho * theta * self.dt**2 * np.exp(-kappa * self.dt)
            deriv_array[0, 3] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            deriv_array[1, 1, 2, 2] = 4 * self.dt**2

            B11 = (1 - np.exp(-2 * kappa * self.dt)) * sigma / kappa
            B12 = rho / kappa * (1 - np.exp(-kappa * self.dt))
            B13 = sigma / kappa**2 * (1 + 4 * rho**2 - np.exp(-kappa * self.dt)) * (1 - np.exp(-kappa * self.dt)) - 4 / kappa * rho**2 * sigma * self.dt * np.exp(-kappa * self.dt)
            B22 = 0
            B23 = 3 * rho / kappa**2 * (np.exp(-kappa * self.dt) - 1) + 3 * rho / kappa * self.dt
            B33 = -sigma / kappa**3 * (1 - np.exp(-kappa * self.dt))**2 - 6 * sigma / kappa**3 * (1 + 8 * rho**2) * (1 - np.exp(-kappa * self.dt)) + 24 * sigma / kappa**2 * rho**2 * self.dt * (1 + np.exp(-kappa * self.dt)) + 6 * sigma / kappa**2 * self.dt
            deriv_array[1, 2] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = 0
            B12 = sigma / kappa * (1 - np.exp(-kappa * self.dt))
            B13 = 4 * sigma**2 / kappa**2 * rho * (1 - np.exp(-kappa * self.dt)) - 4 / kappa * rho * sigma**2 * self.dt * np.exp(-kappa * self.dt)
            B22 = 0
            B23 = 3 * sigma / kappa**2 * (np.exp(-kappa * self.dt) - 1) + 3 * sigma / kappa * self.dt
            B33 = -48 * sigma**2 / kappa**3 * rho * (1 - np.exp(-kappa * self.dt)) + 24 * sigma**2 / kappa**2 * rho * self.dt * (1 + np.exp(-kappa * self.dt))
            deriv_array[1, 3] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = (1 - np.exp(-2 * kappa * self.dt)) / kappa * theta
            B13 = 1 / kappa ** 2 * (1 + 4 * rho ** 2 - np.exp(-kappa * self.dt)) * theta * (1 - np.exp(-kappa * self.dt)) - 4 / kappa * rho ** 2 * theta * self.dt * np.exp(-kappa * self.dt)
            B33 = -1 / kappa ** 3 * theta * (1 - np.exp(-kappa * self.dt)) ** 2 - 6 / kappa ** 3 * theta * (1 + 8 * rho ** 2) * (1 - np.exp(-kappa * self.dt)) + 24 / kappa ** 2 * rho ** 2 * theta * self.dt * (1 + np.exp(-kappa * self.dt)) + 6 / kappa ** 2 * theta * self.dt
            B22 = 0
            B12 = 0
            B23 = 0
            deriv_array[2, 2] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = 0
            B13 = 8 * sigma / kappa**2 * rho * theta * (1 - np.exp(-kappa * self.dt)) - 8 * sigma / kappa * rho * theta * self.dt * np.exp(-kappa * self.dt)
            B33 = - 96 * sigma / kappa ** 3 * theta * rho * (1 - np.exp(-kappa * self.dt)) + 48 * sigma / kappa ** 2 * rho * theta * self.dt * (1 + np.exp(-kappa * self.dt))
            B22 = 0
            B12 = theta / kappa * (1 - np.exp(-kappa * self.dt))
            B23 = 3 / kappa ** 2 * theta * (np.exp(-kappa * self.dt) - 1) + 3 / kappa * theta * self.dt
            deriv_array[2, 3] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            B11 = 0
            B12 = 0
            B13 = 4 * sigma**2 / kappa**2 * theta * (1 - np.exp(-kappa * self.dt)) - 4 / kappa * sigma**2 * theta * self.dt * np.exp(-kappa * self.dt)
            B22 = 0
            B23 = 0
            B33 = -48 * sigma**2 / kappa**3 * theta * (1 - np.exp(-kappa * self.dt)) + 24 * sigma**2 / kappa**2 * theta * self.dt * (1 + np.exp(-kappa * self.dt))
            deriv_array[3, 3] = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])

            deriv_array = deriv_array[np.ix_(wrt, wrt)]
            return deriv_array[np.triu_indices(len(wrt))]

    def Q(self, param, num):
        kappa, theta, sigma, rho = param
        if num == 2:
            result = np.zeros((9, 9))
            result[-1, 0] = 2 / kappa**2 * (1 - np.exp(-kappa * self.dt))**2
            return result
        elif num == 1:
            N11 = 1 / kappa * np.exp(-kappa * self.dt) * (1 - np.exp(-kappa * self.dt)) * sigma**2
            N12 = rho * sigma * self.dt * np.exp(-kappa * self.dt)
            N13 = -sigma**2 / kappa**2 * np.exp(-kappa * self.dt) * (1 - np.exp(-kappa * self.dt)) + sigma**2 / kappa * (1 + kappa * rho**2 * self.dt) * self.dt * np.exp(-kappa * self.dt)
            N22 = 1 / kappa * (1 - np.exp(-kappa * self.dt))
            N23 = 3 * rho * sigma / kappa**2 * (1 - np.exp(-kappa * self.dt) - kappa * np.exp(-kappa * self.dt))
            N33 = 1 / kappa**3 * (6 * sigma**2 * (1 + 2 * rho**2) + 4 * kappa**2 * theta * self.dt) * (1 - np.exp(-kappa * self.dt)) - 1 / kappa**3 * (3 * sigma**2 + 4 * kappa * theta) * (1 - np.exp(-kappa * self.dt))**2 - 6 * sigma**2 / kappa**2 * (1 + 2 * rho**2 + self.dt * rho**2 * kappa) * self.dt * np.exp(-kappa * self.dt)
            result = np.zeros((9, 3))
            result[:, 0] = np.array([N11, N12, N13, N12, N22, N23, N13, N23, N33])
            return result
        elif num == 0:
            N11 = (1 - np.exp(-kappa * self.dt))**2 * sigma**2 / (2 * kappa) * theta
            N12 = theta / kappa * rho * sigma * (1 - np.exp(-kappa * self.dt) - kappa * self.dt * np.exp(-kappa * self.dt))
            N13 = sigma**2 * theta / (2 * kappa**2) * (1 + 4 * rho**2 + np.exp(-kappa * self.dt)) * (1 - np.exp(-kappa * self.dt)) - sigma**2 * theta / kappa * (1 + (kappa * self.dt + 2) * rho**2) * self.dt * np.exp(-kappa * self.dt)
            N22 = theta * self.dt + theta / kappa * (np.exp(-kappa * self.dt) - 1)
            N23 = 3 * rho * sigma / kappa**2 * theta * (kappa - 2 + (kappa * self.dt + 2) * np.exp(-kappa * self.dt))
            N33 = 1 / kappa**3 * (9 * sigma**2 * theta * (1 + 4 * rho**2) + 4 * kappa**2 * theta**2 * self.dt) * (np.exp(-kappa * self.dt) - 1) + 1 / kappa**3 * (2 * kappa * theta**2 + 3 / 2 * sigma**2 * theta) * (np.exp(-kappa * self.dt) - 1)**2 + 6 * sigma**2 / kappa**2 * theta * (1 + 4 * rho**2 + kappa * rho**2 * self.dt) * self.dt * np.exp(-kappa * self.dt) + 3 * sigma**2 / kappa**2 * theta * (1 + 4 * rho**2) * self.dt + 2 * theta**2 * self.dt ** 2
            return np.array([N11, N12, N13, N12, N22, N23, N13, N23, N33])

    def generate_observations(self, t_max, inter_steps, seed=None, verbose=0):
        dt = self.dt / inter_steps
        if seed is not None:
            np.random.seed(seed)

        if not isinstance(inter_steps, int):
            raise TypeError('Attribute inter_steps needs to be integer.')

        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        if self.observations is not None:
            warnings.warn('There are already observations. New observations will be appended to the end.')
            v0, Y_disc_0, Y_disc2_0 = self.observations[-1]
        else:
            v0, Y_disc_0, Y_disc2_0 = self.init.sample(param=self.true_param, n=1)
        kappa, theta, sigma, rho = self.true_param
        S0 = 0

        if 2 * theta * kappa < sigma ** 2:
            warnings.warn('Feller Positivity Condition is not met (2 * theta * kappa < sigma**2)')

        steps = int(np.ceil(np.round(t_max / dt, 7))) + 1
        W = np.random.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=steps)
        Wdt = W * np.sqrt(dt)
        dW1, dW2 = Wdt[:, 0], Wdt[:, 1]
        S, v = [S0], [v0]
        tr = trange(1, steps, desc='Generating Observations') if verbose == 1 else range(1, steps)
        for timestep in tr:
            S0 = S0 + np.sqrt(v0) * dW2[timestep - 1]
            v0 = np.maximum(v0 + kappa * (theta - v0) * dt + sigma * np.sqrt(v0) * dW1[timestep - 1], 0)
            if timestep % inter_steps == 0:
                S.append(S0)
                v.append(v0)

        Y_disc = np.append(Y_disc_0, np.diff(S))
        Y_disc2 = Y_disc ** 2

        observations = np.vstack((v, Y_disc, Y_disc2)).T
        if self.observations is not None:
            observations = np.vstack((self.observations, observations[1:]))
            self.seed = str(self.seed) + '+' + str(seed) if self.seed is not None else seed
            t_max = observations.shape[0] - 1
        else:
            self.seed = seed

        if self.seed is not None:
            np.savetxt('./saves/HestonModel/Observations/observations_{}_seed{}_{}obs.txt'.format(self.savestring, self.seed, observations.shape[0] - 1), observations)
        else:
            np.savetxt('./saves/HestonModel/Observations/observations_{}_{}obs.txt'.format(self.savestring, observations.shape[0] - 1), observations)
        self.observations = observations


class OUNIGModel(PolynomialModel):
    def __init__(self, first_observed, init, dt, true_param=None, wrt=None):
        if true_param is None:
            true_param = np.repeat(np.nan, 3)

        super().__init__(first_observed, init, dt, true_param)
        self.dim = 2
        self.params_names = np.array(['lambda', 'kappa', 'delta'])
        self.params_bounds = np.array([[0.001, 10], [0.0001, 30], [0.0001, 10]])

        if not np.isnan(self.true_param).any():
            self.savestring = 'par=[{:.3f}, {:.3f}, {:.3f}]_dt={:.1e}'.format(self.true_param[0], self.true_param[1], self.true_param[2], self.dt)
            if wrt is not None:
                self.setup_filter(wrt)
        elif wrt is not None:
            warnings.warn('Argument wrt was not used since the whole parameter has not yet been estimated. Please use method "setup_filter" after this has been done.', Warning)

    def a(self, param, order=0, wrt=np.array([0, 1, 2])):
        wrt = np.atleast_1d(wrt)
        if order == 0:
            return np.zeros(2)
        elif order == 1:
            deriv_array = np.zeros((3, 2))
            return deriv_array[wrt]
        elif order == 2:
            deriv_array = np.zeros((3, 3, 2))
            deriv_array = deriv_array[np.ix_(wrt, wrt)]
            return deriv_array[np.triu_indices(len(wrt))]

    def A(self, param, order=0, wrt=np.array([0, 1, 2])):
        wrt = np.atleast_1d(wrt)
        lamb, kappa, delta = param
        if order == 0:
            return np.array([[np.exp(-lamb * self.dt), 0], [kappa * (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa), np.exp(-kappa * self.dt)]])
        elif order == 1:
            deriv_array = np.zeros((3, 2, 2))
            deriv_array[0] = np.array([[-self.dt * np.exp(-lamb * self.dt), 0], [self.dt * kappa * np.exp(-lamb * self.dt) / (lamb - kappa) + kappa * (np.exp(-lamb * self.dt) - np.exp(-kappa * self.dt)) / (lamb - kappa)**2, 0]])
            deriv_array[1] = np.array([[0, 0], [((1 - self.dt * kappa) * np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa) + kappa * (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa)**2, -self.dt * np.exp(-kappa * self.dt)]])
            return deriv_array[wrt]
        elif order == 2:
            deriv_array = np.zeros((3, 3, 2, 2))
            A_lamb_lamb = np.array([[self.dt ** 2 * np.exp(-lamb * self.dt), 0], [2 * kappa * (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa)**3 - 2 * self.dt * kappa * np.exp(-lamb * self.dt) / (lamb - kappa)**2 - self.dt ** 2 * kappa * np.exp(-lamb * self.dt) / (lamb - kappa), 0]])
            A_lamb_kappa = np.array([[0, 0], [self.dt * (lamb * np.exp(-lamb * self.dt) + kappa * np.exp(-kappa * self.dt)) / (lamb - kappa)**2 + (lamb + kappa) / (lamb - kappa)**3 * (np.exp(-lamb * self.dt) - np.exp(-kappa * self.dt)), 0]])
            A_kappa_kappa = np.array([[0, 0], [self.dt * (self.dt * kappa - 2) * np.exp(-kappa * self.dt) / (lamb - kappa) + ((1 - 2 * kappa * self.dt) * np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa)**2 + (lamb + kappa) / (lamb - kappa)**3 * (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)), self.dt**2 * np.exp(-kappa * self.dt)]])

            deriv_array[0, 0] = A_lamb_lamb
            deriv_array[0, 1] = A_lamb_kappa
            deriv_array[1, 1] = A_kappa_kappa

            deriv_array = deriv_array[np.ix_(wrt, wrt)]
            return deriv_array[np.triu_indices(len(wrt))]

    def B(self, param, order):
        lamb, kappa, delta = param
        dicts = return_dict(2, order)
        alpha = 1

        OU_A = np.zeros((n_dim(2, order), n_dim(2, order)))
        OU_A[mult_to_ind([1, 0], dicts), mult_to_ind([1, 0], dicts)] = -lamb
        OU_A[mult_to_ind([0, 1], dicts), mult_to_ind([1, 0], dicts)] = kappa
        OU_A[mult_to_ind([0, 1], dicts), mult_to_ind([0, 1], dicts)] = -kappa

        OU_A[mult_to_ind([2, 2], dicts), 0] = delta / alpha ** 3
        OU_A[mult_to_ind([2, 0], dicts), 0] = delta / alpha
        OU_A[mult_to_ind([0, 2], dicts), 0] = delta / alpha
        OU_A[mult_to_ind([4, 0], dicts), 0] = 3 * delta / alpha ** 3
        OU_A[mult_to_ind([0, 4], dicts), 0] = 3 * delta / alpha ** 3

        OU_Bc = np.zeros(OU_A.shape)
        for i in range(OU_A.shape[0]):
            for j in range(OU_Bc.shape[0]):
                masks = mask(dicts[i], dicts, typ='leq') & mask(dicts[j], dicts, typ='leq')
                lamb_ell = (ind_to_mult(i, dicts) - dicts)[masks]
                mu_ell = (ind_to_mult(j, dicts) - dicts)[masks]
                OU_Bc[i, j] = (multi_binom(dicts[i], dicts[masks]) * OU_A[mult_to_ind(lamb_ell, dicts), mult_to_ind(mu_ell, dicts)]).sum()

        OU_B = expm(OU_Bc * self.dt)
        return OU_B

    def C(self, param, t):
        lamb, kappa, delta = param
        lamb, kappa, delta = param
        alpha = 1
        psi = (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa) + np.exp(-kappa * self.dt) / (lamb + kappa)
        B11 = 1 - np.exp(-2 * lamb * self.dt)
        B12 = kappa / (lamb + kappa) - kappa * np.exp(-lamb * self.dt) * psi
        B22 = np.exp(-2 * kappa * self.dt) * (kappa / (lamb + kappa))**2 + (1 - np.exp(-2 * kappa * self.dt)) * (kappa / (lamb + kappa) + lamb / kappa) - kappa**2 * psi**2
        return (delta / alpha) / (2 * lamb) * np.array([[B11, B12], [B12, B22]])

    def C_lim(self, param, order=0, wrt=np.array([0, 1, 2])):
        wrt = np.atleast_1d(wrt)
        lamb, kappa, delta = param
        alpha = 1
        psi = (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa) + np.exp(-kappa * self.dt) / (lamb + kappa)
        if order == 0:
            B11 = 1 - np.exp(-2 * lamb * self.dt)
            B12 = kappa / (lamb + kappa) - kappa * np.exp(-lamb * self.dt) * psi
            B22 = np.exp(-2 * kappa * self.dt) * (kappa / (lamb + kappa)) ** 2 + (1 - np.exp(-2 * kappa * self.dt)) * (kappa / (lamb + kappa) + lamb / kappa) - kappa ** 2 * psi ** 2
            return (delta / alpha) / (2 * lamb) * np.array([[B11, B12], [B12, B22]])
        elif order == 1:
            psi_l = self.dt * np.exp(-lamb * self.dt) / (lamb - kappa) + (np.exp(-lamb * self.dt) - np.exp(-kappa * self.dt)) / (lamb - kappa) ** 2 - np.exp(-kappa * self.dt) / (lamb + kappa) ** 2
            psi_k = (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa) ** 2 - 2 * self.dt * np.exp(-kappa * self.dt) * lamb / (lamb ** 2 - kappa ** 2) - np.exp(-kappa * self.dt) / (lamb + kappa) ** 2

            deriv_array = np.zeros((3, 2, 2))
            B11 = (delta / alpha) / lamb * self.dt * np.exp(-2 * lamb * self.dt) - (delta / alpha) / (2 * lamb ** 2) * (1 - np.exp(-2 * lamb * self.dt))
            B12 = (delta / alpha) / (2 * lamb ** 2) * (kappa * np.exp(-lamb * self.dt) * psi - kappa / (lamb + kappa)) + (delta / alpha) / (2 * lamb) * (kappa * np.exp(-lamb * self.dt) * (self.dt * psi - psi_l) - kappa / (lamb + kappa) ** 2)
            B22 = (delta / alpha) / (2 * lamb) * ((1 - np.exp(-2 * kappa * self.dt)) * (1 / kappa - kappa / (lamb + kappa) ** 2) - 2 * np.exp(-2 * kappa * self.dt) * kappa ** 2 / (lamb + kappa) ** 3 - 2 * kappa ** 2 * psi * psi_l) - (delta / alpha) / (2 * lamb ** 2) * (np.exp(-2 * kappa * self.dt) * (kappa / (lamb + kappa)) ** 2 + (1 - np.exp(-2 * kappa * self.dt)) * (kappa / (lamb + kappa) + lamb / kappa) - kappa ** 2 * psi ** 2)
            deriv_array[0] = np.array([[B11, B12], [B12, B22]])

            B11 = 0
            B12 = (delta / alpha) / (2 * lamb) * (lamb / (lamb + kappa) ** 2 - np.exp(-lamb * self.dt) * psi - kappa * np.exp(-lamb * self.dt) * psi_k)
            B22 = (delta / alpha) / lamb * (np.exp(-2 * kappa * self.dt) * (lamb * kappa / (lamb + kappa) ** 3 - self.dt * kappa ** 2 / (lamb + kappa) ** 2 + self.dt * kappa / (lamb + kappa) + self.dt * lamb / kappa) + (1 - np.exp(-2 * kappa * self.dt)) * lamb / 2 * (1 / (lamb + kappa) ** 2 - 1 / kappa ** 2) - kappa * psi ** 2 - kappa ** 2 * psi * psi_k)
            deriv_array[1] = np.array([[B11, B12], [B12, B22]])

            B11 = 1 - np.exp(-2 * lamb * self.dt)
            B12 = kappa / (lamb + kappa) - kappa * np.exp(-lamb * self.dt) * psi
            B22 = np.exp(-2 * kappa * self.dt) * (kappa / (lamb + kappa)) ** 2 + (1 - np.exp(-2 * kappa * self.dt)) * (kappa / (lamb + kappa) + lamb / kappa) - kappa ** 2 * psi ** 2
            deriv_array[2] = (1 / alpha) / (2 * lamb) * np.array([[B11, B12], [B12, B22]])

            return deriv_array[wrt]
        elif order == 2:
            psi_l = self.dt * np.exp(-lamb * self.dt) / (lamb - kappa) + (np.exp(-lamb * self.dt) - np.exp(-kappa * self.dt)) / (lamb - kappa) ** 2 - np.exp(-kappa * self.dt) / (lamb + kappa) ** 2
            psi_k = (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa) ** 2 - 2 * self.dt * np.exp(-kappa * self.dt) * lamb / (lamb ** 2 - kappa ** 2) - np.exp(-kappa * self.dt) / (lamb + kappa) ** 2
            psi_ll = self.dt * ((kappa - lamb) * self.dt - 2) * np.exp(-lamb * self.dt) / (lamb - kappa) ** 2 - 2 * (np.exp(-lamb * self.dt) - np.exp(-kappa * self.dt)) / (lamb - kappa) ** 3 + 2 * np.exp(-kappa * self.dt) / (lamb + kappa) ** 3
            psi_lk = self.dt * (np.exp(-kappa * self.dt) + np.exp(-lamb * self.dt)) / (lamb - kappa) ** 2 + 2 * (np.exp(-lamb * self.dt) - np.exp(-kappa * self.dt)) / (lamb - kappa) ** 3 + np.exp(-kappa * self.dt) / (lamb + kappa) ** 3 * (2 + (lamb + kappa) * self.dt)
            psi_kk = 2 * ((np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa) ** 3 + np.exp(-kappa * self.dt) / (lamb + kappa) ** 3 + self.dt**2 * lamb * np.exp(-kappa * self.dt) / (lamb ** 2 - kappa ** 2) - 4 * self.dt * np.exp(-kappa * self.dt) * lamb * kappa / (lamb ** 2 - kappa ** 2) ** 2)

            deriv_array = np.zeros((3, 3, 2, 2))

            B11 = (delta / alpha) / lamb ** 3 * (1 - np.exp(-2 * lamb * self.dt)) - 2 * (delta / alpha) / lamb ** 2 * self.dt * (1 + lamb * self.dt) * np.exp(-2 * lamb * self.dt)
            B12 = -(delta / alpha) / lamb ** 3 * (kappa * np.exp(-lamb * self.dt) * psi - kappa / (lamb + kappa)) + (delta / alpha) / lamb ** 2 * (kappa * np.exp(-lamb * self.dt) * (psi_l - psi * self.dt) + kappa / (lamb + kappa) ** 2) + (delta / alpha) / (2 * lamb) * (2 * kappa / (lamb + kappa) ** 3 - kappa * np.exp(-lamb * self.dt) * (self.dt**2 * psi - 2 * self.dt * psi_l + psi_ll))
            B22 = (delta / alpha) / lamb * ((1 - np.exp(-2 * kappa * self.dt)) * kappa / (lamb + kappa) ** 3 + 3 * np.exp(-2 * kappa * self.dt) * kappa ** 2 / (lamb + kappa) ** 4 - kappa ** 2 * psi_l ** 2 - kappa ** 2 * psi * psi_ll) - (delta / alpha) / lamb ** 2 * ((1 - np.exp(-2 * kappa * self.dt)) * (1 / kappa - kappa / (lamb + kappa) ** 2) - 2 * np.exp(-2 * kappa * self.dt) * kappa ** 2 / (lamb + kappa) ** 3 - 2 * kappa ** 2 * psi * psi_l) + (delta / alpha) / lamb ** 3 * (np.exp(-2 * kappa * self.dt) * (kappa / (lamb + kappa)) ** 2 + (1 - np.exp(-2 * kappa * self.dt)) * (kappa / (lamb + kappa) + lamb / kappa) - kappa ** 2 * psi ** 2)
            deriv_array[0, 0] = np.array([[B11, B12], [B12, B22]])

            B11 = 0
            B12 = (delta / alpha) / (2 * lamb ** 2) * (np.exp(-lamb * self.dt) * psi + kappa * np.exp(-lamb * self.dt) * psi_k - lamb / (lamb + kappa) ** 2) + (delta / alpha) / (2 * lamb) * (np.exp(-lamb * self.dt) * (self.dt * psi - psi_l) + kappa * np.exp(-lamb * self.dt) * (self.dt * psi_k - psi_lk) - (lamb - kappa) / (lamb + kappa) ** 3)
            B22 = (delta / alpha) / lamb * (np.exp(-2 * kappa * self.dt) * (kappa * (kappa - 2 * lamb) / (lamb + kappa) ** 4 + 2 * self.dt * kappa ** 2 / (lamb + kappa) ** 3 - self.dt * kappa / (lamb + kappa) ** 2 + self.dt / kappa) + (1 - np.exp(-2 * kappa * self.dt)) * ((kappa - lamb) / (2 * (lamb + kappa) ** 3) - 1 / (2 * kappa ** 2)) - 2 * kappa * psi * psi_l - kappa ** 2 * psi_l * psi_k - kappa ** 2 * psi * psi_lk) - (delta / alpha) / lamb ** 2 * (np.exp(-2 * kappa * self.dt) * (lamb * kappa / (lamb + kappa) ** 3 - self.dt * kappa ** 2 / (lamb + kappa) ** 2 + kappa * self.dt / (lamb + kappa) + lamb * self.dt / kappa) + (1 - np.exp(-2 * kappa * self.dt)) * lamb / 2 * (1 / (lamb + kappa) ** 2 - 1 / kappa ** 2) - kappa * psi ** 2 - kappa ** 2 * psi * psi_k)
            deriv_array[0, 1] = np.array([[B11, B12], [B12, B22]])

            B11 = (1 / alpha) / lamb * self.dt * np.exp(-2 * lamb * self.dt) - (1 / alpha) / (2 * lamb ** 2) * (1 - np.exp(-2 * lamb * self.dt))
            B12 = (1 / alpha) / (2 * lamb ** 2) * (kappa * np.exp(-lamb * self.dt) * psi - kappa / (lamb + kappa)) + (1 / alpha) / (2 * lamb) * (kappa * np.exp(-lamb * self.dt) * (self.dt * psi - psi_l) - kappa / (lamb + kappa) ** 2)
            B22 = (1 / alpha) / (2 * lamb) * ((1 - np.exp(-2 * kappa * self.dt)) * (1 / kappa - kappa / (lamb + kappa) ** 2) - 2 * np.exp(-2 * kappa * self.dt) * kappa ** 2 / (lamb + kappa) ** 3 - 2 * kappa ** 2 * psi * psi_l) - (1 / alpha) / (2 * lamb ** 2) * (np.exp(-2 * kappa * self.dt) * (kappa / (lamb + kappa)) ** 2 + (1 - np.exp(-2 * kappa * self.dt)) * (kappa / (lamb + kappa) + lamb / kappa) - kappa ** 2 * psi ** 2)
            deriv_array[0, 2] = np.array([[B11, B12], [B12, B22]])

            B11 = 0
            B12 = - (delta / alpha) / (2 * lamb) * (2 * lamb / (lamb + kappa) ** 3 + 2 * np.exp(-lamb * self.dt) * psi_k + kappa * np.exp(-lamb * self.dt) * psi_kk)
            B22 = (delta / alpha) / lamb * (np.exp(-2 * kappa * self.dt) * (2 * (kappa ** 2 * self.dt**2 + lamb * self.dt) / (lamb + kappa) ** 2 - 4 * self.dt * kappa * lamb / (lamb + kappa) ** 3 + lamb * (lamb - 2 * kappa) / (lamb + kappa) ** 4 - 2 * kappa * self.dt**2 / (lamb + kappa) - 2 * lamb / kappa ** 2 * self.dt * (1 + kappa * self.dt)) + (1 - np.exp(-2 * kappa * self.dt)) * lamb * (1 / kappa ** 3 - 1 / (lamb + kappa) ** 3) - psi ** 2 - 4 * kappa * psi * psi_k - kappa ** 2 * psi_k ** 2 - kappa ** 2 * psi * psi_kk)
            deriv_array[1, 1] = np.array([[B11, B12], [B12, B22]])

            B11 = 0
            B12 = (1 / alpha) / (2 * lamb) * (lamb / (lamb + kappa) ** 2 - np.exp(-lamb * self.dt) * psi - kappa * np.exp(-lamb * self.dt) * psi_k)
            B22 = (1 / alpha) / lamb * (np.exp(-2 * kappa * self.dt) * (lamb * kappa / (lamb + kappa) ** 3 - self.dt * kappa ** 2 / (lamb + kappa) ** 2 + self.dt * kappa / (lamb + kappa) + self.dt * lamb / kappa) + (1 - np.exp(-2 * kappa * self.dt)) * lamb / 2 * (1 / (lamb + kappa) ** 2 - 1 / kappa ** 2) - kappa * psi ** 2 - kappa ** 2 * psi * psi_k)
            deriv_array[1, 2] = np.array([[B11, B12], [B12, B22]])

            deriv_array = deriv_array[np.ix_(wrt, wrt)]
            return deriv_array[np.triu_indices(len(wrt))]

    def Q(self, param, num):
        lamb, kappa, delta = param
        alpha = 1
        if num == 2:
            return np.zeros((4, 4))
        elif num == 1:
            return np.zeros((4, 2))
        elif num == 0:
            psi = (np.exp(-kappa * self.dt) - np.exp(-lamb * self.dt)) / (lamb - kappa) + np.exp(-kappa * self.dt) / (lamb + kappa)
            B11 = 1 - np.exp(-2 * lamb * self.dt)
            B12 = kappa / (lamb + kappa) - kappa * np.exp(-lamb * self.dt) * psi
            B22 = np.exp(-2 * kappa * self.dt) * (kappa / (lamb + kappa)) ** 2 + (1 - np.exp(-2 * kappa * self.dt)) * (kappa / (lamb + kappa) + lamb / kappa) - kappa ** 2 * psi ** 2
            return (delta / alpha) / (2 * lamb) * np.array([B11, B12, B12, B22])

    def generate_observations(self, t_max, inter_steps, seed=None, verbose=0):
        dt = self.dt / inter_steps

        if seed is not None:
            np.random.seed(seed)

        if not isinstance(inter_steps, int):
            raise TypeError('Attribute inter_steps needs to be integer.')

        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        if self.observations is not None:
            warnings.warn('There are already observations. New observations will be appended to the end.')
            x = self.observations[-1]
        else:
            x = self.init.sample(param=self.true_param, n=1)
        lamb, kappa, delta = self.true_param
        alpha = 1

        steps = int(np.ceil(np.round(t_max / dt, 7))) + 1
        W = np.random.standard_normal(size=(steps - 1, self.dim))
        dIG = invgauss_bn(xi=delta * dt, eta=alpha, size=steps - 1, seed=seed)
        dNIG = np.sqrt(dIG)[:, None] * W

        expQ = np.array([[np.exp(-lamb * dt), 0], [kappa * (np.exp(-kappa * dt) - np.exp(-lamb * dt)) / (lamb - kappa), np.exp(-kappa * dt)]])
        X = x
        observations = [X]
        tr = trange(1, steps, desc='Generating Observations') if verbose == 1 else range(1, steps)
        for timestep in tr:
            X = expQ @ (X + dNIG[timestep - 1])
            if timestep % inter_steps == 0:
                observations.append(X)

        observations = np.stack(observations)
        if self.observations is not None:
            observations = np.vstack((self.observations, observations[1:]))
            self.seed = str(self.seed) + '+' + str(seed) if self.seed is not None else seed
            t_max = observations.shape[0] - 1
        else:
            self.seed = seed

        if self.seed is not None:
            np.savetxt('./saves/OUNIGModel/Observations/observations_{}_seed{}_{}obs.txt'.format(self.savestring, self.seed, observations.shape[0] - 1), observations)
        else:
            np.savetxt('./saves/OUNIGModel/Observations/observations_{}_{}obs.txt'.format(self.savestring, observations.shape[0] - 1), observations)
        self.observations = observations


# init = InitialDistribution(dist='Dirac', hyper=[0.3**2, 0, 0])
init = InitialDistribution(dist='Dirac', hyper=[0.5, 1])
# init = InitialDistribution(dist='Gamma_Dirac', hyper=[0, 0])

## Test the new restructuring #1 (Simulation study, no observations needed)
heston = HestonModel(first_observed=1, init=init, dt=1/24000, true_param=np.array([1, 0.4**2, 0.3, -0.5]), wrt=1)
V, Std, Corr = heston.compute_V()
ou = OUNIGModel(first_observed=1, init=init, dt=1, true_param=np.array([1, 0.5, 1.5]), wrt=2)
V, Std, Corr = ou.compute_V()


## Test the new restructuring #2 (Simulation study with observations)
heston = HestonModel.from_observations(first_observed=0, init=init, dt=1/24000, obs=200000, inter_steps=250, true_param=np.array([1, 0.4**2, 0.3, -0.5]), seed=20)
result = heston.fit_qml(fit_parameter=2, initial=0.3, t=200000)
result = heston.fit_qml_sequence(fit_parameter=[0, 2], initial=[1, 0.3], t_max=1500)
V, Std, Corr = heston.compute_V(kind='estimate', wrt=2, verbose=1)

fits = []
for i in range(10):
    ou = OUNIGModel.from_observations(first_observed=1, init=init, dt=1/24000, obs=10000, inter_steps=250, true_param=np.array([1, 0.5, 1.5]), seed=i)
    fits.append(ou.fit_qml(fit_parameter=1, initial=0.5, t=200000))
V, Std, Corr = ou.compute_V(kind='estimate', wrt=1, verbose=1)


## Test the new restructuring #3 (Semi-Real-world scenario)
heston = HestonModel.from_observations(first_observed=1, init=init, obs='observations_par=[1.000, 0.160, 0.300, -0.500]_seed5_200000obs.txt', dt=1/250, true_param=np.array([1, 0.4**2, np.nan, -0.5]))
result = heston.fit_qml(fit_parameter=2, initial=0.3, t=10000)
result = heston.fit_qml(fit_parameter=2, initial=0.3, t=12000, update_estimate=True)
V, Std, Corr = heston.compute_V(kind='estimate', wrt=2, verbose=1, filter_unobserved=True)
heston.setup_filter(wrt=2)
V, Std, Corr = heston.compute_V()


## Test the new restructuring #3 (Real-world scenario)
heston = HestonModel.from_observations(first_observed=1, init=init, obs='observations_1.000_0.160_0.300_-0.500_seed5_200000obs.txt')
seq = heston.fit_qml_sequence(fit_parameter=[0, 1, 2, 3], initial=[1, 0.4**2, 0.3, -0.5], t_max=200)
res = heston.fit_qml(fit_parameter=[0, 1, 2, 3], initial=[1, 0.4**2, 0.3, -0.5], t=10000, update_estimate=True)


## Heston wrt = 2 --> 82.97s
## Heston wrt = [2, 3] --> 909.72s
## Heston wrt = [1, 2, 3] --> 7555.98s
## Heston wrt = [0, 1, 2, 3] --> 58115.90s
tic = time()
hestonf = FilteredHestonModel(first_observed=1, kappa=1, theta_vol=0.4, sig=0.3, rho=-0.5, v0=0.4**2, wrt=3)
V, Std, Corr = hestonf.calculate_V()
toc = time()
print(toc - tic)
