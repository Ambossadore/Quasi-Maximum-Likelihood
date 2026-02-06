import os
import pickle as pkl
import warnings
from functools import partial
from itertools import product, compress
from pathlib import Path
from time import time

from joblib import Parallel, delayed, cpu_count
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.special import factorial
from tqdm import tqdm, trange
from tqdm_joblib import tqdm_joblib

from functions import *


class KalmanFilter:
    """
    This class builds the Kalman filter covariance matrices and contains methods to compute the Kalman
    filter and its derivatives with respect to k parameters for some given polynomial state space model
    under some fixed true parameter.
    """
    def __init__(self, dim, a, A, C, E_0, Cov_0, first_observed=0):
        """
        :param dim: Dimension of the underlying polynomial state space model
        :param a: State transition vector of the underlying polynomial state space model. Array of shape (1 + k + k*(k+1)/2, dim).
            a[0] is the state transition vector, the following k rows are the first derivatives of a[0] with respect to the parameters
            1,...,k and the following k*(k+1)/2 rows are the distinct second derivatives of a[0] with respect to the parameters 11,12,...,kk.
        :param A: State transition matrix of the underlying polynomial state space model. Array of shape (1 + k + k*(k+1)/2, dim, dim).
            A[0] is the state transition matrix, the following k rows are the first derivatives of A[0] with respect to the parameters
            1,...,k and the following k*(k+1)/2 rows are the distinct second derivatives of A[0] with respect to the parameters 11,12,...,kk.
        :param C: Limiting Covariance matrix of the noise sequence of the polynomial state space model. Array of shape (k + 1, dim, dim).
            C[0] is the covariance matrix, the following k rows are the first derivatives of C[0] with respect to the parameters
            1,...,k and the following k*(k+1)/2 rows are the distinct second derivatives of C[0] with respect to the parameters 11,12,...,kk.
        :param E_0: Function that returns E(X(0)). Shoud accept arguments deriv_order=1,2 and wrt, the latter being an integer or list
            of integers between 1 and k-1, to return the first or second derivatives of E(X(0)) with respect to the parameters given
            by the argument wrt.
        :param Cov_0: Function that returns Cov(X(0)). Should accept arguments deriv_order=1,2 and wrt, the latter being an integer
            or list of integers between 1 and k-1, to return the first or second derivatives of Cov(X(0)) with respect to the parameters
            given by the argument wrt.
        :param first_observed: Integer between 0,..., dim-1 that denotes the first observed component of the state space model
        """
        self.dim = dim
        self.a = a
        self.A = A
        self.C = C
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
        """
        Computes the sequence of Kalman filter covariance matrices Sigma(t+1, t) and Sigma(t, t).
        :param t_max: Maximal number of t in the sequence
        """
        Cov_0 = self.Cov_0()
        Sig_tp1_t_list = [Cov_0]
        Sig_tt_list = [Cov_0 - Cov_0[:, self.first_observed:] @ np.linalg.pinv(Cov_0[self.first_observed:, self.first_observed:]) @ Cov_0[:, self.first_observed:].T]
        # tt_inv = np.eye(self.dim) - Cov_0 @ self.H.T @ np.linalg.pinv(Cov_0[self.first_observed:, self.first_observed:]) @ self.H
        # Sig_tt_list = [tt_inv @ Cov_0 @ tt_inv.T]
        Sig_tp1_t_list.append(self.A[0] @ Sig_tt_list[0] @ self.A[0].T + self.C[0])
        Sig_tp1_t = Sig_tp1_t_list[-1]
        for t in range(1, t_max):
            Sig_tt = Sig_tp1_t - Sig_tp1_t[:, self.first_observed:] @ np.linalg.inv(Sig_tp1_t[self.first_observed:, self.first_observed:]) @ Sig_tp1_t[:, self.first_observed:].T
            # tt_inv = np.eye(self.dim) - Sig_tp1_t @ self.H.T @ np.linalg.inv(Sig_tp1_t[self.first_observed:, self.first_observed:]) @ self.H
            # Sig_tt = tt_inv @ Sig_tp1_t @ tt_inv.T
            # Sig_tt[self.first_observed:, self.first_observed:] = 0.
            # Sig_tt[:self.first_observed, self.first_observed:] = 0.
            # Sig_tt[self.first_observed:, :self.first_observed] = 0.
            Sig_tp1_t = self.A[0] @ Sig_tt @ self.A[0].T + self.C[0]
            if np.isclose(Sig_tp1_t, Sig_tp1_t_list[-1], atol=0, rtol=1e-8).all():
                break
            Sig_tp1_t_list.append(Sig_tp1_t)
            Sig_tt_list.append(Sig_tt)
        if len(Sig_tt_list) == t_max:
            warnings.warn('Kalman filter covariance matrices have not converged.')
        self.Sig_tp1_t_list = np.stack(Sig_tp1_t_list)
        self.Sig_tt_list = np.stack(Sig_tt_list)
        self.Sig_tp1_t_lim = Sig_tp1_t_list[-1]
        self.Sig_tt_lim = Sig_tt_list[-1]
        self.K_lim = self.A[0] @ self.Sig_tp1_t_lim[:, self.first_observed:] @ np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
        self.F_lim = self.A[0] - self.K_lim @ self.H

    def S_star(self, wrt):
        """
        Computes the limiting Kalman filter covariance matrix derivatives S_j for j=1,...,k (see Lemma 3.7)
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters differentiated with respect to.
        :return: Array of shape (len(wrt), dim, dim) with different derivatives in different rows.
        """
        wrt = np.atleast_1d(wrt)
        BB_lim_partial = self.C[1 + wrt]
        right_side = np.einsum('jk, kl, lmi -> ijm', self.A[0], self.Sig_tt_lim, self.A[1 + wrt].T) + np.einsum('ijk, kl, lm -> ijm', self.A[1 + wrt], self.Sig_tt_lim, self.A[0].T) + BB_lim_partial
        S_star_vectorized = (np.linalg.inv(np.eye(self.dim ** 2) - np.kron(self.F_lim, self.F_lim)) @ right_side.reshape(np.size(wrt), self.dim**2, order='F').T).T
        return S_star_vectorized.reshape(np.size(wrt), self.dim, self.dim, order='F')

    def R_star(self, wrt):
        """
        Computes the limiting Kalman filter covariance matrix second derivatives R_ij for i,j = 1,...,k
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters differentiated with respect to.
        :return: Array of shape (len(wrt) * (len(wrt) + 1) / 2, dim, dim) with second derivatives 11,12,...,kk in different rows.
        """
        wrt = np.atleast_1d(wrt)
        k = int(np.sqrt(9 / 4 + 2 * (self.a.shape[0] - 1)) - 3 / 2)
        wrt2 = np.where(np.isin(np.triu_indices(k)[0], wrt) & np.isin(np.triu_indices(k)[1], wrt))[0]

        BB_lim_partial2 = self.C[1 + k + wrt2]
        partial_A_i = self.A[1 + wrt][np.triu_indices(np.size(wrt))[0]]
        partial_A_j = self.A[1 + wrt][np.triu_indices(np.size(wrt))[1]]
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
        r_tilde = np.einsum('ijk, kl, lmi -> ijm', partial_A_i, self.Sig_tt_lim, partial_A_j.T) + np.einsum('jk, ikl, lmi -> ijm', self.A[0], s_star_i_tt, partial_A_j.T) + np.einsum('jk, ikl, lmi -> ijm', self.A[0], s_star_j_tt, partial_A_i.T) + np.einsum('jk, kl, ilm -> ijm', self.A[0], self.Sig_tt_lim, self.A[1 + k + wrt2])
        r_bar = np.einsum('ijk, kl -> ijl', s_hat, self.Sig_tp1_t_lim[:, self.first_observed:].T) - np.einsum('ijk, kl, lmi -> ijm', s_star_j[:, :, self.first_observed:], Sig_inv, s_star_i[:, :, self.first_observed:].T)
        right_side = sym(r_tilde) - np.einsum('jk, ikl, lm, mn -> ijn', self.K_lim, s_hat_o, self.Sig_tp1_t_lim[:, self.first_observed:].T, self.A[0].T) + np.einsum('jk, ikl, lm -> ijm', self.A[0], sym(r_bar), self.A[0].T) + BB_lim_partial2
        R_star_vectorized = (np.linalg.inv(np.eye(self.dim ** 2) - np.kron(self.F_lim, self.F_lim)) @ right_side.reshape(BB_lim_partial2.shape[0], self.dim**2, order='F').T).T
        return R_star_vectorized.reshape(BB_lim_partial2.shape[0], self.dim, self.dim, order='F')

    def build_kalman_filter(self, observations, t_max=None, verbose=0, close_pb=True):
        """
        Computes the Kalman filter sequences X(t+1, t) and X(t, t).
        :param observations: Array of shape (n, dim) containing the observations from the model
        :param t_max: Maximal number of t in the sequence
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
            Can also be an instance of class tqdm.
        :param close_pb: If verbose is an instance of class tqdm, this parameter specifies if the progressbar should
            be closed after computing the Kalman filter. Default to True.
        """
        if t_max is None:
            t_max = observations.shape[0]
        if self.Sig_tp1_t_list is None:
            raise Exception('Method build_covariance needs to be called before method kalman_filter.')
        observations = observations[:, self.first_observed:]
        X_hat_tp1_t_list = [self.E_0()]
        X_hat_tt_list = [self.E_0() + self.Cov_0()[:, self.first_observed:] @ np.linalg.pinv(self.Cov_0()[self.first_observed:, self.first_observed:]) @ (observations[0] - self.E_0()[self.first_observed:])]
        X_hat_tp1_t_list.append(self.a[0] + self.A[0] @ X_hat_tt_list[0])
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
            X_hat_tp1_t = self.a[0] + self.A[0] @ X_hat_tt
            X_hat_tp1_t_list.append(X_hat_tp1_t)
            X_hat_tt_list.append(X_hat_tt)
            if verbose:
                tr.update(1)
        for t in range(t_conv, t_max):
            X_hat_tt = X_hat_tp1_t + self.Sig_tp1_t_lim[:, self.first_observed:] @ Sig_inv[-1] @ (observations[t] - X_hat_tp1_t[self.first_observed:])
            X_hat_tp1_t = self.a[0] + self.A[0] @ X_hat_tt
            X_hat_tp1_t_list.append(X_hat_tp1_t)
            X_hat_tt_list.append(X_hat_tt)
            if verbose:
                tr.update(1)
        if verbose and close_pb:
            tr.close()
        self.X_hat_tp1_t_list = np.stack(X_hat_tp1_t_list)
        self.X_hat_tt_list = np.stack(X_hat_tt_list)

    def build_kalman_filter_hom(self, observations, t_max=None, verbose=0, close_pb=True):
        """
        Computes the Kalman filter sequences X^hom(t+1, t) and X^hom(t, t) with limiting covariance matrices in place of time-dependent ones.
        :param observations: Array of shape (n, dim) containing the observations from the model
        :param t_max: Maximal number of t in the sequence
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
            Can also be an instance of class tqdm.
        :param close_pb: If verbose is an instance of class tqdm, this parameter specifies if the progressbar should
            be closed after computing the Kalman filter. Default to True.
        """
        if t_max is None:
            t_max = observations.shape[0]
        if self.Sig_tp1_t_lim is None:
            raise Exception('Method build_covariance needs to be called before method kalman_filter.')
        observations = observations[:, self.first_observed:]
        X_hat_tp1_t_list_hom = [self.E_0()]
        X_hat_tt_list_hom = [self.E_0() + self.Sig_tp1_t_lim[:, self.first_observed:] @ np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:]) @ (observations[0] - self.E_0()[self.first_observed:])]
        X_hat_tp1_t_list_hom.append(self.a[0] + self.A[0] @ X_hat_tt_list_hom[0])
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
            X_hat_tp1_t = self.a[0] + self.A[0] @ X_hat_tt
            X_hat_tp1_t_list_hom.append(X_hat_tp1_t)
            X_hat_tt_list_hom.append(X_hat_tt)
            if verbose:
                tr.update(1)
        if verbose and close_pb:
            tr.close()
        self.X_hat_tp1_t_list_hom = np.stack(X_hat_tp1_t_list_hom)
        self.X_hat_tt_list_hom = np.stack(X_hat_tt_list_hom)

    def deriv_filter_hom(self, observations, wrt, t_max=None, verbose=0, close_pb=True):
        """
        Computes the Kalman filter derivative sequences V^hom(t+1, t) and V^hom(t, t) with limiting covariance matrices in place of time-dependent ones.
        :param observations: Array of shape (n, dim) containing the observations from the model
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters differentiated with respect to.
        :param t_max: Maximal number of t in the sequence
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
            Can also be an instance of class tqdm.
        :param close_pb: If verbose is an instance of class tqdm, this parameter specifies if the progressbar should
            be closed after computing the Kalman filter. Default to True.
        """
        wrt = np.atleast_1d(wrt)
        if t_max is None:
            t_max = observations.shape[0]
        s_star = self.S_star(wrt=wrt)
        partial_a = self.a[1 + wrt]
        partial_A = self.A[1 + wrt]
        observations = observations[:, self.first_observed:]
        Sig_inv = np.linalg.inv(self.Sig_tp1_t_lim[self.first_observed:, self.first_observed:])
        s_tilde = np.einsum('jk, ikl, lm -> ijm', Sig_inv, s_star[:, self.first_observed:, self.first_observed:], Sig_inv)
        k_tilde = self.Sig_tp1_t_lim[:, self.first_observed:] @ Sig_inv
        V_tp1_t_list = [self.E_0(deriv_order=1, wrt=wrt)]
        V_tt_list = [self.E_0(deriv_order=1, wrt=wrt) + (s_star[:, :, self.first_observed:] @ Sig_inv - np.einsum('jk, ikl -> ijl', self.Sig_tp1_t_lim[:, self.first_observed:], s_tilde)) @ (observations[0] - self.X_hat_tp1_t_list_hom[0, self.first_observed:]) - np.einsum('jk, ik -> ij', k_tilde, self.E_0(deriv_order=1, wrt=wrt)[:, self.first_observed:])]
        V_tp1_t_list.append(partial_a + partial_A @ self.X_hat_tt_list_hom[0] + np.einsum('jk, ik -> ij', self.A[0], V_tt_list[0]))
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
            V_tp1_t = partial_a + partial_A @ self.X_hat_tt_list_hom[t] + np.einsum('jk, ik -> ij', self.A[0], V_tt)
            V_tp1_t_list.append(V_tp1_t)
            V_tt_list.append(V_tt)
            if verbose:
                tr.update(1)
        if verbose and close_pb:
            tr.close()
        return np.stack(V_tp1_t_list), np.stack(V_tt_list)

    def deriv2_filter_hom(self, observations, wrt, t_max=None, verbose=0, close_pb=True, deriv_filters=None):
        """
        Computes the Kalman filter second derivative sequences W^hom(t+1, t) and W^hom(t, t) with limiting covariance matrices in place of time-dependent ones.
        :param observations: Array of shape (n, dim) containing the observations from the model
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters differentiated with respect to.
        :param t_max: Maximal number of t in the sequence
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
            Can also be an instance of class tqdm.
        :param close_pb: If verbose is an instance of class tqdm, this parameter specifies if the progressbar should
            be closed after computing the Kalman filter. Default to True.
        :param deriv_filters: If the derivatives sequences V^hom(t+1, t), V^hom(t, t) of the Kalman filter have been precomputed, they can be supplied as a tuple.
        """
        wrt = np.atleast_1d(wrt)
        k = int(np.sqrt(9 / 4 + 2 * (self.a.shape[0] - 1)) - 3 / 2)
        wrt2 = np.where(np.isin(np.triu_indices(k)[0], wrt) & np.isin(np.triu_indices(k)[1], wrt))[0]

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
        partial_a = self.a[1 + k + wrt2]
        partial_A = self.A[1 + k + wrt2]
        partial_A_i = self.A[1 + wrt][np.triu_indices(np.size(wrt))[0]]
        partial_A_j = self.A[1 + wrt][np.triu_indices(np.size(wrt))[1]]
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

        W_tp1_t_list = [self.E_0(deriv_order=2, wrt=wrt)]
        W_tt_list = [W_tp1_t_list[0] + M @ (observations[0] - self.X_hat_tp1_t_list_hom[0, self.first_observed:]) + np.einsum('ijk, ik -> ij', N_j, V_tp1_t_i[0, :, self.first_observed:]) + np.einsum('ijk, ik -> ij', N_i, V_tp1_t_j[0, :, self.first_observed:]) - np.einsum('jk, ik -> ij', k_tilde, W_tp1_t_list[0][:, self.first_observed:])]
        W_tp1_t_list.append(partial_a + partial_A @ self.X_hat_tt_list_hom[0] + np.einsum('ijk, ik -> ij', partial_A_j, V_tt_i[0]) + np.einsum('ijk, ik -> ij', partial_A_i, V_tt_j[0]) + np.einsum('jk, ik -> ij', self.A[0], W_tt_list[0]))
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
            W_tp1_t = partial_a + partial_A @ self.X_hat_tt_list_hom[t] + np.einsum('ijk, ik -> ij', partial_A_j, V_tt_i[t]) + np.einsum('ijk, ik -> ij', partial_A_i, V_tt_j[t]) + np.einsum('jk, ik -> ij', self.A[0], W_tt)
            W_tp1_t_list.append(W_tp1_t)
            W_tt_list.append(W_tt)
            if verbose:
                tr.update(1)
        if verbose and close_pb:
            tr.close()
        return np.stack(W_tp1_t_list), np.stack(W_tt_list)


class PolynomialModel:
    """
    This class instantiates a parametric polynomial state space model with respect to some fixed true parameter and contains methods
    to compute QML estimators and to calculate the limiting QML estimator covariance matrices. For concrete use, this class needs to
    be subclassed by an instance of a concrete model. This subclass has to contain a method `poly_A` for the polynomial semimartingale
    characteristics matrix A in the sense of Eberlein and Kallsen [3, Theorem 6.25]
    """
    def __init__(self, first_observed, init, dt, signature, params_names, params_bounds=None, true_param=None, wrt=None, scaling=1, warn=True):
        """
        :param first_observed: Integer between 0,..., dim-1 that denotes the first observed component of the state space model
        :param init: Instance of class InitialDistribution for the distribution of X(0)
        :param dt: Time increment of the discrete-time state space model
        :param signature: String that specifies which components are differenced components and which components occur which which powers.
            For example, the signature '1[1]_2d[1, 2]' specifies that the second component is differenced and occurs with powers 1 and 2,
            i.e. the model (X1(t), ΔX2(t), (ΔX2)(t)^2) is used. The signature '1[1, 2], 2[1, 2, 4], 3d[1]' specifies that the model
            (X1(t), X1(t)^2, X2(t), X2(t)^2, X2(t)^4, ΔX3(t)) is used.
        :param params_names: Array of length k containing strings for the parameter names
        :param params_bounds: Array of shape (k, 2) containing lower and upper bounds for each parameter in each row.
        :param true_param: Array of length k containing the true parameter. Can be None if the true parameter is unknown.
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters that are to be estimated.
        :param scaling: Integer or Array of length dim_c containing scaling factors for the components of the model. A scaling factor different from 1
            can be of advantage for numerical purposes. If a PolynomialModel is instantiated with scaling=1, an optimal scaling factor is printed
            to the console.
        :param warn: Boolean specifying whether or not to suppress warnings.
        """
        self.first_observed = first_observed

        if true_param is None:
            self.true_param = np.repeat(np.nan, np.size(params_names))
        else:
            self.true_param = true_param

        self.init = init
        self.dt = dt
        self.params_names = params_names
        self.params_bounds = params_bounds
        self.observations = None
        self.seed = None
        self.wrt = wrt

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

        if 'd' in signature and warn:
            warnings.warn('The current implementation of differenced components assumes that no other components depend on these.')
        self.signature_string = signature
        signature = signature.split('_')
        self.dim_c = len(signature)
        self.differenced_components = np.array([i for i, s in enumerate(signature) if 'd' in s])
        self.undiff_components = np.setdiff1d(np.arange(self.dim_c), self.differenced_components)
        for i, itm in enumerate(signature):
            if ':' in itm:
                signature[i] = np.arange(int(itm.split(':')[0][-1]), int(itm.split(':')[1][0]) + 1).tolist()
            else:
                signature[i] = eval(signature[i][signature[i].index('['):])
        self.signature = signature
        self.dim = len(sum(signature, []))
        counter = 0
        self.signature_indices = []

        for sublist in self.signature:
            new_sublist = []
            for _ in sublist:
                new_sublist.append(counter)
                counter += 1
            self.signature_indices.append(new_sublist)

        if np.ndim(scaling) == 0:
            self.scaling = np.repeat(scaling, self.dim_c).astype('float')
        elif np.size(scaling) != self.dim_c:
            raise Exception('Dimension of scaling argument is not compatible with dimensionality of underlying process.')
        else:
            self.scaling = np.array(scaling).astype('float')


        init_path = './saves/' + self.__class__.__name__
        paths = [init_path + string for string in ['/Covariance/Estimated', '/Covariance/Explicit', '/Observations', '/Polynomial Matrices', '/QML Sequences']]

        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

        if not np.isnan(self.true_param).any():
            self.savestring = 'par=[' + ', '.join('{:.3f}'.format(item) for item in true_param) + ']_dt={:.1e}_sig{}_sc['.format(self.dt, self.signature_string) + ', '.join('{:.1f}'.format(sc) for sc in self.scaling) + ']'
            if wrt is not None:
                self.setup_filter(wrt)
        elif wrt is not None and warn:
            warnings.warn('Argument wrt was not used since the whole parameter has not yet been estimated. Please use method "setup_filter" after this has been done.', Warning)

    def state_space_params(self, param, deriv_order=0, wrt=None, return_stack=False):
        """
        Computes the state transition vector a, state transition matrix A and limiting noise covariance matrix C for the polynomial state space model
        from the polynomial semimartingale characteristics matrix specified by the method 'poly_A'
        :param param: Array of length k. Parameter at which to compute a, A and C.
        :param deriv_order: Integer between 0 and 2. Specifies whether no derivatives, 0th and 1st derivatives or 0th, 1st and 2nd derivatives are to be computed.
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters differentiated with respect to.
        :param return_stack: Boolean. If True, the outputs for a, A and C have all derivatives stacked on top of each other in the first dimension. If False and
            deriv_order>=1, the outputs for a, A and C are lists of length 2 or 3 that contain the 0th, 1st (and 2nd) derivatives, respectively.
        :return: a, A and C with shape according to the parameter 'return_stack'.
        """
        wrt = np.atleast_1d(wrt)
        k = np.size(wrt)
        n_components = 1 + k * (deriv_order >= 1) + int(k * (k + 1) / 2) * (deriv_order == 2)

        poly_B = self.poly_B(param, poly_order=2, deriv_order=deriv_order, wrt=wrt, return_stack=True)

        a = poly_B[:, 1:(1 + self.dim), 0]
        A = poly_B[:, 1:(1 + self.dim), 1:(1 + self.dim)]

        n = poly_B.shape[-1]
        M = poly_B[:, 1:, 1:]
        m = poly_B[:, 1:, 0]

        lim_expec2 = np.zeros((n_components, poly_B.shape[-1]))
        lim_expec2[0] = np.append(1, np.linalg.inv(np.eye(n - 1) - M[0]) @ m[0])

        if deriv_order >= 1:
            inv = np.linalg.inv(np.eye(n - 1) - M[0])
            lim_expec2[1:(k + 1), 1:] = inv @ M[1:(1 + k)] @ inv @ m[0] + np.einsum('ij, kj -> ki', inv, m[1:(1 + k)])

        if deriv_order == 2:
            i_ind, j_ind = np.triu_indices(k)
            mat1, mat2 = inv @ M[1 + i_ind] @ inv, inv @ M[1 + j_ind] @ inv
            vec1, vec2 = m[1 + j_ind] + M[1 + j_ind] @ inv @ m[0], m[1 + i_ind] + M[1 + i_ind] @ inv @ m[0]
            lim_expec2[(k + 1):, 1:] = np.einsum('ijk, ik -> ij', mat1, vec1) + np.einsum('ijk, ik -> ij', mat2, vec2) + np.einsum('ij, kj -> ki', inv, m[(1 + k):] + M[(1 + k):] @ inv @ m[0])

        Sigma_lim = np.zeros((n_components, self.dim, self.dim))
        Sigma_lim[:, np.triu_indices(self.dim)[0], np.triu_indices(self.dim)[1]] = lim_expec2[:, (1 + self.dim):]
        Sigma_lim[:, np.tril_indices(self.dim)[0], np.tril_indices(self.dim)[1]] = np.transpose(Sigma_lim, (0, 2, 1))[:, np.tril_indices(self.dim)[0], np.tril_indices(self.dim)[1]]
        mu_lim = lim_expec2[:, 1:(1 + self.dim)]

        C = np.zeros((n_components, self.dim, self.dim))
        C[0] = Sigma_lim[0] - a[:1].T @ a[:1] - sym((A[:1] @ mu_lim[0]).T @ a[:1]) - A[0] @ Sigma_lim[0] @ A[0].T

        if deriv_order >= 1:
            C[1:(k + 1)] = Sigma_lim[1:(k + 1)] - np.einsum('jk, ikl, ml -> ijm', A[0], Sigma_lim[1:(k + 1)], A[0]) - sym(np.einsum('ij, l -> ijl', a[1:(k + 1)], a[0]) + np.einsum('ijk, k, l -> ijl ', A[1:(k + 1)], mu_lim[0], a[0]) + np.einsum('jk, ik, l -> ijl', A[0], mu_lim[1:(k + 1)], a[0]) +  np.einsum('jk, k, il -> ijl', A[0], mu_lim[0], a[1:(k + 1)]) + A[1:(k + 1)] @ Sigma_lim[0] @ A[0].T)

        if deriv_order == 2:
            Mij = np.einsum('ij, ik -> ijk', a[1 + i_ind], a[1 + j_ind]) + np.einsum('ijk, kl, iml -> ijm', A[1 + i_ind], Sigma_lim[0], A[1 + j_ind]) + sym(np.einsum('ijk, ik, l -> ijl ', A[1 + i_ind], mu_lim[1 + j_ind], a[0]) + np.einsum('ijk, k, il -> ijl', A[1 + i_ind], mu_lim[0], a[1 + j_ind]) +  np.einsum('jk, ik, il -> ijl', A[0], mu_lim[1 + i_ind], a[1 + j_ind]) + np.einsum('jk, ikl, iml -> ijm', A[0], Sigma_lim[1 + i_ind], A[1 + j_ind]))
            Mji = np.einsum('ij, ik -> ijk', a[1 + j_ind], a[1 + i_ind]) + np.einsum('ijk, kl, iml -> ijm', A[1 + j_ind], Sigma_lim[0], A[1 + i_ind]) + sym(np.einsum('ijk, ik, l -> ijl ', A[1 + j_ind], mu_lim[1 + i_ind], a[0]) + np.einsum('ijk, k, il -> ijl', A[1 + j_ind], mu_lim[0], a[1 + i_ind]) +  np.einsum('jk, ik, il -> ijl', A[0], mu_lim[1 + j_ind], a[1 + i_ind]) + np.einsum('jk, ikl, iml -> ijm', A[0], Sigma_lim[1 + j_ind], A[1 + i_ind]))
            C[(k + 1):] = Sigma_lim[(k + 1):] - np.einsum('jk, ikl, ml -> ijm', A[0], Sigma_lim[(k + 1):], A[0]) - Mij - Mji - sym(np.einsum('ij, l -> ijl', a[(k + 1):], a[0]) + np.einsum('ijk, k, l -> ijl ', A[(k + 1):], mu_lim[0], a[0]) + np.einsum('jk, ik, l -> ijl', A[0], mu_lim[(k + 1):], a[0]) +  np.einsum('jk, k, il -> ijl', A[0], mu_lim[0], a[(k + 1):]) + A[(k + 1):] @ Sigma_lim[0] @ A[0].T)

        if return_stack:
            return a, A, C

        if (deriv_order == 0):
            return a.squeeze(), A.squeeze(), C.squeeze()
        elif deriv_order == 1:
            return [a[0], a[1:]], [A[0], A[1:]], [C[0], C[1:]]
        else:
            return [a[0], a[1:(k + 1)], a[(k + 1):]], [A[0], A[1:(k + 1)], A[(k + 1):]], [C[0], C[1:(k + 1)], C[(k + 1):]]

    def poly_A(self, param, poly_order, deriv_order=0, wrt=None):
        """
        Computes the polynomial semimartingale characteristics matrix A in the sense of Eberlein and Kallsen [3, Theorem 6.25] for the underlying
        time-continuous model. Needs to be overwritten by a subclass for a specific model.
        :param param: Array of length k. Parameter at which to compute the matrix.
        :param poly_order: Polynomial order up to which to evaluate the matrix
        :param deriv_order: Integer between 0 and 2. Specifies whether no derivatives, 0th and 1st derivatives or 0th, 1st and 2nd derivatives are to be computed.
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters differentiated with respect to.
        :return: Array of shape (..., n, n). The rows of the array contain all derivatives stacked on top of each other. The other two dimensions of shape n contain
        the elements of the polynomial matrix (or its derivatives).
        """
        pass

    def poly_B(self, param, poly_order, deriv_order=0, wrt=None, return_stack=False):
        """
        Computes the polynomial moment matrix of the discrete-time state space model in the sense of Kallsen and Richert [4, Lemma 2.10]. The signature of the model
        specified by the init argument 'signature' is taken into account, i.e. the computed moment matrix already corresponds to the model with differenced components
        and with multiple powers of components.
        :param param: Array of length k. Parameter at which to compute the matrix.
        :param poly_order: Polynomial order up to which to evaluate the matrix.
        :param deriv_order: Integer between 0 and 2. Specifies whether no derivatives, 0th and 1st derivatives or 0th, 1st and 2nd derivatives are to be computed.
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters differentiated with respect to.
        :param return_stack: Boolean. If True, the output for the matrix has all derivatives stacked on top of each other in the first dimension. If False and
            deriv_order>=1, the output for the matrix is a list of length 2 or 3 that contains the 0th, 1st (and 2nd) derivatives, respectively.
        :return: The polynomial moment matrix with shape according to the parameter 'return_stack'.
        """
        wrt = np.atleast_1d(wrt)
        k = np.size(wrt)
        poly_A = self.poly_A(param, max(sum(self.signature, [])) * poly_order, deriv_order, wrt)

        dicts = return_dict(self.dim_c, max(sum(self.signature, [])) * poly_order)
        poly_B_gen = np.zeros_like(poly_A)

        if is_run():
            def get_poly_B_entry(i):
                    results = []
                    for j in range(poly_B_gen.shape[-1]):
                        masks = mask(dicts[i], dicts, typ='leq') & mask(dicts[j], dicts, typ='leq')
                        lamb_ell = (ind_to_mult(i, dicts) - dicts)[masks]
                        mu_ell = (ind_to_mult(j, dicts) - dicts)[masks]
                        results.append((multi_binom(dicts[i], dicts[masks]) * poly_A[..., mult_to_ind(lamb_ell, dicts), mult_to_ind(mu_ell, dicts)]).sum(axis=-1))
                    return np.repeat(i, poly_B_gen.shape[-1]), np.arange(poly_B_gen.shape[-1]), np.array(results)

            result = Parallel(n_jobs=cpu_count())(delayed(get_poly_B_entry)(i) for i in range(poly_B_gen.shape[-2]))
            result = list(zip(*result))
            i_s, j_s = np.hstack(result[0]), np.hstack(result[1])
            poly_B_gen[..., i_s, j_s] = np.hstack(tuple(map(np.transpose, result[2])))
        else:
            for i in range(poly_B_gen.shape[-2]):
                for j in range(poly_B_gen.shape[-1]):
                    masks = mask(dicts[i], dicts, typ='leq') & mask(dicts[j], dicts, typ='leq')
                    lamb_ell = (ind_to_mult(i, dicts) - dicts)[masks]
                    mu_ell = (ind_to_mult(j, dicts) - dicts)[masks]
                    poly_B_gen[..., i, j] = (multi_binom(dicts[i], dicts[masks]) * poly_A[..., mult_to_ind(lamb_ell, dicts), mult_to_ind(mu_ell, dicts)]).sum(axis=-1)

        triu_indices = np.less.outer(dicts.sum(axis=-1), dicts.sum(axis=-1))
        scaling_factor = mult_pow(self.scaling, (dicts[:, None, :] - dicts[None, :, :]))
        scaling_factor[triu_indices] = 1
        poly_B_gen *= scaling_factor

        if deriv_order == 0:
            poly_B = expm(poly_B_gen * self.dt)
            poly_B_sig = np.zeros((n_dim(self.dim, poly_order), n_dim(self.dim, poly_order)))
        if deriv_order >= 1:
            poly_B = np.zeros_like(poly_B_gen)
            for i in range(k):
                expon, deriv = expm_frechet(poly_B_gen[0] * self.dt, poly_B_gen[1 + i] * self.dt)
                poly_B[1 + i] = deriv
            poly_B[0] = expon
            poly_B_sig = np.zeros((poly_B.shape[0], n_dim(self.dim, poly_order), n_dim(self.dim, poly_order)))
        if deriv_order == 2:
            first_frechet = np.array([expm_frechet(poly_B_gen[0] * self.dt, poly_B_gen[1 + k + i] * self.dt, compute_expm=False) for i in range(int(k * (k + 1) / 2))])

            E1_indices, E2_indices = np.triu_indices(k)
            E1 = poly_B_gen[1 + E1_indices] * self.dt
            E2 = poly_B_gen[1 + E2_indices] * self.dt
            X1 = np.kron(np.eye(2), poly_B_gen[0] * self.dt) + np.kron(np.array([[0, 1], [0, 0]]), E1)
            X2 = np.kron(np.eye(2), X1) + np.kron(np.array([[0, 1], [0, 0]]), np.kron(np.eye(2), E2))
            second_frechet = expm(X2)[:, :poly_B.shape[-2], -poly_B.shape[-1]:]
            poly_B[(k + 1):] = first_frechet + second_frechet

        dicts_sig = return_dict(self.dim, poly_order)
        cols = np.sort(np.unique(np.vstack([np.sum(self.signature[j] * dicts_sig[:, self.signature_indices[j]], axis=-1) for j in self.undiff_components]), axis=-1, return_index=True)[1])
        # cols = reduce(np.union1d, [np.unique(np.sum(self.signature[j] * dicts_sig[:, self.signature_indices[j]], axis=-1), return_index=True)[1] for j in self.undiff_components])

        diff_cols_zero_orig = [(dicts[:, i] == 0) for i in self.differenced_components]
        diff_cols_zero_orig.append(np.repeat(True, dicts.shape[0]))
        diff_cols_zero_orig = np.all(diff_cols_zero_orig, axis=0)
        powers_undiff = np.vstack([np.sum(dicts_sig[np.ix_(cols, self.signature_indices[i])] * np.array(self.signature[i]), axis=1) for i in self.undiff_components])
        undiff_cols_duplicates_orig = np.array([np.where(np.all([(dicts[diff_cols_zero_orig, k] == i) for k, i in zip(self.undiff_components, powers_undiff[:, j])], axis=0))[0][0] for j in range(powers_undiff.shape[1])])

        for i in range(n_dim(self.dim, poly_order)):
            lamb = ind_to_mult(i, dicts_sig)
            lamb_tilde = np.array([np.sum(lamb[ind] * sig) for ind, sig in zip(self.signature_indices, self.signature)])
            ind = mult_to_ind(lamb_tilde, dicts)
            poly_B_sig[..., i, cols] = poly_B[..., ind, diff_cols_zero_orig][..., undiff_cols_duplicates_orig]

        if return_stack:
            if deriv_order == 0:
                return poly_B_sig[None, ...]
            else:
                return poly_B_sig

        if (deriv_order == 0):
            return poly_B_sig
        elif deriv_order == 1:
            return [poly_B_sig[0], poly_B_sig[1:]]
        else:
            return [poly_B_sig[0], poly_B_sig[1:(k + 1)], poly_B_sig[k + 1:]]

    def calc_filter_B(self, poly_order):
        """
        Computes the polynomial moment matrix of the filtered state space model overline{X}(t) or the filtered state space model underline{X}(t) from Section 3.3 of
        Kallsen and Richert [5] using the formulas from Proposition 2.14 of Kallsen and Richert [4] and stores them in the directory saves/[ModelName]/Polynomial Matrices
        :param poly_order: Integer 2 or 4. If poly_order=2, the polynomial order 2 moment matrix for underline{X}(t) is computed. If poly_order=4, the polynomial
            order 4 moment matrix for overline{X}(t) is computed.
        :return: Polynomial moment matrix of shape (n, n)
        """
        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        if (self.filter_c4 is None) | (self.filter_C4 is None) | (self.filter_Y4 is None):
            raise Exception('Method setup_filter has to be called first.')

        is_run_bool = is_run()

        if poly_order == 4:
            C = self.filter_C4
            c = self.filter_c4
            Y = self.filter_Y4
            filepath = './saves/' + self.__class__.__name__ + '/Polynomial Matrices/B4{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        elif poly_order == 2:
            C = self.filter_C2
            c = self.filter_c2
            Y = self.filter_Y2
            filepath = './saves/' + self.__class__.__name__ + '/Polynomial Matrices/B2{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        else:
            raise Exception('Argument order has to be 4 or 2.')

        k, d = C.shape
        B = self.poly_B(param=self.true_param, poly_order=poly_order)

        if np.any(c):
            C = np.vstack((C, np.repeat(0, d)))
            Y = np.vstack((Y, np.repeat(0, k)))
            c = np.append(c, 1)[:, None]
            Y = np.hstack((Y, c))
            k += 1
            if is_run_bool:
                total = (np.ceil(n_dim(Y.shape[0], poly_order) / 30) + np.ceil(n_dim(C.shape[0], poly_order) / 30)).astype('int') + n_dim(d + k, poly_order) + n_dim(d + k - 1, poly_order)
            else:
                total = n_dim(Y.shape[0], poly_order) + n_dim(C.shape[0], poly_order) + n_dim(d + k, poly_order) + n_dim(d + k - 1, poly_order)
        else:
            if is_run_bool:
                total = (np.ceil(n_dim(Y.shape[0], poly_order) / 30) + np.ceil(n_dim(C.shape[0], poly_order) / 30)).astype('int') + n_dim(d + k, poly_order)
            else:
                total = n_dim(Y.shape[0], poly_order) + n_dim(C.shape[0], poly_order) + n_dim(d + k, poly_order) + n_dim(d + k - 1, poly_order)

        dictd = return_dict(d, poly_order)
        dictk = return_dict(k, poly_order)

        B_large = np.zeros((n_dim(d + k, poly_order), n_dim(d + k, poly_order)))
        dict_large = return_dict(d + k, poly_order)

        def S_func(trans, order, pbar=None):
            dim1, dim2 = trans.shape
            dict1, dict2 = return_dict(dim1, order), return_dict(dim2, order)
            raw_collections1 = np.array([np.sort(dict1[i][dict1[i] != 0]).tolist() for i in range(n_dim(dim1, order))], dtype=object)
            raw_collections2 = np.array([np.sort(dict2[i][dict2[i] != 0]).tolist() for i in range(n_dim(dim2, order))], dtype=object)
            locations1 = [np.where(np.isin(dict1[i], raw_collections1[i]))[0][np.argsort(dict1[i][dict1[i] != 0])] for i in range(n_dim(dim1, order))]
            locations2 = [np.where(np.isin(dict2[i], raw_collections2[i]))[0][np.argsort(dict2[i][dict2[i] != 0])] for i in range(n_dim(dim2, order))]
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

            if is_run_bool:
                i_packages = np.split(np.arange(n_dim(dim1, order)), np.arange(30, n_dim(dim1, order), 30))

                def get_S_entry_from_sols(i_s):
                    results = []
                    for i in i_s:
                        index_in_raw_comb = np.where((raw_combinations == np.expand_dims(np.array([collections1[coll_locator1[I[:, i]]], collections2[coll_locator2[J[:, i]]]]).T, -2)).all(-1))[-1]
                        sol_locator = raw_comb_locator[index_in_raw_comb]
                        sol = solutions[sol_locator]
                        indices = np.where(sol_locator != -1)[0]
                        for j in indices:
                            large_solution = np.zeros((sol[j].shape[0], dim1, dim2))
                            large_solution[np.ix_(np.arange(sol[j].shape[0]), locations1[i], locations2[j])] = sol[j]
                            results.append([i, j, np.prod(multinom(large_solution) * mult_pow(trans, large_solution), axis=1).sum()])
                    return results

                result = Parallel(n_jobs=cpu_count())(delayed(get_S_entry_from_sols)(i_s) for i_s in i_packages)
                result = np.array(sum(result, []))
                S[result[:, 0].astype('int'), result[:, 1].astype('int')] = result[:, 2]
            else:
                for i in range(n_dim(dim1, order)):
                    pbar.update(1)
                    index_in_raw_comb = np.where((raw_combinations == np.expand_dims(np.array([collections1[coll_locator1[I[:, i]]], collections2[coll_locator2[J[:, i]]]]).T, -2)).all(-1))[-1]
                    sol_locator = raw_comb_locator[index_in_raw_comb]
                    sol = solutions[sol_locator]
                    for j in np.where(sol_locator != -1)[0]:
                        large_solution = np.zeros((sol[j].shape[0], dim1, dim2))
                        large_solution[np.ix_(np.arange(sol[j].shape[0]), locations1[i], locations2[j])] = sol[j]
                        S[i, j] = np.prod(multinom(large_solution) * mult_pow(trans, large_solution), axis=1).sum()

            S[0, 0] = 1
            return S

        with tqdm_joblib(desc='Order-{} Matrix: Calculating S'.format(poly_order), total=total) as progress_bar:
            if is_run_bool:
                S_mat = S_func(Y, poly_order)
                S_mat_d = S_func(C, poly_order)
            else:
                S_mat = S_func(Y, poly_order, progress_bar)
                S_mat_d = S_func(C, poly_order, progress_bar)

            progress_bar.set_description('Order-{} Matrix: Calculating B'.format(poly_order))

            if is_run_bool:
                def get_mu(i):
                    large_ind = dict_large[i]
                    mu_inds, mu_locs = dict_large[mask(large_ind, dict_large, typ='leq_abs')], np.where(mask(large_ind, dict_large, typ='leq_abs'))[0]
                    return mu_inds, mu_locs

                def get_B_entry_from_mus(i, mu_inds, mu_locs):
                    results = []
                    large_ind = dict_large[i]
                    lamb, lamb_tilde = np.split(large_ind, [d])
                    lamb_ind = mult_to_ind(lamb, dictd)
                    for mu_ind, mu_loc in zip(mu_inds, mu_locs):
                        mu, mu_tild = np.split(mu_ind, [d])  # np.split(mu_inds, [d], axis=-1)
                        mu_tild_ind = mult_to_ind(mu_tild, dictk)
                        nus, nus_ind = dictd[mask(mu, dictd, typ='leq') & mask(lamb_tilde - mu_tild, dictd, typ='eq_abs')], np.where(mask(mu, dictd, typ='leq') & mask(lamb_tilde - mu_tild, dictd, typ='eq_abs'))[0]
                        etas, etas_ind = dictk[mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs')], np.where(mask(lamb_tilde, dictk, typ='leq') & mask(lamb_tilde - mu_tild, dictk, typ='eq_abs'))[0]
                        lamb_eta_ind = mult_to_ind(lamb_tilde - etas, dictk)
                        mu_nu_ind = mult_to_ind(mu - nus, dictd)
                        prods, prods_int, prods_int2 = product(nus, etas), product(nus_ind, etas_ind), product(mu_nu_ind, lamb_eta_ind)
                        results.append(np.sum([multi_binom(lamb_tilde, prod[1]) * S_mat[prod_int2[1], mu_tild_ind] * S_mat_d[prod_int[1], prod_int[0]] * B[lamb_ind, prod_int2[0]] for prod, prod_int, prod_int2 in zip(prods, prods_int, prods_int2)]))
                    return np.vstack((np.repeat(i, len(mu_locs)), mu_locs, np.array(results)))

                result = Parallel(n_jobs=cpu_count())(delayed(get_B_entry_from_mus)(i, *get_mu(i)) for i in range(dict_large.shape[0]))
                result = np.hstack(result).T
                B_large[result[:, 0].astype('int'), result[:, 1].astype('int')] = result[:, 2]
            else:
                for i in range(dict_large.shape[0]):
                    progress_bar.update(1)
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
                B_final = np.zeros((n_dim(d + k - 1, poly_order), n_dim(d + k - 1, poly_order)))
                dict_final = return_dict(d + k - 1, poly_order)
                progress_bar.set_description('Order-{} Matrix: Incorporating c'.format(poly_order))

                if is_run_bool:
                    def get_mu_final(i):
                        lamb_ind = dict_final[i]
                        mu_inds, mu_locs = dict_final[mask(lamb_ind, dict_final, typ='leq_abs')], np.where(mask(lamb_ind, dict_final, typ='leq_abs'))[0]
                        return mu_inds, mu_locs

                    def get_B_final_from_mus(i, mu_inds, mu_locs):
                        results = []
                        lamb_ind = dict_final[i]
                        for mu_ind, mu_loc in zip(mu_inds, mu_locs):
                            results.append(B_large[mult_to_ind(np.append(lamb_ind, 0), dict_large), np.where((dict_large[:, :-1] == np.atleast_2d(mu_ind)[:, None]).all(-1))[1]].sum())
                        return np.vstack((np.repeat(i, len(mu_locs)), mu_locs, np.array(results)))

                    result = Parallel(n_jobs=cpu_count())(delayed(get_B_final_from_mus)(i, *get_mu_final(i)) for i in range(n_dim(d + k - 1, poly_order)))
                    result = np.hstack(result).T
                    B_final[result[:, 0].astype('int'), result[:, 1].astype('int')] = result[:, 2]
                else:
                    B_final = np.zeros((n_dim(d + k - 1, poly_order), n_dim(d + k - 1, poly_order)))
                    dict_final = return_dict(d + k - 1, poly_order)
                    for i in range(n_dim(d + k - 1, poly_order)):
                        progress_bar.update(1)
                        lamb_ind = dict_final[i]
                        mu_inds, mu_locs = dict_final[mask(lamb_ind, dict_final, typ='leq_abs')], np.where(mask(lamb_ind, dict_final, typ='leq_abs'))[0]
                        for mu_ind, mu_loc in zip(mu_inds, mu_locs):
                            B_final[i, mu_loc] = B_large[mult_to_ind(np.append(lamb_ind, 0), dict_large), np.where((dict_large[:, :-1] == np.atleast_2d(mu_ind)[:, None]).all(-1))[1]].sum()
            else:
                B_final = B_large

            progress_bar.set_description('Order-{} Matrix: Calculating limiting power expectations'.format(poly_order))
            if poly_order == 4:
                self.filter_B4 = B_final
                self.lim_expec4 = np.append(1, np.linalg.inv(np.eye(B_final.shape[0] - 1) - B_final[1:, 1:]) @ B_final[1:, 0])
                np.savetxt(filepath, self.filter_B4)
            elif poly_order == 2:
                self.filter_B2 = B_final
                self.lim_expec2 = np.append(1, np.linalg.inv(np.eye(B_final.shape[0] - 1) - B_final[1:, 1:]) @ B_final[1:, 0])
                np.savetxt(filepath, self.filter_B2)

    def setup_filter(self, wrt):
        """
        Sets up the matrices Y(t) and C(t) as well as the vector c(t) in the sense of Kallsen and Richert [4, equation (2.9)] for the filtered state space models
        overline{X}(t) and underline{X}(t). If no 'true_param' has been specified when initializing the PolynomialModel class, the setup_filter method has to be called
        before computing asymptotic QML estimator covariance matrices via 'compute_V'.
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters differentiated with respect to.
        """
        if np.isnan(self.true_param).any():
            raise Exception('Full underlying parameter has to be given or has to be estimated first.')

        self.wrt = np.atleast_1d(wrt)

        filepath_U = './saves/' + self.__class__.__name__ + '/Covariance/Explicit/U{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        filepath_W = './saves/' + self.__class__.__name__ + '/Covariance/Explicit/W{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        self.U = np.atleast_2d(np.loadtxt(filepath_U)) if os.path.exists(filepath_U) else None
        self.W = np.atleast_2d(np.loadtxt(filepath_W)) if os.path.exists(filepath_W) else None

        self.dicts = return_dict(self.dim * (np.size(wrt) + 2), order=4)
        self.dicts2 = return_dict(self.dim * (int(np.size(wrt) * (np.size(wrt) + 1) / 2) + np.size(wrt) + 2), order=2)

        a, A, C = self.state_space_params(self.true_param, deriv_order=2, wrt=np.arange(np.size(self.true_param)), return_stack=True)
        partial_E_0 = partial(self.init.E_0, param=self.true_param)
        partial_Cov_0 = partial(self.init.Cov_0, param=self.true_param)
        self.kalman_filter = KalmanFilter(dim=self.dim, a=a, A=A, C=C, E_0=partial_E_0, Cov_0=partial_Cov_0, first_observed=self.first_observed)
        self.kalman_filter.build_covariance()

        if np.all(self.scaling == 1):
            def objective(Ms, S):
                M_diag = np.hstack([Ms[i] ** sig for i, sig in enumerate(self.signature)])
                M = np.diag(M_diag)
                return np.linalg.cond(M @ S @ M.T)

            Sig = self.kalman_filter.Sig_tp1_t_lim
            result = minimize(objective, np.array([1, 1]), args=(Sig,), bounds=[(0, None) for i in range(self.dim_c)], method='Nelder-Mead')
            print('Covariance condition optimal scaling: [' + ', '.join('{:.1f}'.format(sc) for sc in result.x) + '] \nCondition number reduction: {:.3e} -> {:.3e}'.format(objective(self.scaling, Sig), objective(result.x, Sig)))

        k = np.size(self.wrt)
        wrt2 = np.where(np.isin(np.triu_indices(np.size(self.true_param))[0], self.wrt) & np.isin(np.triu_indices(np.size(self.true_param))[1], self.wrt))[0]

        a_0 = a[0]
        a_1 = a[1 + self.wrt]
        a_2 = a[1 + np.size(self.true_param) + wrt2]
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

        A_1 = (A[1 + self.wrt] @ K_tilde + A[0] @ SS) @ self.kalman_filter.H
        A_10 = A[1 + self.wrt] @ (np.eye(self.dim) - K_tilde @ self.kalman_filter.H) - A[0] @ SS @ self.kalman_filter.H
        A_11 = self.kalman_filter.F_lim

        prod_i = np.einsum('ijk, lkm -> iljm', A[1 + self.wrt], SS) @ self.kalman_filter.H
        prod_ij = (prod_i + np.transpose(prod_i, (1, 0, 2, 3)))[np.triu_indices(k)]
        M = R[:, :, self.first_observed:] @ Sig_inv - S_hat + np.einsum('jk, ikl -> ijl', K_tilde, S_hat_o) - np.einsum('jk, ikl, lm -> ijm', K_tilde, R[:, self.first_observed:, self.first_observed:], Sig_inv)
        N_i = np.einsum('jk, ikl -> ijl', Sig[:, self.first_observed:], S_i_tilde) - S_i[:, :, self.first_observed:] @ Sig_inv
        N_j = np.einsum('jk, ikl -> ijl', Sig[:, self.first_observed:], S_j_tilde) - S_j[:, :, self.first_observed:] @ Sig_inv

        A_2 = A[1 + np.size(self.true_param) + wrt2] @ K_tilde @ self.kalman_filter.H + prod_ij + A[0] @ M @ self.kalman_filter.H
        A_20 = A[1 + np.size(self.true_param) + wrt2] @ (np.eye(self.dim) - K_tilde @ self.kalman_filter.H) - prod_ij - A[0] @ M @ self.kalman_filter.H
        A_21_i = A[1 + self.wrt][np.triu_indices(k)[0]] @ (np.eye(self.dim) - K_tilde @ self.kalman_filter.H) + A[0] @ N_i @ self.kalman_filter.H
        A_21_j = A[1 + self.wrt][np.triu_indices(k)[1]] @ (np.eye(self.dim) - K_tilde @ self.kalman_filter.H) + A[0] @ N_j @ self.kalman_filter.H
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
            if is_run():
                print('Multiprocessing is used for polynomial matrix calculations')
                already_printed = True
            self.calc_filter_B(poly_order=4)

        filepath_B2 = './saves/' + self.__class__.__name__ + '/Polynomial Matrices/B2{}_m={}_{}.txt'.format(np.atleast_1d(self.wrt).tolist(), self.first_observed, self.savestring)
        if os.path.exists(filepath_B2):
            self.filter_B2 = np.loadtxt(filepath_B2)
            self.lim_expec2 = np.append(1, np.linalg.inv(np.eye(self.filter_B2.shape[0] - 1) - self.filter_B2[1:, 1:]) @ self.filter_B2[1:, 0])
        else:
            if is_run() and not already_printed:
                print('Multiprocessing is used for polynomial matrix calculations')
            self.calc_filter_B(poly_order=2)

    def generate_observations(self, t_max, inter_steps, seed, verbose):
        """
        Generates a discretised trajectory with stepsize self.dt of the underlying time-continuous model and stores it in the directory
        saves/[ModelName]/Observations. The variable self.true_param must not be None to use this method. If there are already observations,
        new observations are appended to the end of the old ones. Needs to be overwritten by a subclass for a specific model.
        :param t_max: Maximal number of t for the simulated X(t)
        :param inter_steps: Number of steps to compute internally between steps of size self.dt to reduce the simulation bias
        :param seed: Random seed to use
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
        """
        pass

    @classmethod
    def from_observations(cls, first_observed, init, dt, signature, obs, inter_steps=None, true_param=None, wrt=None, scaling=1, seed=None, warn=True):
        """
        Initializes an instance of the class PolynomialModel by either simulating observations via the generate_observations method or via importing
        observations data from a csv or txt file.
        :param first_observed: Integer between 0,..., dim-1 that denotes the first observed component of the state space model
        :param init: Instance of class InitialDistribution for the distribution of X(0)
        :param dt: Time increment of the discrete-time state space model
        :param signature: String that specifies which components are differenced components and which components occur which which powers.
            For example, the signature '1[1]_2d[1, 2]' specifies that the second component is differenced and occurs with powers 1 and 2,
            i.e. the model (X1(t), ΔX2(t), (ΔX2)(t)^2) is used. The signature '1[1, 2], 2[1, 2, 4], 3d[1]' specifies that the model
            (X1(t), X1(t)^2, X2(t), X2(t)^2, X2(t)^4, ΔX3(t)) is used.
        :param obs: Number of observations
        :param inter_steps: Number of steps to compute internally between steps of size self.dt to reduce the simulation bias. Only required if
            observations are simulated.
        :param true_param: Array of length k containing the true parameter. Only required if observations are simulated.
        :param wrt: Integer or list of integers between 0 and k-1 that specifies the parameters that are to be estimated.
        :param scaling: Integer or Array of length dim_c containing scaling factors for the components of the model. A scaling factor different from 1
            can be of advantage for numerical purposes. If a PolynomialModel is instantiated with scaling=1, an optimal scaling factor is printed
            to the console.
        :param seed: Random seed to use
        :param warn: Boolean specifying whether or not to suppress warnings.
        :return: Instance of PolynomialModel class
        """
        obj = cls(first_observed=first_observed, init=init, dt=dt, signature=signature, true_param=true_param, wrt=wrt, scaling=scaling, warn=warn)

        if isinstance(obs, str):
            filename = './saves/' + cls.__name__ + '/Observations/' + obs
        else:
            if seed is not None:
                filename = './saves/' + cls.__name__ + '/Observations/observations_par=[' + ', '.join('{:.3f}'.format(item) for item in true_param) + ']_dt={:.1e}_sig{}_seed{}_{}obs.txt'.format(dt, signature, seed, obs)
            else:
                filename = './saves/' + cls.__name__ + '/Observations/observations_par=[' + ', '.join('{:.3f}'.format(item) for item in true_param) + ']_dt={:.1e}_sig{}_{}obs.txt'.format(dt, signature, obs)

        if os.path.exists(filename):
            observations = np.loadtxt(filename)
            obj.observations = observations * np.hstack([obj.scaling[i] ** sig for i, sig in enumerate(obj.signature)])
        else:
            obj.generate_observations(t_max=obs * dt, inter_steps=inter_steps, seed=seed, verbose=1)

        obj.seed = seed
        return obj

    def log_lik(self, param, t, verbose=0):
        """
        Computes the log-likelihood of the Gaussian equivalent of the polynomial state space model.
        :param param: Array of length k. Parameter at which to compute the matrix.
        :param t: Integer. Number of observations of the polynomial state space model to use for computation of the log-likelihood.
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
            Can also be an instance of class tqdm.
        :return: Value of the log-likelihood.
        """
        a, A, C = self.state_space_params(param=param, return_stack=True)
        kfilter = KalmanFilter(dim=self.dim, a=a, A=A, C=C, E_0=partial(self.init.E_0, param=param), Cov_0=partial(self.init.Cov_0, param=param), first_observed=self.first_observed)
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
        """
        Determines the quasi-maximum likelihood estimator of the model.
        :param fit_parameter: Integer or list of integers between 0 and k-1 that specifies which components of the parameter are estimated.
        :param initial: float or array-like that specifies the parameter guess
        :param t: Integer. Number of observations of the polynomial state space model to use for computation of the log-likelihood.
        :param verbose: If verbose is 1, a minimization callback showing the minimization progress is shown.
        :param update_estimate: Boolean. If update_estimate=True, the argument self.true_param of the PolynomialModel class is updated
            by the estimated parameter.
        :return: float or array containing the quasi-maximum likelihood estimate
        """
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
            self.savestring =  'par=[' + ', '.join('{:.3f}'.format(item) for item in self.true_param) + ']_dt={:.1e}_sig{}_sc['.format(self.dt, self.signature_string) + ', '.join('{:.1f}'.format(sc) for sc in self.scaling) + ']'

        return result

    def fit_qml_sequence(self, fit_parameter, initial, t_max=None, every=50, verbose=1, update_estimate=False):
        """
        Computes a sequence of quasi-maximum likelihood estimators of the model. Each current estimate is used as an initial guess for the subsequent estimate.
        Stores the computed qml sequence in the directory saves/[ModelName]/QML Sequences
        :param fit_parameter: Integer or list of integers between 0 and k-1 that specifies which components of the parameter are estimated.
        :param initial: float or array-like that specifies the parameter guess
        :param t_max: Integer. Maximum number of observations of X(t) to use in the sequence
        :param every: Integer. Increment of observations of X(t) between two qml estimators of the sequence.
        :param verbose: If verbose is 1, either a progressbar is shown (if the program is run) or a console callback is shown.
        :param update_estimate: Boolean. If update_estimate=True, the argument self.true_param of the PolynomialModel class is updated
            by the ultimate estimated parameter from the sequence.
        :return: A sequence of qml estimators corresponding to every, 2*every, ..., t_max observations
        """
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
                self.savestring = 'par=[' + ', '.join('{:.3f}'.format(item) for item in self.true_param) + ']_dt={:.1e}_sig{}_sc['.format(self.dt, self.signature_string) + ', '.join('{:.1f}'.format(sc) for sc in self.scaling) + ']'

        else:
            t_range = np.arange(t_max + every)[::every][1:]
            if is_run():
                if verbose == 1:
                    t_range = tqdm(t_range)
                qml_list = Parallel(n_jobs=cpu_count())(delayed(self.fit_qml)(fit_parameter=fit_parameter, initial=initial, t=t, verbose=0) for t in t_range)
            else:
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
                self.savestring = 'par=[' + ', '.join('{:.3f}'.format(item) for item in self.true_param) + ']_dt={:.1e}_sig{}_sc['.format(self.dt, self.signature_string) + ', '.join('{:.1f}'.format(sc) for sc in self.scaling) + ']'

            np.savetxt(filename, qml_list)

        end = time()
        if verbose == 1:
            print('Total Elapsed Time: ' + format_time(end - start))
        return qml_list

    def compute_U(self, kind='explicit', wrt=None, t_max=None, verbose=0, filter_unobserved=True, kfilter=None, deriv_filters=None, close_pb=True, save_raw=False):
        """
        Either performs Algorithm 3.12 the compute the matrix U_\vartheta from Theorem 3.4 explicitly or computes a sequence of the estimators (3.5)
        for U_\vartheta.Uses self.true_param for \vartheta. Stores the output in the directory saves/[ModelName]/Covariance
        :param kind: String 'explicit' or 'estimate' that specifies whether to use Algorithm 3.12 or the estimator (3.5)
        :param wrt: Integer or list of integers between 0 and k-1 that specifies which components of the parameter are estimated. Can be None if
            the setup_filter has been called first.
        :param t_max: Integer. Maximmal number of observations of the polynomial state space model to use for computation of U. Only required if kind='estimate'.
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
            Can also be an instance of class tqdm.
        :param filter_unobserved: Boolean. If True, the modification from Remark 3.10 is used where unavailable unobservable components are replaced
            by their Kalman filter. Only used if kind='estimate'.
        :param kfilter: A pre-instantiated object of class KalmanFilter can be provided here. If kfilter=None, a new KalmanFilter object is initialized.
        :param deriv_filters: If the derivatives sequences V^hom(t+1, t), V^hom(t, t) of the Kalman filter have been precomputed, they can be supplied as a tuple.
        :param close_pb: If verbose is an instance of class tqdm, this parameter specifies if the progressbar should
            be closed after computing the Kalman filter. Default to True. Only required if kind='estimate'.
        :param save_raw: Boolean. If save_raw=True the summands of the estimator (3.5) are stored in the directory saves/[ModelName]/Covariance instead
            of their means. Only used if kind='estimate'.
        :return: Array containing the matrix U_\vartheta (if kind='explicit') or a sequence of estimators (3.5) for U_\vartheta (if kind='estimate').
        """
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

            a, A, C = self.state_space_params(param=param, deriv_order=1, wrt=np.arange(np.size(param)), return_stack=True)
            if kfilter is None:
                E_0 = partial(self.init.E_0, param=param)
                Cov_0 = partial(self.init.Cov_0, param=param)

                kfilter = KalmanFilter(dim=self.dim, a=a, A=A, C=C, E_0=E_0, Cov_0=Cov_0, first_observed=self.first_observed)
                kfilter.build_covariance()
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

            wrt = np.atleast_1d(wrt)
            K_prime = A[1 + wrt] @ kfilter.Sig_tp1_t_lim[:, self.first_observed:] @ Sig_inv + np.einsum('jk, ikl, lm -> ijm', A[0], s_star[:, :, self.first_observed:], Sig_inv) - np.einsum('jk, kl, lm, imn, nr -> ijr', A[0], kfilter.Sig_tp1_t_lim[:, self.first_observed:], Sig_inv, s_star[:, self.first_observed:, self.first_observed:], Sig_inv)
            F_prime = A[1 + wrt] - K_prime @ kfilter.H
            a_hat = np.hstack((a[0], a[0], a[1 + wrt].flatten()))
            A_hat = np.zeros((self.dim * (k + 2), self.dim * (k + 2)))
            A_hat[:self.dim, :self.dim] = A[0]
            A_hat[self.dim:2 * self.dim, :2 * self.dim] = np.hstack((kfilter.K_lim @ kfilter.H, kfilter.F_lim))
            A_hat[2 * self.dim:, :] = np.hstack(((K_prime @ kfilter.H).reshape(k * self.dim, self.dim), F_prime.reshape(k * self.dim, self.dim), np.kron(np.eye(k), kfilter.F_lim)))

            j = np.arange(self.dim ** 2) // self.dim
            i = np.arange(self.dim ** 2) - j * self.dim
            i, j = np.maximum(i, j), np.minimum(i, j)
            l = (i + j * (self.dim - (j + 1) / 2)).astype('int')

            poly_B = self.poly_B(param, poly_order=2, deriv_order=0, return_stack=True)
            Q2 = np.zeros((self.dim ** 2, self.dim ** 2))
            Q2[:, np.unique(l, return_index=True)[1]] = poly_B[0, (self.dim + 1):, (self.dim + 1):][l]
            Q2 -= np.kron(A[0], A[0])
            Q1 = poly_B[0, (self.dim + 1):, 1:(1 + self.dim)][l] - kron_sym(a[:1].T, A[0])
            q = poly_B[0, (self.dim + 1):, 0][l] - np.kron(a[0], a[0])

            Q2_hat = tracy_singh(np.diag(unit_vec(k + 2, 1)), np.kron(np.diag(unit_vec(k + 2, 1)), Q2), (k + 2, k + 2), (self.dim, self.dim))
            Q_hat = tracy_singh(np.diag(unit_vec(k + 2, 1)), np.kron(unit_vec(k + 2, 1, as_column=True), Q1), (k + 2, k + 2), (self.dim, self.dim))
            q_hat = tracy_singh(unit_vec(k + 2, 1, as_column=True), np.kron(unit_vec(k + 2, 1, as_column=True), q.reshape(-1, 1)), (k + 2, 1), (self.dim, 1)).squeeze()
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
                with open(filepath, 'wb') as file:
                    pkl.dump(raw, file)

            return cummean(raw, axis=0)
        else:
            raise Exception('Argument kind needs to be set to "explicit" or "estimate".')

    def compute_W(self, kind='explicit', wrt=None, t_max=None, verbose=0, kfilter=None, deriv_filters=None, close_pb=True, save_raw=False):
        """
        Either performs Algorithm 3.12 the compute the matrix W(\vartheta) from Theorem 3.4 explicitly or computes a sequence of estimators described
        before Remark 3.10 for W(\vartheta). Uses self.true_param for \vartheta. Stores the output in the directory saves/[ModelName]/Covariance
        :param kind: String 'explicit' or 'estimate' that specifies whether to use Algorithm 3.12 or the estimator described before Remark 3.10.
        :param wrt: Integer or list of integers between 0 and k-1 that specifies which components of the parameter are estimated. Can be None if
            the setup_filter has been called first.
        :param t_max: Integer. Maximmal number of observations of the polynomial state space model to use for computation of W. Only required if kind='estimate'.
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
            Can also be an instance of class tqdm.
        :param kfilter: A pre-instantiated object of class KalmanFilter can be provided here. If kfilter=None, a new KalmanFilter object is initialized.
        :param deriv_filters: If the derivatives sequences V^hom(t+1, t), V^hom(t, t) of the Kalman filter have been precomputed, they can be supplied as a tuple.
        :param close_pb: If verbose is an instance of class tqdm, this parameter specifies if the progressbar should
            be closed after computing the Kalman filter. Default to True. Only required if kind='estimate'.
        :param save_raw: Boolean. If save_raw=True the summands of the estimator described before Remark 3.10 are stored in the directory
            saves/[ModelName]/Covariance instead of their means. Only used if kind='estimate'.
        :return: Array containing the matrix W(\vartheta) (if kind='explicit') or a sequence of estimators for W(\vartheta) (if kind='estimate').
        """
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

            a, A, C = self.state_space_params(param=param, deriv_order=2, wrt=np.arange(np.size(param)), return_stack=True)
            if kfilter is None:
                E_0 = partial(self.init.E_0, param=param)
                Cov_0 = partial(self.init.Cov_0, param=param)

                kfilter = KalmanFilter(dim=self.dim, a=a, A=A, C=C, E_0=E_0, Cov_0=Cov_0, first_observed=self.first_observed)
                kfilter.build_covariance()
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
                with open(filepath, 'wb') as file:
                    pkl.dump(result, file)

            return cummean(result, axis=0)
        else:
            raise Exception('Argument kind needs to be set to "explicit" or "estimate".')

    def compute_V(self, kind='explicit', wrt=None, t_max=None, verbose=0, filter_unobserved=True, from_raw=None, every=None):
        """
        Either performs Algorithm 3.12 the compute the matrix V_\vartheta from Theorem 3.4 explicitly or computes a sequence of estimators described
        before Remark 3.10 for V_\vartheta. Uses self.true_param for \vartheta. Stores the output in the directory saves/[ModelName]/Covariance
        :param kind: String 'explicit' or 'estimate' that specifies whether to use Algorithm 3.12 or the estimator described before Remark 3.10.
        :param wrt: Integer or list of integers between 0 and k-1 that specifies which components of the parameter are estimated. Can be None if
            the setup_filter has been called first.
        :param t_max: Integer. Maximmal number of observations of the polynomial state space model to use for computation of V. Only required if kind='estimate'.
        :param verbose: If verbose is 0, no progress is visually tracked. If verbose is 1, a progressbar is shown.
            Can also be an instance of class tqdm.
        :param filter_unobserved: Boolean. If True, the modification from Remark 3.10 is used where unavailable unobservable components are replaced
            by their Kalman filter. Only used if kind='estimate'.
        :param from_raw: Boolean. If from_raw=True the computation used the raw summands for the U and W estimators stored in the directory
            saves/[ModelName]/Covariance. Only used if kind='estimate'.
        :param every: Increment of observations of X(t) used for the estimate sequence for V_\vartheta. Returns a sequence corresponding to
            every, 2*every, ..., t_max observations. Only used if kind='estimate'.
        :return: Array containing the matrix W(\vartheta) (if kind='explicit') or a sequence of estimators for W(\vartheta) (if kind='estimate').
        """
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
                    a, A, C = self.state_space_params(param=param, deriv_order=2, wrt=np.arange(np.size(param)), return_stack=True)
                    E_0 = partial(self.init.E_0, param=param)
                    Cov_0 = partial(self.init.Cov_0, param=param)

                    if t_max >= self.observations.shape[0]:
                        raise Exception('Not enough observations available for t_max = {}'.format(t_max))

                    U_missing = not os.path.exists(filepath_U)
                    W_missing = not os.path.exists(filepath_W)

                    if U_missing | W_missing:
                        if (verbose == 1) & (not W_missing):
                            verbose = tqdm(total=2 * t_max, desc='Computing filter')
                        elif (verbose == 1) & W_missing:
                            verbose = tqdm(total=3 * t_max, desc='Computing filter')
                        kfilter = KalmanFilter(dim=self.dim, a=a, A=A, C=C, E_0=E_0, Cov_0=Cov_0, first_observed=self.first_observed)
                        kfilter.build_covariance()
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

            with open(filepath, 'wb') as file:
                pkl.dump([V.squeeze(), Std.squeeze(), Corr.squeeze()], file)

            return V.squeeze(), Std.squeeze(), Corr.squeeze()
        else:
            raise Exception('Argument kind needs to be set to "explicit" or "estimate".')


class HestonModel(PolynomialModel):
    def __init__(self, first_observed, init, dt, signature, true_param=None, wrt=None, scaling=1, warn=True):
        params_names = np.array(['kappa', 'theta', 'sigma', 'rho'])
        params_bounds = np.array([[0.0001, 10], [0.0001 ** 2, 1], [0.0001, 1], [-1, 1]])
        super().__init__(first_observed, init, dt, signature, params_names, params_bounds, true_param, wrt, scaling, warn)

    def poly_A(self, param, poly_order, deriv_order=0, wrt=None):
        wrt = np.atleast_1d(wrt)
        kappa, theta, sigma, rho = param
        mu, delta = 0, 0
        dicts = return_dict(self.dim_c, poly_order)
        k = np.size(wrt)
        n_components = 1 + k * (deriv_order >= 1) + int(k * (k + 1) / 2) * (deriv_order == 2)

        heston_A = np.zeros((n_components, n_dim(self.dim_c, poly_order), n_dim(self.dim_c, poly_order)))
        heston_A[0, mult_to_ind([1, 0], dicts), 0] = kappa * theta
        heston_A[0, mult_to_ind([1, 0], dicts), mult_to_ind([1, 0], dicts)] = -kappa
        heston_A[0, mult_to_ind([0, 1], dicts), 0] = mu
        heston_A[0, mult_to_ind([0, 1], dicts), mult_to_ind([1, 0], dicts)] = delta
        heston_A[0, mult_to_ind([2, 0], dicts), mult_to_ind([1, 0], dicts)] = sigma ** 2
        heston_A[0, mult_to_ind([1, 1], dicts), mult_to_ind([1, 0], dicts)] = sigma * rho
        heston_A[0, mult_to_ind([0, 2], dicts), mult_to_ind([1, 0], dicts)] = 1

        if deriv_order >= 1:
            deriv_A = np.zeros((4, n_dim(self.dim_c, poly_order), n_dim(self.dim_c, poly_order)))
            deriv_A[0, mult_to_ind([1, 0], dicts), 0] = theta
            deriv_A[0, mult_to_ind([1, 0], dicts), mult_to_ind([1, 0], dicts)] = -1
            deriv_A[1, mult_to_ind([1, 0], dicts), 0] = kappa
            deriv_A[2, mult_to_ind([2, 0], dicts), mult_to_ind([1, 0], dicts)] = 2 * sigma
            deriv_A[2, mult_to_ind([1, 1], dicts), mult_to_ind([1, 0], dicts)] = rho
            deriv_A[3, mult_to_ind([1, 1], dicts), mult_to_ind([1, 0], dicts)] = sigma
            heston_A[1:(1 + k)] = deriv_A[wrt]
        if deriv_order == 2:
            deriv2_A = np.zeros((4, 4, n_dim(self.dim_c, poly_order), n_dim(self.dim_c, poly_order)))
            deriv2_A[0, 1, mult_to_ind([1, 0], dicts), 0] = 1
            deriv2_A[2, 2, mult_to_ind([2, 0], dicts), mult_to_ind([1, 0], dicts)] = 2
            deriv2_A[2, 3, mult_to_ind([1, 1], dicts), mult_to_ind([1, 0], dicts)] = 1
            deriv2_A = deriv2_A[np.ix_(wrt, wrt)]
            heston_A[(1 + k):] = deriv2_A[np.triu_indices(len(wrt))]

        return heston_A.squeeze()

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
            obs0 = self.observations[-1]
        else:
            obs0 = self.init.sample(param=self.true_param, n=1)
        kappa, theta, sigma, rho = self.true_param
        S0 = 0 if 1 in self.differenced_components else obs0[self.signature_indices[1][0]]
        v0 = obs0[0]

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

        v = np.array(v) * self.scaling[0]
        S = np.array(S) * self.scaling[1]

        if 1 in self.differenced_components:
            Y_disc = np.append(obs0[self.signature_indices[1][0]], np.diff(S))
        else:
            Y_disc = S

        v = np.power.outer(v, self.signature[0])
        Y_disc = np.power.outer(Y_disc, self.signature[1])

        observations = np.hstack((v, Y_disc))
        if self.observations is not None:
            observations = np.vstack((self.observations, observations[1:]))
            self.seed = str(self.seed) + '+' + str(seed) if self.seed is not None else seed
        else:
            self.seed = seed

        if self.seed is not None:
            np.savetxt('./saves/HestonModel/Observations/observations_par=[' + ', '.join('{:.3f}'.format(item) for item in self.true_param) + ']_dt={:.1e}_sig{}_seed{}_{}obs.txt'.format(self.dt, self.signature_string, self.seed, observations.shape[0] - 1), observations / np.hstack([self.scaling[i] ** sig for i, sig in enumerate(self.signature)]))
        else:
            np.savetxt('./saves/HestonModel/Observations/observations_par=[' + ', '.join('{:.3f}'.format(item) for item in self.true_param) + ']_dt={:.1e}_sig{}_{}obs.txt'.format(self.dt, self.signature_string, observations.shape[0] - 1), observations / np.hstack([self.scaling[i] ** sig for i, sig in enumerate(self.signature)]))
        self.observations = observations


class OUNIGModel(PolynomialModel):
    def __init__(self, first_observed, init, dt, signature, true_param=None, wrt=None, scaling=1, warn=True):
        params_names = np.array(['lambda', 'kappa', 'delta'])
        params_bounds = np.array([[0.001, 10], [0.0001, 30], [0.0001, 10]])
        super().__init__(first_observed, init, dt, signature, params_names, params_bounds, true_param, wrt, scaling, warn)

    def poly_A(self, param, poly_order, deriv_order=0, wrt=None):
        if poly_order > 4:
            raise Exception('Polynomial orders greater than 4 have not been implemented yet.')

        wrt = np.atleast_1d(wrt)
        lamb, kappa, delta = param
        alpha = 1
        dicts = return_dict(self.dim_c, poly_order)
        k = np.size(wrt)
        n_components = 1 + k * (deriv_order >= 1) + int(k * (k + 1) / 2) * (deriv_order == 2)

        OU_A = np.zeros((n_components, n_dim(self.dim_c, poly_order), n_dim(self.dim_c, poly_order)))
        OU_A[0, mult_to_ind([1, 0], dicts), mult_to_ind([1, 0], dicts)] = -lamb
        OU_A[0, mult_to_ind([0, 1], dicts), mult_to_ind([1, 0], dicts)] = kappa
        OU_A[0, mult_to_ind([0, 1], dicts), mult_to_ind([0, 1], dicts)] = -kappa
        OU_A[0, mult_to_ind([2, 2], dicts), 0] = delta / alpha ** 3
        OU_A[0, mult_to_ind([2, 0], dicts), 0] = delta / alpha
        OU_A[0, mult_to_ind([0, 2], dicts), 0] = delta / alpha
        OU_A[0, mult_to_ind([4, 0], dicts), 0] = 3 * delta / alpha ** 3
        OU_A[0, mult_to_ind([0, 4], dicts), 0] = 3 * delta / alpha ** 3

        if deriv_order >= 1:
            deriv_A = np.zeros((3, n_dim(self.dim_c, poly_order), n_dim(self.dim_c, poly_order)))
            deriv_A[0, mult_to_ind([1, 0], dicts), mult_to_ind([1, 0], dicts)] = -1
            deriv_A[1, mult_to_ind([0, 1], dicts), mult_to_ind([1, 0], dicts)] = 1
            deriv_A[1, mult_to_ind([0, 1], dicts), mult_to_ind([0, 1], dicts)] = -1
            deriv_A[2, mult_to_ind([2, 2], dicts), 0] = 1 / alpha ** 3
            deriv_A[2, mult_to_ind([2, 0], dicts), 0] = 1 / alpha
            deriv_A[2, mult_to_ind([0, 2], dicts), 0] = 1 / alpha
            deriv_A[2, mult_to_ind([4, 0], dicts), 0] = 3 / alpha ** 3
            deriv_A[2, mult_to_ind([0, 4], dicts), 0] = 3 / alpha ** 3
            OU_A[1:(1 + k)] = deriv_A[wrt]

        return OU_A.squeeze()

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
            obs0 = self.init.sample(param=self.true_param, n=1)
            x = np.array([obs0[self.signature_indices[0][0]], obs0[self.signature_indices[1][0]]])
        lamb, kappa, delta = self.true_param
        alpha = 1

        steps = int(np.ceil(np.round(t_max / dt, 7))) + 1
        W = np.random.standard_normal(size=(steps - 1, 2))
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
        first_comp = np.power.outer(observations[:, 0] * self.scaling[0], self.signature[0])
        second_comp = np.power.outer(observations[:, 1] * self.scaling[1], self.signature[1])
        observations = np.hstack((first_comp, second_comp))

        if self.observations is not None:
            observations = np.vstack((self.observations, observations[1:]))
            self.seed = str(self.seed) + '+' + str(seed) if self.seed is not None else seed
        else:
            self.seed = seed

        if self.seed is not None:
            np.savetxt('./saves/OUNIGModel/Observations/observations_par=[' + ', '.join('{:.3f}'.format(item) for item in self.true_param) + ']_dt={:.1e}_sig{}_seed{}_{}obs.txt'.format(self.dt, self.signature_string, self.seed, observations.shape[0] - 1), observations / np.hstack([self.scaling[i] ** sig for i, sig in enumerate(self.signature)]))
        else:
            np.savetxt('./saves/OUNIGModel/Observations/observations_par=[' + ', '.join('{:.3f}'.format(item) for item in self.true_param) + ']_dt={:.1e}_sig{}_{}obs.txt'.format(self.dt, self.signature_string, observations.shape[0] - 1), observations / np.hstack([self.scaling[i] ** sig for i, sig in enumerate(self.signature)]))

        self.observations = observations


if __name__ == '__main__':
    ### Isolated computation of the asymptotic standard deviation of the vol-vol estimator [dt = 1, based on (v, ΔY, (ΔY)^2), no observations needed]
    init = InitialDistribution(dist='Dirac', hyper=[0.3**2, 0, 0])
    model = HestonModel(first_observed=1, init=init, dt=1, signature='1[1]_2d[1, 2]', true_param=np.array([1, 0.4**2, 0.3, -0.5]), wrt=2, warn=False)
    V, Std, Corr = model.compute_V()

    ### Isolated computation of the asymptotic standard deviation of the vol-vol estimator [dt = 1/24000, based on (v, v^2, ΔY, (ΔY)^2, (ΔY)^4), no observations needed]
    # init = InitialDistribution(dist='Dirac', hyper=[0.3**2, 0.3**4, 0, 0, 0])
    # model = HestonModel(first_observed=1, init=init, dt=1/24000, signature='1[1, 2]_2d[1, 2, 4]', true_param=np.array([1, 0.4**2, 0.3, -0.5]), wrt=2, scaling=[20, 140], warn=False)
    # V, Std, Corr = model.compute_V()

    ### Joint computation of the asymptotic covariance matrix of the estimator [dt = 1, based on (v, ΔY, (ΔY)^2), no observations needed]
    # init = InitialDistribution(dist='Dirac', hyper=[0.3**2, 0, 0])
    # model = HestonModel(first_observed=1, init=init, dt=1, signature='1[1]_2d[1, 2]', true_param=np.array([1, 0.4**2, 0.3, -0.5]), wrt=[0, 1, 2, 3], warn=False)
    # V, Std, Corr = model.compute_V()

    ### Initialize model by simulating observations, compute QML estimators and estimate the asymptotic QML estimator standard deviation
    # init = InitialDistribution(dist='Dirac', hyper=[0.3 ** 2, 0, 0])
    # model = HestonModel.from_observations(first_observed=1, init=init, dt=1, signature='1[1]_2d[1, 2]', obs=200000, inter_steps=250, true_param=np.array([1, 0.4 ** 2, 0.3, -0.5]), seed=20, warn=False)
    # result = model.fit_qml(fit_parameter=2, initial=0.5, t=200000)
    # seq = model.fit_qml_sequence(fit_parameter=[0, 2], initial=[1, 0.5], t_max=1500)
    # V, Std, Corr = model.compute_V(kind='estimate', wrt=2, verbose=1)

    ### Initialize model with observations externally given, compute QML estimators and compute the asymptotic QML estimator standard deviation
    # init = InitialDistribution(dist='Dirac', hyper=[0.3 ** 2, 0, 0])
    # model = HestonModel.from_observations(first_observed=1, init=init, obs='observations_par=[1.000, 0.160, 0.300, -0.500]_dt=1.0e+00_sig1[1]_2d[1, 2]_seed20_200000obs.txt', dt=1, signature='1[1]_2d[1, 2]')
    # result = model.fit_qml(fit_parameter=[0, 1, 2, 3], initial=[1, 0.4**2, 0.3, -0.5], t=10000, update_estimate=True)
    # model.setup_filter(wrt=2)
    # V, Std, Corr = model.compute_V()
