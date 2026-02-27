from main import *
import numpy as np
from scipy.special import iv, ive
from scipy.stats import truncnorm
from numba import njit, prange
from time import time
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator):
    """
    Systematic resampling. weights must sum to 1.
    Returns ancestor indices of shape (N,).
    """
    N = weights.size
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # guard against roundoff
    u0 = rng.random() / N
    u = u0 + np.arange(N) / N
    return np.searchsorted(cdf, u, side="left")


def _heston_char(u, dt, v, kappa, m, sigma, rho):
    B = kappa - rho * sigma * 1j * u
    D = np.sqrt(B ** 2 + sigma ** 2 * u ** 2)
    lnG = np.log((np.exp(-D * dt) + 1) / 2 - B * (np.exp(-D * dt) - 1) / (2 * D))
    Psi0 = kappa * m / sigma ** 2 * ((B - D) * dt - 2 * lnG)
    Psi1 = u ** 2 * (np.exp(-D * dt) - 1) / (D * (np.exp(-D * dt) + 1) - B * (np.exp(-D * dt) - 1))  # u ** 2 * (sp.exp(-D * T) - 1) / (D * (sp.exp(-D * T) + 1) - B * (sp.exp(-D * T) - 1))
    return np.exp(Psi0 + Psi1 * v)


@njit(cache=True, parallel=True)
def _log_p_dYn_given_vnm1_vn(n, vnm1, vn, dt, kappa, m, sigma, rho, u, E, pref, lam, R_table, p_min, p_max, logp0, inv_dlogp, P, log_pi):
    """
    Parallel batch evaluation for fixed n.
    Returns:
      ok    : (M,) uint8  (1 = success, 0 = failed/non-positive inversion)
      logp  : (M,)
      grad  : (M,4)
      hdiag : (M,4)
    """
    M = vnm1.shape[0]
    Ku = u.shape[0]

    ok = np.ones(M, dtype=np.uint8)
    logp = np.empty(M, dtype=np.float64)

    # Same observed return for all pairs in this batch
    Erow = E[n]  # shape (Ku,), contiguous if E is C-order and shape (T,Ku)

    for i in prange(M):
        a = vnm1[i]
        b = vn[i]

        # default outputs if failure
        logp_i = np.nan

        if a <= 0.0 or b <= 0.0:
            ok[i] = 0
            logp[i] = logp_i
            continue

        # endpoint combinations
        s = a + b
        d = b - a
        p = np.sqrt(a * b)

        # interpolate index on log p-grid
        if p <= p_min:
            j = 0
            t = 0.0
        elif p >= p_max:
            j = P - 2
            t = 1.0
        else:
            x = np.log(p)
            y = (x - logp0) * inv_dlogp
            j = int(y)
            if j < 0:
                j = 0
                t = 0.0
            elif j >= P - 1:
                j = P - 2
                t = 1.0
            else:
                t = y - j

        omt = 1.0 - t

        # mu_const = (rho/sigma) * (d - kappa*m*dt)
        c = d - kappa * m * dt
        mu_const = (rho / sigma) * c

        # accumulator for Re(S0)
        S0_re = 0.0

        for k in range(Ku):
            uk = u[k]

            # interpolate R at p
            Rk = omt * R_table[j, k] + t * R_table[j + 1, k]

            # psi_k = pref_k * exp(lam_k*s + i*u_k*mu_const) * Rk
            exp_arg = lam[k] * s + (1j * uk * mu_const)
            psi_k = pref[k] * np.exp(exp_arg) * Rk

            z = psi_k * Erow[k]
            zr = z.real
            S0_re += zr

        # finalize
        if (not np.isfinite(S0_re)) or (S0_re <= 0.0):
            ok[i] = 0
        else:
            logp_i = np.log(S0_re) - log_pi

        logp[i] = logp_i
    return ok, logp


@njit(cache=True, parallel=True)
def _log_p_dYn_given_vnm1_vn_with_derivs(n, vnm1, vn, dt, kappa, m, sigma, rho, u, E, pref, lam, dlogpref, d2logpref, dlam, d2lam, R_table, dlogR_table, d2logR_table, p_min, p_max, logp0, inv_dlogp, P, log_pi):
    """
    Parallel batch evaluation for fixed n.
    Returns:
      ok    : (M,) uint8  (1 = success, 0 = failed/non-positive inversion)
      logp  : (M,)
      grad  : (M,4)
      hdiag : (M,4)
    """
    M = vnm1.shape[0]
    Ku = u.shape[0]

    ok = np.ones(M, dtype=np.uint8)
    logp = np.empty(M, dtype=np.float64)
    grad = np.empty((M, 4), dtype=np.float64)
    hdiag = np.empty((M, 4), dtype=np.float64)

    # Same observed return for all pairs in this batch
    Erow = E[n]  # shape (Ku,), contiguous if E is C-order and shape (T,Ku)

    for i in prange(M):
        a = vnm1[i]
        b = vn[i]

        # default outputs if failure
        logp_i = np.nan
        g0 = np.nan
        g1 = np.nan
        g2 = np.nan
        g3 = np.nan
        h0 = np.nan
        h1 = np.nan
        h2 = np.nan
        h3 = np.nan

        if a <= 0.0 or b <= 0.0:
            ok[i] = 0
            logp[i] = logp_i
            grad[i, 0] = g0
            grad[i, 1] = g1
            grad[i, 2] = g2
            grad[i, 3] = g3
            hdiag[i, 0] = h0
            hdiag[i, 1] = h1
            hdiag[i, 2] = h2
            hdiag[i, 3] = h3
            continue

        # endpoint combinations
        s = a + b
        d = b - a
        p = np.sqrt(a * b)

        # interpolate index on log p-grid
        if p <= p_min:
            j = 0
            t = 0.0
        elif p >= p_max:
            j = P - 2
            t = 1.0
        else:
            x = np.log(p)
            y = (x - logp0) * inv_dlogp
            j = int(y)
            if j < 0:
                j = 0
                t = 0.0
            elif j >= P - 1:
                j = P - 2
                t = 1.0
            else:
                t = y - j

        omt = 1.0 - t

        # mu_const and theta-derivatives (diagonal 2nd only)
        # mu = (rho/sigma) * (d - kappa*m*dt)
        c = d - kappa * m * dt
        mu_const = (rho / sigma) * c

        dmu0 = -(rho / sigma) * m * dt  # d/dkappa
        dmu1 = -(rho / sigma) * kappa * dt  # d/dm
        dmu2 = -(rho / (sigma * sigma)) * c  # d/dsigma
        dmu3 = c / sigma  # d/drho

        d2mu0 = 0.0
        d2mu1 = 0.0
        d2mu2 = 2.0 * rho * c / (sigma * sigma * sigma)
        d2mu3 = 0.0

        # accumulators for Re(S0), Re(S1_l), Re(S2_l)
        S0_re = 0.0

        S1_0 = 0.0
        S1_1 = 0.0
        S1_2 = 0.0
        S1_3 = 0.0
        S2_0 = 0.0
        S2_1 = 0.0
        S2_2 = 0.0
        S2_3 = 0.0

        for k in range(Ku):
            uk = u[k]

            # interpolate R and derivative tables at p
            Rk = omt * R_table[j, k] + t * R_table[j + 1, k]

            # psi_k = pref_k * exp(lam_k*s + i*u_k*mu_const) * Rk
            exp_arg = lam[k] * s + (1j * uk * mu_const)
            psi_k = pref[k] * np.exp(exp_arg) * Rk

            z = psi_k * Erow[k]
            zr = z.real
            S0_re += zr

            # ----- theta component 0: kappa -----
            dlogR0 = omt * dlogR_table[j, 0, k] + t * dlogR_table[j + 1, 0, k]
            d2logR0 = omt * d2logR_table[j, 0, k] + t * d2logR_table[j + 1, 0, k]
            G0 = (1j * uk * dmu0) + dlogpref[0, k] + s * dlam[0, k] + dlogR0
            H0 = (1j * uk * d2mu0) + d2logpref[0, k] + s * d2lam[0, k] + d2logR0
            S1_0 += (z * G0).real
            S2_0 += (z * (H0 + G0 * G0)).real

            # ----- theta component 1: m -----
            dlogR1 = omt * dlogR_table[j, 1, k] + t * dlogR_table[j + 1, 1, k]
            d2logR1 = omt * d2logR_table[j, 1, k] + t * d2logR_table[j + 1, 1, k]
            G1 = (1j * uk * dmu1) + dlogpref[1, k] + s * dlam[1, k] + dlogR1
            H1 = (1j * uk * d2mu1) + d2logpref[1, k] + s * d2lam[1, k] + d2logR1
            S1_1 += (z * G1).real
            S2_1 += (z * (H1 + G1 * G1)).real

            # ----- theta component 2: sigma -----
            dlogR2 = omt * dlogR_table[j, 2, k] + t * dlogR_table[j + 1, 2, k]
            d2logR2 = omt * d2logR_table[j, 2, k] + t * d2logR_table[j + 1, 2, k]
            G2 = (1j * uk * dmu2) + dlogpref[2, k] + s * dlam[2, k] + dlogR2
            H2 = (1j * uk * d2mu2) + d2logpref[2, k] + s * d2lam[2, k] + d2logR2
            S1_2 += (z * G2).real
            S2_2 += (z * (H2 + G2 * G2)).real

            # ----- theta component 3: rho -----
            dlogR3 = omt * dlogR_table[j, 3, k] + t * dlogR_table[j + 1, 3, k]
            d2logR3 = omt * d2logR_table[j, 3, k] + t * d2logR_table[j + 1, 3, k]
            G3 = (1j * uk * dmu3) + dlogpref[3, k] + s * dlam[3, k] + dlogR3
            H3 = (1j * uk * d2mu3) + d2logpref[3, k] + s * d2lam[3, k] + d2logR3
            S1_3 += (z * G3).real
            S2_3 += (z * (H3 + G3 * G3)).real

        # finalize
        if (not np.isfinite(S0_re)) or (S0_re <= 0.0):
            ok[i] = 0
        else:
            logp_i = np.log(S0_re) - log_pi

            g0 = S1_0 / S0_re
            g1 = S1_1 / S0_re
            g2 = S1_2 / S0_re
            g3 = S1_3 / S0_re

            h0 = S2_0 / S0_re - g0 * g0
            h1 = S2_1 / S0_re - g1 * g1
            h2 = S2_2 / S0_re - g2 * g2
            h3 = S2_3 / S0_re - g3 * g3

        logp[i] = logp_i
        grad[i, 0] = g0
        grad[i, 1] = g1
        grad[i, 2] = g2
        grad[i, 3] = g3
        hdiag[i, 0] = h0
        hdiag[i, 1] = h1
        hdiag[i, 2] = h2
        hdiag[i, 3] = h3

    return ok, logp, grad, hdiag


def _log_p_vn_given_vnm1(vn, vnm1, dt, kappa, m, sigma):
    """
    Exact log density of the CIR transition:
        X_t | X_0 = x_prev
    for the SDE
        dX_t = kappa (m - X_t) dt + sigma sqrt(X_t) dW_t
    """
    ek = np.exp(-kappa * dt)
    one_minus_ek = -np.expm1(-kappa * dt)  # 1 - e^{-kappa dt}
    c = 2.0 * kappa / (sigma ** 2 * one_minus_ek)
    q = 2.0 * kappa * m / sigma ** 2 - 1.0  # Bessel order nu

    u = c * vnm1 * ek
    v = c * vn
    z = 2.0 * np.sqrt(u * v)

    # log I_q(z) using scaled Bessel ive: ive(q,z) = exp(-z) I_q(z) for z>0
    # Hence log I_q(z) = log(ive(q,z)) + z
    ive_q_z = ive(q, z)
    logI = np.log(ive_q_z) + z

    # Continuous-part log-density
    return np.log(c) - u - v + 0.5 * q * (np.log(v) - np.log(u)) + logI


class Proposal:
    def __init__(self, dt, kappa, m, sigma, rho, rng=None):
        self.dt = dt
        self.kappa = kappa
        self.m = m
        self.sigma = sigma
        self.rho = rho
        self.rng = rng if rng is not None else np.random.default_rng()

    def log_q_dYn_given_vnm1(self, dYn, vnm1):
        """Gaussian from Euler discretization"""
        return -0.5 * (np.log(2 * np.pi) + np.log(vnm1 * self.dt) + dYn ** 2 / (vnm1 * self.dt))

    def log_q_vn_given_vnm1_dYn(self, vn, vnm1, dYn):
        """Truncated Gaussian from Euler discretization"""
        mu_q = vnm1 + self.kappa * (self.m - vnm1) * self.dt + self.sigma * self.rho * dYn
        std_q = self.sigma * np.sqrt((1.0 - self.rho ** 2) * vnm1 * self.dt)
        trunc = truncnorm(a=-mu_q / std_q, b=np.inf, loc=mu_q, scale=std_q)
        return trunc.logpdf(vn)

    def sample_q_vn_given_vnm1_dYn(self, vnm1, dYn) -> np.ndarray:
        """Truncated Gaussian from Euler discretization"""
        mu_q = vnm1 + self.kappa * (self.m - vnm1) * self.dt + self.sigma * self.rho * dYn
        std_q = self.sigma * np.sqrt((1.0 - self.rho ** 2) * vnm1 * self.dt)
        trunc = truncnorm(a=-mu_q / std_q, b=np.inf, loc=mu_q, scale=std_q)
        return trunc.rvs(size=mu_q.shape, random_state=self.rng)


class HestonTransition:
    def __init__(self, dY_seq, dt, kappa, m, sigma, rho, *, U=200.0, K=256, p_min=1e-10, p_max=5.0, P=32768, p_grid=None, fd_eps=1e-4):
        # --- observed returns
        self.dY_seq = np.asarray(dY_seq, dtype=np.float64)
        self.T = self.dY_seq.size

        self.dt = float(dt)
        self.kappa = float(kappa)
        self.m = float(m)
        self.sigma = float(sigma)
        self.rho = float(rho)

        # --- quadrature grid in u
        self.U = float(U)
        self.K = int(K)

        u = np.linspace(0.0, self.U, self.K + 1, dtype=np.float64)
        du = u[1] - u[0]
        w = np.full_like(u, du, dtype=np.float64)
        w[0] *= 0.5
        w[-1] *= 0.5

        self.u = u
        self.iu = 1j * u
        self.w = w
        self.u2 = u * u

        # --- precompute E_{k,n} = w_k * exp(-i u_k r_n), shape (T, Ku)
        self.E = np.ascontiguousarray(w[None, :] * np.exp(-self.iu[None, :] * self.dY_seq[:, None]))

        # --- p-grid (for p = sqrt(a*b))
        if p_grid is None:
            if p_min <= 0:
                raise ValueError("p_min must be > 0")
            if p_max <= p_min:
                raise ValueError("p_max must be > p_min")
            self.p_grid = np.geomspace(p_min, p_max, P)
        else:
            self.p_grid = np.asarray(p_grid, dtype=np.float64)
            if np.any(self.p_grid <= 0):
                raise ValueError("p_grid must be strictly positive")
            if np.any(np.diff(self.p_grid) <= 0):
                raise ValueError("p_grid must be strictly increasing")

        self.P = self.p_grid.size
        self.logp_grid = np.log(self.p_grid)

        # cache for fast scalar interpolation index lookup
        self._p_min = self.p_grid[0]
        self._p_max = self.p_grid[-1]

        # geomspace => uniform spacing in log-space
        self._logp0 = self.logp_grid[0]
        self._dlogp = self.logp_grid[1] - self.logp_grid[0]
        self._inv_dlogp = 1.0 / self._dlogp

        # --- base u-dependent terms + base R-table
        base_terms = self._compute_u_terms(self.kappa, self.m, self.sigma, self.rho)
        self.nu = base_terms["nu"]
        self.pref = base_terms["pref"]  # (Ku,)
        self.lam = base_terms["lam"]  # (Ku,)
        self.R_table = self._compute_R_table(base_terms)  # (Ku,P)

        # --- finite-difference steps for theta-derivatives
        self.fd_eps = fd_eps * np.maximum([self.kappa, self.m, self.sigma, np.abs(self.rho)], 1.)

        # --- precompute derivative tables wrt theta = (kappa, m, sigma, rho)
        self._precompute_theta_derivative_tables()

    def log_p_dYn_given_vnm1_vn(self, n, vnm1, vn, *, raise_on_fail=True):
        """
        Batch version for fixed observation index n.
        Evaluates log pdf for many (a,b) pairs.

        Inputs:
          n      : int
          a_vec  : array-like
          b_vec  : array-like (same shape as a_vec or broadcastable)

        Returns:
          logp   : shape S
        """

        n = int(n)
        if n < 0 or n >= self.T:
            raise IndexError(f"n must be in [0, {self.T - 1}]")

        ok, logp = _log_p_dYn_given_vnm1_vn(n, vnm1, vn, self.dt, self.kappa, self.m, self.sigma, self.rho, self.u, self.E, self.pref, self.lam, self.R_table, self._p_min, self._p_max, self._logp0, self._inv_dlogp, self.P, np.log(np.pi))

        if raise_on_fail and np.any(ok == 0):
            bad = int(np.sum(ok == 0))
            raise FloatingPointError(
                f"Batch kernel failed for {bad} pair(s) "
                "(non-positive/invalid Fourier inversion result or invalid a,b)."
            )

        return logp

    def log_p_dYn_given_vnm1_vn_with_derivs(self, n, vnm1, vn, *, raise_on_fail=True):
        """
        Batch version for fixed observation index n.
        Evaluates (log pdf, gradient, diagonal Hessian) for many (a,b) pairs.

        Inputs:
          n      : int
          a_vec  : array-like
          b_vec  : array-like (same shape as a_vec or broadcastable)

        Returns:
          logp   : shape S
          grad   : shape S + (4,)
          hdiag  : shape S + (4,)
        """

        n = int(n)
        if n < 0 or n >= self.T:
            raise IndexError(f"n must be in [0, {self.T - 1}]")

        ok, logp, grad, hdiag = _log_p_dYn_given_vnm1_vn_with_derivs(n, vnm1, vn, self.dt, self.kappa, self.m, self.sigma, self.rho, self.u, self.E, self.pref, self.lam, self.dlogpref, self.d2logpref, self.dlam, self.d2lam, self.R_table, self.dlogR_table, self.d2logR_table, self._p_min, self._p_max, self._logp0, self._inv_dlogp, self.P, np.log(np.pi))

        if raise_on_fail and np.any(ok == 0):
            bad = int(np.sum(ok == 0))
            raise FloatingPointError(
                f"Batch kernel failed for {bad} pair(s) "
                "(non-positive/invalid Fourier inversion result or invalid a,b)."
            )

        return logp, grad, hdiag

    def log_p_vn_given_vnm1(self, vn, vnm1):
        """
        Exact CIR transition log-density and theta-derivatives (diag Hessian only)
        for theta = (kappa, m, sigma, rho), where rho does NOT enter the transition.

        Returns
        -------
        logf : ndarray or scalar
            log f_theta(x | x_prev)
        grad : ndarray
            gradient wrt theta, shape (..., 4)
            order = [d/dkappa, d/dm, d/dsigma, d/drho]
        hess_diag : ndarray
            diagonal second derivatives wrt theta, shape (..., 4)
            order = [d2/dkappa2, d2/dm2, d2/dsigma2, d2/drho2]

        Notes
        -----
        - Derivatives are computed by centered finite differences of the exact log-density.
        - rho-derivatives are exactly zero.
        """

        # Base log-density
        logf0 = _log_p_vn_given_vnm1(vn, vnm1, self.dt, self.kappa, self.m, self.sigma)

        # Return scalars if scalar input
        if logf0.ndim == 0:
            return float(logf0)

        return logf0

    def log_p_vn_given_vnm1_with_derivs(self, vn, vnm1):
        """
        Exact CIR transition log-density and theta-derivatives (diag Hessian only)
        for theta = (kappa, m, sigma, rho), where rho does NOT enter the transition.

        Returns
        -------
        logf : ndarray or scalar
            log f_theta(x | x_prev)
        grad : ndarray
            gradient wrt theta, shape (..., 4)
            order = [d/dkappa, d/dm, d/dsigma, d/drho]
        hess_diag : ndarray
            diagonal second derivatives wrt theta, shape (..., 4)
            order = [d2/dkappa2, d2/dm2, d2/dsigma2, d2/drho2]

        Notes
        -----
        - Derivatives are computed by centered finite differences of the exact log-density.
        - rho-derivatives are exactly zero.
        """
        out_shape = np.shape(vn)

        # Base log-density
        logf0 = _log_p_vn_given_vnm1(vn, vnm1, self.dt, self.kappa, self.m, self.sigma)

        # Initialize outputs
        grad = np.full(out_shape + (4,), np.nan, dtype=np.float64)
        hdiag = np.full(out_shape + (4,), np.nan, dtype=np.float64)

        # Parameter 0: kappa
        h = self.fd_eps[0]
        lp = _log_p_vn_given_vnm1(vn, vnm1, self.dt, self.kappa + h, self.m, self.sigma)
        lm = _log_p_vn_given_vnm1(vn, vnm1, self.dt, self.kappa - h, self.m, self.sigma)
        grad[..., 0] = ((lp - lm) / (2.0 * h))
        hdiag[..., 0] = ((lp - 2.0 * logf0 + lm) / h ** 2)

        # Parameter 1: m
        h = self.fd_eps[1]
        lp = _log_p_vn_given_vnm1(vn, vnm1, self.dt, self.kappa, self.m + h, self.sigma)
        lm = _log_p_vn_given_vnm1(vn, vnm1, self.dt, self.kappa, self.m - h, self.sigma)
        grad[..., 1] = ((lp - lm) / (2.0 * h))
        hdiag[..., 1] = ((lp - 2.0 * logf0 + lm) / h ** 2)

        # Parameter 2: sigma
        h = self.fd_eps[2]
        lp = _log_p_vn_given_vnm1(vn, vnm1, self.dt, self.kappa, self.m, self.sigma + h)
        lm = _log_p_vn_given_vnm1(vn, vnm1, self.dt, self.kappa, self.m, self.sigma - h)
        grad[..., 2] = ((lp - lm) / (2.0 * h))
        hdiag[..., 2] = ((lp - 2.0 * logf0 + lm) / h ** 2)

        # Parameter 3: rho (transition independent of rho)
        grad[..., 3] = 0.0
        hdiag[..., 3] = 0.0

        # Return scalars if scalar input
        if logf0.ndim == 0:
            return float(logf0), grad.reshape(4), hdiag.reshape(4)

        return logf0, grad, hdiag

    def log_p_vn_dYn_given_vnm1(self, n, vn, vnm1):
        p = self.log_p_dYn_given_vnm1_vn(n, vnm1, vn)
        f = self.log_p_vn_given_vnm1(vn, vnm1)
        return p + f

    def log_p_vn_dYn_given_vnm1_with_derivs(self, n, vn, vnm1):
        p, dp, d2p = self.log_p_dYn_given_vnm1_vn_with_derivs(n, vnm1, vn)
        f, df, d2f = self.log_p_vn_given_vnm1_with_derivs(vn, vnm1)
        return p + f, dp + df, d2p + d2f

    def _compute_u_terms(self, kappa, m, sigma, rho):
        """
        Compute u-dependent BK ingredients at alpha(u)=i*s(u):
            alpha(u) = (rho*kappa/sigma) u + i * 0.5*(1-rho^2) u^2
        Returns dict with:
            nu, pref(u), lam(u), c_num(u), c_den
        where
            Φ(alpha) = pref(u) * exp( lam(u)*(a+b) ) * R(u,p)
        and p = sqrt(ab).
        """

        # BK Bessel order
        df = 4.0 * kappa * m / sigma ** 2
        nu = 0.5 * df - 1.0

        # alpha(u) = i * s(u), where
        # s(u)=0.5(1-rho^2)u^2 - i u rho kappa/sigma
        alpha = (rho * kappa / sigma) * self.u + 1j * (0.5 * (1.0 - rho ** 2) * self.u2)

        # gamma(alpha) = sqrt(kappa^2 - 2 sigma^2 i alpha)
        gamma = np.sqrt(kappa ** 2 - 2.0 * sigma ** 2 * 1j * alpha)

        ek = np.exp(-kappa * self.dt)
        one_minus_ek = -np.expm1(-kappa * self.dt)

        eg = np.exp(-gamma * self.dt)
        one_minus_eg = -np.expm1(-gamma * self.dt)

        # BK prefactor
        pref = (gamma * np.exp(-0.5 * (gamma - kappa) * self.dt) * one_minus_ek) / (kappa * one_minus_eg)

        # endpoint exponential factor: exp(lam(u)*(a+b))
        term_k = kappa * (1.0 + ek) / one_minus_ek
        term_g = gamma * (1.0 + eg) / one_minus_eg
        lam = (term_k - term_g) / sigma ** 2

        # Bessel arguments:
        # z_num(u,p) = p * c_num(u), z_den(p) = p * c_den
        c_num = (4.0 * gamma * np.exp(-0.5 * gamma * self.dt)) / (sigma ** 2 * one_minus_eg)
        c_den = (4.0 * kappa * np.exp(-0.5 * kappa * self.dt)) / (sigma ** 2 * one_minus_ek)

        return {
            "nu": nu,
            "pref": pref,
            "lam": lam,
            "c_num": c_num,
            "c_den": c_den,
        }

    def _compute_R_table(self, terms):
        """
        Precompute R(u_k,p_j) = I_nu(z_num)/I_nu(z_den) on (u,p)-grid.
        """
        nu = terms["nu"]
        c_num = terms["c_num"]  # (Ku,)
        c_den = terms["c_den"]  # scalar

        # z_den real positive
        z_den = c_den * self.p_grid
        den_scaled = ive(nu, z_den)  # exp(-z_den) I_nu(z_den)

        # z_num complex
        z_num = c_num[:, None] * self.p_grid[None, :]  # (Ku,P)
        num = iv(nu, z_num)

        # Stable ratio
        num_scaled = num * np.exp(-np.real(z_num))
        ratio = num_scaled * np.exp(np.real(z_num) - z_den[None, :]) / den_scaled[None, :]

        # At u=0, ratio should be exactly 1
        ratio[0] = 1.0 + 0.0j
        return np.ascontiguousarray(ratio.T)

    def _precompute_theta_derivative_tables(self):
        """
        Precompute (wrt theta=(kappa,m,sigma,rho)):

          dlogpref[ell,k]    = d/dtheta_ell log(pref_k)
          d2logpref[ell,k]   = d^2/dtheta_ell^2 log(pref_k)

          dlam[ell,k]        = d/dtheta_ell lam_k
          d2lam[ell,k]       = d^2/dtheta_ell^2 lam_k

          dlogR_table[ell,k,j]   = d/dtheta_ell log(R_{k,j})
          d2logR_table[ell,k,j]  = d^2/dtheta_ell^2 log(R_{k,j})

        using centered finite differences.
        """
        theta0 = np.array([self.kappa, self.m, self.sigma, self.rho])

        Ku = self.K + 1
        P = self.P

        dlogpref = np.empty((4, Ku), dtype=np.complex128)
        d2logpref = np.empty((4, Ku), dtype=np.complex128)

        dlam = np.empty((4, Ku), dtype=np.complex128)
        d2lam = np.empty((4, Ku), dtype=np.complex128)

        dlogR_table = np.empty((P, 4, Ku), dtype=np.complex128)
        d2logR_table = np.empty((P, 4, Ku), dtype=np.complex128)

        # Base arrays in high precision
        pref0 = self.pref
        lam0 = self.lam
        R0 = self.R_table

        for ell in trange(4, desc='Precomputing derivatives'):
            h = self.fd_eps[ell]

            tp = theta0.copy()
            tm = theta0.copy()
            tp[ell] += h
            tm[ell] -= h

            # Perturbed terms/tables
            terms_p = self._compute_u_terms(*tp)
            terms_m = self._compute_u_terms(*tm)

            pref_p = terms_p["pref"]
            pref_m = terms_m["pref"]
            lam_p = terms_p["lam"]
            lam_m = terms_m["lam"]

            Rp = self._compute_R_table(terms_p)
            Rm = self._compute_R_table(terms_m)

            # First and second derivatives of pref
            pref1 = (pref_p - pref_m) / (2.0 * h)
            pref2 = (pref_p - 2.0 * pref0 + pref_m) / (h * h)

            dlogpref[ell, :] = pref1 / pref0
            d2logpref[ell, :] = pref2 / pref0 - (pref1 / pref0) ** 2

            # First and second derivatives of lam
            dlam[ell, :] = (lam_p - lam_m) / (2.0 * h)
            d2lam[ell, :] = (lam_p - 2.0 * lam0 + lam_m) / (h * h)

            # First and second derivatives of R
            R1 = (Rp - Rm) / (2.0 * h)
            R2 = (Rp - 2.0 * R0 + Rm) / (h * h)

            dlogR_table[:, ell, :] = R1 / R0
            d2logR_table[:, ell, :] = R2 / R0 - (R1 / R0) ** 2

        self.dlogpref = dlogpref
        self.d2logpref = d2logpref
        self.dlam = dlam
        self.d2lam = d2lam
        self.dlogR_table = dlogR_table
        self.d2logR_table = d2logR_table


def paris_backward_indices_ar(
    x_prev: np.ndarray,          # shape (N,)
    w_prev: np.ndarray,          # shape (N,), must be normalized
    x_next: np.ndarray,          # shape (N,)  (children)
    f_density,                   # callable: f_density(x_next_vec, x_prev_vec) -> pdf vec
    eps_bar: float,              # bound: f_density(x'|x) <= eps_bar on relevant support
    N_tilde: int,                # number of backward draws per child
    rng: np.random.Generator,
    *,
    max_rounds: int = 10_000,    # max AR iterations per j (batch loop)
    fallback_exact: bool = True, # if True, exact O(N) sampling for remaining children
) -> np.ndarray:
    """
    OW Algorithm 3 (Appendix B.2) in your notation:
      sample J_{t+1}^{(i,j)} ~ proportional to w_prev[l] * f_theta(x_next[i] | x_prev[l])

    Returns:
      J : ndarray of shape (N, N_tilde), where J[i, j] is the sampled ancestor index for child i.
    """
    x_prev = np.asarray(x_prev, dtype=np.float64)
    w_prev = np.asarray(w_prev, dtype=np.float64)
    x_next = np.asarray(x_next, dtype=np.float64)

    N = x_prev.size
    if x_next.size != N:
        raise ValueError("x_prev and x_next must have the same length N")
    if w_prev.size != N:
        raise ValueError("w_prev must have length N")
    if N_tilde < 1:
        raise ValueError("N_tilde must be >= 1")
    if not np.isfinite(eps_bar) or eps_bar <= 0:
        raise ValueError("eps_bar must be finite and > 0")

    # normalize weights (robust)
    s = w_prev.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("w_prev must sum to a positive finite number")
    w_prev = w_prev / s

    J = np.empty((N, N_tilde), dtype=np.int64)

    # For each j, fill one column J[:, j]
    for j in range(N_tilde):
        # indices of children not yet accepted for this j
        L = np.arange(N, dtype=np.int64)

        rounds = 0
        while L.size > 0:
            rounds += 1
            if rounds > max_rounds:
                if not fallback_exact:
                    raise RuntimeError("Accept-reject did not finish; increase eps_bar or max_rounds.")
                # Exact fallback for the remaining children: O(|L|*N)
                for idx in L:
                    # unnormalized weights proportional to w_prev[l] * f(x_next[idx] | x_prev[l])
                    fvals = f_density(np.full(N, x_next[idx], dtype=np.float64), x_prev)  # (N,)
                    w = w_prev * fvals
                    tot = w.sum()
                    if tot <= 0 or not np.isfinite(tot):
                        # if this happens, something is wrong with f_density or numerical underflow
                        raise FloatingPointError("Fallback exact sampling failed: invalid weights.")
                    w = w / tot
                    J[idx, j] = rng.choice(N, p=w)
                L = L[:0]
                break

            # propose ancestors I_k ~ Categorical(w_prev) for each remaining child
            I = rng.choice(N, size=L.size, replace=True, p=w_prev)
            U = rng.random(L.size)

            # compute acceptance probabilities f(x_next[L] | x_prev[I]) / eps_bar
            fvals = f_density(x_next[L], x_prev[I])  # must return shape (len(L),)
            # If eps_bar is a true bound, probs <= 1 always. If not, you must fix eps_bar.
            probs = fvals / eps_bar

            # Guard: if eps_bar is violated, you can either raise or clip. Raising is safer.
            if np.any(probs > 1.0 + 1e-12):
                raise ValueError("eps_bar is not a valid bound (found f/eps_bar > 1).")

            accept = U <= np.minimum(probs, 1.0)
            # assign accepted indices
            if np.any(accept):
                J[L[accept], j] = I[accept]
            # keep only rejected children
            L = L[~accept]

    return J


if __name__ == '__main__':
    for seed in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
        init = InitialDistribution(dist='Dirac', hyper=[0.3**2, 0.])
        model1 = HestonModel(first_observed=1, init=init, dt=1., signature='1[1]_2d[1]', true_param=np.array([1, 0.4**2, 0.3, -0.5]), warn=False)
        model1.generate_observations(t_max=200000, inter_steps=250, seed=seed, verbose=1)

        #-----------------------------#
        #### Algorithm 2 of OW #######
        #-----------------------------#

        dY_seq = model1.observations[1:, 1]
        kappa, m, sigma, rho = 1., 0.16, 0.3, -0.5
        dt = 1.
        N = 1000
        N_tilde = 100
        eps_bar = np.exp(-minimize(lambda v: -_log_p_vn_given_vnm1(v[1], v[0], dt=1, kappa=1., m=0.16, sigma=0.3), x0=[0.1, 0.1], bounds=((1e-18, 5), (1e-18, 5))).fun)
        rng = np.random.default_rng(seed=20)
        K = len(dY_seq)
        proposal = Proposal(dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho, rng=rng)
        transition = HestonTransition(dY_seq=dY_seq, dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)

        # Warm-up compile
        _ = transition.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)
        _ = transition.log_p_dYn_given_vnm1_vn_with_derivs(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)

        # Initial particles
        vnm1 = np.repeat(0.3 ** 2, N)
        m_nm1 = np.zeros((N, 4))
        M_nm1 = np.zeros((N, 4))
        logweights_nm1 = np.repeat(-np.log(N), N)

        S = np.zeros((K, 4))
        Sigma_diag = np.zeros((K, 4))

        for n in trange(K, desc='Running particle filter'):
            ## Step A
            resample_logweights = logweights_nm1 + proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
            resample_weights = np.exp(resample_logweights - resample_logweights.max())
            resample_weights = resample_weights / resample_weights.sum()
            resample_indices = _systematic_resample(resample_weights, rng)

            ## Step B & C
            vn = proposal.sample_q_vn_given_vnm1_dYn(vnm1[resample_indices], dY_seq[n])
            logpdf = transition.log_p_vn_dYn_given_vnm1(n, vn, vnm1[resample_indices])
            logweights_n = logpdf - proposal.log_q_vn_given_vnm1_dYn(vn, vnm1[resample_indices], dY_seq[n]) - proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1[resample_indices])
            weights_n = np.exp(logweights_n - logweights_n.max())
            weights_n = weights_n / weights_n.sum()

            J_ij = paris_backward_indices_ar(vnm1, weights_nm1, vn, lambda vn, vnm1: np.exp(transition.log_p_vn_given_vnm1(vn, vnm1)), eps_bar, N_tilde, rng=rng)
            logpdf, deriv, hess = transition.log_p_vn_dYn_given_vnm1_with_derivs(n, vn[:, None], vnm1[J_ij])

            m_n = np.mean(m_nm1[J_ij] + deriv, axis=1)
            M_n = np.mean(M_nm1[J_ij] + 2 * m_nm1[J_ij] * deriv + deriv ** 2 + hess, axis=1)
            S[n] = weights_n @ m_n
            Sigma_diag[n] = S[n] ** 2 - weights_n @ M_n

            vnm1, m_nm1, M_nm1, logweights_nm1, weights_nm1 = vn, m_n, M_n, logweights_n, weights_n

        #-----------------------------#
        #### Algorithm 1 of PDS #######
        #-----------------------------#

        # dY_seq = model1.observations[1:, 1]
        # kappa, m, sigma, rho = 1., 0.16, 0.3, -0.5
        # dt = 1.
        # N = 1000
        # rng = np.random.default_rng(seed=20)
        # K = len(dY_seq)
        # proposal = Proposal(dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho, rng=rng)
        # transition = HestonTransition(dY_seq=dY_seq, dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)
        #
        # # Warm-up compile
        # _ = transition.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)
        #
        # # Initial particles
        # vnm1 = np.repeat(0.3 ** 2, N)
        # alpha_nm1 = np.zeros((N, 4))
        # beta_nm1 = np.zeros((N, 4))
        # logweights = np.repeat(-np.log(N), N)
        #
        # S = np.zeros((K, 4))
        # Sigma_diag = np.zeros((K, 4))
        #
        # for n in trange(K, desc='Running particle filter'):
        #     resample_logweights = logweights + proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
        #     resample_weights = np.exp(resample_logweights - resample_logweights.max())
        #     resample_weights = resample_weights / resample_weights.sum()
        #     resample_indices = _systematic_resample(resample_weights, rng)
        #     vnm1, alpha_nm1, beta_nm1 = vnm1[resample_indices], alpha_nm1[resample_indices], beta_nm1[resample_indices]
        #
        #     vn = proposal.sample_q_vn_given_vnm1_dYn(vnm1, dY_seq[n])
        #     logpdf, deriv, hess = transition.log_p_vn_dYn_given_vnm1(n, vn, vnm1)
        #     logweights = logpdf - proposal.log_q_vn_given_vnm1_dYn(vn, vnm1, dY_seq[n]) - proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
        #     weights = np.exp(logweights - logweights.max())
        #     weights = weights / weights.sum()
        #
        #     alpha_n = alpha_nm1 + deriv
        #     beta_n = beta_nm1 + hess
        #     S[n] = weights @ alpha_n
        #     Sigma_diag[n] = S[n] ** 2 - weights @ (alpha_n ** 2 + beta_n)
        #
        #     vnm1, alpha_nm1, beta_nm1 = vn, alpha_n, beta_n
        #
        # print("finished")


        #-----------------------------#
        #### Algorithm 2 of NFM #######
        #-----------------------------#

        dY_seq = model1.observations[1:, 1]
        kappa, m, sigma, rho = 1., 0.16, 0.3, -0.5
        dt = 1.
        N = 20000
        lamb = 0.99
        h2 = 1 - lamb ** 2
        rng = np.random.default_rng(seed=seed)
        K = len(dY_seq)
        proposal = Proposal(dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho, rng=rng)
        transition = HestonTransition(dY_seq=dY_seq, dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)
        # transition2 = HestonTransitionTrap(dY_seq=dY_seq, dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)

        # Warm-up compile
        _ = transition.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)
        # _ = transition2.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)

        # Initial particles
        vnm1 = np.repeat(0.3 ** 2, N)
        m_nm1 = np.zeros((N, 4))
        n_nm1 = np.zeros((N, 4))
        logweights = np.repeat(-np.log(N), N)
        weights = np.repeat(1 / N, N)

        V_n = np.zeros(4)
        B_nm1 = np.zeros(4)

        S = np.zeros((K + 1, 4))
        I_diag = np.zeros((K + 1, 4))

        for n in trange(K, desc='Running particle filter'):
            V_n += weights @ ((m_nm1 - S[n]) ** 2)

            resample_logweights = logweights + proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
            resample_weights = np.exp(resample_logweights - resample_logweights.max())
            resample_weights = resample_weights / resample_weights.sum()
            resample_indices = _systematic_resample(resample_weights, rng)
            vnm1, m_nm1, n_nm1 = vnm1[resample_indices], m_nm1[resample_indices], n_nm1[resample_indices]

            vn = proposal.sample_q_vn_given_vnm1_dYn(vnm1, dY_seq[n])
            logpdf, deriv, hess = transition.log_p_vn_dYn_given_vnm1(n, vn, vnm1)
            # logpdf2, deriv2, hess2 = transition2.log_p_vn_dYn_given_vnm1(n, vn, vnm1)
            logweights = logpdf - proposal.log_q_vn_given_vnm1_dYn(vn, vnm1, dY_seq[n]) - proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
            weights = np.exp(logweights - logweights.max())
            weights = weights / weights.sum()

            m_n = lamb * m_nm1 + (1 - lamb) * S[n] + deriv
            n_n = lamb * n_nm1 + (1 - lamb) * B_nm1 + hess

            S[n + 1] = weights @ m_n
            I_diag[n + 1] = S[n + 1] ** 2 - weights @ (m_n ** 2 + n_n) - h2 * V_n

            B_nm1 = weights @ n_n
            vnm1, m_nm1, n_nm1 = vn, m_n, n_n

        print("finished")

        np.savetxt('./S_N=20000_K=200000_seed={}_Ku=256_P=32768.txt'.format(seed), S)
        np.savetxt('./I_diag_N=20000_K=200000_seed={}_Ku=256_P=32768.txt'.format(seed), I_diag)

        # S_list, I_diag_list = [], []
        # score_list, score2_list, I_list = [], [], []
        # K = 200000

        # for seed in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
        #     S_list.append(np.loadtxt('./S_N=10000_K=200000_seed={}_Ku=256_P=32768.txt'.format(seed)))
        #     I_diag_list.append(np.loadtxt('./I_diag_N=10000_K=200000_seed={}_Ku=256_P=32768.txt'.format(seed)))
        #     I_list.append(I_diag_list[-1][1:] / np.arange(1, K + 1)[:, None])
        #     score_list.append(S_list[-1][1:] / np.arange(1, K + 1)[:, None])
        #     S_inc = np.diff(S_list[-1], axis=0)
        #     score2_list.append((S_inc ** 2).cumsum(axis=0) / np.arange(1, K + 1)[:, None])
        #
        # for i in range(12):
        #     plt.plot(1 / I_list[i][:, 0], color='C0', alpha=0.5, linewidth=1)
        #     plt.plot(1 / score2_list[i][:, 0], color='C1', alpha=0.5, linewidth=1)
        # plt.ylim(0, 40)
        #
        # for i in range(12):
        #     plt.plot(1 / I_list[i][:, 1], color='C0', alpha=0.5, linewidth=1)
        #     plt.plot(1 / score2_list[i][:, 1], color='C1', alpha=0.5, linewidth=1)
        # plt.ylim(0.03, 0.07)
        #
        # for i in range(12):
        #     plt.plot(1 / I_list[i][:, 2], color='C0', alpha=0.5, linewidth=1)
        #     plt.plot(1 / score2_list[i][:, 2], color='C1', alpha=0.5, linewidth=1)
        # plt.ylim(0, 1.5)
        #
        # for i in range(12):
        #     plt.plot(1 / I_list[i][:, 3], color='C0', alpha=0.5, linewidth=1)
        #     plt.plot(1 / score2_list[i][:, 3], color='C1', alpha=0.5, linewidth=1)
        # plt.ylim(3, 9)


    for seed in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
        init = InitialDistribution(dist='Dirac', hyper=[0.3**2, 0.])
        model1 = HestonModel(first_observed=1, init=init, dt=1., signature='1[1]_2d[1]', true_param=np.array([1, 0.4**2, 0.3, -0.5]), warn=False)
        model1.generate_observations(t_max=200000, inter_steps=250, seed=seed, verbose=1)

        #-----------------------------#
        #### Algorithm 1 of PDS #######
        #-----------------------------#

        # dY_seq = model1.observations[1:, 1]
        # kappa, m, sigma, rho = 1., 0.16, 0.3, -0.5
        # dt = 1.
        # N = 1000
        # rng = np.random.default_rng(seed=20)
        # K = len(dY_seq)
        # proposal = Proposal(dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho, rng=rng)
        # transition = HestonTransition(dY_seq=dY_seq, dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)
        #
        # # Warm-up compile
        # _ = transition.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)
        #
        # # Initial particles
        # vnm1 = np.repeat(0.3 ** 2, N)
        # alpha_nm1 = np.zeros((N, 4))
        # beta_nm1 = np.zeros((N, 4))
        # logweights = np.repeat(-np.log(N), N)
        #
        # S = np.zeros((K, 4))
        # Sigma_diag = np.zeros((K, 4))
        #
        # for n in trange(K, desc='Running particle filter'):
        #     resample_logweights = logweights + proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
        #     resample_weights = np.exp(resample_logweights - resample_logweights.max())
        #     resample_weights = resample_weights / resample_weights.sum()
        #     resample_indices = _systematic_resample(resample_weights, rng)
        #     vnm1, alpha_nm1, beta_nm1 = vnm1[resample_indices], alpha_nm1[resample_indices], beta_nm1[resample_indices]
        #
        #     vn = proposal.sample_q_vn_given_vnm1_dYn(vnm1, dY_seq[n])
        #     logpdf, deriv, hess = transition.log_p_vn_dYn_given_vnm1(n, vn, vnm1)
        #     logweights = logpdf - proposal.log_q_vn_given_vnm1_dYn(vn, vnm1, dY_seq[n]) - proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
        #     weights = np.exp(logweights - logweights.max())
        #     weights = weights / weights.sum()
        #
        #     alpha_n = alpha_nm1 + deriv
        #     beta_n = beta_nm1 + hess
        #     S[n] = weights @ alpha_n
        #     Sigma_diag[n] = S[n] ** 2 - weights @ (alpha_n ** 2 + beta_n)
        #
        #     vnm1, alpha_nm1, beta_nm1 = vn, alpha_n, beta_n
        #
        # print("finished")


        #-----------------------------#
        #### Algorithm 2 of NFM #######
        #-----------------------------#

        dY_seq = model1.observations[1:, 1]
        kappa, m, sigma, rho = 1., 0.16, 0.3, -0.5
        dt = 1.
        N = 10000
        lamb = 0.995
        h2 = 1 - lamb ** 2
        rng = np.random.default_rng(seed=seed)
        K = len(dY_seq)
        proposal = Proposal(dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho, rng=rng)
        transition = HestonTransition(dY_seq=dY_seq, dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)
        # transition2 = HestonTransitionTrap(dY_seq=dY_seq, dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)

        # Warm-up compile
        _ = transition.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)
        # _ = transition2.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)

        # Initial particles
        vnm1 = np.repeat(0.3 ** 2, N)
        m_nm1 = np.zeros((N, 4))
        n_nm1 = np.zeros((N, 4))
        logweights = np.repeat(-np.log(N), N)
        weights = np.repeat(1 / N, N)

        V_n = np.zeros(4)
        B_nm1 = np.zeros(4)

        S = np.zeros((K + 1, 4))
        I_diag = np.zeros((K + 1, 4))

        for n in trange(K, desc='Running particle filter'):
            V_n += weights @ ((m_nm1 - S[n]) ** 2)

            resample_logweights = logweights + proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
            resample_weights = np.exp(resample_logweights - resample_logweights.max())
            resample_weights = resample_weights / resample_weights.sum()
            resample_indices = _systematic_resample(resample_weights, rng)
            vnm1, m_nm1, n_nm1 = vnm1[resample_indices], m_nm1[resample_indices], n_nm1[resample_indices]

            vn = proposal.sample_q_vn_given_vnm1_dYn(vnm1, dY_seq[n])
            logpdf, deriv, hess = transition.log_p_vn_dYn_given_vnm1(n, vn, vnm1)
            # logpdf2, deriv2, hess2 = transition2.log_p_vn_dYn_given_vnm1(n, vn, vnm1)
            logweights = logpdf - proposal.log_q_vn_given_vnm1_dYn(vn, vnm1, dY_seq[n]) - proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
            weights = np.exp(logweights - logweights.max())
            weights = weights / weights.sum()

            m_n = lamb * m_nm1 + (1 - lamb) * S[n] + deriv
            n_n = lamb * n_nm1 + (1 - lamb) * B_nm1 + hess

            S[n + 1] = weights @ m_n
            I_diag[n + 1] = S[n + 1] ** 2 - weights @ (m_n ** 2 + n_n) - h2 * V_n

            B_nm1 = weights @ n_n
            vnm1, m_nm1, n_nm1 = vn, m_n, n_n

        print("finished")

        np.savetxt('./S_N=10000_K=200000_seed={}_Ku=256_P=32768_lam=0995.txt'.format(seed), S)
        np.savetxt('./I_diag_N=10000_K=200000_seed={}_Ku=256_P=32768_lam=0995.txt'.format(seed), I_diag)

        # S_list, I_diag_list = [], []
        # score_list, score2_list, I_list = [], [], []
        # K = 200000

        # for seed in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
        #     S_list.append(np.loadtxt('./S_N=10000_K=200000_seed={}_Ku=256_P=32768.txt'.format(seed)))
        #     I_diag_list.append(np.loadtxt('./I_diag_N=10000_K=200000_seed={}_Ku=256_P=32768.txt'.format(seed)))
        #     I_list.append(I_diag_list[-1][1:] / np.arange(1, K + 1)[:, None])
        #     score_list.append(S_list[-1][1:] / np.arange(1, K + 1)[:, None])
        #     S_inc = np.diff(S_list[-1], axis=0)
        #     score2_list.append((S_inc ** 2).cumsum(axis=0) / np.arange(1, K + 1)[:, None])
        #
        # for i in range(12):
        #     plt.plot(1 / I_list[i][:, 0], color='C0', alpha=0.5, linewidth=1)
        #     plt.plot(1 / score2_list[i][:, 0], color='C1', alpha=0.5, linewidth=1)
        # plt.ylim(0, 40)
        #
        # for i in range(12):
        #     plt.plot(1 / I_list[i][:, 1], color='C0', alpha=0.5, linewidth=1)
        #     plt.plot(1 / score2_list[i][:, 1], color='C1', alpha=0.5, linewidth=1)
        # plt.ylim(0.03, 0.07)
        #
        # for i in range(12):
        #     plt.plot(1 / I_list[i][:, 2], color='C0', alpha=0.5, linewidth=1)
        #     plt.plot(1 / score2_list[i][:, 2], color='C1', alpha=0.5, linewidth=1)
        # plt.ylim(0, 1.5)
        #
        # for i in range(12):
        #     plt.plot(1 / I_list[i][:, 3], color='C0', alpha=0.5, linewidth=1)
        #     plt.plot(1 / score2_list[i][:, 3], color='C1', alpha=0.5, linewidth=1)
        # plt.ylim(3, 9)


    # #
    # S = np.loadtxt('./S_N=10000_K=50000_seed=20_Ku=512_P=32768.txt')
    # I_diag = np.loadtxt('./I_diag_N=10000_K=50000_seed=20_Ku=512_P=32768.txt')
    # S_1000 = np.loadtxt('./S_N=10000_K=50000_seed=21_Ku=256_P=32768.txt')
    # I_diag_1000 = np.loadtxt('./I_diag_N=10000_K=50000_seed=21_Ku=256_P=32768.txt')
    # #
    # I = I_diag[1:] / np.arange(1, K + 1)[:, None]
    # score = S[1:] / np.arange(1, K + 1)[:, None]
    # S_inc = np.diff(S, axis=0)
    # score2 = (S_inc ** 2).cumsum(axis=0) / np.arange(1, K + 1)[:, None]
    #
    # I_1000 = I_diag_1000[1:] / np.arange(1, K + 1)[:, None]
    # score_1000 = S_1000[1:] / np.arange(1, K + 1)[:, None]
    # S_inc_1000 = np.diff(S_1000, axis=0)
    # score2_1000 = (S_inc_1000 ** 2).cumsum(axis=0) / np.arange(1, K + 1)[:, None]
    #
    # plt.plot(I)
    # plt.plot(I_1000)
    # plt.figure()
    # plt.plot(score2)
    # plt.plot(score2_1000)
    # plt.figure()
    # plt.plot(score)
    # plt.plot(score_1000)
    # plt.show()
