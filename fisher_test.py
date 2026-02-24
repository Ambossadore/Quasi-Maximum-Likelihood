import numpy as np
from scipy.special import iv, ive
from numba import njit, prange
from time import time
from tqdm import trange
import matplotlib.pyplot as plt


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
def _log_p_dYn_given_vnm1_vn(n, vnm1, vn, dt, kappa, m, sigma, rho, u, E, pref, lam, dlogpref, d2logpref, dlam, d2lam, R_table, dlogR_table, d2logR_table, p_min, p_max, logp0, inv_dlogp, P, log_pi):
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
    Erow = E[n]   # shape (Ku,), contiguous if E is C-order and shape (T,Ku)

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
            grad[i, 0] = g0; grad[i, 1] = g1; grad[i, 2] = g2; grad[i, 3] = g3
            hdiag[i, 0] = h0; hdiag[i, 1] = h1; hdiag[i, 2] = h2; hdiag[i, 3] = h3
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

        dmu0 = -(rho / sigma) * m * dt              # d/dkappa
        dmu1 = -(rho / sigma) * kappa * dt          # d/dm
        dmu2 = -(rho / (sigma * sigma)) * c         # d/dsigma
        dmu3 = c / sigma                            # d/drho

        d2mu0 = 0.0
        d2mu1 = 0.0
        d2mu2 = 2.0 * rho * c / (sigma * sigma * sigma)
        d2mu3 = 0.0

        # accumulators for Re(S0), Re(S1_l), Re(S2_l)
        S0_re = 0.0

        S1_0 = 0.0; S1_1 = 0.0; S1_2 = 0.0; S1_3 = 0.0
        S2_0 = 0.0; S2_1 = 0.0; S2_2 = 0.0; S2_3 = 0.0

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
        grad[i, 0] = g0; grad[i, 1] = g1; grad[i, 2] = g2; grad[i, 3] = g3
        hdiag[i, 0] = h0; hdiag[i, 1] = h1; hdiag[i, 2] = h2; hdiag[i, 3] = h3

    return ok, logp, grad, hdiag


def _log_p_vn_given_vnm1(vn, vnm1, dt, kappa, m, sigma):
    """
    Exact log density of the CIR transition:
        X_t | X_0 = x_prev
    for the SDE
        dX_t = kappa (m - X_t) dt + sigma sqrt(X_t) dW_t
    """
    vn = np.maximum(vn, 0.)
    ek = np.exp(-kappa * dt)
    one_minus_ek = -np.expm1(-kappa * dt)  # 1 - e^{-kappa dt}
    c = 2.0 * kappa / (sigma ** 2 * one_minus_ek)
    q = 2.0 * kappa * m / sigma ** 2 - 1.0  # Bessel order nu

    u = c * vnm1 * ek
    v = c * vn
    z = 2.0 * np.sqrt(u * v)

    # log I_q(z) using scaled Bessel ive: ive(q,z) = exp(-z) I_q(z) for z>0
    # Hence log I_q(z) = log(ive(q,z)) + z
    ive_q_z = ive(q, z)   # TODO: Stopped here
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
        integrand = lambda u: np.real(np.exp(-1j * u * dYn) * _heston_char(u, self.dt, vnm1, self.kappa, self.m, self.sigma, self.rho))
        return np.log(quad_vec(integrand, 0, np.inf)[0]) - np.log(np.pi)

    def log_q_vn_given_vnm1_dYn(self, vn, vnm1, dYn):
        mu_q = vnm1 + self.kappa * (self.m - vnm1) * self.dt + self.sigma * self.rho * dYn
        var_q = self.sigma ** 2 * (1.0 - self.rho ** 2) * vnm1 * self.dt
        return -0.5 * (np.log(2 * np.pi) + np.log(var_q) + (vn - mu_q) ** 2 / var_q)

    def sample_q_vn_given_vnm1_dYn(self, vnm1, dYn):
        mu_q = vnm1 + self.kappa * (self.m - vnm1) * self.dt + self.sigma * self.rho * dYn
        var_q = self.sigma ** 2 * (1.0 - self.rho ** 2) * vnm1 * self.dt
        return mu_q + np.sqrt(var_q) * self.rng.standard_normal(size=np.shape(mu_q))


class HestonTransition:
    def __init__(self, dY_seq, dt, kappa, m, sigma, rho, *, U=200.0, K=4096, p_min=1e-10, p_max=5.0, P=2048, p_grid=None, fd_eps=1e-4):
        # --- observed returns
        self.r_seq = np.asarray(dY_seq, dtype=np.float64)
        self.T = self.r_seq.size

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
        self.E = (w[None, :] * np.exp(-self.iu[None, :] * self.r_seq[:, None]))

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
        self.pref = base_terms["pref"]    # (Ku,)
        self.lam = base_terms["lam"]      # (Ku,)
        self.R_table = self._compute_R_table(base_terms)  # (Ku,P)

        # --- finite-difference steps for theta-derivatives
        self.fd_eps = fd_eps * np.array([self.kappa, self.m, self.sigma, np.abs(self.rho)])

        # --- precompute derivative tables wrt theta = (kappa, m, sigma, rho)
        self._precompute_theta_derivative_tables()

    def log_p_dYn_given_vnm1_vn(self, n, vnm1, vn, *, raise_on_fail=True):
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

        vnm1, vn = np.broadcast_arrays(np.asarray(vnm1, dtype=np.float64), np.asarray(vn, dtype=np.float64))
        orig_shape = vnm1.shape

        vnm1_flat = np.ascontiguousarray(vnm1.ravel())
        vn_flat = np.ascontiguousarray(vn.ravel())

        ok, logp_flat, grad_flat, hdiag_flat = _log_p_dYn_given_vnm1_vn(n, vnm1_flat, vn_flat, self.dt, self.kappa, self.m, self.sigma, self.rho, self.u, self.E, self.pref, self.lam, self.dlogpref, self.d2logpref, self.dlam, self.d2lam, self.R_table, self.dlogR_table, self.d2logR_table, self._p_min, self._p_max, self._logp0, self._inv_dlogp, self.P, np.log(np.pi))

        if raise_on_fail and np.any(ok == 0):
            bad = int(np.sum(ok == 0))
            raise FloatingPointError(
                f"Batch kernel failed for {bad} pair(s) "
                "(non-positive/invalid Fourier inversion result or invalid a,b)."
            )

        logp = logp_flat.reshape(orig_shape)
        grad = grad_flat.reshape(orig_shape + (4,))
        hdiag = hdiag_flat.reshape(orig_shape + (4,))

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
        p, dp, d2p = self.log_p_dYn_given_vnm1_vn(n, vnm1, vn)
        f, df, d2f = self.log_p_vn_given_vnm1(vn, vnm1)
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
        c_num = terms["c_num"]     # (Ku,)
        c_den = terms["c_den"]     # scalar

        # z_den real positive
        z_den = c_den * self.p_grid
        den_scaled = ive(nu, z_den)                # exp(-z_den) I_nu(z_den)

        # z_num complex
        z_num = c_num[:, None] * self.p_grid[None, :]        # (Ku,P)
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


#-----------------------------#
#### Algorithm 1 of PDS #######
#-----------------------------#

dY_seq = ...
kappa, m, sigma, rho = 1., 0.16, 0.3, -0.5
dt = 1.
rng = np.random.default_rng()
K = len(dY_seq)
proposal = Proposal(dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)
transition = HestonTransition(dY_seq=dY_seq, dt=dt, kappa=kappa, m=m, sigma=sigma, rho=rho)

# Warm-up compile
_ = transition.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)

# Initial particles
vnm1 = 0.3**2
alpha_nm1 = 0
beta_nm1 = 0
logweights = np.repeat(-np.log(N), N)

S = np.zeros((K, 4))
Sigma_diag = np.zeros((K, 4))

for n in range(len(dY_seq)):
    resample_logweights = logweights + proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
    resample_weights = np.exp(log_weights)
    resample_weights = resample_weights / resample_weights.sum()
    resample_indices = _systematic_resample(resample_weights, rng)
    vnm1, alpha_nm1, beta_nm1 = vnm1[resample_indices], alpha_nm1[resample_indices], beta_nm1[resample_indices]

    vn = proposal.sample_q_vn_given_vnm1_dYn(vnm1, dY_seq[n])
    logpdf, deriv, hess = transition.log_p_vn_dYn_given_vnm1(n, vn, vnm1)
    logweights = logpdf - proposal.log_q_vn_given_vnm1_dYn(vn, vnm1, dY_seq[n]) - proposal.log_q_dYn_given_vnm1(dY_seq[n], vnm1)
    weights = np.exp(logweights)
    weights = weights / weights.sum()
    alpha_n = alpha_nm1 + deriv
    beta_n = beta_nm1 + hess
    S[n] = weights @ alpha_n
    Sigma_diag[n] = S[n] ** 2 - weights @ (alpha_n ** 2 + beta_n)

    v_nm1, alpha_nm1, beta_nm1 = v_n, alpha_n, beta_n





# def pds_algorithm1_diag(
#     r_seq,
#     N,
#     *,
#     # Initial particles:
#     # either provide x0_particles directly, or provide sample_x0 callback
#     x0_particles=None,
#     sample_x0=None,
#     # Proposal q(x_n | r_n, x_{n-1}) sampler
#     sample_qx,
#     # Callbacks returning (log, grad, hess_diag), all wrt theta
#     # Shapes:
#     #   log_*:      (M,)
#     #   grad_*:     (M, d)
#     #   hessdiag_*: (M, d)
#     log_h_and_derivs,
#     log_qx_and_derivs,
#     log_qr_and_derivs,
#     # Optional: if proposal derivatives are ignored, set these to False
#     include_qx_derivs=True,
#     include_qr_derivs=True,
#     # RNG
#     rng=None,
#     # Return extra internals
#     return_particle_state=False,
# ):
#     """
#     PDS Algorithm 1 (path-space, O(N)) adapted to joint density h_theta.
#
#     Model increment target density:
#         h_theta(x_n, r_n | x_{n-1})
#
#     Proposal/auxiliary terms:
#         q_theta(x_n | r_n, x_{n-1})
#         q_theta(r_n | x_{n-1})
#
#     Increment log-weight (and complete-data score/Hessian increment):
#         ell_n = log h - log qx - log qr
#
#     The algorithm computes:
#       - particle filter / APF approximation
#       - score estimate S_n = E[ A_n | r_{1:n} ] approx sum_i w_n^i A_n^i
#       - diagonal observed information estimate
#             I_n,diag = S_n^2 - E[ B_n + A_n^2 | r_{1:n} ]_diag
#         where
#             A_n = sum_t ∇ log increment_t
#             B_n = sum_t diag(∇^2 log increment_t)
#
#     Parameters
#     ----------
#     r_seq : array-like, shape (T,)
#         Observed returns r_1,...,r_T.
#     N : int
#         Number of particles.
#     x0_particles : array-like, optional
#         Initial latent particles x_0^{(i)}. If None, `sample_x0` must be provided.
#     sample_x0 : callable, optional
#         sample_x0(N, rng) -> array shape (N,)
#     sample_qx : callable
#         sample_qx(x_prev, r_n, rng) -> x_new, shape (N,)
#     log_h_and_derivs : callable
#         log_h_and_derivs(x_new, r_n, x_prev) -> (logh, dlogh, d2logh_diag)
#     log_qx_and_derivs : callable
#         log_qx_and_derivs(x_new, x_prev, r_n) -> (logqx, dlogqx, d2logqx_diag)
#     log_qr_and_derivs : callable
#         log_qr_and_derivs(r_n, x_prev) -> (logqr, dlogqr, d2logqr_diag)
#
#     Returns
#     -------
#     out : dict
#         Keys:
#           - "score":    array, shape (T, d)
#           - "info_diag": array, shape (T, d)
#           - "weights":  final weights, shape (N,)
#           - "x_particles": final particles, shape (N,)
#           - "A_particles": final path score accumulators, shape (N,d)
#           - "Bdiag_particles": final path Hessian-diag accumulators, shape (N,d)
#         plus, if return_particle_state=False, only the first two are guaranteed.
#     """
#     if rng is None:
#         rng = np.random.default_rng()
#
#     r_seq = np.asarray(r_seq, dtype=np.float64)
#     T = r_seq.size
#     if T == 0:
#         raise ValueError("r_seq must be non-empty")
#     if N <= 0:
#         raise ValueError("N must be positive")
#
#     # ----- Initialize x_0 particles
#     if x0_particles is None:
#         if sample_x0 is None:
#             raise ValueError("Provide either x0_particles or sample_x0.")
#         x_prev = np.asarray(sample_x0(N, rng), dtype=np.float64)
#     else:
#         x_prev = np.asarray(x0_particles, dtype=np.float64)
#         if x_prev.shape != (N,):
#             raise ValueError(f"x0_particles must have shape ({N},)")
#
#     # Uniform initial weights
#     w_prev = np.full(N, 1.0 / N, dtype=np.float64)
#
#     # Need parameter dimension d from a probe call on the first observation
#     # (using x_prev as both x_new/x_prev just to infer shape; values don't matter)
#     r0 = float(r_seq[0])
#     logh0, dlogh0, d2h0 = log_h_and_derivs(x_prev, r0, x_prev)
#     dlogh0 = np.asarray(dlogh0)
#     d2h0 = np.asarray(d2h0)
#
#     if dlogh0.ndim != 2 or dlogh0.shape[0] != N:
#         raise ValueError("log_h_and_derivs must return grad shape (N,d)")
#     if d2h0.shape != dlogh0.shape:
#         raise ValueError("log_h_and_derivs must return hess_diag shape (N,d)")
#
#     d = dlogh0.shape[1]
#
#     # Path accumulators (complete-data score and diagonal Hessian sums)
#     A_prev = np.zeros((N, d), dtype=np.float64)      # A_n^i
#     Bdiag_prev = np.zeros((N, d), dtype=np.float64)  # B_n^i
#
#     # Output trajectories
#     score_hist = np.empty((T, d), dtype=np.float64)
#     info_diag_hist = np.empty((T, d), dtype=np.float64)
#
#     # ----- Main loop over observations r_1,...,r_T
#     for t in range(T):
#         r_n = float(r_seq[t])
#
#         # 1) First-stage weights ~ w_{n-1}^i * q(r_n | x_{n-1}^i)
#         logqr_prev, dlogqr_prev, d2logqr_prev = log_qr_and_derivs(r_n, x_prev)
#
#         logqr_prev = np.asarray(logqr_prev, dtype=np.float64)
#         dlogqr_prev = np.asarray(dlogqr_prev, dtype=np.float64)
#         d2logqr_prev = np.asarray(d2logqr_prev, dtype=np.float64)
#
#         if logqr_prev.shape != (N,):
#             raise ValueError("log_qr_and_derivs must return logqr shape (N,)")
#         if dlogqr_prev.shape != (N, d) or d2logqr_prev.shape != (N, d):
#             raise ValueError("log_qr_and_derivs grad/hess_diag must have shape (N,d)")
#
#         # log w_{n-1} may contain zeros -> use -inf safely
#         logw_prev = np.where(w_prev > 0.0, np.log(w_prev), -np.inf)
#         log_first_stage = logw_prev + logqr_prev
#         w_first, _ = _normalize_logweights(log_first_stage)
#
#         # 2) Resample ancestors according to first-stage weights
#         anc = _systematic_resample(w_first, rng)
#
#         x_anc = x_prev[anc]
#         A_anc = A_prev[anc]
#         Bdiag_anc = Bdiag_prev[anc]
#
#         # Need proposal-qr derivatives at chosen ancestors
#         logqr_anc = logqr_prev[anc]
#         dlogqr_anc = dlogqr_prev[anc]
#         d2logqr_anc = d2logqr_prev[anc]
#
#         # 3) Propagate x_n^i ~ q(x_n | r_n, x_{n-1}^{anc_i})
#         x_new = np.asarray(sample_qx(x_anc, r_n, rng), dtype=np.float64)
#         if x_new.shape != (N,):
#             raise ValueError("sample_qx must return shape (N,)")
#
#         # 4) Evaluate target and proposal densities and derivatives
#         logh, dlogh, d2logh = log_h_and_derivs(x_new, r_n, x_anc)
#         logqx, dlogqx, d2logqx = log_qx_and_derivs(x_new, x_anc, r_n)
#
#         logh = np.asarray(logh, dtype=np.float64)
#         dlogh = np.asarray(dlogh, dtype=np.float64)
#         d2logh = np.asarray(d2logh, dtype=np.float64)
#
#         logqx = np.asarray(logqx, dtype=np.float64)
#         dlogqx = np.asarray(dlogqx, dtype=np.float64)
#         d2logqx = np.asarray(d2logqx, dtype=np.float64)
#
#         if logh.shape != (N,) or logqx.shape != (N,):
#             raise ValueError("log_h/log_qx must return shape (N,)")
#         if dlogh.shape != (N, d) or d2logh.shape != (N, d):
#             raise ValueError("log_h grad/hess_diag must have shape (N,d)")
#         if dlogqx.shape != (N, d) or d2logqx.shape != (N, d):
#             raise ValueError("log_qx grad/hess_diag must have shape (N,d)")
#
#         # 5) Increment log-weight and complete-data derivative increments
#         log_inc = logh - logqx - logqr_anc
#
#         if include_qx_derivs:
#             dlog_inc = dlogh - dlogqx
#             d2log_inc = d2logh - d2logqx
#         else:
#             dlog_inc = dlogh.copy()
#             d2log_inc = d2logh.copy()
#
#         if include_qr_derivs:
#             dlog_inc -= dlogqr_anc
#             d2log_inc -= d2logqr_anc
#
#         # 6) Path-space recursion (PDS Algorithm 1)
#         A_new = A_anc + dlog_inc
#         Bdiag_new = Bdiag_anc + d2log_inc
#
#         # 7) Second-stage weights
#         w_new, _ = _normalize_logweights(log_inc)
#
#         # 8) Score and diagonal observed information estimate
#         # Score: S_n = sum_i w_i A_i
#         score_n = np.sum(w_new[:, None] * A_new, axis=0)
#
#         # Diagonal observed information:
#         # I_diag = S^2 - E[B_diag + A^2 | y]
#         # (all operations elementwise in parameter dimension)
#         exp_term = np.sum(w_new[:, None] * (Bdiag_new + A_new * A_new), axis=0)
#         info_diag_n = score_n * score_n - exp_term
#
#         score_hist[t] = score_n
#         info_diag_hist[t] = info_diag_n
#
#         # 9) Advance
#         x_prev = x_new
#         w_prev = w_new
#         A_prev = A_new
#         Bdiag_prev = Bdiag_new
#
#     out = {
#         "score": score_hist,
#         "info_diag": info_diag_hist,
#     }
#
#     if return_particle_state:
#         out.update({
#             "weights": w_prev,
#             "x_particles": x_prev,
#             "A_particles": A_prev,
#             "Bdiag_particles": Bdiag_prev,
#         })
#
#     return out
#


eng = HestonTransition(dY_seq=np.linspace(-1, 1, 100), dt=1, kappa=1, m=0.3 ** 2, sigma=0.3, rho=-0.5)

# Warm-up compile (small batch is enough)
_ = eng.log_p_dYn_given_vnm1_vn(n=0, vnm1=np.array([0.03, 0.04]), vn=np.array([0.05, 0.06]), raise_on_fail=False)

tic = time()
f = eng.log_p_vn_dYn_given_vnm1(74, np.linspace(0.3 ** 2, 0.4 ** 2, 10000), np.linspace(0.4 ** 2, 0.5 ** 2, 10000))
toc = time()
print(toc - tic)
print(f[1].shape)
