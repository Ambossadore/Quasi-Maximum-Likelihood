from main import *
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from scipy.integrate import quad
from scipy.special import iv, ive
from scipy.interpolate import CubicSpline

from scipy.special import iv, ive
from tqdm import trange

import numpy as np
from numba import njit


@njit(cache=True)
def _logpdf_grad_hessdiag_kernel(
    n, a, b,
    # theta
    dt, kappa, m, sigma, rho,
    # precomputed grids / tables
    u, E,                      # u: (Ku,), E: (T,Ku)
    pref, lam,                 # (Ku,), (Ku,)
    dlogpref, d2logpref,       # (4,Ku), (4,Ku)
    dlam, d2lam,               # (4,Ku), (4,Ku)
    R_table,                   # (P,Ku)
    dlogR_table, d2logR_table, # (P,4,Ku), (P,4,Ku)
    # interpolation cache (log-grid)
    p_min, p_max, logp0, inv_dlogp, P
):
    """
    Returns:
      ok_flag (1 if success, 0 if numerical issue),
      log_pdf,
      grad (4,),
      hess_diag (4,)
    """
    grad = np.zeros(4, dtype=np.float64)
    hdiag = np.zeros(4, dtype=np.float64)

    # Basic checks (Numba-friendly)
    if a <= 0.0 or b <= 0.0:
        return 0, np.nan, grad, hdiag

    # Endpoint combinations
    s = a + b
    d = b - a
    p = np.sqrt(a * b)

    # Interpolation index j and weight t for p-grid (uniform in log p)
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

    # mu_const and its first/second diagonal theta-derivatives
    # mu_const = (rho/sigma) * (d - kappa*m*dt)
    c = d - kappa * m * dt
    mu_const = (rho / sigma) * c

    dmu = np.empty(4, dtype=np.float64)
    dmu[0] = -(rho / sigma) * m * dt
    dmu[1] = -(rho / sigma) * kappa * dt
    dmu[2] = -(rho / (sigma * sigma)) * c
    dmu[3] = c / sigma

    d2mu = np.zeros(4, dtype=np.float64)
    d2mu[2] = 2.0 * rho * c / (sigma * sigma * sigma)  # only sigma diagonal is nonzero

    # Accumulators for Re(S0), Re(S1_l), Re(S2_l)
    S0_re = 0.0
    S1_re = np.zeros(4, dtype=np.float64)
    S2_re = np.zeros(4, dtype=np.float64)

    Erow = E[n]      # contiguous if E is (T,Ku)
    Ku = u.shape[0]

    one_minus_t = 1.0 - t

    # Main fused loop over Fourier nodes
    for k in range(Ku):
        # Interpolate R and derivative tables on the fly
        Rk = one_minus_t * R_table[j, k] + t * R_table[j + 1, k]

        # Build psi_k = pref_k * exp(lam_k*s + i*u_k*mu_const) * Rk
        exp_arg = lam[k] * s + (1j * u[k] * mu_const)
        psi_k = pref[k] * np.exp(exp_arg) * Rk

        z = psi_k * Erow[k]   # contributes to Fourier sums
        zr = z.real
        S0_re += zr

        uk = u[k]

        # Loop over theta components
        for ell in range(4):
            dlogR = one_minus_t * dlogR_table[j, ell, k] + t * dlogR_table[j + 1, ell, k]
            d2logR = one_minus_t * d2logR_table[j, ell, k] + t * d2logR_table[j + 1, ell, k]

            # G = d log Psi
            G = (1j * uk * dmu[ell]) + dlogpref[ell, k] + s * dlam[ell, k] + dlogR

            # H = d^2 log Psi (diagonal only)
            H = (1j * uk * d2mu[ell]) + d2logpref[ell, k] + s * d2lam[ell, k] + d2logR

            S1_re[ell] += (z * G).real
            S2_re[ell] += (z * (H + G * G)).real

    # Density must be positive (up to numerical inversion error)
    if S0_re <= 0.0 or not np.isfinite(S0_re):
        return 0, np.nan, grad, hdiag

    # log p = log(Re(S0)) - log(pi)
    log_pdf = np.log(S0_re) - np.log(np.pi)

    # grad = Re(S1)/Re(S0)
    for ell in range(4):
        grad[ell] = S1_re[ell] / S0_re

    # hdiag = Re(S2)/Re(S0) - grad^2
    for ell in range(4):
        hdiag[ell] = S2_re[ell] / S0_re - grad[ell] * grad[ell]

    return 1, log_pdf, grad, hdiag


class BridgeReturnPDF:
    """
    Exact (up to 1D quadrature) p(r_n | v_{n-1}=a, v_n=b) using Broadie–Kaya CF,
    optimized for repeated evaluations at observed r_n.

    Added methods:
      - deriv_log_pdf(n, a, b)             -> gradient wrt theta=(kappa,m,sigma,rho)
      - log_pdf_and_deriv(n, a, b)         -> (log_pdf, grad)
      - deriv2_log_pdf_diag(n, a, b)       -> diagonal 2nd derivatives wrt theta
      - log_pdf_deriv_hessdiag(n, a, b)    -> (log_pdf, grad, hess_diag)

    All theta-derivative tables are precomputed in __init__ via centered finite differences
    on the CF building blocks.
    """

    def __init__(self, r_seq, dt, kappa, m, sigma, rho, *, U=200.0, K=4096, p_min=1e-10, p_max=5.0, P=2048, p_grid=None, fd_eps=1e-4):
        # --- observed returns
        self.r_seq = np.asarray(r_seq, dtype=np.float64)
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

    def log_pdf_deriv_hessdiag(self, n, a, b):
        ok, logp, grad, hdiag = _logpdf_grad_hessdiag_kernel(
            int(n), float(a), float(b),
            self.dt, self.kappa, self.m, self.sigma, self.rho,
            self.u, self.E,
            self.pref, self.lam,
            self.dlogpref, self.d2logpref,
            self.dlam, self.d2lam,
            self.R_table,
            self.dlogR_table, self.d2logR_table,
            self._p_min, self._p_max, self._logp0, self._inv_dlogp, self.P
        )
        if ok == 0:
            raise FloatingPointError(
                "Kernel returned non-positive/invalid density. "
                "Increase U/K or improve p-grid range/resolution."
            )
        return float(logp), np.asarray(grad), np.asarray(hdiag)

    def pdf_deriv_deriv2(self, n, a, b):
        """
        Single runtime pass to compute pdf and gradient, diagonal Hessian of logpdf
        of log p(r_n | a,b).
        """
        n = int(n)
        if n < 0 or n >= self.T:
            raise IndexError(f"n must be in [0, {self.T-1}]")

        a = float(a)
        b = float(b)
        if a <= 0.0 or b <= 0.0:
            raise ValueError("log-derivative methods require a>0 and b>0")

        # Endpoint combinations
        s = a + b
        d = b - a
        p = np.sqrt(a * b)

        # Interpolate R and its precomputed log-derivatives at p
        Rk = self._interp_first_axis(self.R_table, p)                   # (Ku,)
        dlogR = self._interp_first_axis(self.dlogR_table, p)            # (4,Ku)
        d2logR = self._interp_first_axis(self.d2logR_table, p)      # (4,Ku)

        # Build Psi_k = phase * pref * exp(lam*s) * R
        mu_const = (self.rho / self.sigma) * (d - self.kappa * self.m * self.dt)
        phase = np.exp(self.iu * mu_const)                             # (Ku,)
        exp_lam_s = np.exp(self.lam * s)                               # (Ku,)
        psi = (phase * self.pref * exp_lam_s * Rk)   # (Ku,)

        # Base Fourier sum (1/pi drops out of log-derivatives)
        Ecol = self.E[n]                                            # (Ku,)
        S0 = np.dot(psi, Ecol)                                         # complex
        p_raw = np.real(S0)

        if p_raw <= 0.0:
            raise FloatingPointError(
                "Non-positive Fourier inversion result for density. "
                "Increase U/K and/or improve p-grid range/resolution."
            )

        pdf = p_raw / np.pi

        # First log-derivative of phase: d log phase = i*u * d(mu_const)
        dmu = self._d_mu_const_dtheta(d)                               # (4,)
        dlog_phase = self.iu[None, :] * dmu[:, None]                   # (4,Ku)

        # G = d log Psi = d log phase + d log pref + s*d lam + d log R
        G = dlog_phase + self.dlogpref + s * self.dlam + dlogR         # (4,Ku)

        # S1_l = sum_k psi_k E_k G_{l,k}
        S1 = (G * psi[None, :]) @ Ecol                                 # (4,) complex
        num1 = np.real(S1)                                             # (4,)
        grad = num1 / p_raw                                            # (4,)

        # Second log-derivative of phase diagonal: d2 log phase = i*u * d2(mu_const)
        d2mu = self._d2_mu_const_diag_dtheta2(d)                       # (4,)
        d2log_phase = self.iu[None, :] * d2mu[:, None]                 # (4,Ku)

        # Hdiag = d2 log Psi (diagonal only)
        Hdiag = d2log_phase + self.d2logpref + s * self.d2lam + d2logR # (4,Ku)

        # S2_l = sum_k psi_k E_k [ H_l + G_l^2 ]
        S2 = ((Hdiag + G * G) * psi[None, :]) @ Ecol                   # (4,) complex
        num2 = np.real(S2)                                             # (4,)

        # d2 log p = (Re S2)/p_raw - ((Re S1)/p_raw)^2
        hess_diag = num2 / p_raw - grad ** 2

        return (
            pdf,
            grad,
            hess_diag,
        )

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

    def _interp_first_axis(self, table, p_val):
        """
        Fast interpolation along axis 0 (the p-grid axis).

        Assumes table has shape (P, ...), e.g.
          - (P, Ku)
          - (P, 4, Ku)

        Returns shape (...), i.e. the table interpolated at scalar p_val.

        Uses direct index arithmetic when interp_logp=True (geomspace p-grid).
        Falls back to searchsorted only if needed.
        """
        p = float(p_val)

        # Clamp to grid range (cheap and avoids extrapolation)
        if p <= self._p_min:
            return table[0]
        if p >= self._p_max:
            return table[self.P - 1]

        # Fast path: uniform in log(p) (typical case with geomspace)
        x = np.log(p)
        y = (x - self._logp0) * self._inv_dlogp  # fractional index
        j = int(y)

        # guard against roundoff at boundaries
        if j < 0:
            j = 0
            t = 0.0
        elif j >= self.P - 1:
            j = self.P - 2
            t = 1.0
        else:
            t = y - j

        left = table[j]  # contiguous slice if table is C-order and shape (P,...)
        right = table[j + 1]
        return (1.0 - t) * left + t * right

    def _d_mu_const_dtheta(self, d):
        """
        mu_const = (rho/sigma) * (d - kappa*m*dt)
        First derivatives wrt theta=(kappa,m,sigma,rho).
        """
        dt = self.dt
        kappa = self.kappa
        m = self.m
        sigma = self.sigma
        rho = self.rho

        c = d - kappa * m * dt

        d_kappa = -(rho / sigma) * m * dt
        d_m = -(rho / sigma) * kappa * dt
        d_sigma = -(rho / (sigma * sigma)) * c
        d_rho = c / sigma

        return np.array([d_kappa, d_m, d_sigma, d_rho])

    def _d2_mu_const_diag_dtheta2(self, d):
        """
        Diagonal second derivatives wrt theta=(kappa,m,sigma,rho) of
            mu_const = (rho/sigma) * (d - kappa*m*dt)
        """
        dt = self.dt
        kappa = self.kappa
        m = self.m
        sigma = self.sigma
        rho = self.rho

        c = d - kappa * m * dt

        d2_kappa = 0.0
        d2_m = 0.0
        d2_sigma = 2.0 * rho * c / (sigma ** 3)
        d2_rho = 0.0

        return np.array([d2_kappa, d2_m, d2_sigma, d2_rho])

eng = BridgeReturnPDF(r_seq=np.linspace(-1, 1, 100), dt=1, kappa=1, m=0.3**2, sigma=0.3, rho=-0.5)
eng.pdf_deriv_deriv2(74, 0.3**2, 0.4**2)

tic = time()
f = [eng.pdf_deriv_deriv2(74, 0.3**2, 0.4**2) for i in range(10000)]
toc = time()
print(toc - tic)


def heston_char(u, dt, v, param):
    kappa, m, sigma, rho = param
    B = kappa - rho * sigma * 1j * u
    D = np.sqrt(B ** 2 + sigma ** 2 * u ** 2)
    lnG = np.log((np.exp(-D * dt) + 1) / 2 - B * (np.exp(-D * dt) - 1) / (2 * D))
    Psi0 = kappa * m / sigma ** 2 * ((B - D) * dt - 2 * lnG)
    Psi1 = u ** 2 * (np.exp(-D * dt) - 1) / (D * (np.exp(-D * dt) + 1) - B * (np.exp(-D * dt) - 1))  # u ** 2 * (sp.exp(-D * T) - 1) / (D * (sp.exp(-D * T) + 1) - B * (sp.exp(-D * T) - 1))
    return np.exp(Psi0 + Psi1 * v)


def p_dYn_given_vnm1(x, dt, v, param):
    integrand = lambda u: np.real(np.exp(-1j * u * x) * heston_char(u, dt, v, param))
    return 1 / np.pi * quad_vec(integrand, 0, np.inf)[0]



def _phi_intvar_cond_endpoints(alpha, v0, vT, dt, kappa, m, sigma):
    """
    Broadie–Kaya (2006) Eq (13):
      Φ(α) = E[ exp(i α ∫ v_s ds) | v(0)=v0, v(dt)=vT ]
    for CIR: dv = kappa(m - v)dt + sigma*sqrt(v)dW.

    alpha can be complex scalar or complex numpy array.
    Returns complex numpy array broadcast to alpha.
    """
    alpha = np.asarray(alpha, dtype=np.complex128)

    # degrees-of-freedom parameter and Bessel order
    d = 4.0 * kappa * m / (sigma * sigma)
    nu = 0.5 * d - 1.0

    # stable expm1 helpers
    ek = np.exp(-kappa * dt)
    one_minus_ek = -np.expm1(-kappa * dt)  # 1 - e^{-kappa dt}

    # gamma(alpha) = sqrt(kappa^2 - 2 sigma^2 i alpha)
    gamma = np.sqrt(kappa * kappa - 2.0 * (sigma * sigma) * 1j * alpha)
    eg = np.exp(-gamma * dt)
    one_minus_eg = -np.expm1(-gamma * dt)  # 1 - e^{-gamma dt}

    # prefactor
    pref = (gamma * np.exp(-0.5 * (gamma - kappa) * dt) * one_minus_ek) / (kappa * one_minus_eg)

    # exponential term
    term_k = kappa * (1.0 + ek) / one_minus_ek
    term_g = gamma * (1.0 + eg) / one_minus_eg
    expo = np.exp((v0 + vT) / (sigma * sigma) * (term_k - term_g))

    # Bessel ratio arguments
    sqrt_v0vT = np.sqrt(v0 * vT)
    z_num = sqrt_v0vT * (4.0 * gamma * np.exp(-0.5 * gamma * dt)) / ((sigma * sigma) * one_minus_eg)  # complex
    z_den = sqrt_v0vT * (4.0 * kappa * np.exp(-0.5 * kappa * dt)) / ((sigma * sigma) * one_minus_ek)  # real > 0

    # Stabilize denominator with ive (scaled Iν for real positive z_den)
    den_scaled = ive(nu, z_den)  # = exp(-z_den) * Iν(z_den)

    # Numerator can be complex; stabilize magnitude by removing exp(Re(z_num))
    num_scaled = iv(nu, z_num) * np.exp(-np.real(z_num))

    # Iν(z_num)/Iν(z_den) = [num_scaled * exp(Re(z_num))] / [exp(z_den)*den_scaled]
    #                    = num_scaled * exp(Re(z_num) - z_den) / den_scaled
    ratio = num_scaled * np.exp(np.real(z_num) - z_den) / den_scaled

    return pref * expo * ratio


def _psi_dY_cond_endpoints(u, v0, vT, dt, kappa, m, sigma, rho):
    """
    Ψ(u) = E[ exp(i u ΔY) | v(0)=v0, v(dt)=vT ]  (scalar u)
    """
    u = float(u)

    # constant mean part (from shared W^(1) term)
    mu_const = (rho / sigma) * (vT - v0 - kappa * m * dt)

    # s(u) such that E[exp(iuΔY)|v0,vT] = exp(iu*mu_const) * E[exp(-s(u) I)|v0,vT]
    s = 0.5 * (1.0 - rho * rho) * (u * u) - 1j * u * (rho * kappa / sigma)

    # Laplace = Φ(i s)
    alpha = 1j * s
    phi_I = _phi_intvar_cond_endpoints(alpha, v0, vT, dt, kappa, m, sigma)

    return np.exp(1j * u * mu_const) * phi_I


def bridge_return_pdf_quad(
    r, v0, vT, dt, kappa, m, sigma, rho,
    mode="weighted_inf",
    U=200.0,
    epsabs=1e-8,
    epsrel=1e-6,
    limit=200,
):
    """
    Compute p(r | v0, vT) using 1D Fourier inversion with scipy.integrate.quad.

    mode:
      - "finite":      integrate on [0, U]  -> p ≈ (1/π)∫_0^U Re(e^{-iur}Ψ(u)) du
      - "weighted_inf": integrate on [0, ∞) using oscillatory weights:
            Re(e^{-iur}Ψ) = Re(Ψ)cos(ur) + Im(Ψ)sin(ur)
        so ∫_0^∞ Re(e^{-iur}Ψ) du = ∫_0^∞ Re(Ψ)cos(ur) du + ∫_0^∞ Im(Ψ)sin(ur) du

    Returns: float (density value)
    """
    r = float(r)
    v0 = float(v0)
    vT = float(vT)

    if v0 <= 0.0 or vT <= 0.0:
        return 0.0  # CIR support

    if mode == "finite":
        def integrand(u):
            psi = _psi_dY_cond_endpoints(u, v0, vT, dt, kappa, m, sigma, rho)
            return float(np.real(np.exp(-1j * u * r) * psi))

        val, err = quad(integrand, 0.0, float(U), epsabs=epsabs, epsrel=epsrel, limit=limit)
        return float(val / np.pi)

    if mode == "weighted_inf":
        # ∫ Re(Ψ(u)) cos(ur) du
        def fcos(u):
            psi = _psi_dY_cond_endpoints(u, v0, vT, dt, kappa, m, sigma, rho)
            return float(np.real(psi))

        # ∫ Im(Ψ(u)) sin(ur) du
        def fsin(u):
            psi = _psi_dY_cond_endpoints(u, v0, vT, dt, kappa, m, sigma, rho)
            return float(np.imag(psi))

        Icos, err1 = quad(fcos, 0.0, np.inf, weight="cos", wvar=r,
                          epsabs=epsabs, epsrel=epsrel, limit=limit)
        Isin, err2 = quad(fsin, 0.0, np.inf, weight="sin", wvar=r,
                          epsabs=epsabs, epsrel=epsrel, limit=limit)
        return float((Icos + Isin) / np.pi)

    raise ValueError("mode must be 'finite' or 'weighted_inf'")


def _phi_intvar_cond_endpoints(alpha, v0, vT, dt, kappa, m, sigma):
    """
    Φ(α) = E[ exp(i α ∫ v_s ds) | v(0)=v0, v(dt)=vT ]  (Broadie–Kaya Eq. 13)

    Vectorized in alpha, v0, vT via NumPy broadcasting.

    alpha: complex array-like, shape (..., )
    v0, vT: real array-like, broadcastable to the same leading shape
    returns: complex array of broadcasted shape
    """
    alpha = np.asarray(alpha, dtype=np.complex128)
    v0 = np.asarray(v0, dtype=np.float64)
    vT = np.asarray(vT, dtype=np.float64)

    d = 4.0 * kappa * m / (sigma * sigma)
    nu = 0.5 * d - 1.0

    ek = np.exp(-kappa * dt)
    one_minus_ek = -np.expm1(-kappa * dt)  # 1 - e^{-kappa dt} > 0

    gamma = np.sqrt(kappa * kappa - 2.0 * (sigma * sigma) * 1j * alpha)  # principal branch
    eg = np.exp(-gamma * dt)
    one_minus_eg = -np.expm1(-gamma * dt)

    pref = (gamma * np.exp(-0.5 * (gamma - kappa) * dt) * one_minus_ek) / (kappa * one_minus_eg)

    term_k = kappa * (1.0 + ek) / one_minus_ek
    term_g = gamma * (1.0 + eg) / one_minus_eg
    expo = np.exp((v0 + vT) / (sigma * sigma) * (term_k - term_g))

    sqrt_v0vT = np.sqrt(v0 * vT)

    z_num = sqrt_v0vT * (4.0 * gamma * np.exp(-0.5 * gamma * dt)) / ((sigma * sigma) * one_minus_eg)
    z_den = sqrt_v0vT * (4.0 * kappa * np.exp(-0.5 * kappa * dt)) / ((sigma * sigma) * one_minus_ek)  # real >=0

    # Denominator stable: ive is for real z_den
    den_scaled = ive(nu, z_den)  # exp(-z_den) I_nu(z_den)

    # Numerator can be complex; scale out exp(Re(z_num))
    num_scaled = iv(nu, z_num) * np.exp(-np.real(z_num))

    ratio = num_scaled * np.exp(np.real(z_num) - z_den) / den_scaled

    return pref * expo * ratio


class BridgeReturnPDF:
    """
    Exact (up to 1D quadrature) p(r_n | v_{n-1}=a, v_n=b) using Broadie–Kaya CF,
    optimized for repeated evaluations at observed r_n.

    Precomputes:
      - u-grid and trapezoidal weights w_k
      - E_{k,n} = w_k * exp(-i u_k r_n)  for all observed r_n
      - R_{k,j} = I_nu(z_num(u_k, p_j)) / I_nu(z_den(p_j)) on a 1D p-grid (p = sqrt(a b))

    Then:
      pdf(n, a, b) = (1/pi) * Re( sum_k  Psi_k(a,b) * E_{k,n} )
    """

    def __init__(self, r_seq, dt, kappa, m, sigma, rho, *, U=200.0, K=1024, p_min=1e-10, p_max=5.0, P=512, p_grid=None, interp_logp=True):
        self.r_seq = np.asarray(r_seq, dtype=np.float64)
        self.T = self.r_seq.size

        self.dt = float(dt)
        self.kappa = float(kappa)
        self.m = float(m)
        self.sigma = float(sigma)
        self.rho = float(rho)

        self.U = float(U)
        self.K = int(K)

        # --- u-grid [0, U] with K intervals -> K+1 points
        u = np.linspace(0.0, self.U, self.K + 1)
        du = u[1] - u[0]

        w = np.full_like(u, du, dtype=np.float64)
        w[0] *= 0.5
        w[-1] *= 0.5

        self.u = u                          # (Ku,)
        self.w = w                          # (Ku,)
        self.u2 = u * u

        # --- precompute E_{k,n} = w_k * exp(-i u_k r_n)
        # shape: (Ku, T)
        self.E = w[:, None] * np.exp(-1j * u[:, None] * self.r_seq[None, :])

        # --- CIR degrees-of-freedom parameter for BK Bessel order
        df = 4.0 * self.kappa * self.m / (self.sigma * self.sigma)
        self.nu = 0.5 * df - 1.0

        # --- p-grid
        if p_grid is None:
            if p_min <= 0:
                raise ValueError("p_min must be > 0")
            if p_max <= p_min:
                raise ValueError("p_max must be > p_min")
            # geometric grid is usually better than linear for variance scales
            self.p_grid = np.geomspace(p_min, p_max, P)
        else:
            self.p_grid = np.asarray(p_grid)
            if np.any(self.p_grid <= 0):
                raise ValueError("p_grid must be strictly positive")
            if np.any(np.diff(self.p_grid) <= 0):
                raise ValueError("p_grid must be strictly increasing")

        self.P = self.p_grid.size
        self.interp_logp = bool(interp_logp)
        self.logp_grid = np.log(self.p_grid)

        # --- precompute u-dependent BK ingredients at alpha(u) = i*s(u)
        self._precompute_u_dependent_terms()

        # --- precompute R_{k,j} on (u_k, p_j)
        self._precompute_R_table()

        if interp_logp:
            self.spline = CubicSpline(self.logp_grid, self.R_table, axis=-1)
        else:
            self.spline = CubicSpline(self.p_grid, self.R_table, axis=-1)

    def _precompute_u_dependent_terms(self):
        """
        Precompute arrays over u-grid that do NOT depend on endpoints (a,b),
        evaluated at alpha(u) = i*s(u), where
            s(u) = 0.5(1-rho^2)u^2 - i u * (rho*kappa/sigma)
        so alpha(u) = i*s(u) = (rho*kappa/sigma)u + i*0.5(1-rho^2)u^2.
        """
        u = self.u
        dt = self.dt
        kappa = self.kappa
        sigma = self.sigma
        rho = self.rho

        # alpha(u) = i*s(u)
        alpha = (rho * kappa / sigma) * u + 1j * (0.5 * (1.0 - rho * rho) * (u * u))
        alpha = alpha.astype(np.complex128)

        # gamma(alpha) = sqrt(kappa^2 - 2 sigma^2 i alpha)
        gamma = np.sqrt(kappa * kappa - 2.0 * (sigma * sigma) * 1j * alpha)

        ek = np.exp(-kappa * dt)
        one_minus_ek = -np.expm1(-kappa * dt)  # 1 - e^{-kappa dt}

        eg = np.exp(-gamma * dt)
        one_minus_eg = -np.expm1(-gamma * dt)  # 1 - e^{-gamma dt}

        # prefactor in BK Φ(α): independent of (a,b)
        pref = (gamma * np.exp(-0.5 * (gamma - kappa) * dt) * one_minus_ek) / (kappa * one_minus_eg)

        # term_k is constant; term_g depends on u
        term_k = kappa * (1.0 + ek) / one_minus_ek
        term_g = gamma * (1.0 + eg) / one_minus_eg

        # expo in BK Φ is exp( (a+b)/sigma^2 * (term_k - term_g) )
        lam = (term_k - term_g) / (sigma * sigma)  # complex, depends on u only

        # z_num(u,p) = p * c_num(u)
        c_num = (4.0 * gamma * np.exp(-0.5 * gamma * dt)) / ((sigma * sigma) * one_minus_eg)  # complex, u-dependent

        # z_den(p) = p * c_den  (independent of u)
        c_den = (4.0 * kappa * np.exp(-0.5 * kappa * dt)) / ((sigma * sigma) * one_minus_ek)  # real positive

        self.gamma = gamma.astype(np.complex128)
        self.pref = pref
        self.lam = lam
        self.c_num = c_num
        self.c_den = np.real(c_den)

    def _precompute_R_table(self):
        """
        Precompute R(u_k, p_j) = I_nu(z_num(u_k,p_j)) / I_nu(z_den(p_j)),
        where
          z_num(u,p) = p * c_num(u),
          z_den(p)   = p * c_den.
        """
        p = self.p_grid  # (P,)
        nu = self.nu

        # z_den depends only on p (real)
        z_den = self.c_den * p  # (P,)

        # Stable denominator using ive: ive(nu,z) = exp(-z) I_nu(z)
        den_scaled = ive(nu, z_den)  # (P,), real for real positive z_den

        # z_num is (Ku,P)
        z_num = self.c_num[:, None] * p[None, :] # (Ku,P)

        # Numerator Bessel: I_nu(z_num), complex
        num = iv(nu, z_num)

        # Stabilize ratio:
        # ratio = I_nu(z_num)/I_nu(z_den)
        # with den_scaled = exp(-z_den) I_nu(z_den),
        # compute:
        #   num_scaled = I_nu(z_num) * exp(-Re(z_num))
        #   ratio = num_scaled * exp(Re(z_num) - z_den) / den_scaled
        num_scaled = num * np.exp(-np.real(z_num))
        ratio = num_scaled * np.exp(np.real(z_num) - z_den[None, :]) / den_scaled[None, :]

        # Enforce the u=0 row to be exactly 1 (removes tiny numerical noise)
        # At u=0: alpha=0 -> gamma=kappa -> z_num=z_den -> ratio=1.
        ratio[0, :] = 1.0 + 0.0j

        self.R_table = ratio # (Ku,P)

    def _interp_R(self, p_val):
        """
        Interpolate R_table[:, j] in p at p_val (scalar), returning (Ku,) complex.
        Uses linear interpolation in log(p) by default.
        """
        # Clamp p to grid range to avoid extrapolation blowups
        if p_val <= self.p_grid[0]:
            return self.R_table[:, 0]
        if p_val >= self.p_grid[-1]:
            return self.R_table[:, -1]

        if self.interp_logp:
            x = np.log(p_val)
            grid = self.logp_grid
        else:
            x = p_val
            grid = self.p_grid

        j = int(np.searchsorted(grid, x) - 1)
        # ensure in [0, P-2]
        j = max(0, min(self.P - 2, j))

        x0 = grid[j]
        x1 = grid[j + 1]
        t = (x - x0) / (x1 - x0)

        return (1.0 - t) * self.R_table[:, j] + t * self.R_table[:, j + 1]

    def pdf(self, n, a, b):
        """
        Compute p(r_n | v_{n-1}=a, v_n=b) using precomputed E_{k,n} and R_table.

        Args:
          n: integer index into the observed r-sequence (0 <= n < T)
          a: v_{n-1} > 0
          b: v_n > 0
        """
        n = int(n)
        s = a + b
        d = b - a
        p = np.sqrt(a * b)

        Rk = self._interp_R(p)
        mu_const = (self.rho / self.sigma) * (d - self.kappa * self.m * self.dt)
        phase = np.exp(1j * self.u * mu_const)  # (Ku,), complex128

        explam = np.exp(self.lam * s)
        psi = phase * self.pref * explam * Rk
        E_n = self.E[:, n]
        return np.real(np.dot(psi, E_n)) / np.pi

import numpy as np
from scipy.special import iv, ive


class BridgeReturnPDF:
    """
    Exact (up to 1D quadrature) p(r_n | v_{n-1}=a, v_n=b) using Broadie–Kaya CF,
    optimized for repeated evaluations at observed r_n.

    Added:
      - deriv_log_pdf(n, a, b): gradient wrt theta = (kappa, m, sigma, rho)

    Notes:
      - Theta-derivatives are precomputed in __init__ via central finite differences
        on the *CF building blocks* (not on the final pdf).
      - This keeps runtime fast (interpolation + dot products).
    """

    def __init__(
        self,
        r_seq,
        dt,
        kappa,
        m,
        sigma,
        rho,
        *,
        U=200.0,
        K=1024,                 # number of u-intervals; points = K+1
        p_min=1e-10,
        p_max=5.0,
        P=512,                  # number of p-grid points
        p_grid=None,            # optional explicit p-grid
        interp_logp=True,       # interpolate tables in log(p)
        complex_dtype=np.complex64,
        fd_eps=None,            # finite-diff steps for theta-derivs (scalar or len-4)
    ):
        # --- store observed returns
        self.r_seq = np.asarray(r_seq, dtype=np.float64)
        self.T = self.r_seq.size

        # --- base theta
        self.dt = float(dt)
        self.kappa = float(kappa)
        self.m = float(m)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self._check_theta(self.kappa, self.m, self.sigma, self.rho)

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
        self.complex_dtype = complex_dtype

        # --- precompute E_{k,n} = w_k * exp(-i u_k r_n)
        # shape (Ku, T)
        self.E = (w[:, None] * np.exp(-1j * u[:, None] * self.r_seq[None, :])).astype(complex_dtype)

        # --- p-grid
        if p_grid is None:
            if p_min <= 0:
                raise ValueError("p_min must be > 0")
            if p_max <= p_min:
                raise ValueError("p_max must be > p_min")
            self.p_grid = np.geomspace(p_min, p_max, P).astype(np.float64)
        else:
            self.p_grid = np.asarray(p_grid, dtype=np.float64)
            if np.any(self.p_grid <= 0):
                raise ValueError("p_grid must be strictly positive")
            if np.any(np.diff(self.p_grid) <= 0):
                raise ValueError("p_grid must be strictly increasing")

        self.P = self.p_grid.size
        self.interp_logp = bool(interp_logp)
        self.logp_grid = np.log(self.p_grid)

        # --- base u-dependent terms + base R-table
        base_terms = self._compute_u_terms(self.kappa, self.m, self.sigma, self.rho)
        self.nu = base_terms["nu"]
        self.pref = base_terms["pref"].astype(complex_dtype)    # shape (Ku,)
        self.lam = base_terms["lam"].astype(complex_dtype)      # shape (Ku,)
        self.R_table = self._compute_R_table(base_terms).astype(complex_dtype)  # (Ku,P)

        # --- finite-difference steps for theta-derivatives
        self.fd_eps = self._parse_fd_eps(fd_eps)

        # --- precompute derivative tables wrt theta = (kappa, m, sigma, rho)
        # d/dtheta log(pref), d/dtheta lam, d/dtheta log(R)
        self._precompute_theta_derivative_tables()

    # =====================================================================
    # Public methods
    # =====================================================================

    def pdf(self, n, a, b):
        """
        Compute p(r_n | v_{n-1}=a, v_n=b).
        """
        n = int(n)
        if n < 0 or n >= self.T:
            raise IndexError(f"n must be in [0, {self.T-1}]")

        a = float(a)
        b = float(b)
        if a <= 0.0 or b <= 0.0:
            return 0.0

        psi = self._psi_vector(a, b)               # (Ku,)
        val = np.real(np.dot(psi, self.E[:, n])) / np.pi
        return float(val)

    def deriv_log_pdf(self, n, a, b):
        """
        Compute gradient wrt theta = (kappa, m, sigma, rho) of

            log p(r_n | v_{n-1}=a, v_n=b).

        Returns:
            grad: np.ndarray, shape (4,)
                  order = [d/dkappa, d/dm, d/dsigma, d/drho]
        """
        n = int(n)
        if n < 0 or n >= self.T:
            raise IndexError(f"n must be in [0, {self.T-1}]")

        a = float(a)
        b = float(b)
        if a <= 0.0 or b <= 0.0:
            raise ValueError("deriv_log_pdf requires a>0 and b>0")

        # Runtime scalars
        s = a + b
        d = b - a
        p = np.sqrt(a * b)

        # Interpolate R and dlogR at p
        Rk = self._interp_last_axis(self.R_table, p)            # (Ku,)
        dlogR = self._interp_last_axis(self.dlogR_table, p)     # (4, Ku)

        # Build Psi_k = phase * pref * exp(lam*s) * R
        mu_const = (self.rho / self.sigma) * (d - self.kappa * self.m * self.dt)
        phase = np.exp(self.iu * mu_const)                      # (Ku,)
        psi = (phase.astype(self.complex_dtype)
               * self.pref
               * np.exp(self.lam * s)
               * Rk)                                            # (Ku,)

        # Base Fourier sum (without 1/pi, which cancels in log-derivative)
        Ecol = self.E[:, n]                                     # (Ku,)
        S0 = np.dot(psi, Ecol)                                  # complex scalar
        denom = np.real(S0)

        if denom <= 0.0:
            raise FloatingPointError(
                "Encountered non-positive Fourier inversion result for pdf. "
                "Increase K/U or improve interpolation range/resolution."
            )

        # d/dtheta log phase = i u * d(mu_const)/dtheta
        dmu = self._d_mu_const_dtheta(d)                        # (4,)
        dlog_phase = self.iu[None, :] * dmu[:, None]            # (4, Ku)

        # d/dtheta log Psi = dlog_phase + dlogpref + s*dlam + dlogR
        G = dlog_phase + self.dlogpref + s * self.dlam + dlogR  # (4, Ku)

        # Numerators: Re sum_k [psi_k * E_k * G_{ell,k}]
        # Vectorized over ell:
        nums = np.real((G * psi[None, :]) @ Ecol)               # (4,)

        grad = nums / denom
        return np.asarray(grad, dtype=np.float64)

    # =====================================================================
    # Internal helpers: parameter checks / finite-diff settings
    # =====================================================================

    @staticmethod
    def _check_theta(kappa, m, sigma, rho):
        if kappa <= 0:
            raise ValueError("kappa must be > 0")
        if m <= 0:
            raise ValueError("m must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if not (-1.0 < rho < 1.0):
            raise ValueError("rho must be in (-1, 1)")

    def _parse_fd_eps(self, fd_eps):
        """
        Finite-difference steps for theta=(kappa,m,sigma,rho).
        If fd_eps is None, choose mild relative defaults.
        """
        if fd_eps is None:
            # Conservative defaults; tune if needed
            eps = np.array([
                1e-4 * max(1.0, abs(self.kappa)),
                1e-4 * max(1.0, abs(self.m)),
                1e-4 * max(1.0, abs(self.sigma)),
                1e-4 * max(1.0, abs(self.rho), 1.0),  # rho gets absolute-ish step
            ], dtype=np.float64)
        else:
            arr = np.asarray(fd_eps, dtype=np.float64)
            if arr.ndim == 0:
                eps = np.full(4, float(arr), dtype=np.float64)
            elif arr.shape == (4,):
                eps = arr.copy()
            else:
                raise ValueError("fd_eps must be None, scalar, or length-4 array")
        if np.any(eps <= 0):
            raise ValueError("All fd_eps must be > 0")

        # Ensure centered differences stay in admissible region
        if self.kappa - eps[0] <= 0:
            raise ValueError("fd_eps for kappa too large (kappa-h <= 0)")
        if self.m - eps[1] <= 0:
            raise ValueError("fd_eps for m too large (m-h <= 0)")
        if self.sigma - eps[2] <= 0:
            raise ValueError("fd_eps for sigma too large (sigma-h <= 0)")
        if abs(self.rho) + eps[3] >= 1:
            raise ValueError("fd_eps for rho too large (rho±h leaves (-1,1))")
        return eps

    # =====================================================================
    # Internal helpers: BK u-dependent terms and R-table
    # =====================================================================

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
        self._check_theta(kappa, m, sigma, rho)

        u = self.u
        dt = self.dt

        # BK order parameter
        df = 4.0 * kappa * m / (sigma * sigma)
        nu = 0.5 * df - 1.0

        # alpha(u) = i * s(u), where
        # s(u)=0.5(1-rho^2)u^2 - i u rho kappa/sigma
        alpha = (rho * kappa / sigma) * u + 1j * (0.5 * (1.0 - rho * rho) * (u * u))
        alpha = alpha.astype(np.complex128)

        # gamma(alpha) = sqrt(kappa^2 - 2 sigma^2 i alpha)
        gamma = np.sqrt(kappa * kappa - 2.0 * (sigma * sigma) * 1j * alpha)

        ek = np.exp(-kappa * dt)
        one_minus_ek = -np.expm1(-kappa * dt)

        eg = np.exp(-gamma * dt)
        one_minus_eg = -np.expm1(-gamma * dt)

        # BK prefactor (u-dependent, endpoint-independent)
        pref = (gamma * np.exp(-0.5 * (gamma - kappa) * dt) * one_minus_ek) / (kappa * one_minus_eg)

        # Exponential endpoint term: exp( lam(u) * (a+b) )
        term_k = kappa * (1.0 + ek) / one_minus_ek
        term_g = gamma * (1.0 + eg) / one_minus_eg
        lam = (term_k - term_g) / (sigma * sigma)

        # Bessel arguments:
        # z_num(u,p) = p * c_num(u), z_den(p) = p * c_den
        c_num = (4.0 * gamma * np.exp(-0.5 * gamma * dt)) / ((sigma * sigma) * one_minus_eg)
        c_den = (4.0 * kappa * np.exp(-0.5 * kappa * dt)) / ((sigma * sigma) * one_minus_ek)

        return {
            "nu": float(np.real(nu)),
            "pref": np.asarray(pref, dtype=np.complex128),
            "lam": np.asarray(lam, dtype=np.complex128),
            "c_num": np.asarray(c_num, dtype=np.complex128),
            "c_den": float(np.real(c_den)),
        }

    def _compute_R_table(self, terms):
        """
        Precompute R(u_k,p_j) = I_nu(z_num)/I_nu(z_den) on (u,p)-grid.
        """
        nu = terms["nu"]
        c_num = terms["c_num"]     # (Ku,)
        c_den = terms["c_den"]     # scalar
        p = self.p_grid            # (P,)

        # z_den real positive
        z_den = c_den * p                          # (P,)
        den_scaled = ive(nu, z_den)                # exp(-z_den) I_nu(z_den)

        # z_num complex
        z_num = c_num[:, None] * p[None, :]        # (Ku,P)
        num = iv(nu, z_num)

        # Stable ratio
        num_scaled = num * np.exp(-np.real(z_num))
        ratio = num_scaled * np.exp(np.real(z_num) - z_den[None, :]) / den_scaled[None, :]

        # At u=0, ratio should be exactly 1
        ratio[0, :] = 1.0 + 0.0j
        return ratio

    # =====================================================================
    # Internal helpers: precompute theta-derivative tables (finite differences)
    # =====================================================================

    def _precompute_theta_derivative_tables(self):
        """
        Precompute:
          dlogpref[ell, k]   = d/dtheta_ell log(pref_k)
          dlam[ell, k]       = d/dtheta_ell lam_k
          dlogR_table[ell,k,j]= d/dtheta_ell log(R_{k,j})

        via centered finite differences in theta=(kappa,m,sigma,rho).
        """
        theta0 = np.array([self.kappa, self.m, self.sigma, self.rho], dtype=np.float64)

        Ku = self.K + 1
        P = self.P

        dlogpref = np.empty((4, Ku), dtype=np.complex128)
        dlam = np.empty((4, Ku), dtype=np.complex128)
        dlogR_table = np.empty((4, Ku, P), dtype=np.complex128)

        # Base arrays in high precision for finite-difference ratios
        pref0 = np.asarray(self.pref, dtype=np.complex128)
        lam0 = np.asarray(self.lam, dtype=np.complex128)
        R0 = np.asarray(self.R_table, dtype=np.complex128)

        for ell in range(4):
            h = self.fd_eps[ell]

            tp = theta0.copy()
            tm = theta0.copy()
            tp[ell] += h
            tm[ell] -= h

            # Build perturbed u-terms and R-tables
            terms_p = self._compute_u_terms(*tp)
            terms_m = self._compute_u_terms(*tm)

            pref_p = terms_p["pref"]
            pref_m = terms_m["pref"]
            lam_p = terms_p["lam"]
            lam_m = terms_m["lam"]

            Rp = self._compute_R_table(terms_p)
            Rm = self._compute_R_table(terms_m)

            # d/dtheta log(pref) = (pref' / pref) approx (pref_p - pref_m)/(2h*pref0)
            dlogpref[ell, :] = (pref_p - pref_m) / (2.0 * h * pref0)

            # d/dtheta lam
            dlam[ell, :] = (lam_p - lam_m) / (2.0 * h)

            # d/dtheta log(R) = (R' / R) approx (Rp - Rm)/(2h*R0)
            dlogR_table[ell, :, :] = (Rp - Rm) / (2.0 * h * R0)

        self.dlogpref = dlogpref.astype(self.complex_dtype)
        self.dlam = dlam.astype(self.complex_dtype)
        self.dlogR_table = dlogR_table.astype(self.complex_dtype)

    # =====================================================================
    # Internal helpers: runtime interpolation / psi / derivatives
    # =====================================================================

    def _interp_last_axis(self, table, p_val):
        """
        Interpolate along the last axis (which is the p-grid axis).
        Works for:
          table shape (..., P)
        returns shape (...,)

        Uses linear interpolation in log(p) if self.interp_logp=True.
        """
        p_val = float(p_val)

        # Clamp to grid range
        if p_val <= self.p_grid[0]:
            return np.take(table, 0, axis=-1)
        if p_val >= self.p_grid[-1]:
            return np.take(table, self.P - 1, axis=-1)

        if self.interp_logp:
            x = np.log(p_val)
            grid = self.logp_grid
        else:
            x = p_val
            grid = self.p_grid

        j = int(np.searchsorted(grid, x) - 1)
        j = max(0, min(self.P - 2, j))

        x0 = grid[j]
        x1 = grid[j + 1]
        t = (x - x0) / (x1 - x0)

        left = np.take(table, j, axis=-1)
        right = np.take(table, j + 1, axis=-1)
        return (1.0 - t) * left + t * right

    def _psi_vector(self, a, b):
        """
        Build Psi(u_k | a,b) as a vector over u-grid.
        """
        s = a + b
        d = b - a
        p = np.sqrt(a * b)

        Rk = self._interp_last_axis(self.R_table, p)   # (Ku,)
        mu_const = (self.rho / self.sigma) * (d - self.kappa * self.m * self.dt)
        phase = np.exp(self.iu * mu_const)

        psi = (phase.astype(self.complex_dtype)
               * self.pref
               * np.exp(self.lam * s)
               * Rk)
        return psi

    def _d_mu_const_dtheta(self, d):
        """
        Derivatives wrt theta=(kappa,m,sigma,rho) of
            mu_const = (rho/sigma) * (d - kappa*m*dt)
        """
        dt = self.dt
        kappa = self.kappa
        m = self.m
        sigma = self.sigma
        rho = self.rho

        common = (d - kappa * m * dt)

        d_kappa = -(rho / sigma) * m * dt
        d_m = -(rho / sigma) * kappa * dt
        d_sigma = -(rho / (sigma * sigma)) * common
        d_rho = common / sigma

        return np.array([d_kappa, d_m, d_sigma, d_rho], dtype=np.float64)



import numpy as np
from scipy.special import iv, ive



eng2 = BridgeReturnPDF(r_seq=np.linspace(-1, 1, 100), dt=1, kappa=1 + 1e-4, m=0.3**2, sigma=0.3, rho=-0.5, U=200, K=4096, P=2500, complex_dtype=np.complex128)
eng3 = BridgeReturnPDF(r_seq=np.linspace(-1, 1, 100), dt=1, kappa=1 - 1e-7, m=0.3**2, sigma=0.3, rho=-0.5, U=200, K=4096, P=2500, complex_dtype=np.complex128)

tic = time()
f = [eng.pdf_deriv_deriv2(74, 0.3**2, 0.4**2) for i in range(10000)]
toc = time()
print(toc - tic)

tic = time()
f = [eng.deriv_log_pdf(74, 0.3**2, 0.4**2) for i in range(100)]
toc = time()
print(toc - tic)

val = eng.pdf(n=10, a=0.03, b=0.05)
grad = eng.deriv_log_pdf(n=10, a=0.03, b=0.05)



eng = BridgeReturnPDF(r_seq=np.linspace(-1, 1, 2000), dt=1, kappa=1, m=0.3**2, sigma=0.3, rho=-0.5, U=200, K=4096, P=2500)

tic = time()
f = [eng.pdf(749, 0.3**2, 0.4**2) for i in range(10000)]
toc = time()
print(toc - tic)

tic = time()
f = eng.pdf_pairs2(749, np.linspace(0.3**2, 0.4**2, 10000), np.linspace(0.4**2, 0.5**2, 10000))
toc = time()
print(toc - tic, f)

bridge_return_pdf_quad(0.49494949494949503, 0.3**2, 0.4**2, dt=1, kappa=1, m=0.3**2, sigma=0.3, rho=-0.5)
