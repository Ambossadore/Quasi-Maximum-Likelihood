from main import *
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from scipy.integrate import quad
from scipy.special import iv, ive
from scipy.interpolate import CubicSpline


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
        # store complex64 by default to reduce RAM
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



eng = BridgeReturnPDF(r_seq=np.linspace(-1, 1, 2000), dt=1, kappa=1, m=0.3**2, sigma=0.3, rho=-0.5, U=200, K=4096, P=2500)

tic = time()
f = [eng.pdf(749, 0.3**2, 0.4**2) for i in range(10000)]
toc = time()
print(toc - tic)

tic = time()
f = eng.pdf_pairs2(749, np.linspace(0.3**2, 0.4**2, 10000), np.linspace(0.4**2, 0.5**2, 10000))
toc = time()
print(toc - tic, f)

bridge_return_pdf_quad(0.49949949949949946, 0.3**2, 0.4**2, dt=1, kappa=1, m=0.3**2, sigma=0.3, rho=-0.5)
