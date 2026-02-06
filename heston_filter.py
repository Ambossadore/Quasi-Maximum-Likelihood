from main import *
import matplotlib.pyplot as plt
from scipy.stats import gamma

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
})

def particle_filter_heston(N, param, dt, Delta_Y):

    kappa, m, sigma, rho = param
    K = np.size(Delta_Y)
    v_particles = gamma.rvs((2 * kappa * m) / sigma ** 2, scale=sigma ** 2 / (2 * kappa), size=N)

    log_weights = np.zeros(N) - np.log(N)
    v_filtered = np.zeros(K)
    v_filtered[0] = np.mean(v_particles)

    for i in trange(1, K):
        var_obs = np.maximum(v_particles, 1e-9) * dt
        log_likelihood = -0.5 * np.log(var_obs) - (Delta_Y[i] ** 2) / (2 * var_obs)
        log_weights += log_likelihood

        max_log_w = np.max(log_weights)
        weights_unnorm = np.exp(log_weights - max_log_w)
        weights = weights_unnorm / np.sum(weights_unnorm)

        ESS = 1.0 / np.sum(weights ** 2)  # Effective Sample Size (ESS)

        if ESS < N / 2.0:
            indices = systematic_resample(weights)
            v_particles = v_particles[indices]
            log_weights = np.zeros(N) - np.log(N)
            weights = np.ones(N) / N
        else:
            log_weights = log_weights - np.log(np.sum(weights_unnorm)) - max_log_w

        Z = np.random.normal(0, 1, size=N)
        v_prev_plus = np.maximum(v_particles, 0)
        drift = kappa * (m - v_prev_plus) * dt
        diffusion_observed = sigma * rho * Delta_Y[i]
        diffusion_unobserved = sigma * np.sqrt(1 - rho ** 2) * np.sqrt(v_prev_plus) * np.sqrt(dt) * Z

        v_particles = v_particles + drift + diffusion_observed + diffusion_unobserved
        v_filtered[i] = np.sum(v_particles * weights)

    return v_filtered


def systematic_resample(weights):
    N = np.size(weights)
    positions = (np.arange(N) + np.random.random()) / N
    indexes = np.zeros(N, dtype='int')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


if __name__ == '__main__':
    init = InitialDistribution(dist='Dirac', hyper=[0.3**2, 0., 0.])
    model1 = HestonModel(first_observed=1, init=init, dt=1 / 250, signature='1[1]_2d[1, 2]', true_param=np.array([1, 0.4**2, 0.3, -0.5]), warn=False)
    model1.generate_observations(t_max=8, inter_steps=250, seed=20, verbose=1)

    model1.init = InitialDistribution(dist='Gamma_Dirac', hyper=[0., 0.])
    a1, A1, C1 = model1.state_space_params(model1.true_param, return_stack=True)
    partial_E_0 = partial(model1.init.E_0, param=model1.true_param)
    partial_Cov_0 = partial(model1.init.Cov_0, param=model1.true_param)
    model1.kalman_filter = KalmanFilter(dim=model1.dim, a=a1, A=A1, C=C1, E_0=partial_E_0, Cov_0=partial_Cov_0, first_observed=model1.first_observed)
    model1.kalman_filter.build_covariance()
    model1.kalman_filter.build_kalman_filter(observations=model1.observations)

    init2 = InitialDistribution(dist='Gamma_Dirac', hyper=[0.])
    model2 = HestonModel(first_observed=1, init=init2, dt=1 / 250, signature='1[1]_2d[1]', true_param=np.array([1, 0.4**2, 0.3, -0.5]), warn=False)
    model2.observations = model1.observations[:, :-1]
    a2, A2, C2 = model2.state_space_params(model2.true_param, return_stack=True)
    partial_E_0 = partial(model2.init.E_0, param=model2.true_param)
    partial_Cov_0 = partial(model2.init.Cov_0, param=model2.true_param)
    model2.kalman_filter = KalmanFilter(dim=model2.dim, a=a2, A=A2, C=C2, E_0=partial_E_0, Cov_0=partial_Cov_0, first_observed=model2.first_observed)
    model2.kalman_filter.build_covariance()
    model2.kalman_filter.build_kalman_filter(observations=model2.observations)

    v_particle_filter = particle_filter_heston(N=100000, param=model1.true_param, dt=1 / 250, Delta_Y=model1.observations[:, 1])

    # init3 = InitialDistribution(dist='Dirac', hyper=[0.3**2, 0., 0., 0.])
    # scaling = [1, 1]
    # model3 = HestonModel(first_observed=1, init=init3, dt=1/250, signature='1[1]_2d[1, 2, 4]', true_param=np.array([1, 0.4**2, 0.3, -0.5]), scaling=scaling, warn=False)
    # model3.observations = np.hstack((model1.observations, model1.observations[:, 1:2]**4))
    # model3.observations[:, 0] *= scaling[0]
    # model3.observations[:, 1] *= scaling[1]
    # model3.observations[:, 2] *= scaling[1]**2
    # model3.observations[:, 3] *= scaling[1]**4
    # a3, A3, C3 = model3.state_space_params(model3.true_param, return_stack=True)
    # partial_E_0 = lambda: model3.init.E_0(param=model3.true_param) * scaling[0]
    # partial_Cov_0 = lambda: model3.init.Cov_0(param=model3.true_param) * scaling[1]
    # model3.kalman_filter = KalmanFilter(dim=model3.dim, a=a3, A=A3, C=C3, E_0=partial_E_0, Cov_0=partial_Cov_0, first_observed=model3.first_observed)
    # model3.kalman_filter.build_covariance()
    # model3.kalman_filter.build_kalman_filter(observations=model3.observations)

    t = np.arange(0, 8 + 1 / 250, 1 / 250)
    plt.plot(t, model1.observations[:, 0], linewidth=1)
    plt.plot(t, model2.kalman_filter.X_hat_tt_list[:, 0], linewidth=1, color='C2')
    plt.plot(t, model1.kalman_filter.X_hat_tt_list[:, 0], linewidth=1, color='C1')
    plt.plot(t, v_particle_filter, linewidth=1, color='C3')
    plt.legend([r'$v(t)$', r'$\widehat v(t, t)^{(1)}$', r'$\widehat v(t, t)^{(2)}$', r'$\widehat v(t, t)^{\mathrm{part.}}$'])
    plt.grid(alpha=0.4)
    plt.xlabel('$t$')
    plt.ylabel(r'$v$')

    plt.figure()
    plt.plot(model2.kalman_filter.Sig_tt_list[:, 0, 0])
    plt.plot(model1.kalman_filter.Sig_tt_list[:, 0, 0])

    params = DriftlessHestonParams(kappa=1, theta=0.16, sigma_v=0.3, rho=-0.5)
    dt = 1/250
    v0 = 0.16
    dY = model2.observations[:100, 1]

    # dY is your observed ΔY_0,...,ΔY_T; ΔY_0 is ignored in the loop
    v_filt_mean, diag = pf_driftless_heston_exact_auxI(
        N=1000,
        params=params,
        dt=dt,
        dY=dY,
        v0=v0,
        I_h=0.08,
        I_n_terms=150,
        I_bisect_iters=40,
    )
