from main import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gamma

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
})


def particle_filter_heston(N, param, dt, Delta_Y, verbose=1):

    kappa, m, sigma, rho = param
    K = np.size(Delta_Y)
    v_particles = gamma.rvs((2 * kappa * m) / sigma ** 2, scale=sigma ** 2 / (2 * kappa), size=N)

    log_weights = np.zeros(N) - np.log(N)
    v_filtered, v_cond_variance = np.zeros(K), np.zeros(K)
    v_filtered[0] = np.mean(v_particles)
    v_cond_variance[0] = np.var(v_particles)

    t = trange(1, K) if verbose else range(1, K)
    for i in t:
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
        v_cond_variance[i] = np.sum(v_particles**2 * weights) - v_filtered[i]**2

    return v_filtered, v_cond_variance


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

    v_particle_filter, _ = particle_filter_heston(N=10000, param=model1.true_param, dt=1 / 250, Delta_Y=model1.observations[:, 1])

    errors = np.zeros((100000, 751))
    for i in trange(100000):
        init = InitialDistribution(dist='Gamma_Dirac', hyper=[0.])
        model = HestonModel(first_observed=1, init=init, dt=1 / 250, signature='1[1]_2d[1]', true_param=np.array([1, 0.4 ** 2, 0.3, -0.5]), warn=False)
        model.generate_observations(t_max=3, inter_steps=250, seed=i, verbose=0)
        _, v_cond_variance = particle_filter_heston(N=10000, param=model.true_param, dt=1 / 250, Delta_Y=model.observations[:, 1], verbose=0)
        errors[i] = v_cond_variance

    particle_mse = errors.mean(axis=0)
    np.savetxt('particle_mse.txt', particle_mse)

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
    plt.plot(t, model1.observations[:, 0], linewidth=0.8)
    plt.plot(t, model2.kalman_filter.X_hat_tt_list[:, 0], linewidth=0.8, color='C2')
    plt.plot(t, model1.kalman_filter.X_hat_tt_list[:, 0], linewidth=0.8, color='C1')
    plt.plot(t, v_particle_filter, linewidth=0.8, color='C3')
    plt.legend([r'$v(t)$', r'$\widehat v(t, t)^{(1)}$', r'$\widehat v(t, t)^{(2)}$', r'$\widehat v(t, t)^{\mathrm{part.}}$'])
    plt.grid(alpha=0.4)
    plt.xlabel('$t$')
    plt.ylabel(r'$v$')

    t = np.arange(0, 8 + 1 / 250, 1 / 250)
    Sig_tt_1 = np.append(model1.kalman_filter.Sig_tt_list[:, 0, 0], np.repeat(model1.kalman_filter.Sig_tt_list[-1, 0, 0], len(t) - len(model1.kalman_filter.Sig_tt_list[:, 0, 0])))
    Sig_tt_2 = np.append(model2.kalman_filter.Sig_tt_list[:, 0, 0], np.repeat(model2.kalman_filter.Sig_tt_list[-1, 0, 0], len(t) - len(model2.kalman_filter.Sig_tt_list[:, 0, 0])))

    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, model1.observations[:, 0], linewidth=0.8)
    ax1.plot(t, model2.kalman_filter.X_hat_tt_list[:, 0], linewidth=0.8, color='C2')
    ax1.plot(t, model1.kalman_filter.X_hat_tt_list[:, 0], linewidth=0.8, color='C1')
    ax1.plot(t, v_particle_filter, linewidth=0.8, color='C3')
    ax1.legend([r'$v(t)$', r'$\widehat v(t, t)^{(1)}$', r'$\widehat v(t, t)^{(2)}$', r'$\widehat v(t, t)^{\mathrm{part.}}$'])
    ax1.grid(alpha=0.4)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel(r'$v$')

    ax2 = fig.add_subplot(gs[2, :])
    t2 = np.arange(0, 3 + 1 / 250, 1 / 250)
    Sig_tt_11 = np.append(model1.kalman_filter.Sig_tt_list[:, 0, 0], np.repeat(model1.kalman_filter.Sig_tt_list[-1, 0, 0], len(particle_mse) - len(model1.kalman_filter.Sig_tt_list[:, 0, 0])))
    Sig_tt_22 = model2.kalman_filter.Sig_tt_list[:len(particle_mse), 0, 0]
    ax2.loglog(t2, np.sqrt(Sig_tt_11), color='C1')
    ax2.loglog(t2, np.sqrt(Sig_tt_22), color='C2')
    ax2.loglog(t2, np.sqrt(particle_mse), color='C3')
    ax2.set_xlim(t[1], 4)
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$\sqrt{\mathbb{E}[(v(t) - \widehat v(t, t))^2]}$')
    ax2.legend([r'$\widehat v(t, t)^{(1)}$', r'$\widehat v(t, t)^{(2)}$', r'$\widehat v(t, t)^{\mathrm{part.}}$'])
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
    ax2.grid(which='both', alpha=.4)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, model1.observations[:, 0], linewidth=0.8)
    ax3.plot(t, model2.kalman_filter.X_hat_tt_list[:, 0], linewidth=0.8, color='C2')
    ax3.plot(t, model2.kalman_filter.X_hat_tt_list[:, 0] + np.sqrt(Sig_tt_2), linewidth=0.8, color='grey', alpha=0.4)
    ax3.plot(t, model2.kalman_filter.X_hat_tt_list[:, 0] - np.sqrt(Sig_tt_2), linewidth=0.8, color='grey', alpha=0.4)
    plt.fill_between(t, model2.kalman_filter.X_hat_tt_list[:, 0] - np.sqrt(Sig_tt_2), model2.kalman_filter.X_hat_tt_list[:, 0] + np.sqrt(Sig_tt_2), color='C2', alpha=0.4)
    ax3.set_xlabel(r'$t$')
    ax3.set_ylabel(r'$\widehat v(t, t) \pm \sqrt{\mathbb{E}[(v(t) - \widehat v(t, t))^2]}$')
    ax3.legend([r'$v(t)$', r'$\widehat v(t, t)^{(1)}$'])
    ax3.grid(alpha=0.4)

    ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
    ax4.plot(t, model1.observations[:, 0], linewidth=0.8)
    ax4.plot(t, model1.kalman_filter.X_hat_tt_list[:, 0], linewidth=0.8, color='C1')
    ax4.plot(t, model1.kalman_filter.X_hat_tt_list[:, 0] + np.sqrt(Sig_tt_1), linewidth=0.8, color='grey', alpha=0.4)
    ax4.plot(t, model1.kalman_filter.X_hat_tt_list[:, 0] - np.sqrt(Sig_tt_1), linewidth=0.8, color='grey', alpha=0.4)
    plt.fill_between(t, model1.kalman_filter.X_hat_tt_list[:, 0] - np.sqrt(Sig_tt_1), model1.kalman_filter.X_hat_tt_list[:, 0] + np.sqrt(Sig_tt_1), color='C1', alpha=0.4)
    ax4.set_xlabel(r'$t$')
    ax4.grid(alpha=0.4)
    plt.setp(ax4.get_yticklabels(), visible=False)
    ax4.legend([r'$v(t)$', r'$\widehat v(t, t)^{(2)}$'])
    gs.tight_layout(fig, rect=[0., 0., 0.97, 0.97])
    fig.savefig('./heston_filter.png', dpi=1000)
