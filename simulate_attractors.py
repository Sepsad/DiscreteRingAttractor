import numpy as np
from ring_attractor import (
    simulate,
    bump_init,
    compute_bump_state,
    plot_dynamics,
    plot_attractor_map,
    pca_from_scratch,
    plot_pca,
    energy_landscape,
    plot_energy_landscape,
)


def build_connectivity(N, JE=3.0, JI=-1.0, delta=np.pi/2, tau=0.1):
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    Ws = (JI + JE * np.cos(theta[:, None] - theta[None, :])) / N
    Wa = np.cos(theta[:, None] - theta[None, :] + delta) / N
    W = (Ws - np.eye(N)) / tau
    W_cw = Wa / tau
    W_ccw = Wa / tau
    return theta, W, W_cw, W_ccw


def main(plot=False):
    N = 6
    tau = 0.1
    C0 = 1.0
    A = 1.0
    width = 3 * 2 * np.pi / N  # activate roughly 3 neurons
    JE = 3.0
    JI = -1.0
    theta, W, W_cw, W_ccw = build_connectivity(N, JE=JE, JI=JI, tau=tau)

    centers = np.linspace(0, np.pi / N, N, endpoint=False)
    centers = np.concatenate([centers + k * (np.pi / N) for k in range(2 * N)])

    attractors = []
    final_states = []
    example_t = None
    example_h = None
    for idx, psi in enumerate(centers):
        h0 = bump_init(theta, psi, width, A)
        t, h = simulate(
            W,
            np.zeros_like(W_cw),
            np.zeros_like(W_ccw),
            C0,
            tau,
            v_fn=lambda t: 0.0,
            h0=h0,
            t_span=(0.0, 5.0),
            dt=0.01,
        )
        psi_traj, _, _ = compute_bump_state(h)
        attractors.append(psi_traj[-1])
        final_states.append(h[:, -1])

        if idx == 0:
            example_t = t
            example_h = h

    final_states = np.stack(final_states)

    np.save("attractors.npy", np.array(attractors))
    np.save("attractor_activity.npy", final_states)

    eigvals, eigvecs, scores = pca_from_scratch(final_states)

    psi_samples = np.linspace(0, 2 * np.pi, 200)
    width_samples = np.linspace(np.pi / N, (N - 1) * np.pi / N, 200)
    E_land = energy_landscape(JE, JI, C0, theta, psi_samples, width_samples)
    np.save("energy_landscape.npy", E_land)

    if plot and example_t is not None:
        plot_dynamics(example_t, example_h, theta)
        plot_attractor_map(centers, attractors)
        plot_pca(scores, centers)
        plot_energy_landscape(E_land, psi_samples, width_samples)
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main(plot=True)
