import numpy as np
from scipy.integrate import solve_ivp


def poslin(x):
    """Threshold linear function."""
    return np.maximum(0.0, x)


def dyn_sys(t, h, W, W_cw, W_ccw, C0, tau, v_fn, noise_fn):
    """Continuous-time dynamics for the ring attractor.

    Parameters
    ----------
    t : float
        Current time.
    h : ndarray
        Activity vector of size N.
    W, W_cw, W_ccw : ndarray
        Connectivity matrices.
    C0 : float
        Constant feedforward input.
    tau : float
        Neuronal time constant.
    v_fn : callable
        Function returning the velocity input at time t.
    noise_fn : callable
        Function returning an additive noise vector at time t.
    """
    v = v_fn(t)
    noise = noise_fn(t)

    v_cw = max(v, 0.0)
    v_ccw = max(-v, 0.0)

    r = poslin(h)
    dh = W @ r + v_cw * (W_cw @ r) + v_ccw * (W_ccw @ r) + noise
    dh = (dh - h + C0) / tau
    return dh


def simulate(W, W_cw, W_ccw, C0, tau, v_fn, h0, t_span, dt, noise_fn=None):
    """Simulate the network dynamics."""
    if noise_fn is None:
        noise_fn = lambda t: np.zeros_like(h0)

    sol = solve_ivp(
        dyn_sys,
        (t_span[0], t_span[1]),
        h0,
        t_eval=np.arange(t_span[0], t_span[1] + dt, dt),
        args=(W, W_cw, W_ccw, C0, tau, v_fn, noise_fn),
        vectorized=False,
    )
    return sol.t, sol.y


def bump_init(theta, psi, width, amplitude):
    """Cosine shaped bump initialization."""
    return amplitude * poslin(np.cos(theta - psi) - np.cos(width / 2.0))


def compute_bump_state(h):
    """Return bump orientation and width from activity matrix."""
    N = h.shape[0]
    dft = np.fft.fft(h, axis=0)
    H0 = dft[0] / N
    rho = np.abs(dft[1]) / N
    psi = -np.angle(dft[1])
    thc = np.arccos(-H0 / (2.0 * rho))
    return psi, rho, thc


def plot_dynamics(t, h, theta=None):
    """Plot activity and bump orientation over time.

    Parameters
    ----------
    t : ndarray
        Time points returned by :func:`simulate`.
    h : ndarray
        Activity matrix with shape ``(N, len(t))``.
    theta : ndarray, optional
        Neuron angle for visualizing the heatmap. If ``None`` a uniform grid
        over ``[0, 2π)`` is used.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure instance.
    """
    import matplotlib.pyplot as plt

    psi, _, _ = compute_bump_state(h)
    if theta is None:
        theta = np.linspace(0.0, 2 * np.pi, h.shape[0], endpoint=False)

    fig, (ax_h, ax_psi) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    im = ax_h.imshow(
        h,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], theta[0], theta[-1]],
        cmap="viridis",
    )
    ax_h.set_ylabel("Orientation")
    fig.colorbar(im, ax=ax_h, label="Activity")

    ax_psi.plot(t, psi, color="tab:red")
    ax_psi.set_ylabel("Bump orientation")
    ax_psi.set_xlabel("Time")

    fig.tight_layout()
    return fig


def plot_attractor_map(initial_orientations, final_orientations):
    """Scatter plot of initial vs. final bump orientations."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(initial_orientations, final_orientations, color="tab:blue")
    lims = [0, 2 * np.pi]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Initial orientation")
    ax.set_ylabel("Final orientation")
    ax.set_title("Attractor orientations")
    fig.tight_layout()
    return fig


def pca_from_scratch(X):
    """Compute PCA using eigen-decomposition of the covariance matrix.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows correspond to observations and columns to
        variables (neurons).

    Returns
    -------
    eigvals : ndarray
        Eigenvalues in descending order.
    eigvecs : ndarray
        Corresponding eigenvectors (columns).
    scores : ndarray
        Projection of the centered data onto the eigenvectors.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    scores = Xc @ eigvecs
    return eigvals, eigvecs, scores


def plot_pca(scores, orientations):
    """Scatter points in the PC1-PC2 plane colored by orientation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=orientations, cmap="hsv")
    fig.colorbar(sc, ax=ax, label="Initial orientation")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of attractor states")
    fig.tight_layout()
    return fig


def compute_energy(W, C0, h):
    """Return the network energy for a given activity state.

    Parameters
    ----------
    W : ndarray
        Symmetric connectivity matrix of shape ``(N, N)``.
    C0 : float
        Constant feedforward input.
    h : ndarray
        Activity vector of shape ``(N,)`` or matrix ``(N, T)``.

    Returns
    -------
    ndarray
        Energy for each column of ``h``.
    """
    r = poslin(h)
    if r.ndim == 1:
        r = r[:, None]
    Erec = -0.5 * np.einsum("ik,ij,jk->k", r, W, r)
    Eext = -C0 * np.sum(r, axis=0)
    Eleak = 0.5 * np.sum(r * r, axis=0)
    return Erec + Eext + Eleak


def _calculate_functions(psi, width, theta):
    """Helper implementing the MATLAB ``calculateFunctions`` routine."""
    N = len(theta)
    p = psi % (2 * np.pi)
    act = np.where((theta > p - width) & (theta < p + width))[0]
    if p + width >= 2 * np.pi:
        act = np.concatenate([act, np.where(theta < p + width - 2 * np.pi)[0]])
    if p - width < 0:
        act = np.concatenate([act, np.where(theta > p - width + 2 * np.pi)[0]])
    if act.size == 0:
        return np.nan, np.nan, np.nan
    diff = np.cos(theta[act] - p) - np.cos(width)
    fH0 = diff.mean()
    frho = (diff * np.cos(theta[act] - p)).mean()
    fpsi = (diff * np.sin(theta[act] - p)).mean()
    return fH0 / N, frho / N, fpsi / N


def energy_landscape(J1, J0, C0, theta, psi_samples, width_samples):
    """Compute the energy landscape over ``psi`` and ``width``.

    Parameters
    ----------
    J1, J0 : float
        Amplitude and baseline of the cosine connectivity profile.
    C0 : float
        Constant feedforward input.
    theta : ndarray
        Preferred orientations of the neurons (size ``N``).
    psi_samples : ndarray
        Orientations at which to evaluate the energy.
    width_samples : ndarray
        Bump half widths.

    Returns
    -------
    ndarray
        Energy with shape ``(len(width_samples), len(psi_samples))``.
    """
    N = len(theta)
    W = (J0 + J1 * np.cos(theta[:, None] - theta[None, :])) / N
    E = np.zeros((len(width_samples), len(psi_samples)))
    for i, psi in enumerate(psi_samples):
        for j, w in enumerate(width_samples):
            fH0, _, _ = _calculate_functions(psi, w, theta)
            rho = -C0 / (2 * (np.cos(w) + J0 * fH0))
            h = 2 * rho * (np.cos(theta - psi) - np.cos(w))
            E[j, i] = compute_energy(W, C0, h)
    return E


def plot_energy_landscape(E, psi_samples, width_samples):
    """Visualize an energy landscape heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        E,
        aspect="auto",
        origin="lower",
        extent=[psi_samples[0], psi_samples[-1], 2 * width_samples[0], 2 * width_samples[-1]],
        cmap="gray",
    )
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Bump width")
    fig.colorbar(im, ax=ax, label="Energy")
    ax.set_title("Energy landscape")
    fig.tight_layout()
    return fig
