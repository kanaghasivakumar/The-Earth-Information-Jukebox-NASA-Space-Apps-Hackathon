"""Latent space and posterior visualization utilities."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def most_active_dimensions(mu, count=2):
    """Return latent dimensions with the largest across-frame mean variance."""
    mu = np.asarray(mu)
    if mu.ndim != 2 or mu.shape[1] == 0:
        raise ValueError("Latent means must be a nonempty (frames, dimensions) array.")
    count = min(int(count), mu.shape[1])
    return np.argsort(np.var(mu, axis=0))[-count:][::-1]


def posterior_surface(mu, logvar, path, title="Posterior", latent_dim=None, dimensions=None):
    mu = np.asarray(mu).reshape(-1)
    lv = np.asarray(logvar).reshape(-1)
    dims = np.asarray(dimensions if dimensions is not None else [0, 1], dtype=int)
    if len(dims) != 2 or np.any(dims < 0) or np.any(dims >= len(mu)):
        raise ValueError("Posterior surface requires two valid latent dimension indices.")
    mu = mu[dims]
    lv = lv[dims]
    std = np.exp(0.5 * lv[:2])
    lim = max(3.0, float(np.max(np.abs(mu[:2]) + 3 * std)))
    x = np.linspace(-lim, lim, 90)
    y = np.linspace(-lim, lim, 90)
    X, Y = np.meshgrid(x, y)

    def density(mx, my, sx, sy):
        return np.exp(-0.5 * ((X - mx) ** 2 / sx**2 + (Y - my) ** 2 / sy**2)) / (
            2 * np.pi * sx * sy
        )

    prior = density(0, 0, 1, 1)
    posterior = density(mu[0], mu[1], std[0], std[1])
    fig = plt.figure(figsize=(11, 4))
    for n, (Z, label) in enumerate(((prior, "Standard normal"), (posterior, title)), 1):
        ax = fig.add_subplot(1, 2, n, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, alpha=0.9)
        if n == 2:
            ax.scatter(
                [mu[0]],
                [mu[1]],
                [posterior.max()],
                color="red",
                s=28,
                label="posterior mean",
            )
            ax.legend()
        ax.set(
            xlabel=f"z{dims[0] + 1}",
            ylabel=f"z{dims[1] + 1}",
            zlabel="density",
            title=label,
        )
    if latent_dim and latent_dim > 2:
        fig.suptitle(
            f"Marginal over z{dims[0] + 1} and z{dims[1] + 1} of {latent_dim}D posterior"
        )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def latent_space_density_3d(mu, timestamps, path, grid_size=65, chunk_size=128):
    """Plot the empirical 2D latent space as a 3D density surface.

    The horizontal plane uses the two latent dimensions with the greatest
    across-frame variance; the vertical axis is estimated frame density, not
    another latent variable. Frame means are shown on the base plane and
    colored by timestamp.
    """
    mu = np.asarray(mu, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if mu.ndim != 2 or mu.shape[1] < 2:
        raise ValueError("A 3D latent-space density plot requires at least two latent dimensions.")
    if len(mu) == 0 or len(mu) != len(ts):
        raise ValueError("Latent means and timestamps must contain the same nonzero number of frames.")
    if not np.isfinite(mu).all() or not np.isfinite(ts).all():
        raise ValueError("Latent means and timestamps must be finite.")

    dimensions = most_active_dimensions(mu, 2)
    points = mu[:, dimensions]
    count = len(points)
    spread = points.std(axis=0)
    span = np.ptp(points, axis=0)
    # Scott's rule for a 2D Gaussian KDE, with a small floor for collapsed
    # dimensions so the plot remains visible when the encoder barely varies.
    bandwidth = np.maximum(1.06 * spread * count ** (-1 / 6), np.maximum(span * 0.02, 1e-5))
    half_width = np.maximum(span * 0.5, bandwidth * 3.5)
    center = points.mean(axis=0)
    x = np.linspace(center[0] - half_width[0], center[0] + half_width[0], grid_size)
    y = np.linspace(center[1] - half_width[1], center[1] + half_width[1], grid_size)
    X, Y = np.meshgrid(x, y)
    grid_x = X.ravel()
    grid_y = Y.ravel()
    density = np.zeros_like(grid_x)

    # Chunk the calculation so the plot stays light on memory for long videos.
    for start in range(0, count, chunk_size):
        sample = points[start : start + chunk_size]
        dx = (grid_x[None, :] - sample[:, 0, None]) / bandwidth[0]
        dy = (grid_y[None, :] - sample[:, 1, None]) / bandwidth[1]
        density += np.exp(-0.5 * (dx * dx + dy * dy)).sum(axis=0)

    density /= count * 2 * np.pi * bandwidth[0] * bandwidth[1]
    peak = density.max()
    if peak > 0:
        density /= peak
    density = density.reshape(X.shape)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, density, cmap="viridis", linewidth=0, antialiased=True, alpha=0.88)
    frame_points = ax.scatter(
        points[:, 0],
        points[:, 1],
        np.full(count, -0.04),
        c=ts,
        cmap="plasma",
        s=10,
        alpha=0.6,
        depthshade=False,
    )
    ax.set(
        xlabel=f"latent z{dimensions[0] + 1} (std={mu[:, dimensions[0]].std():.3g})",
        ylabel=f"latent z{dimensions[1] + 1} (std={mu[:, dimensions[1]].std():.3g})",
        zlabel="relative frame density",
        title="VAE latent space: most active dimensions (color: time)",
    )
    ax.set_zlim(-0.08, 1.03)
    ax.view_init(elev=30, azim=-55)
    fig.colorbar(frame_points, ax=ax, pad=0.12, shrink=0.7, label="timestamp (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return dimensions


def latent_plots(mu, timestamps, source_ids, out_dir):
    mu = np.asarray(mu)
    ts = np.asarray(timestamps)
    ids = np.asarray(source_ids)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dimensions = latent_space_density_3d(mu, ts, out / "latent_space_3d.png")
    points = mu[:, dimensions]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(points[:, 0], points[:, 1], c=ts, cmap="viridis", s=12, alpha=0.7)
    ax.set(
        xlabel=f"latent z{dimensions[0] + 1}",
        ylabel=f"latent z{dimensions[1] + 1}",
        title="Most active latent means (color: timestamp)",
    )
    fig.colorbar(ax.collections[0], ax=ax, label="timestamp (s)")
    fig.tight_layout()
    fig.savefig(out / "latent_scatter.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    for sid in np.unique(ids):
        ix = np.where(ids == sid)[0]
        order = ix[np.argsort(ts[ix])]
        ax.plot(points[order, 0], points[order, 1], "-o", markersize=2, label=f"source {sid}")
    ax.set(
        xlabel=f"latent z{dimensions[0] + 1}",
        ylabel=f"latent z{dimensions[1] + 1}",
        title="Connected trajectories in the most active dimensions",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "latent_trajectory.png")
    plt.close(fig)
    time_dimensions = most_active_dimensions(mu, min(4, mu.shape[1]))
    fig, axs = plt.subplots(
        len(time_dimensions), 1, figsize=(9, 2.5 * len(time_dimensions)),
        sharex=True, squeeze=False,
    )
    for d, ax in zip(time_dimensions, axs[:, 0]):
        for sid in np.unique(ids):
            ix = np.where(ids == sid)[0]
            ix = ix[np.argsort(ts[ix])]
            ax.plot(ts[ix], mu[ix, d], ".-", label=f"source {sid}")
        ax.set_ylabel(f"z{d + 1} (std={mu[:, d].std():.3g})")
    axs[-1, 0].set_xlabel("timestamp (s)")
    fig.tight_layout()
    fig.savefig(out / "latent_vs_time.png")
    plt.close(fig)
    return dimensions
