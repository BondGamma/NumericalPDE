"""Universal path plotter.

``plot_paths`` draws one-dimensional trajectories onto a single axes and
supports the three selection modes needed across the experiments:

  * plot ALL paths (default),
  * plot a random SAMPLE of ``k`` paths,
  * plot a SELECTED subset by index.

It accepts three input shapes so it can serve both Monte-Carlo ensembles (many
paths on one shared time grid) and overlay figures (a few paths each on their
own grid, e.g. a fine reference path plus coarse samples):

  1. ``paths`` as a 2-D array ``(n_paths, n_pts)`` with a shared ``t``;
  2. ``paths`` as a list of 1-D arrays with a shared ``t``;
  3. ``paths`` as a list of ``(t_i, y_i)`` or ``(t_i, y_i, style)`` tuples,
     each with its own time grid (``t=None`` here; ``t_i`` must be an ndarray).

Two advanced ensemble plotters complement ``plot_paths``:

  * ``plot_paths_band`` draws a smooth Monte-Carlo confidence band (the region
    between polyfitted top/bottom quantiles at each time step) together with a
    random sample of paths;
  * ``plot_paths_band_slices`` builds on the band and additionally lays the
    cross-section distribution (discrete histogram + a fitted continuous
    density) at selected moments directly on the graph.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde as _gaussian_kde
except Exception:                                   # scipy is optional
    _gaussian_kde = None


def _broadcast(v, n):
    """Broadcast a scalar-or-sequence style value to a length-n list."""
    if v is None:
        return [None] * n
    if np.isscalar(v):
        return [v] * n
    return list(v)


def _is_path_spec(e):
    """True for a form-3 entry ``(t, y)`` / ``(t, y, style)``."""
    return (isinstance(e, (tuple, list)) and len(e) in (2, 3)
            and isinstance(e[0], np.ndarray))


def plot_paths(paths, t=None, *, sample=None, indices=None, rng=None, ax=None,
               figsize=(7, 4.5), labels=None, colors=None, markers=None,
               linestyles=None, linewidths=None, marker_sizes=None, alpha=None,
               title=None, xlabel="t", ylabel=None, legend=True, grid=True,
               hline=None, save_path=None, show=False, dpi=150):
    """Plot paths (all, a random sample, or a selected subset).

    Parameters
    ----------
    paths : ndarray or list
        One of the three forms documented in the module docstring.
    t : ndarray, optional
        Shared time grid (required for forms 1 and 2; ``None`` for form 3).
    sample : int, optional
        Plot a random subset of ``sample`` paths (seeded by ``rng``).
    indices : sequence of int, optional
        Plot only these path indices (mutually exclusive with ``sample``).
    rng : int or numpy.random.Generator, optional
        Seed / generator used for the random ``sample``.
    ax : matplotlib.axes.Axes, optional
        Draw on an existing axes; a new figure is created when ``None``.
    labels, colors, markers, linestyles, linewidths, marker_sizes, alpha :
        Per-path style, broadcast from a scalar (applied to all) or a sequence.
    hline : float, optional
        Horizontal reference line at this y-position.
    save_path, show, dpi : save / display controls.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Normalise `paths` to a list of (t_i, y_i, style) triples.
    if isinstance(paths, np.ndarray) and paths.ndim == 2:
        if t is None:
            raise ValueError("a shared time grid `t` is required for a 2-D paths array")
        specs = [(t, row, {}) for row in paths]
    elif isinstance(paths, (list, tuple)):
        entries = list(paths)
        if entries and _is_path_spec(entries[0]):          # form 3: per-path grids
            specs = []
            for e in entries:
                if len(e) == 2:
                    specs.append((e[0], e[1], {}))
                else:
                    specs.append((e[0], e[1], dict(e[2])))
        else:                                              # form 2: shared grid
            if t is None:
                raise ValueError("a shared time grid `t` is required for a list of 1-D arrays")
            specs = [(t, y, {}) for y in entries]
    else:
        raise ValueError("unsupported `paths` type: %r" % type(paths))

    # Selection: all / sample / indices.
    if sample is not None and indices is not None:
        raise ValueError("`sample` and `indices` are mutually exclusive")
    if sample is not None:
        gen = np.random.default_rng(rng)
        idx = gen.choice(len(specs), size=min(sample, len(specs)), replace=False)
        specs = [specs[i] for i in idx]
    elif indices is not None:
        specs = [specs[i] for i in indices]

    n = len(specs)
    labels = _broadcast(labels, n)
    colors = _broadcast(colors, n)
    markers = _broadcast(markers, n)
    linestyles = _broadcast(linestyles, n)
    linewidths = _broadcast(linewidths, n)
    marker_sizes = _broadcast(marker_sizes, n)
    alphas = _broadcast(alpha, n)

    for (ti, yi, style), lab, col, mk, ls, lw, ms, al in zip(
            specs, labels, colors, markers, linestyles, linewidths, marker_sizes, alphas):
        kw = dict(style)
        if lab is not None:
            kw["label"] = lab
        if col is not None:
            kw["color"] = col
        if mk is not None:
            kw["marker"] = mk
        if ls is not None:
            kw["linestyle"] = ls
        if lw is not None:
            kw["linewidth"] = lw
        if ms is not None:
            kw["markersize"] = ms
        if al is not None:
            kw["alpha"] = al
        ax.plot(ti, yi, **kw)

    if hline is not None:
        ax.axhline(hline, color="0.3", lw=0.8)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="small")
    if grid:
        ax.grid(True, ls=":", alpha=0.5)

    if save_path is not None or show:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
    return fig, ax


# --------------------------------------------------------------------------- #
# Advanced ensemble plotters (confidence band + cross-section distributions).  #
# --------------------------------------------------------------------------- #


def _band_curves(paths, t, q, poly_deg, n_dense=512):
    """Pointwise top/bottom quantiles, optionally smoothed by a polyfit.

    The confidence band is the region between the ``q`` and ``1 - q``
    empirical quantiles of the ensemble at every time step.  Fitting a
    polynomial through each quantile sequence turns the raw step-by-step band
    into a smooth, continuous region (the "continuous confidence interval").

    Returns
    -------
    t_band, lo, hi, mid : ndarray
        ``t_band`` is a dense grid when ``poly_deg`` is given, else ``t``.
        ``lo`` / ``hi`` bound the band; ``mid`` is the mean centreline.
    """
    t = np.asarray(t, dtype=float)
    lo = np.quantile(paths, q, axis=0)
    hi = np.quantile(paths, 1.0 - q, axis=0)
    mid = np.mean(paths, axis=0)
    if poly_deg is None:
        return t, lo, hi, mid
    t_band = np.linspace(t[0], t[-1], n_dense)
    lo = np.polyval(np.polyfit(t, lo, poly_deg), t_band)
    hi = np.polyval(np.polyfit(t, hi, poly_deg), t_band)
    mid = np.polyval(np.polyfit(t, mid, poly_deg), t_band)
    return t_band, lo, hi, mid


def _sample_indices(n_paths, sample, rng):
    """Random path indices for the overlay sample (empty when ``sample`` None)."""
    if sample is None:
        return []
    gen = np.random.default_rng(rng)
    return gen.choice(n_paths, size=min(int(sample), n_paths), replace=False)


def _fit_density(samples, v, fit):
    """Continuous density estimate of a cross-section on the grid ``v``.

    ``fit="kde"`` uses a Gaussian kernel-density estimate (shape-free; needs
    scipy, otherwise it falls back to a Gaussian).  ``fit="normal"`` fits a
    Gaussian with the sample mean / standard deviation.
    """
    if fit == "normal":
        mu = float(np.mean(samples))
        sd = float(np.std(samples))
        if sd <= 0.0:
            sd = float(np.ptp(samples))
            if sd <= 0.0:
                sd = 1e-12
        return np.exp(-0.5 * ((v - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))
    if fit == "kde":
        if _gaussian_kde is None:
            return _fit_density(samples, v, "normal")
        try:
            return _gaussian_kde(samples)(v)
        except Exception:
            return _fit_density(samples, v, "normal")
    raise ValueError("unknown `fit` %r; expected 'kde' or 'normal'" % (fit,))


def _cross_section(samples, vmin, vmax, nbins, fit, n_curve=200):
    """Discrete histogram + continuous density of one time cross-section.

    Both are normalised to a peak of 1 so they can share a single horizontal
    scale (``slice_scale`` sets how far that peak extends in data units).

    Returns
    -------
    centers, heights : ndarray
        Histogram bin centres (values) and normalised bar heights.
    v_curve, density : ndarray
        The continuous fit evaluated on a fine grid, normalised to peak 1.
    """
    samples = np.asarray(samples, dtype=float)
    counts, edges = np.histogram(samples, bins=nbins, range=(vmin, vmax))
    counts = counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = counts.max()
    heights = counts / peak if peak > 0.0 else np.zeros_like(counts)

    v_curve = np.linspace(vmin, vmax, n_curve)
    density = _fit_density(samples, v_curve, fit)
    dmax = density.max()
    density = density / dmax if dmax > 0.0 else np.zeros_like(density)
    return centers, heights, v_curve, density


def _draw_slice(ax, samples, tau, vmin, vmax, nbins, slice_scale, fit, color):
    """Lay one cross-section (discrete + continuous) onto the axes at ``tau``.

    The distribution is rotated so the value axis runs vertically: the
    histogram bars and the fitted density extend horizontally to the right of
    ``tau``.
    """
    centers, heights, v_curve, density = _cross_section(
        samples, vmin, vmax, nbins, fit)
    bin_width = (vmax - vmin) / nbins
    ax.barh(centers, heights * slice_scale, left=tau, height=bin_width,
            align="center", color=color, alpha=0.30, edgecolor="none")
    ax.plot(tau + density * slice_scale, v_curve, color=color, lw=1.6,
            label="density @ t=%.2f" % tau)


def _draw_band_and_samples(ax, paths, t, q, poly_deg, sample, rng, mean,
                           mean_color, mean_lw, band_color, band_alpha,
                           band_label, path_color, path_alpha, path_lw):
    """Shared body: confidence band + mean centreline + a path sample."""
    t_band, lo, hi, mid = _band_curves(paths, t, q, poly_deg)
    ax.fill_between(t_band, lo, hi, color=band_color, alpha=band_alpha,
                    label=band_label)
    if mean:
        ax.plot(t_band, mid,
                color=mean_color if mean_color is not None else band_color,
                lw=mean_lw, ls="--", label="mean")
    for k, i in enumerate(_sample_indices(paths.shape[0], sample, rng)):
        ax.plot(t, paths[i], color=path_color, alpha=path_alpha, lw=path_lw,
                label="sample path" if k == 0 else None)


def _finalize_axes(fig, ax, title, xlabel, ylabel, legend, grid,
                   save_path, show, dpi):
    """Apply the common title / labels / legend / grid / save block."""
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="small")
    if grid:
        ax.grid(True, ls=":", alpha=0.5)
    if save_path is not None or show:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()


def plot_paths_band(paths, t, *, q=0.05, poly_deg=8, sample=10, rng=None,
                    mean=True, mean_color=None, mean_lw=1.2,
                    band_color="C0", band_alpha=0.25, band_label=None,
                    path_color="0.4", path_alpha=0.6, path_lw=1.0,
                    ax=None, figsize=(7, 4.5), title=None, xlabel="t",
                    ylabel=None, legend=True, grid=True, save_path=None,
                    show=False, dpi=150):
    """Plot an ensemble with a smooth confidence band and a path sample.

    Method 1 of the advanced path plotters.  Given a Monte-Carlo ensemble of
    one-dimensional trajectories (pure Brownian motion, exact / EM / Milstein
    GBM, or the NA-SV price S), it draws

      * the continuous confidence region: at every discrete time step the top
        and bottom empirical quantiles are taken, a polynomial is fitted
        through each quantile sequence, and the space between the two fitted
        curves is shaded (coverage ``1 - 2 q``); and
      * a random sample of ``sample`` paths on the same axes.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_pts)
        Ensemble of trajectories on the shared grid ``t``.
    t : ndarray, shape (n_pts,)
        Shared time grid.
    q : float
        Lower tail quantile; the band spans ``[q, 1 - q]`` at each time step
        (``q=0.05`` gives a 90% interval).
    poly_deg : int or None
        Degree of the polynomial fitted through the top/bottom quantiles to
        smooth the band; ``None`` draws the raw stepwise band.
    sample : int or None
        Number of randomly drawn paths to overlay (``None`` for none).
    rng : int or numpy.random.Generator, optional
        Seed / generator for the random sample.
    mean : bool
        Draw the mean centreline.
    mean_color, mean_lw : colour / width of the mean line.
    band_color, band_alpha, band_label : style of the shaded region.
    path_color, path_alpha, path_lw : style of the sampled paths.
    ax : matplotlib.axes.Axes, optional
        Draw on an existing axes; a new figure is created when ``None``.
    title, xlabel, ylabel, legend, grid : labels / display controls.
    save_path, show, dpi : save / display controls.

    Returns
    -------
    fig, ax
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("`paths` must be a 2-D (n_paths, n_pts) array")
    t = np.asarray(t, dtype=float)
    if paths.shape[1] != t.shape[0]:
        raise ValueError("paths has %d points but t has %d"
                         % (paths.shape[1], t.shape[0]))
    if not 0.0 < q < 0.5:
        raise ValueError("`q` must be in (0, 0.5); got %r" % (q,))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _draw_band_and_samples(ax, paths, t, q, poly_deg, sample, rng, mean,
                           mean_color, mean_lw, band_color, band_alpha,
                           band_label, path_color, path_alpha, path_lw)
    _finalize_axes(fig, ax, title, xlabel, ylabel, legend, grid, save_path,
                   show, dpi)
    return fig, ax


def plot_paths_band_slices(paths, t, *, times=(1.0 / 3.0, 2.0 / 3.0, 1.0),
                           q=0.05, poly_deg=8, sample=10, rng=None, nbins=30,
                           slice_scale=None, fit="kde", slice_color="C3",
                           mean=True, mean_color=None, mean_lw=1.2,
                           band_color="C0", band_alpha=0.25, band_label=None,
                           path_color="0.4", path_alpha=0.6, path_lw=1.0,
                           ax=None, figsize=(7, 4.5), title=None, xlabel="t",
                           ylabel=None, legend=True, grid=True, save_path=None,
                           show=False, dpi=150):
    """Confidence band + path sample, plus cross-section distributions.

    Method 2, built on :func:`plot_paths_band`.  On top of the band and the
    sampled paths it selects a few moments ``times`` (default 1/3, 2/3, 1;
    each snapped to the nearest computed time), takes the ensemble's
    cross-section at each, and lays its distribution directly on the graph:

      * the discrete distribution — a histogram rotated so value runs
        vertically and counts run horizontally to the right of the moment; and
      * a continuous density fitted on top of it (KDE by default; a Gaussian
        with ``fit="normal"``).

    Parameters
    ----------
    paths, t, q, poly_deg, sample, rng, mean, mean_color, mean_lw,
    band_color, band_alpha, band_label, path_color, path_alpha, path_lw,
    ax, figsize, title, xlabel, ylabel, legend, grid, save_path, show, dpi :
        As in :func:`plot_paths_band`.
    times : sequence of float
        Moments at which to draw a cross-section; each is snapped to the
        nearest grid point of ``t``.
    nbins : int
        Histogram bins for the discrete cross-section.
    slice_scale : float or None
        Horizontal extent (in time units) of the tallest slice bar / density
        peak; defaults to 20% of the time span.
    fit : {"kde", "normal"}
        Continuous fit: Gaussian kernel-density estimate or a Gaussian.
    slice_color : str
        Colour of the cross-section bars and fitted density.

    Returns
    -------
    fig, ax
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("`paths` must be a 2-D (n_paths, n_pts) array")
    t = np.asarray(t, dtype=float)
    if paths.shape[1] != t.shape[0]:
        raise ValueError("paths has %d points but t has %d"
                         % (paths.shape[1], t.shape[0]))
    if not 0.0 < q < 0.5:
        raise ValueError("`q` must be in (0, 0.5); got %r" % (q,))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _draw_band_and_samples(ax, paths, t, q, poly_deg, sample, rng, mean,
                           mean_color, mean_lw, band_color, band_alpha,
                           band_label, path_color, path_alpha, path_lw)

    if slice_scale is None:
        slice_scale = 0.2 * (t[-1] - t[0])
    # Shared value range across slices so the three histograms are comparable.
    vmin = float(np.quantile(paths.ravel(), q))
    vmax = float(np.quantile(paths.ravel(), 1.0 - q))
    if vmax <= vmin:
        vmax = vmin + 1e-12

    for tau in times:
        j = int(np.argmin(np.abs(t - tau)))
        t_act = float(t[j])
        _draw_slice(ax, paths[:, j], t_act, vmin, vmax, nbins, slice_scale,
                    fit, slice_color)
        ax.axvline(t_act, color=slice_color, ls=":", lw=0.8, alpha=0.6)

    # Make room for the rightmost slice instead of letting it clip.
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(min(xlo, t[0]), max(xhi, t[-1] + slice_scale))

    _finalize_axes(fig, ax, title, xlabel, ylabel, legend, grid, save_path,
                   show, dpi)
    return fig, ax
