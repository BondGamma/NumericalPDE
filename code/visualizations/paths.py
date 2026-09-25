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
"""

import numpy as np
import matplotlib.pyplot as plt


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
