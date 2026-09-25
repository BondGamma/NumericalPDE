"""Distribution / curve plotter with highlighted regions.

``plot_distribution`` draws a 1-D curve ``y(z)`` — most often a density — and
optionally shades one or more regions of the domain (e.g. the negative region
of a normal density), draws vertical marker lines, a horizontal reference line,
and scatter points.  Despite the name it is general enough to draw any curve
of ``z`` (e.g. the Milstein bracket parabola).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(z, y, *, ax=None, figsize=(7, 4.0), regions=None,
                      region_color="C3", region_alpha=0.5, vlines=None,
                      vline_color="C3", vline_ls="--", vline_lw=1.2,
                      hline=None, points=None, curve_color="0.2", curve_lw=1.6,
                      curve_label=None, title=None, xlabel="z", ylabel="density",
                      xlim=None, ylim=None, legend=True, grid=True,
                      save_path=None, show=False, dpi=150):
    """Plot a curve y(z) with optional shaded region(s) and reference lines.

    Parameters
    ----------
    z, y : array_like
        The curve to draw (``y`` evaluated on the grid ``z``).
    regions : sequence of (lo, hi[, label]), optional
        Shade ``y`` over ``lo <= z <= hi`` with ``fill_between``; ``lo``/``hi``
        may be ``-inf`` / ``+inf`` (clamped to the grid).
    region_color, region_alpha : colour / alpha of the shaded region(s).
    vlines : sequence of float or (float, label) tuples, optional
        Vertical dashed lines at these ``z`` positions.
    vline_color, vline_ls, vline_lw : colour / style of the vertical lines.
    hline : float, optional
        Horizontal line at this ``y`` position.
    points : (xs, ys) pair, optional
        Scatter markers.
    curve_color, curve_lw, curve_label : style of the y(z) curve.
    xlim, ylim : (lo, hi) tuples, optional.
    ax : matplotlib.axes.Axes, optional
        Draw on an existing axes; a new figure is created when ``None``.
    save_path, show, dpi : save / display controls.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    ax.plot(z, y, color=curve_color, lw=curve_lw, label=curve_label)

    if regions is not None:
        for region in regions:
            lo, hi = region[0], region[1]
            label = region[2] if len(region) > 2 else None
            if lo is None or lo == -np.inf:
                lo = z.min()
            if hi is None or hi == np.inf:
                hi = z.max()
            ax.fill_between(z, 0, y, where=((z >= lo) & (z <= hi)),
                            color=region_color, alpha=region_alpha, label=label)

    if vlines is not None:
        for v in vlines:
            if isinstance(v, (tuple, list)):
                xv, lab = v[0], v[1]
            else:
                xv, lab = v, None
            ax.axvline(xv, color=vline_color, ls=vline_ls, lw=vline_lw, label=lab)

    if hline is not None:
        ax.axhline(hline, color="0.3", lw=0.8)

    if points is not None:
        xs, ys = points
        ax.plot(xs, ys, "ko", ms=5)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
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
