"""Shared plotting helpers used across the experiments.

Public API:
    plot_paths             universal path plotter (all / sample / selected)
    plot_paths_band        ensemble confidence band + sampled paths
    plot_paths_band_slices confidence band + cross-section distributions
    plot_distribution      density/curve plotter with highlighted regions
"""

from code.visualizations.paths import plot_paths
from code.visualizations.paths import plot_paths_band
from code.visualizations.paths import plot_paths_band_slices
from code.visualizations.distributions import plot_distribution

__all__ = ["plot_paths", "plot_paths_band", "plot_paths_band_slices",
           "plot_distribution"]
