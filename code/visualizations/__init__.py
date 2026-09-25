"""Shared plotting helpers used across the experiments.

Public API:
    plot_paths          universal path plotter (all / sample / selected)
    plot_distribution   density/curve plotter with highlighted regions
"""

from code.visualizations.paths import plot_paths
from code.visualizations.distributions import plot_distribution

__all__ = ["plot_paths", "plot_distribution"]
