"""
Experiment 0 — smoke test of the "same Brownian motion" pipeline.

Workflow:
  1. tools/mc.simulate generates fine Brownian increments at the maximum
     resolution n_max and saves gbm_dw_log / nasv_dw_log under data/.
  2. SDEs/bm_engine.extract_dw coarsens a log to any smaller step count n
     (n | n_max) by summing blocks, so every step size sees the SAME
     underlying Brownian motion.
  3. This script plots each simulated Brownian path (fine) and overlays the
     coarse samples at the chosen n's; they must coincide, which confirms the
     unification.

Output: 4 PNGs in figures/ — 2 simulations x 2 BM engines (standard, pairing).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend; safe on any machine
import matplotlib.pyplot as plt

# Make the package importable whether this file is run as a script
# (`python code/experiments/exp0_simulate_bm.py`) or as a module
# (`python -m code.experiments.exp0_simulate_bm`), by putting the project root
# on sys.path.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _PROJECT_ROOT)

from code.tools import mc                       # noqa: E402
from code.SDEs import bm_engine                 # noqa: E402
from code.SDEs import nasv                      # noqa: E402

FIGURES_DIR = os.path.join(_PROJECT_ROOT, "figures")


def _coarse_path(dw_fine, n):
    """Cumulative coarse Brownian path sampled from fine increments at n steps."""
    coarse = bm_engine.extract_dw(dw_fine, n)
    return np.concatenate([[0.0], np.cumsum(coarse)])


def _plot_one(ax, dw_fine, n_max, n_list, dt, title):
    """Plot the fine path and overlay the coarse samples (they should match)."""
    t_fine = dt * np.arange(n_max + 1)
    W_fine = np.concatenate([[0.0], np.cumsum(dw_fine)])
    ax.plot(t_fine, W_fine, lw=1.0, color="0.35", label="fine (n=%d)" % n_max)
    # Larger n -> more sample points -> smaller pins, and plot in ascending-n
    # order so the smallest pins are drawn last (on top).  Otherwise all pins
    # at a shared grid point overlap and only the last colour is visible.
    for size, n in zip(np.linspace(8.0, 4.0, len(n_list)), sorted(n_list)):
        W = _coarse_path(dw_fine, n)
        t = (dt * n_max / n) * np.arange(n + 1)
        ax.plot(t, W, marker="o", ls="", ms=size, label="n=%d" % n)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("W(t)")
    ax.legend(fontsize="small")
    ax.grid(True, ls=":", alpha=0.5)


def main():
    n_sims = 2
    n_max = 1024
    T = 1.0
    rho = nasv.NA_SV_RHO
    n_list = [128, 256, 512]            # each divides n_max
    dt = T / n_max

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1) simulate + save the fine logs, then reload to test the round-trip.
    mc.simulate("standard", n_sims, n_max, T=T, seed=0)
    mc.simulate("pairing", n_sims, n_max, T=T, rho=rho, seed=0)
    gbm_log = mc.load("standard")       # (n_sims, n_max)
    nasv_log = mc.load("pairing")       # (2, n_sims, n_max)

    # 2) + 3) visualise each simulation: fine path + coarse samples overlaid.
    for i in range(n_sims):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot_one(ax, gbm_log[i], n_max, n_list, dt,
                  "Standard BM — simulation %d" % (i + 1))
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "bm_standard_sim%d.png" % (i + 1)),
                    dpi=150)
        plt.close(fig)

    for i in range(n_sims):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        _plot_one(axes[0], nasv_log[0, i], n_max, n_list, dt,
                  "Correlated BM W1 — sim %d" % (i + 1))
        _plot_one(axes[1], nasv_log[1, i], n_max, n_list, dt,
                  "Correlated BM W2 — sim %d" % (i + 1))
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "bm_pairing_sim%d.png" % (i + 1)),
                    dpi=150)
        plt.close(fig)

    print("wrote 4 figures to %s" % FIGURES_DIR)


if __name__ == "__main__":
    main()
