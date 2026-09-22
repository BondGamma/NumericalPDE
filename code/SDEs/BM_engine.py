"""
Brownian-motion engine — Direction 4 (problem pack Section 4).

Two simulators, one for each SDE model:

  (1) standard_bm         a single independent standard Brownian motion
                          (drives GBM):
                              dW_n = sqrt(dt) Z_n,  Z_n iid N(0, 1)

  (2) correlated_bm_pair  a correlated pair (W^(1), W^(2)) with
                          d<W^(1), W^(2)>_t = rho dt (drives NA-SV):
                              dW1_n = sqrt(dt) Z1_n
                              dW2_n = sqrt(dt) (rho Z1_n + sqrt(1 - rho^2) Z2_n)
                          with Z1, Z2 iid N(0, 1), so that
                          corr(dW1, dW2) = rho and Var(dW2) = dt.

Both return arrays of shape (n_paths, n_steps); a full Brownian path is the
cumulative sum (prepend W(0) = 0).  See the problem pack for the increment
convention.
"""

import numpy as np


def standard_bm(n_steps, dt, n_paths=1, seed=None):
    """Independent standard Brownian increments, shape (n_paths, n_steps).

        dW_n = sqrt(dt) * Z_n,   Z_n iid N(0, 1)

    Returns
    -------
    dW : ndarray, shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    return np.sqrt(dt) * Z


def correlated_bm_pair(n_steps, dt, rho, n_paths=1, seed=None):
    """Correlated increments (dW1, dW2) with correlation rho.

        dW1_n = sqrt(dt) Z1_n
        dW2_n = sqrt(dt) (rho Z1_n + sqrt(1 - rho^2) Z2_n),   Z1, Z2 iid N(0,1)

    Returns
    -------
    dW1, dW2 : ndarray, each of shape (n_paths, n_steps)
        With corr(dW1, dW2) = rho and Var(dW2) = dt.
    """
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    dW1 = np.sqrt(dt) * Z1
    dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1.0 - rho * rho) * Z2)
    return dW1, dW2


def bm_path(dW):
    """Cumulative Brownian path W from increments dW (prepends W(0) = 0).

    Parameters
    ----------
    dW : ndarray, shape (n_paths, n_steps)

    Returns
    -------
    W : ndarray, shape (n_paths, n_steps + 1), with W[:, 0] = 0.
    """
    dW = np.asarray(dW, dtype=float)
    if dW.ndim == 1:
        dW = dW[None, :]
    return np.concatenate(
        [np.zeros((dW.shape[0], 1)), np.cumsum(dW, axis=1)], axis=1
    )
