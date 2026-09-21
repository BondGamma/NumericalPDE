"""

Two models, in the notation of the problem pack:

  (1) Geometric Brownian motion (GBM)
        dS = mu S dt + sigma S dW

      Has an exact solution, so it is the validation benchmark.  Working
      in the log-price X = log S, where Ito's lemma gives a *constant*-
      coefficient SDE:

        dX = (mu - sigma^2 / 2) dt + sigma dW

      and the transition is exactly Gaussian:

        X(t + dt) | X(t)  ~  N( X(t) + (mu - sigma^2/2) dt ,  sigma^2 dt )

      which is what makes exact simulation possible:
        X_{n+1} = X_n + (mu - sigma^2/2) dt + sigma sqrt(dt) Z_n
        S_{n+1} = exp(X_{n+1})

      This is the "exact GBM sampler" the problem pack contrasts with the
      Euler-Maruyama scheme.

  (2) Non-affine stochastic volatility (NASV)
        dX = (mu - g(Y)^2 / 2) dt + g(Y) dW^(1)
        dY = kappa (theta - Y) dt + xi sqrt(1 + Y^2) dW^(2)
        g(y) = sigma_min + (sigma_max - sigma_min) / (1 + exp(-y))
        d<W^(1), W^(2)>_t = rho dt

      No closed-form transition and no practical exact sampler.  Numerical
      SDE integration is part of the model solution, not a comparison
      against a known formula.

Why "non-affine"?  The Y-diffusion xi sqrt(1 + Y^2) is not affine in Y,
and the joint process (X, Y) does not have affine drift/diffusion in the
sense that would admit a Riccati ODE for the characteristic function (as
Heston does).  The logistic g(y) keeps instantaneous volatility bounded
between sigma_min and sigma_max; simulating X (not S) keeps S = exp(X) > 0
automatically.

Provided here (no tests, no convergence study):
    - gbm_drift, gbm_diffusion          (log-price form)
    - gbm_exact                         (log-price, exact)
    - gbm_exact_price                   (price, exact)
    - nasv_g, nasv_drift, nasv_diffusion
    - em_step_gbm, em_step_nasv         (one Euler-Maruyama step each)
    - main()                            demo: exact GBM path vs. EM path

Author: Team 11
Course: Numerical Analysis of ODEs and PDEs
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model 1: Geometric Brownian motion
#
#   dS = mu S dt + sigma S dW
#
# In log-price X = log S:
#
#   dX = (mu - sigma^2 / 2) dt + sigma dW
#
# Constant drift and constant diffusion -> the exact solution is a Gaussian
# transition, so GBM is the one model where discretisation
# error can be measured against a *pathwise* exact reference.
# ---------------------------------------------------------------------------

# Default benchmark parameters 
GBM_S0    = 100.0
GBM_MU    = 0.05
GBM_SIGMA = 0.20
GBM_T     = 1.0


def gbm_drift(x, mu=GBM_MU, sigma=GBM_SIGMA):
    """Drift of the log-price SDE:  (mu - sigma^2/2).  Constant in x."""
    return mu - 0.5 * sigma * sigma


def gbm_diffusion(x, mu=GBM_MU, sigma=GBM_SIGMA):
    """Diffusion of the log-price SDE:  sigma.  Constant in x.

    Because this is constant, the Milstein correction vanishes on X.
    To *see* strong order 1/2 for EM and order 1 for Milstein, discretise
    the price S, where the noise is state-dependent (sigma * S).  That is
    what em_step_gbm below does.
    """
    return sigma


def gbm_exact(X0, t, W, mu=GBM_MU, sigma=GBM_SIGMA):
    """Exact log-price path:  X(t) = X0 + (mu - sigma^2/2) t + sigma W(t).

    Parameters
    ----------
    X0 : float or (n_paths,) array
        Initial log-price, X0 = log(S0).
    t  : float or (n_times,) array
        Evaluation times.
    W  : (n_paths, n_times) array
        Brownian path values at those times, W(t) with W(0) = 0.

    Returns
    -------
    X : same shape as W, the exact log-price at the requested times.
    """
    X0 = np.asarray(X0, dtype=float)
    t = np.asarray(t, dtype=float)
    W = np.asarray(W, dtype=float)
    drift = (mu - 0.5 * sigma * sigma) * t
    # Broadcast: X0 and drift along the last axis, W carries the paths.
    if W.ndim == 2:
        return X0[..., None] + drift[None, :] + sigma * W
    return X0 + drift + sigma * W


def gbm_exact_price(S0, t, W, mu=GBM_MU, sigma=GBM_SIGMA):
    """Exact price path:  S(t) = S0 exp((mu - sigma^2/2) t + sigma W(t))."""
    S0 = np.asarray(S0, dtype=float)
    X0 = np.log(S0)
    return np.exp(gbm_exact(X0, t, W, mu=mu, sigma=sigma))


def em_step_gbm(S, dt, dW, mu=GBM_MU, sigma=GBM_SIGMA):
    """One Euler-Maruyama step for GBM on the *price* S.

        S_{n+1} = S_n + mu S_n dt + sigma S_n dW_n

    Noise is state-dependent here, so EM has strong order 1/2.  For the
    positivity-preserving exact sampler, use the log-price update:

        X_{n+1} = X_n + (mu - sigma^2/2) dt + sigma dW_n
        S_{n+1} = exp(X_{n+1})

    which is exact at the grid points and is not a discretisation at all.
    """
    return S + mu * S * dt + sigma * S * dW


def milstein_step_gbm(S, dt, dW, mu=GBM_MU, sigma=GBM_SIGMA):
    """One Milstein step for GBM on the price S.

        S_{n+1} = S_n + mu S_n dt + sigma S_n dW_n
                  + (1/2) sigma^2 S_n (dW_n^2 - dt)

    Strong order 1 on S.  On X this correction is identically zero because
    the X-diffusion is constant.
    """
    return S + mu * S * dt + sigma * S * dW + 0.5 * sigma * sigma * S * (dW * dW - dt)


# ---------------------------------------------------------------------------
# Model 2: Non-affine stochastic volatility (NASV)
#
#   dX = (mu - g(Y)^2 / 2) dt + g(Y) dW^(1)
#   dY = kappa (theta - Y) dt + xi sqrt(1 + Y^2) dW^(2)
#   g(y) = sigma_min + (sigma_max - sigma_min) / (1 + exp(-y))
#   d<W^(1), W^(2)>_t = rho dt
#
# State is z = (X, Y) in R^2.  S = exp(X) stays positive by construction.
# The Y-diffusion xi sqrt(1 + Y^2) is what makes the model non-affine.
# ---------------------------------------------------------------------------

# Default benchmark parameters (problem pack Section 4)
NASV_S0        = 100.0
NASV_Y0        = 0.0
NASV_MU        = 0.05
NASV_KAPPA     = 2.0
NASV_THETA     = -0.2
NASV_XI        = 0.6
NASV_RHO       = -0.7
NASV_SIGMA_MIN = 0.10
NASV_SIGMA_MAX = 0.50
NASV_T         = 1.0


def nasv_g(y, sigma_min=NASV_SIGMA_MIN, sigma_max=NASV_SIGMA_MAX):
    """Bounded logistic instantaneous volatility.

        g(y) = sigma_min + (sigma_max - sigma_min) / (1 + exp(-y))

    Maps R -> (sigma_min, sigma_max).  Smooth, monotone, saturating:
    g(-inf) = sigma_min, g(+inf) = sigma_max, g(0) = (sigma_min+sigma_max)/2.
    """
    return sigma_min + (sigma_max - sigma_min) / (1.0 + np.exp(-y))


def nasv_drift(z, mu=NASV_MU, kappa=NASV_KAPPA, theta=NASV_THETA,
               sigma_min=NASV_SIGMA_MIN, sigma_max=NASV_SIGMA_MAX):
    """Drift of the 2-D NASV state z = (X, Y).

        f_X = mu - g(Y)^2 / 2
        f_Y = kappa (theta - Y)
    """
    X, Y = z
    g = nasv_g(Y, sigma_min, sigma_max)
    return np.array([mu - 0.5 * g * g, kappa * (theta - Y)])


def nasv_diffusion(z, xi=NASV_XI, sigma_min=NASV_SIGMA_MIN,
                   sigma_max=NASV_SIGMA_MAX):
    """Diffusion matrix of the NASV state z = (X, Y).

        D = [[ g(Y)        , 0                 ],
             [ 0           , xi sqrt(1 + Y^2) ]]

    Kept diagonal here: the correlation rho is imposed on the *increments*
    dW^(1), dW^(2), not on the diffusion matrix.  This matches the problem
    pack's increment convention exactly.
    """
    X, Y = z
    g = nasv_g(Y, sigma_min, sigma_max)
    return np.array([[g, 0.0],
                     [0.0, xi * np.sqrt(1.0 + Y * Y)]])


def em_step_nasv(z, dt, dW1, dW2, mu=NASV_MU, kappa=NASV_KAPPA,
                 theta=NASV_THETA, xi=NASV_XI,
                 sigma_min=NASV_SIGMA_MIN, sigma_max=NASV_SIGMA_MAX):
    """One Euler-Maruyama step for NASV on z = (X, Y).

        X_{n+1} = X_n + (mu - g(Y_n)^2 / 2) dt + g(Y_n) dW_n^(1)
        Y_{n+1} = Y_n + kappa (theta - Y_n) dt + xi sqrt(1 + Y_n^2) dW_n^(2)

    The correlated increments dW1, dW2 must come from the Brownian engine:
        dW1 = sqrt(dt) Z1
        dW2 = sqrt(dt) (rho Z1 + sqrt(1 - rho^2) Z2)
    with Z1, Z2 iid N(0, 1).  A full multidimensional Milstein method for
    this two-noise system would need cross terms and simulated iterated
    stochastic integrals, so we do not quote a scalar Milstein order here.
    """
    X, Y = z
    g = nasv_g(Y, sigma_min, sigma_max)
    X_next = X + (mu - 0.5 * g * g) * dt + g * dW1
    Y_next = Y + kappa * (theta - Y) * dt + xi * np.sqrt(1.0 + Y * Y) * dW2
    return np.array([X_next, Y_next])


# ---------------------------------------------------------------------------
# Main: demo only.  Exact GBM path vs. Euler-Maruyama on S, one figure.
# No convergence study, no error tables.
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("sde_models.py - GBM and NASV SDE definitions")
    print("=" * 60)

    # --- GBM sanity: exact mean and variance at T -------------------------
    # E[S(T)] = S0 exp(mu T)   exactly
    # Var[S(T)] = S0^2 exp(2 mu T) (exp(sigma^2 T) - 1)   exactly
    S0, mu, sigma, T = GBM_S0, GBM_MU, GBM_SIGMA, GBM_T
    E_exact = S0 * np.exp(mu * T)
    Var_exact = S0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1.0)
    print(f"[1] GBM exact moments at T={T}:")
    print(f"    E[S(T)]   = {E_exact:.6f}")
    print(f"    Var[S(T)] = {Var_exact:.6f}")

    # --- one path: exact vs. EM, same Brownian increments -----------------
    n_steps = 252
    dt = T / n_steps
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(n_steps)
    dW = np.sqrt(dt) * Z
    W = np.concatenate(([0.0], np.cumsum(dW)))
    t = np.linspace(0.0, T, n_steps + 1)

    S_exact = gbm_exact_price(S0, t, W, mu=mu, sigma=sigma)

    S_em = np.empty(n_steps + 1)
    S_em[0] = S0
    for n in range(n_steps):
        S_em[n + 1] = em_step_gbm(S_em[n], dt, dW[n], mu=mu, sigma=sigma)

    # --- NASV demo: one EM path, just to show the interface --------------
    Y0 = NASV_Y0
    X0 = np.log(NASV_S0)
    z = np.array([X0, Y0])
    Z1 = rng.standard_normal(n_steps)
    Z2 = rng.standard_normal(n_steps)
    dW1 = np.sqrt(dt) * Z1
    dW2 = np.sqrt(dt) * (NASV_RHO * Z1 + np.sqrt(1.0 - NASV_RHO**2) * Z2)
    X_nasv = np.empty(n_steps + 1)
    Y_nasv = np.empty(n_steps + 1)
    X_nasv[0], Y_nasv[0] = z
    for n in range(n_steps):
        z = em_step_nasv(z, dt, dW1[n], dW2[n])
        X_nasv[n + 1], Y_nasv[n + 1] = z
    S_nasv = np.exp(X_nasv)

    print(f"[2] one NASV EM path: S(0)={S_nasv[0]:.4f}  "
          f"S(T)={S_nasv[-1]:.4f}  Y(T)={Y_nasv[-1]:+.4f}")

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    ax[0].plot(t, S_exact, label="GBM exact")
    ax[0].plot(t, S_em, "--", label="GBM Euler-Maruyama")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("S")
    ax[0].set_title("GBM: exact vs. EM on S (same W)")
    ax[0].legend(); ax[0].grid(True, ls=":", alpha=0.6)

    ax[1].plot(t, S_nasv, label="NASV EM  S = exp(X)")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("S")
    ax[1].set_title("NASV: one Euler-Maruyama path")
    ax[1].legend(); ax[1].grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig("sde_models_demo.png", dpi=150)
    plt.close(fig)

    print("=" * 60)
    print("wrote sde_models_demo.png")
    print("=" * 60)


if __name__ == "__main__":
    main()