"""
SMLC point predictor: per-mode Ridge with an explicit unpenalized intercept.

The calibration and sampling of the per-mode mixed likelihood live in
smlc.metrics (calibrate, gaussian_params, sample_smlr, evaluate_nll); this
module provides only the deterministic Ridge fit, which — by the basis
equivariance proposition — is what makes RMSE identical across bases.
"""

import numpy as np


def ridge(X, Z, lam):
    """Closed-form ridge with an explicit UNPENALIZED intercept.
    Returns W of shape (1+p, J): row 0 is the intercept."""
    Xa = np.hstack([np.ones((X.shape[0], 1)), X])
    P  = np.eye(Xa.shape[1]); P[0, 0] = 0.0     # do not penalize the intercept
    A  = Xa.T @ Xa + lam * P
    cond_num = np.linalg.cond(A)
    if cond_num > 1e12:
        print(f'  Warning: ill-conditioned matrix (cond={cond_num:.2e})')
    return np.linalg.solve(A, Xa.T @ Z)


def ridge_predict(X, W):
    """Prediction with intercept: Zhat = 1*W[0] + X W[1:]."""
    return W[0:1, :] + X @ W[1:, :]
