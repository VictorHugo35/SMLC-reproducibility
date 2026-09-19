#!/usr/bin/env python
"""Fast, data-free checks for the released SMLC implementation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smlc.models.smlc import ridge, ridge_predict  # noqa: E402
from smlc.transforms import build_dct  # noqa: E402


def main():
    rng = np.random.default_rng(42)
    n, p, h = 40, 12, 24
    x = rng.normal(size=(n, p))
    y = rng.normal(size=(n, h))
    u = build_dct(h)

    assert u.shape == (h, h)
    assert np.allclose(u.T @ u, np.eye(h), atol=1e-10)

    w_identity = ridge(x, y, lam=1.0)
    w_spectral = ridge(x, y @ u, lam=1.0)
    yhat_identity = ridge_predict(x, w_identity)
    yhat_spectral = ridge_predict(x, w_spectral) @ u.T

    assert w_identity.shape == (p + 1, h)
    assert w_spectral.shape == (p + 1, h)
    assert yhat_spectral.shape == (n, h)
    assert np.allclose(yhat_identity, yhat_spectral, atol=1e-10)

    # Column-vector mathematics and row-stacked NumPy storage are transposes
    # of the same operation: z_t = U^T y_t <=> z_t^T = y_t^T U.
    y_column = y[0][:, None]
    assert np.allclose((u.T @ y_column).T, y[0:1] @ u, atol=1e-12)

    print('PASS: DCT orthogonality, Ridge basis equivariance, array dimensions,')
    print('      and column-vector/row-batch equivalence.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
