"""
QR-Direct baseline: 9 quantile levels x 24 hours = 216 independent linear
quantile regressions on the shared standardized design matrix.

Its independence copula over hourly marginals means no inter-hour dependence is
learned — directly explaining its Variogram Score. Crossing is removed by
monotone rearrangement (see metrics.rearranged_qr_stack) before evaluation.
"""

import numpy as np
from sklearn.linear_model import QuantileRegressor


def fit_qr_direct(X_tr, Y_tr, quantiles, alpha_qr, H):
    """Fit 216 quantile regressions. Returns (qr_W, qr_b) dicts keyed by q."""
    qr_W, qr_b = {}, {}
    print('Training QR-Direct...')
    for q in quantiles:
        Wq = np.zeros((X_tr.shape[1], H))
        bq = np.zeros(H)
        for h in range(H):
            qr = QuantileRegressor(quantile=q, alpha=alpha_qr, solver='highs')
            qr.fit(X_tr, Y_tr[:, h])
            Wq[:, h] = qr.coef_
            bq[h]    = qr.intercept_
        qr_W[q], qr_b[q] = Wq, bq
        print(f'  q={q:.3f} done', end='  ')
    print()
    print(f'  {len(quantiles)} quantile levels x {H} hours = {len(quantiles)*H} models')
    return qr_W, qr_b


def predict_qr(X_block, q, qr_W, qr_b):
    """Return the quantile prediction X W_q + b_q with shape (N, 24)."""
    return X_block @ qr_W[q] + qr_b[q]
