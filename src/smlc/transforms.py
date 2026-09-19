"""
Orthogonal spectral transforms (DCT-II and PCA) and the design-matrix features.

The transforms are fitted on the training block only. Basis equivariance
guarantees the point forecast — and hence RMSE — is
identical across orthogonal bases; these transforms only affect the uncertainty
layer built on top of the fixed predictor.
"""

import numpy as np
from sklearn.decomposition import PCA

from .config import CFG


def build_dct(H):
    """Return the orthogonal DCT-II basis with shape (H, H)."""
    U = np.zeros((H, H))
    for h in range(H):
        for k in range(H):
            U[h, k] = np.sqrt(2 / H) * np.cos(np.pi / H * (h + 0.5) * k)
    U[:, 0] /= np.sqrt(2)
    assert np.allclose(U.T @ U, np.eye(H), atol=1e-10), 'DCT is not orthogonal'
    return U


def build_pca(Y_tr, H, seed=CFG['seed']):
    """PCA basis fitted on the training block only. Returns U of shape (H, H)."""
    pca = PCA(n_components=H, random_state=seed)
    pca.fit(Y_tr)
    U = pca.components_.T
    assert np.allclose(U.T @ U, np.eye(H), atol=1e-10), 'PCA is not orthogonal'
    return U


def make_transforms(part, cfg=CFG):
    """Build DCT/PCA bases and project all three blocks.

    Parameters
    ----------
    part : dict
        Output of data.temporal_partition (Y_tr/ca/te).

    Returns
    -------
    dict with U_DCT, U_PCA and Z_{tr,ca,te}_{dct,pca}.
    """
    H = cfg['n_hours']
    Y_tr, Y_ca, Y_te = part['Y_tr'], part['Y_ca'], part['Y_te']

    U_DCT = build_dct(H)
    Z_tr_dct, Z_ca_dct, Z_te_dct = Y_tr @ U_DCT, Y_ca @ U_DCT, Y_te @ U_DCT

    U_PCA = build_pca(Y_tr, H, cfg['seed'])
    Z_tr_pca, Z_ca_pca, Z_te_pca = Y_tr @ U_PCA, Y_ca @ U_PCA, Y_te @ U_PCA

    # Verify Frobenius-norm invariance under each orthogonal transform.
    for name, Z in [('DCT', Z_tr_dct), ('PCA', Z_tr_pca)]:
        delta = abs(np.linalg.norm(Y_tr, 'fro') - np.linalg.norm(Z, 'fro'))
        print(f'  {name}: | ||Y||_F - ||Z||_F | = {delta:.2e}  (should be ~0)')

    return dict(U_DCT=U_DCT, U_PCA=U_PCA,
                Z_tr_dct=Z_tr_dct, Z_ca_dct=Z_ca_dct, Z_te_dct=Z_te_dct,
                Z_tr_pca=Z_tr_pca, Z_ca_pca=Z_ca_pca, Z_te_pca=Z_te_pca)


def build_X(Y_block, dates_block, y_prev_last):
    """Build X in R^(N x 12); row i uses the curve of day i-1."""
    N     = len(Y_block)
    X     = np.zeros((N, 12))
    Y_ext = np.vstack([y_prev_last[None, :], Y_block])
    for i in range(N):
        date  = dates_block[i]
        dow   = date.dayofweek
        mon   = date.month
        y_prev = Y_ext[i]
        X[i, 0]  = np.sin(2 * np.pi * dow / 7)
        X[i, 1]  = np.cos(2 * np.pi * dow / 7)
        X[i, 2]  = np.sin(2 * np.pi * mon / 12)
        X[i, 3]  = np.cos(2 * np.pi * mon / 12)
        X[i, 4]  = float(dow >= 5)
        X[i, 5]  = y_prev.mean()
        X[i, 6]  = y_prev.std()
        X[i, 7]  = y_prev[0]
        X[i, 8]  = y_prev[6]
        X[i, 9]  = y_prev[12]
        X[i, 10] = y_prev[18]
        X[i, 11] = y_prev[23]
    return X


def make_features(part):
    """Build z-score-standardized design matrices for all three blocks.
    Statistics come from TRAIN only (no leakage). Returns X_tr/ca/te."""
    Y_tr, Y_ca, Y_te = part['Y_tr'], part['Y_ca'], part['Y_te']
    d_tr, d_ca, d_te = part['d_tr'], part['d_ca'], part['d_te']

    X_tr_raw = build_X(Y_tr, d_tr, part['y_prev_tr'])
    X_ca_raw = build_X(Y_ca, d_ca, Y_tr[-1])
    X_te_raw = build_X(Y_te, d_te, Y_ca[-1])

    feat_mu = X_tr_raw.mean(axis=0)
    feat_sd = X_tr_raw.std(axis=0)
    feat_sd[feat_sd < 1e-12] = 1.0

    X_tr = (X_tr_raw - feat_mu) / feat_sd
    X_ca = (X_ca_raw - feat_mu) / feat_sd
    X_te = (X_te_raw - feat_mu) / feat_sd

    print(f'  X_tr in R^{X_tr.shape},  X_ca in R^{X_ca.shape},  X_te in R^{X_te.shape}')
    print(f"  Design-matrix conditioning: raw cond(X'X)={np.linalg.cond(X_tr_raw.T @ X_tr_raw):.2e}"
          f"  ->  standardized cond(X'X)={np.linalg.cond(X_tr.T @ X_tr):.2e}")
    return dict(X_tr=X_tr, X_ca=X_ca, X_te=X_te)
