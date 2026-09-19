"""
Unified evaluation metrics and mixed-likelihood calibration primitives.

Every probabilistic model is represented by the SAME object — an ensemble of M
Monte Carlo sample curves per test day — and every distributional metric is
computed by the SAME estimator across all models, so differences are attributable
to the models, not to the evaluation code.
"""

import numpy as np
from scipy.stats import norm, laplace, t as student_t, multivariate_normal
from scipy.optimize import minimize

from .config import DIST_NORMAL, DIST_LAPLACE, DIST_STUDENT


# ── Base metrics ──────────────────────────────────────────────────────────
def rmse(Y_true, Y_hat):
    """RMSE in the original space; invariant across complete orthogonal bases."""
    return np.sqrt(np.mean((Y_true - Y_hat) ** 2))


# ── Per-mode maximum-likelihood fits ──────────────────────────────────────
def mle_normal(r):
    mu, s = r.mean(), r.std()
    return -np.mean(norm.logpdf(r, loc=mu, scale=s)), (mu, s)


def mle_laplace(r):
    mu = np.median(r); b = np.mean(np.abs(r - mu))
    return -np.mean(laplace.logpdf(r, loc=mu, scale=b)), (mu, b)


def mle_student(r, nu=4):
    f = lambda p: -np.mean(student_t.logpdf(r, df=nu, loc=p[0], scale=abs(p[1]) + 1e-9))
    res = minimize(f, x0=[np.median(r), r.std()], method='Nelder-Mead')
    return res.fun, (res.x[0], abs(res.x[1]))


def calibrate(Z_ca, Zh_ca, nu=4):
    """Fit each mode's residual with the lowest-NLL family (Normal/Laplace/
    Student-t). Returns dict(R, distribs, params, nll_table)."""
    R = Z_ca - Zh_ca
    distribs, params, nll_table = [], [], []
    for j in range(R.shape[1]):
        r = R[:, j]
        n_nll, n_p = mle_normal(r)
        l_nll, l_p = mle_laplace(r)
        t_nll, t_p = mle_student(r, nu)
        table = {DIST_NORMAL: (n_nll, n_p), DIST_LAPLACE: (l_nll, l_p),
                 DIST_STUDENT: (t_nll, t_p)}
        winner = min(table, key=lambda k: table[k][0])
        distribs.append(winner)
        params.append(table[winner][1])          # <- always a (loc, scale) tuple
        nll_table.append({k: v[0] for k, v in table.items()})
    assert all(isinstance(p, tuple) and len(p) == 2 for p in params), \
        'calibrate(): found a non-tuple param — check the MLE functions above'
    return dict(R=R, distribs=distribs, params=params, nll_table=nll_table)


def gaussian_params(Z_ca, Zh_ca):
    """Force Normal on all modes — the Gaussian-baseline parameters.
    Returns (distribs, params) where distribs is [Normal]*H."""
    H = Z_ca.shape[1]
    distribs = [DIST_NORMAL] * H
    params = [((Z_ca[:, j] - Zh_ca[:, j]).mean(),
               np.sqrt(np.mean(((Z_ca[:, j] - Zh_ca[:, j])
                                - (Z_ca[:, j] - Zh_ca[:, j]).mean()) ** 2)))
              for j in range(H)]
    assert all(isinstance(p, tuple) and len(p) == 2 for p in params), \
        'gaussian_params: found a non-tuple entry'
    return distribs, params


# ── Test negative log-likelihood ──────────────────────────────────────────
def evaluate_nll(Z_te, Zh_te, distribs, params, nu=4):
    R = Z_te - Zh_te
    nlls = []
    for j in range(R.shape[1]):
        r, d = R[:, j], distribs[j]
        mu, p = params[j]                       # <- unpack (loc, scale)
        if   d == DIST_NORMAL:  nlls.append(-np.mean(norm.logpdf(r, loc=mu, scale=p)))
        elif d == DIST_LAPLACE: nlls.append(-np.mean(laplace.logpdf(r, loc=mu, scale=p)))
        else:                   nlls.append(-np.mean(student_t.logpdf(r, df=nu, loc=mu, scale=p)))
    return np.mean(nlls)


# ── SMLC predictive samples ───────────────────────────────────────────────
def sample_smlr(Zh_te, distribs, params, U, n=1000, nu=4):
    """Sample r_{t,j} ~ D_j(loc_j, scale_j) and reconstruct Y = (Zhat + R) U^T."""
    N, J = Zh_te.shape
    Rs   = np.zeros((N, n, J))
    for j in range(J):
        d = distribs[j]; mu, p = params[j]     # <- unpack (loc, scale)
        if   d == DIST_NORMAL:  Rs[:, :, j] = norm.rvs(loc=mu, scale=p, size=(N, n))
        elif d == DIST_LAPLACE: Rs[:, :, j] = laplace.rvs(loc=mu, scale=p, size=(N, n))
        else:                   Rs[:, :, j] = student_t.rvs(df=nu, loc=mu, scale=p, size=(N, n))
    return (Zh_te[:, None, :] + Rs) @ U.T


# ── QR-Direct: rearranged quantile stack and full-support sampling ────────
def rearranged_qr_stack(qr_preds, quantiles):
    """Stack (N, n_q, H) of quantile predictions, monotone-rearranged across
    levels (Chernozhukov et al., 2010) via per-(day,hour) sorting."""
    q_arr   = np.array(sorted(quantiles))
    q_stack = np.stack([qr_preds[q] for q in q_arr], axis=1)
    return q_arr, np.sort(q_stack, axis=1)


def sample_qr_full_support(q_arr, q_stack_sorted, n_samples, seed, rng=None):
    """Inverse-CDF sampling on the FULL unit interval: u ~ U(0,1); linear
    interpolation between estimated quantiles; flat extension beyond the
    outermost levels (np.interp clamps to end values). Hours are sampled
    independently — QR-Direct defines no joint structure. Shape (N,n,H)."""
    if rng is None: rng = np.random.default_rng(seed)
    N, n_q, Hh = q_stack_sorted.shape
    u   = rng.uniform(0.0, 1.0, size=(N, n_samples, Hh))
    out = np.empty((N, n_samples, Hh))
    for t in range(N):
        for h in range(Hh):
            out[t, :, h] = np.interp(u[t, :, h], q_arr, q_stack_sorted[t, :, h])
    return out


# ── CRPS — single MC estimator for ALL models ────────────────────────────
def crps_mc_per_day(Y_true, Y_samp):
    """Fair-CRPS MC estimator, averaged over hours, returned per day.
    CRPS = E|X-y| - 0.5 E|X-X'| with the M(M-1) unbiased pairing term.
    Y_true:(N,H), Y_samp:(N,M,H) -> (N,)."""
    t1     = np.abs(Y_samp - Y_true[:, None, :]).mean(axis=1)
    Y_sort = np.sort(Y_samp, axis=1)
    m      = Y_samp.shape[1]
    w      = (2 * np.arange(m) - m + 1) / (m * (m - 1))
    t2     = (Y_sort * w[None, :, None]).sum(axis=1)
    return (t1 - t2).mean(axis=1)


def crps_mc(Y_true, Y_samp):
    return float(crps_mc_per_day(Y_true, Y_samp).mean())


# ── Energy Score (multivariate) — per day and aggregate ──────────────────
def energy_score_per_day(Y_true, Y_samp):
    """ES(F,y) = E||X-y|| - 0.5 E||X-X'|| in R^24 (Gneiting & Raftery, 2007).
    Unbiased M(M-1) pairing. Y_true:(N,H), Y_samp:(N,M,H) -> (N,)."""
    N, M, _ = Y_samp.shape
    es = np.empty(N)
    for t in range(N):
        X, y  = Y_samp[t], Y_true[t]
        term1 = np.mean(np.linalg.norm(X - y[None, :], axis=1))
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        term2 = dists.sum() / (M * (M - 1))
        es[t] = term1 - 0.5 * term2
    return es


def energy_score_mc(Y_true, Y_samp):
    return float(energy_score_per_day(Y_true, Y_samp).mean())


# ── Variogram Score — per day and aggregate ───────────────────────────────
def variogram_score_per_day(Y_true, Y_samp, p=0.5):
    """VS_p(F,y) = sum_{i,j} (|y_i-y_j|^p - E|X_i-X_j|^p)^2, p=0.5, uniform
    weights (Scheuerer & Hamill, 2015). -> (N,)."""
    N, M, Hh = Y_samp.shape
    vs = np.empty(N)
    for t in range(N):
        X, y      = Y_samp[t], Y_true[t]
        obs_diff  = np.abs(y[:, None] - y[None, :]) ** p
        samp_diff = np.abs(X[:, :, None] - X[:, None, :]) ** p
        exp_diff  = samp_diff.mean(axis=0)
        vs[t]     = np.sum((obs_diff - exp_diff) ** 2)
    return vs


def variogram_score_mc(Y_true, Y_samp, p=0.5):
    return float(variogram_score_per_day(Y_true, Y_samp, p).mean())


# ── Multivariate Gaussian NLL (FTS) ───────────────────────────────────────
def nll_gaussian_curves(Y_true, Y_mean, covs):
    """Per-hour NLL for a per-day multivariate Gaussian. covs:(N,24,24)."""
    N, Hh = Y_true.shape; eye = np.eye(Hh); vals = []
    for i in range(N):
        try:
            lp = multivariate_normal.logpdf(Y_true[i], mean=Y_mean[i],
                                            cov=covs[i], allow_singular=False)
        except Exception:
            lp = multivariate_normal.logpdf(Y_true[i], mean=Y_mean[i],
                                            cov=covs[i] + 1e-6 * eye, allow_singular=True)
        vals.append(-lp / Hh)
    return float(np.mean(vals))


# ── Coverage and sharpness — SAME sample-based estimator for all models ───
def coverage_sharpness_mc(Y_true, Y_samp, alphas, alpha_map):
    cov, sh = {}, {}
    for a in alphas:
        lo_q, hi_q = alpha_map[a]
        lo = np.percentile(Y_samp, lo_q * 100, axis=1)
        hi = np.percentile(Y_samp, hi_q * 100, axis=1)
        cov[a] = ((Y_true >= lo) & (Y_true <= hi)).mean()
        sh[a]  = (hi - lo).mean()
    return cov, sh


def cal_err(cov):
    """CalErr = (1/4) sum_alpha |PICP_alpha - alpha|."""
    return np.mean([abs(v - a) for a, v in cov.items()])
