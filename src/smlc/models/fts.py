"""
FTS baseline: functional time-series forecasting via FPCA + per-score ARIMA.

FPCA on the training curves; ARIMA per score (order chosen by AIC) with a
rolling one-step state update; a per-day Gaussian predictive distribution with a
variance scale factor calibrated on the calibration block. K is selected on the
calibration block only.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA

from ..config import CFG
from ..metrics import (rmse, crps_mc, cal_err, coverage_sharpness_mc,
                       nll_gaussian_curves)


def select_arima_by_aic(y, candidate_orders):
    """Select ARIMA(p,d,q) by AIC. Fallback: ARIMA(0,0,0)."""
    best_fit, best_order, best_aic = None, None, np.inf
    y = np.asarray(y, dtype=float)
    for order in candidate_orders:
        try:
            fit = ARIMA(y, order=order).fit()
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic, best_order, best_fit = fit.aic, order, fit
        except Exception:
            continue
    if best_fit is None:
        best_order = (0, 0, 0); best_fit = ARIMA(y, order=best_order).fit()
    return best_order, best_fit


def rolling_arima_1step(fit, y_future):
    """Rolling one-step-ahead forecast, updating state without re-estimating."""
    res = fit; means, vars_ = [], []
    for y_obs in np.asarray(y_future, dtype=float):
        fc = res.get_forecast(steps=1)
        means.append(float(np.asarray(fc.predicted_mean)[0]))
        vars_.append(max(float(np.asarray(fc.var_pred_mean)[0]), 1e-8))
        res = res.append([float(y_obs)], refit=False)
    return np.asarray(means), np.asarray(vars_), res


def build_covs_and_samples_fts(score_means, score_vars, components,
                               mean_curve, resid_var_hour,
                               n_samples=1000, random_state=42):
    """Per-day covariances and sample curves for FTS."""
    rng = np.random.default_rng(random_state)
    N, K = score_means.shape; Hh = components.shape[1]
    Y_hat  = mean_curve[None, :] + score_means @ components
    covs   = np.zeros((N, Hh, Hh))
    Y_samp = np.zeros((N, n_samples, Hh))
    for i in range(N):
        cov_i = components.T @ np.diag(score_vars[i]) @ components
        cov_i = cov_i + np.diag(resid_var_hour) + 1e-6 * np.eye(Hh)
        covs[i] = cov_i
        ss = rng.normal(loc=score_means[i], scale=np.sqrt(np.maximum(score_vars[i], 1e-8)),
                        size=(n_samples, K))
        rs = rng.normal(loc=0.0, scale=np.sqrt(np.maximum(resid_var_hour, 1e-8)),
                        size=(n_samples, Hh))
        Y_samp[i] = mean_curve[None, :] + ss @ components + rs
    return Y_hat, covs, Y_samp


def run_fts_fpca_arima(Y_train, Y_cal, Y_test=None, K=6, candidate_orders=None,
                       alphas=None, alpha_map=None, n_samples=1000,
                       variance_calibration=True, random_state=42, cfg=CFG):
    """Full FTS pipeline. K and variance calibrated without test data."""
    if candidate_orders is None: candidate_orders = cfg['fts_candidate_orders']
    if alphas    is None: alphas    = cfg['alpha_levels']
    if alpha_map is None: alpha_map = cfg['alpha_map']

    pca_fts = PCA(n_components=K, random_state=cfg['seed'])
    scores_train = pca_fts.fit_transform(Y_train)
    scores_cal   = pca_fts.transform(Y_cal)
    components   = pca_fts.components_
    mean_curve   = pca_fts.mean_

    resid_var_hour = np.maximum(np.var(Y_train - pca_fts.inverse_transform(scores_train),
                                       axis=0, ddof=1), 1e-8)

    orders, fits = [], []
    for k in range(K):
        o, f = select_arima_by_aic(scores_train[:, k], candidate_orders)
        orders.append(o); fits.append(f)

    cal_means, cal_vars, fits_after_cal = [], [], []
    for k in range(K):
        m, v, fa = rolling_arima_1step(fits[k], scores_cal[:, k])
        cal_means.append(m); cal_vars.append(v); fits_after_cal.append(fa)
    cal_means = np.column_stack(cal_means)
    cal_vars  = np.column_stack(cal_vars)

    # Variance-scale calibration (calibration block only)
    var_factor = 1.0
    if variance_calibration:
        _, covs_unc, _ = build_covs_and_samples_fts(
            cal_means, cal_vars, components, mean_curve,
            resid_var_hour, n_samples=50, random_state=random_state)
        Y_cal_hat_tmp = mean_curve[None, :] + cal_means @ components
        diag_var = np.maximum(np.diagonal(covs_unc, axis1=1, axis2=2), 1e-8)
        var_factor = float(np.clip(np.nanmean((Y_cal - Y_cal_hat_tmp) ** 2 / diag_var), 0.1, 10.0))

    cal_vars_cal  = cal_vars * var_factor
    resid_var_cal = resid_var_hour * var_factor
    Y_cal_hat, covs_cal, samp_cal = build_covs_and_samples_fts(
        cal_means, cal_vars_cal, components, mean_curve,
        resid_var_cal, n_samples=n_samples, random_state=random_state)

    cov_c, sh_c = coverage_sharpness_mc(Y_cal, samp_cal, alphas, alpha_map)
    out = dict(K=K, orders=orders, variance_factor=var_factor,
               components=components, mean_curve=mean_curve,
               resid_var_hour=resid_var_cal,
               cal=dict(mean=Y_cal_hat, covs=covs_cal, samples=samp_cal,
                        metrics={'RMSE': rmse(Y_cal, Y_cal_hat),
                                 'NLL_per_hour': nll_gaussian_curves(Y_cal, Y_cal_hat, covs_cal),
                                 'CRPS': crps_mc(Y_cal, samp_cal),
                                 'CalErr': cal_err(cov_c), 'coverage': cov_c, 'sharpness': sh_c}))

    if Y_test is not None:
        scores_test = pca_fts.transform(Y_test)
        te_means, te_vars = [], []
        for k in range(K):
            m, v, _ = rolling_arima_1step(fits_after_cal[k], scores_test[:, k])
            te_means.append(m); te_vars.append(v)
        te_means = np.column_stack(te_means)
        te_vars  = np.column_stack(te_vars) * var_factor
        Y_te_hat, covs_te, samp_te = build_covs_and_samples_fts(
            te_means, te_vars, components, mean_curve,
            resid_var_cal, n_samples=n_samples, random_state=random_state + 1000)
        cov_t, sh_t = coverage_sharpness_mc(Y_test, samp_te, alphas, alpha_map)
        out['test'] = dict(mean=Y_te_hat, covs=covs_te, samples=samp_te,
                           metrics={'RMSE': rmse(Y_test, Y_te_hat),
                                    'NLL_per_hour': nll_gaussian_curves(Y_test, Y_te_hat, covs_te),
                                    'CRPS': crps_mc(Y_test, samp_te),
                                    'CalErr': cal_err(cov_t), 'coverage': cov_t, 'sharpness': sh_t})
    return out


def select_k_and_fit(Y_tr, Y_ca, Y_te, cfg=CFG):
    """Select K on the calibration block, then refit at the chosen K and
    evaluate on the test block. Returns (fts_best, fts_best_K, fts_cal_df)."""
    alphas    = cfg['alpha_levels']
    alpha_map = cfg['alpha_map']
    n_s       = cfg['n_mc_samples']
    criterion = cfg['fts_k_criterion']

    print(f'Selecting K on the calibration block (criterion: {criterion})...')
    fts_rows = []
    for K_fts in cfg['fts_k_grid']:
        print(f'  K={K_fts}...', end=' ')
        res_cal = run_fts_fpca_arima(Y_tr, Y_ca, Y_test=None, K=K_fts,
                                     n_samples=max(300, min(n_s, 500)),
                                     random_state=42 + K_fts, cfg=cfg)
        m = res_cal['cal']['metrics']
        fts_rows.append({'K': K_fts, 'NLL_per_hour': m['NLL_per_hour'], 'CRPS': m['CRPS'],
                         'RMSE': m['RMSE'], 'CalErr': m['CalErr'],
                         'variance_factor': res_cal['variance_factor']})
        print(f"{criterion}={fts_rows[-1][criterion]:.4f}")

    fts_cal_df = pd.DataFrame(fts_rows)
    fts_best_K = int(fts_cal_df.sort_values(criterion).iloc[0]['K'])
    print(f'\nSelected K by {criterion}: {fts_best_K}')
    print(fts_cal_df[['K', 'NLL_per_hour', 'CRPS', 'RMSE', 'CalErr', 'variance_factor']].to_string(index=False))

    print('\nRefitting with final K and evaluating on test...')
    fts_best = run_fts_fpca_arima(Y_tr, Y_ca, Y_test=Y_te, K=fts_best_K,
                                  n_samples=n_s, random_state=123, cfg=cfg)
    m_fts = fts_best['test']['metrics']
    print(f"  K={fts_best_K}, RMSE={m_fts['RMSE']:,.2f} Wh, NLL/h={m_fts['NLL_per_hour']:.4f}")
    return fts_best, fts_best_K, fts_cal_df
