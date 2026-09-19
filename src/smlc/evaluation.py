"""Test-block metrics and statistical/computational evaluations."""

import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import QuantileRegressor

from .config import CFG, MODEL_ORDER, REFERENCE_MODEL
from .metrics import (crps_mc_per_day, energy_score_per_day, variogram_score_per_day,
                      coverage_sharpness_mc, cal_err, evaluate_nll, sample_smlr,
                      sample_qr_full_support, rearranged_qr_stack, rmse, calibrate)
from .models.smlc import ridge


def build_ensembles(fit, cfg=CFG):
    """Assemble the (N, M, 24) Monte Carlo ensemble for every model, plus the
    per-day CRPS/ES/VS, coverage/sharpness/CalErr, RMSE and NLL dicts.

    `fit` is the state dictionary produced by the preceding pipeline stages.
    """
    nu    = cfg['nu_student']
    n_s   = cfg['n_mc_samples']
    H     = cfg['n_hours']
    alphas, alpha_map = cfg['alpha_levels'], cfg['alpha_map']
    Y_te  = fit['Y_te']

    # QR-Direct: rearranged quantiles, point forecast, full-support samples
    q_arr, q_stack_sorted = rearranged_qr_stack(fit['qr_preds_raw'], cfg['quantiles'])
    q_stack_raw = np.stack([fit['qr_preds_raw'][q] for q in sorted(cfg['quantiles'])], axis=1)
    crossings   = int(np.any(np.diff(q_stack_raw, axis=1) < 0, axis=(1, 2)).sum())
    Yh_te_qr    = q_stack_sorted[:, list(q_arr).index(0.50), :]

    # RMSE
    rmse_dct = rmse(Y_te, fit['Yh_te_dct'])
    rmse_pca = rmse(Y_te, fit['Yh_te_pca'])
    rmse_qr  = rmse(Y_te, Yh_te_qr)

    # NLL (density-based models only)
    nll_dct   = evaluate_nll(fit['Z_te_dct'], fit['Zh_te_dct'], fit['cal_dct']['distribs'], fit['cal_dct']['params'], nu)
    nll_pca   = evaluate_nll(fit['Z_te_pca'], fit['Zh_te_pca'], fit['cal_pca']['distribs'], fit['cal_pca']['params'], nu)
    nll_gauss = evaluate_nll(fit['Z_te_dct'], fit['Zh_te_dct'], fit['dist_g'], fit['params_g'], nu)

    print('Generating Monte Carlo ensembles...')
    samp_dct       = sample_smlr(fit['Zh_te_dct'], fit['cal_dct']['distribs'], fit['cal_dct']['params'], fit['U_DCT'], n_s, nu)
    samp_pca       = sample_smlr(fit['Zh_te_pca'], fit['cal_pca']['distribs'], fit['cal_pca']['params'], fit['U_PCA'], n_s, nu)
    samp_gauss     = sample_smlr(fit['Zh_te_dct'], fit['dist_g'], fit['params_g'], fit['U_DCT'], n_s, nu)
    samp_gauss_pca = sample_smlr(fit['Zh_te_pca'], fit['dist_g_pca'], fit['params_g_pca'], fit['U_PCA'], n_s, nu)
    samp_qr        = sample_qr_full_support(q_arr, q_stack_sorted, n_s, cfg['seed'])

    ENSEMBLES = {
        'DCT-Mixed'   : samp_dct,
        'PCA-Mixed'   : samp_pca,
        'DCT-Gaussian': samp_gauss,
        'PCA-Gaussian': samp_gauss_pca,
        'QR-Direct'   : samp_qr,
        'FTS'         : fit['samp_fts'],
    }

    print('Computing per-day CRPS / ES / VS for all models...')
    crps_day = {m: crps_mc_per_day(Y_te, S)         for m, S in ENSEMBLES.items()}
    es_day   = {m: energy_score_per_day(Y_te, S)    for m, S in ENSEMBLES.items()}
    vs_day   = {m: variogram_score_per_day(Y_te, S) for m, S in ENSEMBLES.items()}

    cov_all, sh_all = {}, {}
    for m, S in ENSEMBLES.items():
        cov_all[m], sh_all[m] = coverage_sharpness_mc(Y_te, S, alphas, alpha_map)
    ce_all = {m: cal_err(c) for m, c in cov_all.items()}

    nll_gauss_pca = evaluate_nll(fit['Z_te_pca'], fit['Zh_te_pca'], fit['dist_g_pca'], fit['params_g_pca'], nu)

    RMSE_ALL = {'DCT-Mixed': rmse_dct, 'PCA-Mixed': rmse_pca, 'DCT-Gaussian': rmse_dct,
                'PCA-Gaussian': rmse_pca, 'QR-Direct': rmse_qr, 'FTS': fit['rmse_fts']}
    NLL_ALL  = {'DCT-Mixed': nll_dct, 'PCA-Mixed': nll_pca, 'DCT-Gaussian': nll_gauss,
                'PCA-Gaussian': nll_gauss_pca, 'QR-Direct': np.nan, 'FTS': fit['nll_fts']}

    POINT_ALL = {
        'DCT-Mixed'   : fit['Yh_te_dct'],
        'PCA-Mixed'   : fit['Yh_te_pca'],
        'DCT-Gaussian': fit['Yh_te_dct'],
        'PCA-Gaussian': fit['Yh_te_pca'],
        'QR-Direct'   : Yh_te_qr,
        'FTS'         : fit['Yh_te_fts'],
    }

    return dict(ENSEMBLES=ENSEMBLES, POINT_ALL=POINT_ALL,
                crps_day=crps_day, es_day=es_day, vs_day=vs_day,
                cov_all=cov_all, sh_all=sh_all, ce_all=ce_all,
                RMSE_ALL=RMSE_ALL, NLL_ALL=NLL_ALL,
                Yh_te_qr=Yh_te_qr, q_arr=q_arr, q_stack_sorted=q_stack_sorted,
                crossings=crossings, rmse_dct=rmse_dct)


def _mean_se(v):
    return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))


def main_metrics_tables(fit, ev, cfg=CFG):
    """Print the reported metrics and write the main table to CSV and LaTeX."""
    Y_te = fit['Y_te']
    n_s  = cfg['n_mc_samples']
    alphas = cfg['alpha_levels']
    fig_dir = cfg['figures_dir']
    crps_day, es_day, vs_day = ev['crps_day'], ev['es_day'], ev['vs_day']
    NLL_ALL, RMSE_ALL, ce_all = ev['NLL_ALL'], ev['RMSE_ALL'], ev['ce_all']
    cov_all, sh_all = ev['cov_all'], ev['sh_all']

    print('\n' + '=' * 118)
    print(f'Main metrics (test block, n={len(Y_te)} days). '
          f'Mean (standard error across days) for sample-based scores.')
    print('=' * 118)
    hdr = (f"{'Model':<19} {'NLL/h':>8} {'CRPS (Wh)':>20} {'ES (Wh)':>22} "
           f"{'VS':>26} {'RMSE (Wh)':>12} {'CalErr':>8}")
    print(hdr); print('-' * 118)
    rows_t3 = []
    for m in MODEL_ORDER:
        c_m, c_se = _mean_se(crps_day[m])
        e_m, e_se = _mean_se(es_day[m])
        v_m, v_se = _mean_se(vs_day[m])
        nll_s = f'{NLL_ALL[m]:>8.4f}' if not np.isnan(NLL_ALL[m]) else f'{"—":>8}'
        print(f"{m:<19} {nll_s} {c_m:>12,.1f} ({c_se:,.0f}) {e_m:>13,.1f} ({e_se:,.0f}) "
              f"{v_m:>15,.0f} ({v_se:,.0f}) {RMSE_ALL[m]:>12,.1f} {ce_all[m]:>8.4f}")
        rows_t3.append({'Model': m, 'NLL/h': NLL_ALL[m], 'CRPS (Wh)': c_m, 'CRPS SE': c_se,
                        'ES (Wh)': e_m, 'ES SE': e_se, 'VS': v_m, 'VS SE': v_se,
                        'RMSE (Wh)': RMSE_ALL[m], 'CalErr': ce_all[m]})
    print('=' * 118)
    print(f'  Basis-invariant RMSE (spectral models): {ev["rmse_dct"]:,.2f} Wh')
    print(f'  QR-Direct raw quantile crossing: {ev["crossings"]}/{len(Y_te)} days '
          f'({ev["crossings"]/len(Y_te)*100:.1f}%) — corrected by monotone rearrangement before evaluation')
    print(f"  FTS: K={fit['fts_best_K']}, variance factor={fit['fts_best']['variance_factor']:.4f}")

    table3_df = pd.DataFrame(rows_t3)
    table3_df.to_csv(os.path.join(fig_dir, 'table_main_metrics.csv'), index=False)
    with open(os.path.join(fig_dir, 'table_main_metrics.tex'), 'w') as f:
        f.write(table3_df.round(4).to_latex(
            index=False, escape=True,
            caption=(f'Main metrics on the test block ($n={len(Y_te)}$ days). Sample-based '
                     'scores (CRPS, ES, VS) are computed with a common Monte Carlo estimator '
                     f'on $M={n_s}$ sample curves per model per day; standard errors are '
                     'across test days. NLL is reported only for density-based models.'),
            label='tab:main_metrics'))
    print(f"\n  LaTeX: {os.path.join(fig_dir, 'table_main_metrics.tex')}")

    print('\n' + '=' * 62)
    print('Empirical coverage PICP (%)')
    print('=' * 62)
    print(f"{'Model':<19}" + ''.join(f"{str(int(a*100))+'%':>10}" for a in alphas))
    print('-' * 62)
    for m in MODEL_ORDER:
        print(f"{m:<19}" + ''.join(f"{cov_all[m][a]*100:>9.1f}%" for a in alphas))

    print('\n' + '=' * 70)
    print('Sharpness (PI width, Wh)')
    print('=' * 70)
    print(f"{'Model':<19}" + ''.join(f"{str(int(a*100))+'%':>13}" for a in alphas))
    print('-' * 70)
    for m in MODEL_ORDER:
        print(f"{m:<19}" + ''.join(f"{sh_all[m][a]:>13,.1f}" for a in alphas))

    return table3_df


def wilcoxon_tests(fit, ev, cfg=CFG):
    """Paired Wilcoxon signed-rank tests (Holm-corrected). Returns significance_df."""
    Y_te = fit['Y_te']
    fig_dir = cfg['figures_dir']
    crps_day, es_day, vs_day = ev['crps_day'], ev['es_day'], ev['vs_day']
    baselines = [m for m in MODEL_ORDER if m != REFERENCE_MODEL]

    rows, raw_pvals = [], []
    for metric_name, score_dict in [('CRPS', crps_day), ('ES', es_day), ('VS', vs_day)]:
        for base in baselines:
            diff = score_dict[REFERENCE_MODEL] - score_dict[base]
            stat, p_raw = wilcoxon(diff, alternative='two-sided')
            rows.append({
                'Metric'                : metric_name,
                'Baseline'              : base,
                'Mean diff (ref - base)': diff.mean(),
                'Median diff'           : np.median(diff),
                'Win rate (%)'          : 100 * float(np.mean(diff < 0)),
                'Wilcoxon W'            : stat,
                'p (raw)'               : p_raw,
            })
            raw_pvals.append(p_raw)

    significance_df = pd.DataFrame(rows)
    _, p_holm, _, _ = multipletests(raw_pvals, alpha=0.05, method='holm')
    significance_df['p (Holm)'] = p_holm
    significance_df['Significant (a=0.05)'] = significance_df['p (Holm)'] < 0.05

    with pd.option_context('display.float_format', lambda x: f'{x:,.4g}'):
        print('=' * 100)
        print(f'Paired Wilcoxon signed-rank tests — {REFERENCE_MODEL} vs. baselines '
              f'(Holm-corrected over {len(raw_pvals)} tests)')
        print('=' * 100)
        print(significance_df.to_string(index=False))

    with open(os.path.join(fig_dir, 'table_wilcoxon.tex'), 'w') as f:
        f.write(significance_df.round(4).to_latex(
            index=False, escape=True,
            caption=(f'Two-sided paired Wilcoxon signed-rank tests ($n={len(Y_te)}$ days, '
                     f'Holm-corrected over {len(raw_pvals)} comparisons) between '
                     f'{REFERENCE_MODEL} and each baseline on CRPS, Energy Score (ES) and '
                     'Variogram Score (VS). Negative differences indicate the reference '
                     'model outperforms the baseline on that day.'),
            label='tab:wilcoxon'))
    print(f"\nLaTeX: {os.path.join(fig_dir, 'table_wilcoxon.tex')}")
    return significance_df


def _time_call(fn, *args, n_repeats, **kwargs):
    """Run fn n_repeats times; return (last_result, mean_s, std_s)."""
    times, result = [], None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, float(np.mean(times)), float(np.std(times))


def cost_benchmark(fit, ev, cfg=CFG):
    """Wall-clock computational-cost benchmark. Returns cost_df."""
    N_REPEATS = cfg['cost_n_repeats']
    nu, n_s, H = cfg['nu_student'], cfg['n_mc_samples'], cfg['n_hours']
    lam = cfg['lambda_ridge']
    fig_dir = cfg['figures_dir']
    X_tr, Y_tr = fit['X_tr'], fit['Y_tr']
    Z_tr_dct, Z_ca_dct, Zh_ca_dct, Zh_te_dct = fit['Z_tr_dct'], fit['Z_ca_dct'], fit['Zh_ca_dct'], fit['Zh_te_dct']
    U_DCT = fit['U_DCT']

    cost_rows = []

    # Shared ridge point predictor (SMLC variants + DCT-Gaussian baseline)
    _, t_fit_ridge, s_fit_ridge = _time_call(ridge, X_tr, Z_tr_dct, lam, n_repeats=N_REPEATS)
    _, t_cal_mixed, s_cal_mixed = _time_call(calibrate, Z_ca_dct, Zh_ca_dct, nu, n_repeats=N_REPEATS)
    _, t_cal_gauss, s_cal_gauss = _time_call(
        lambda: [((Z_ca_dct[:, j] - Zh_ca_dct[:, j]).mean(),
                 np.sqrt(np.mean(((Z_ca_dct[:, j] - Zh_ca_dct[:, j])
                                  - (Z_ca_dct[:, j] - Zh_ca_dct[:, j]).mean()) ** 2)))
                for j in range(H)], n_repeats=N_REPEATS)
    _, t_samp_smlr, s_samp_smlr = _time_call(
        sample_smlr, Zh_te_dct, fit['cal_dct']['distribs'], fit['cal_dct']['params'], U_DCT, n_s, nu,
        n_repeats=N_REPEATS)
    _, t_samp_g, s_samp_g = _time_call(
        sample_smlr, Zh_te_dct, fit['dist_g'], fit['params_g'], U_DCT, n_s, nu, n_repeats=N_REPEATS)

    cost_rows.append({'Model': 'DCT-Mixed / PCA-Mixed (shared point predictor)',
                      'Fit time (s)': t_fit_ridge, 'Fit std (s)': s_fit_ridge,
                      'Calibration time (s)': t_cal_mixed, 'Calibration std (s)': s_cal_mixed,
                      'Sampling time, test block (s)': t_samp_smlr, 'Sampling std (s)': s_samp_smlr})
    cost_rows.append({'Model': 'DCT-Gaussian',
                      'Fit time (s)': t_fit_ridge, 'Fit std (s)': s_fit_ridge,
                      'Calibration time (s)': t_cal_gauss, 'Calibration std (s)': s_cal_gauss,
                      'Sampling time, test block (s)': t_samp_g, 'Sampling std (s)': s_samp_g})

    # QR-Direct: 216 quantile regressions
    def fit_qr_direct():
        qr_W_tmp, qr_b_tmp = {}, {}
        for q in cfg['quantiles']:
            Wq = np.zeros((X_tr.shape[1], H)); bq = np.zeros(H)
            for h in range(H):
                qr = QuantileRegressor(quantile=q, alpha=cfg['alpha_qr'], solver='highs')
                qr.fit(X_tr, Y_tr[:, h])
                Wq[:, h] = qr.coef_; bq[h] = qr.intercept_
            qr_W_tmp[q], qr_b_tmp[q] = Wq, bq
        return qr_W_tmp, qr_b_tmp

    _, t_fit_qr, s_fit_qr = _time_call(fit_qr_direct, n_repeats=N_REPEATS)
    _, t_samp_qr, s_samp_qr = _time_call(
        sample_qr_full_support, ev['q_arr'], ev['q_stack_sorted'], n_s, cfg['seed'], n_repeats=N_REPEATS)

    cost_rows.append({'Model': 'QR-Direct (216 quantile regressions)',
                      'Fit time (s)': t_fit_qr, 'Fit std (s)': s_fit_qr,
                      'Calibration time (s)': 0.0, 'Calibration std (s)': 0.0,
                      'Sampling time, test block (s)': t_samp_qr, 'Sampling std (s)': s_samp_qr})

    # FTS: K-grid search and final refit
    from .models.fts import run_fts_fpca_arima
    Y_ca, Y_te = fit['Y_ca'], fit['Y_te']
    _, t_fts_search, s_fts_search = _time_call(
        lambda: [run_fts_fpca_arima(Y_tr, Y_ca, Y_test=None, K=k,
                                    n_samples=max(300, min(n_s, 500)), random_state=42 + k, cfg=cfg)
                 for k in cfg['fts_k_grid']], n_repeats=N_REPEATS)
    _, t_fts_refit, s_fts_refit = _time_call(
        run_fts_fpca_arima, Y_tr, Y_ca, Y_te, fit['fts_best_K'], None, None, None, n_s, True, 123, cfg,
        n_repeats=N_REPEATS)

    cost_rows.append({'Model': f'FTS — K grid search ({len(cfg["fts_k_grid"])} candidates)',
                      'Fit time (s)': t_fts_search, 'Fit std (s)': s_fts_search,
                      'Calibration time (s)': np.nan, 'Calibration std (s)': np.nan,
                      'Sampling time, test block (s)': np.nan, 'Sampling std (s)': np.nan})
    cost_rows.append({'Model': f"FTS — final refit (K={fit['fts_best_K']}, incl. sampling)",
                      'Fit time (s)': t_fts_refit, 'Fit std (s)': s_fts_refit,
                      'Calibration time (s)': 0.0, 'Calibration std (s)': 0.0,
                      'Sampling time, test block (s)': np.nan, 'Sampling std (s)': np.nan})

    cost_df = pd.DataFrame(cost_rows)
    cost_df['Total fit time (s)'] = cost_df[['Fit time (s)', 'Calibration time (s)']].sum(axis=1, skipna=True)

    with pd.option_context('display.float_format', lambda x: f'{x:,.4g}'):
        print('=' * 100)
        print('Computational cost (wall-clock, mean over repeated runs; hardware dependent)')
        print('=' * 100)
        print(cost_df.to_string(index=False))

    with open(os.path.join(fig_dir, 'table_computational_cost.tex'), 'w') as f:
        f.write(cost_df.round(4).to_latex(
            index=False, escape=True,
            caption=('Wall-clock computational cost (mean over repeated runs). '
                     'Wall-clock values are hardware dependent. FTS is reported both '
                     'as the full $K$-grid search used for model selection and as the final '
                     'refit at the selected $K$.'),
            label='tab:cost'))
    print(f"\nLaTeX: {os.path.join(fig_dir, 'table_computational_cost.tex')}")
    return cost_df
