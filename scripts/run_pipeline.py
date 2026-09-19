#!/usr/bin/env python
"""
End-to-end SMLC reproduction pipeline (single entry point).

Run from the repository root:

    python scripts/run_pipeline.py

The raw UCI ElectricityLoadDiagrams20112014 file must be placed at
data/LD2011_2014.txt first (not distributed here — see data/README.md). All
tables (.tex/.csv) and the three data-dependent figures (.pdf) are written
to figures/. No figure is
displayed on screen (matplotlib's Agg backend is forced in smlc.figures), so
this runs identically on headless servers, CI, and batch queues.

This orchestrator only wires the modules together in order and passes state
explicitly; all logic lives in the smlc package (config, data, transforms,
models, metrics, evaluation, reserve, figures).
"""

import os
import sys
import warnings
import argparse

# Make the smlc package importable when run as `python scripts/run_pipeline.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from collections import Counter

from smlc.config import CFG, DIST_NORMAL, MODEL_ORDER
from smlc import data as data_mod
from smlc import transforms as tf
from smlc.models.smlc import ridge, ridge_predict
from smlc.models.qr_direct import fit_qr_direct, predict_qr
from smlc.models.fts import select_k_and_fit
from smlc.metrics import calibrate, gaussian_params
from smlc import evaluation as ev_mod
from smlc import reserve as reserve_mod
from smlc import figures as fig_mod


def main(cfg=CFG):
    warnings.filterwarnings('ignore')
    np.random.seed(cfg['seed'])
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # Echo configuration
    print('Configuration loaded:')
    for k, v in cfg.items():
        print(f'  {k}: {v}')

    H  = cfg['n_hours']
    nu = cfg['nu_student']
    lam = cfg['lambda_ridge']

    # ── 1. Data + partition ────────────────────────────────────────────────
    Y, dates = data_mod.load_daily_curves(cfg)
    part = data_mod.temporal_partition(Y, dates, cfg)
    Y_tr, Y_ca, Y_te = part['Y_tr'], part['Y_ca'], part['Y_te']
    d_te = part['d_te']

    # ── 2. Transforms + features ───────────────────────────────────────────
    tr = tf.make_transforms(part, cfg)
    feats = tf.make_features(part)
    X_tr, X_ca, X_te = feats['X_tr'], feats['X_ca'], feats['X_te']

    # ── 3. SMLC ridge point predictor ──────────────────────────────────────
    W_dct = ridge(X_tr, tr['Z_tr_dct'], lam)
    W_pca = ridge(X_tr, tr['Z_tr_pca'], lam)
    Zh_ca_dct = ridge_predict(X_ca, W_dct); Zh_te_dct = ridge_predict(X_te, W_dct)
    Zh_ca_pca = ridge_predict(X_ca, W_pca); Zh_te_pca = ridge_predict(X_te, W_pca)
    Yh_te_dct = Zh_te_dct @ tr['U_DCT'].T
    Yh_te_pca = Zh_te_pca @ tr['U_PCA'].T
    print(f'  W_dct in R^{W_dct.shape},  W_pca in R^{W_pca.shape}')

    # ── 4. QR-Direct ───────────────────────────────────────────────────────
    qr_W, qr_b = fit_qr_direct(X_tr, Y_tr, cfg['quantiles'], cfg['alpha_qr'], H)
    qr_preds_raw = {q: predict_qr(X_te, q, qr_W, qr_b) for q in cfg['quantiles']}

    # ── 5. Mixed-likelihood calibration + Gaussian factorial variants ─────
    cal_dct = calibrate(tr['Z_ca_dct'], Zh_ca_dct, nu)
    cal_pca = calibrate(tr['Z_ca_pca'], Zh_ca_pca, nu)
    dist_g,     params_g     = gaussian_params(tr['Z_ca_dct'], Zh_ca_dct)
    dist_g_pca, params_g_pca = gaussian_params(tr['Z_ca_pca'], Zh_ca_pca)
    print('Calibration OK — all params are (loc, scale) tuples.')
    print(f'\n  {"Mode":>4}  {"DCT":>10}  {"PCA":>10}')
    print('  ' + '-' * 28)
    for j in range(H):
        print(f'  {j+1:>4}  {cal_dct["distribs"][j]:>10}  {cal_pca["distribs"][j]:>10}')
    print(f'\n  DCT summary: {dict(Counter(cal_dct["distribs"]))}')
    print(f'  PCA summary: {dict(Counter(cal_pca["distribs"]))}')
    print('PCA-Gaussian variant calibrated (2x2 basis x distribution design complete).')

    # ── 6. FTS baseline (K selection + refit) ──────────────────────────────
    fts_best, fts_best_K, _ = select_k_and_fit(Y_tr, Y_ca, Y_te, cfg)
    samp_fts  = fts_best['test']['samples']
    Yh_te_fts = fts_best['test']['mean']
    nll_fts   = fts_best['test']['metrics']['NLL_per_hour']
    rmse_fts  = fts_best['test']['metrics']['RMSE']

    # ── Assemble the shared state dict ─────────────────────────────────────
    fit = dict(
        Y_tr=Y_tr, Y_ca=Y_ca, Y_te=Y_te, d_te=d_te,
        X_tr=X_tr, X_ca=X_ca, X_te=X_te,
        U_DCT=tr['U_DCT'], U_PCA=tr['U_PCA'],
        Z_tr_dct=tr['Z_tr_dct'], Z_ca_dct=tr['Z_ca_dct'], Z_te_dct=tr['Z_te_dct'],
        Z_ca_pca=tr['Z_ca_pca'], Z_te_pca=tr['Z_te_pca'],
        W_dct=W_dct, W_pca=W_pca,
        Zh_ca_dct=Zh_ca_dct, Zh_te_dct=Zh_te_dct,
        Zh_ca_pca=Zh_ca_pca, Zh_te_pca=Zh_te_pca,
        Yh_te_dct=Yh_te_dct, Yh_te_pca=Yh_te_pca,
        qr_preds_raw=qr_preds_raw,
        cal_dct=cal_dct, cal_pca=cal_pca,
        dist_g=dist_g, params_g=params_g,
        dist_g_pca=dist_g_pca, params_g_pca=params_g_pca,
        fts_best=fts_best, fts_best_K=fts_best_K,
        samp_fts=samp_fts, Yh_te_fts=Yh_te_fts, nll_fts=nll_fts, rmse_fts=rmse_fts,
    )

    # ── 7. Test evaluation, significance, cost ─────────────────────────────
    ev = ev_mod.build_ensembles(fit, cfg)
    table3_df = ev_mod.main_metrics_tables(fit, ev, cfg)
    ev_mod.wilcoxon_tests(fit, ev, cfg)

    # ── 8. Reserve-aware evaluation ────────────────────────────────────────
    reserve_df, r5vals = reserve_mod.hourly_reserve(fit, ev, cfg)
    reserve_mod.day_level_reserve(fit, ev, reserve_df, r5vals, cfg)

    # ── 9. Cost benchmark + output figures ─────────────────────────────────
    ev_mod.cost_benchmark(fit, ev, cfg)
    fig_mod.make_output_figures(fit, ev, cfg)

    print('\n' + '=' * 100)
    print('Pipeline finished. Tables and data-dependent figures are in:',
          cfg['figures_dir'] + '/')
    print('=' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the complete UCI SMLC experiment.')
    parser.add_argument('--data', help='Path to LD2011_2014.txt.')
    parser.add_argument('--figures-dir', help='Directory for generated tables and figures.')
    parser.add_argument('--cost-repeats', type=int,
                        help='Number of repetitions in the fitting-time benchmark.')
    args = parser.parse_args()

    run_cfg = dict(CFG)
    if args.data:
        run_cfg['data_path'] = os.path.abspath(args.data)
    if args.figures_dir:
        run_cfg['figures_dir'] = os.path.abspath(args.figures_dir)
    if args.cost_repeats is not None:
        if args.cost_repeats < 1:
            parser.error('--cost-repeats must be at least 1')
        run_cfg['cost_n_repeats'] = args.cost_repeats
    main(run_cfg)
