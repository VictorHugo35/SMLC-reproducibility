"""
Reserve-aware operational evaluation.

Reuses the Monte Carlo ensembles already generated for CRPS/ES/VS to compute,
per model and nominal risk level, the procured upward reserve and the empirical
shortage rate — at the hourly-marginal level, the day (full-curve) level, and
under an iso-risk (matched realized shortage) comparison.
"""

import os

import numpy as np
import pandas as pd

from .config import CFG, MODEL_ORDER


def _reserve_risk_curve(Y_true, samples, point_forecast, eps_grid, scope):
    """Return mean reserve and realized shortage along a fine epsilon grid.

    ``scope='hourly'`` evaluates marginal hour-by-hour quantiles.  ``scope='day'``
    evaluates the quantile of each sampled curve's worst-hour excess.  Keeping
    both constructions here makes the matched-risk calculations use exactly the
    same definitions as the nominal-risk tables above.
    """
    reserves, shortages = [], []
    if scope == 'day':
        sample_excess = (samples - point_forecast[:, None, :]).max(axis=2)
        actual_excess = (Y_true - point_forecast).max(axis=1)

    for eps in eps_grid:
        if scope == 'hourly':
            upper = np.quantile(samples, 1 - eps, axis=1)
            reserve = np.clip(upper - point_forecast, 0.0, None)
            shortage = (Y_true > upper).mean() * 100
        elif scope == 'day':
            reserve = np.clip(
                np.quantile(sample_excess, 1 - eps, axis=1), 0.0, None)
            shortage = (actual_excess > reserve).mean() * 100
        else:
            raise ValueError("scope must be 'hourly' or 'day'")
        reserves.append(float(np.mean(reserve)))
        shortages.append(float(shortage))
    return np.asarray(reserves), np.asarray(shortages)


def _reserve_at_matched_risk(reserves, shortages, target_shortage):
    """Interpolate reserve on a deterministic empirical risk frontier.

    Empirical shortage is discrete, so several epsilon values can produce the
    same realized risk.  For each duplicate risk we retain the smallest reserve
    (the non-dominated choice) before interpolation.  This avoids the unstable
    result obtained by passing duplicate x-coordinates directly to ``np.interp``.
    """
    order = np.argsort(shortages, kind='stable')
    risks_sorted = shortages[order]
    reserves_sorted = reserves[order]
    unique_risks = np.unique(risks_sorted)
    frontier_reserves = np.array([
        reserves_sorted[risks_sorted == risk].min() for risk in unique_risks
    ])
    return float(np.interp(target_shortage, unique_risks, frontier_reserves))


def matched_risk_table(fit, ev, reserve_df, day_reserve_df, cfg=CFG):
    """Create hourly/day-level matched-realized-risk comparisons.

    Positive ``Target reserve reduction`` means the target model uses less
    reserve than the comparator at the same realized shortage rate.
    """
    Y_te = fit['Y_te']
    ensembles, points = ev['ENSEMBLES'], ev['POINT_ALL']
    fine_eps = np.linspace(0.002, 0.30, 300)

    nominal = {
        'hourly': reserve_df[reserve_df['eps_nominal (%)'] == 5.0].set_index('Model'),
        'day': day_reserve_df[day_reserve_df['eps_nominal (%)'] == 5.0].set_index('Model'),
    }
    reserve_col = {
        'hourly': 'Mean upward reserve (Wh)',
        'day': 'Mean day-level reserve (Wh)',
    }
    shortage_col = {
        'hourly': 'Empirical shortage rate (%)',
        'day': 'Empirical day-level shortage rate (%)',
    }

    comparisons = [
        # Overall hourly gain at PCA-Mixed's realized risk.
        ('hourly', 'PCA-Mixed', 'DCT-Gaussian'),
        # Basis-only effect: likelihood fixed to Gaussian.
        ('hourly', 'PCA-Gaussian', 'DCT-Gaussian'),
        # Likelihood-only effect: basis fixed to PCA.
        ('hourly', 'PCA-Mixed', 'PCA-Gaussian'),
        # Day-level comparison in both directions because the frontiers cross.
        ('day', 'PCA-Mixed', 'DCT-Gaussian'),
        ('day', 'DCT-Gaussian', 'PCA-Mixed'),
    ]

    curve_cache = {}
    rows = []
    for scope, target, comparator in comparisons:
        key = (scope, comparator)
        if key not in curve_cache:
            curve_cache[key] = _reserve_risk_curve(
                Y_te, ensembles[comparator], points[comparator], fine_eps, scope)
        comp_reserves, comp_shortages = curve_cache[key]

        target_shortage = float(nominal[scope].loc[target, shortage_col[scope]])
        target_reserve = float(nominal[scope].loc[target, reserve_col[scope]])
        comparator_reserve = _reserve_at_matched_risk(
            comp_reserves, comp_shortages, target_shortage)
        reduction = (1 - target_reserve / comparator_reserve) * 100
        rows.append({
            'Scope': scope,
            'Target model': target,
            'Comparator model': comparator,
            'Target nominal eps (%)': 5.0,
            'Target realized shortage (%)': target_shortage,
            'Target reserve (Wh)': target_reserve,
            'Comparator reserve at matched risk (Wh)': comparator_reserve,
            'Target reserve reduction vs comparator (%)': reduction,
        })

    matched_df = pd.DataFrame(rows)
    fig_dir = cfg['figures_dir']
    csv_path = os.path.join(fig_dir, 'table_matched_risk.csv')
    tex_path = os.path.join(fig_dir, 'table_matched_risk.tex')
    matched_df.to_csv(csv_path, index=False)
    with open(tex_path, 'w') as f:
        f.write(matched_df.round(3).to_latex(
            index=False, escape=True,
            caption=('Reserve comparison at matched realized shortage risk. Positive reductions '
                     'mean that the target model procures less reserve than the comparator.'),
            label='tab:matched_risk'))

    print('\n' + '=' * 100)
    print('TABLE — Matched realized shortage risk')
    print('=' * 100)
    print(matched_df.round(3).to_string(index=False))
    print(f'\n  CSV : {csv_path}')
    print(f'  TeX : {tex_path}')
    return matched_df


def hourly_reserve(fit, ev, cfg=CFG):
    """Hourly-marginal reserve table + writes CSV/TeX. Returns (reserve_df, r5)."""
    Y_te = fit['Y_te']
    fig_dir = cfg['figures_dir']
    eps_levels = cfg['eps_levels']
    ENSEMBLES, POINT_ALL = ev['ENSEMBLES'], ev['POINT_ALL']

    reserve_rows = []
    for m in MODEL_ORDER:
        S = ENSEMBLES[m]              # (N_test, M, 24)
        Yhat = POINT_ALL[m]           # (N_test, 24)
        for eps in eps_levels:
            q_upper = np.quantile(S, 1 - eps, axis=1)
            R_up = np.clip(q_upper - Yhat, a_min=0.0, a_max=None)
            violation = (Y_te > q_upper).astype(float)
            reserve_rows.append({
                'Model': m,
                'eps_nominal (%)': eps * 100,
                'Mean upward reserve (Wh)': float(R_up.mean()),
                'Empirical shortage rate (%)': float(violation.mean() * 100),
            })
    reserve_df = pd.DataFrame(reserve_rows)

    print('\n' + '=' * 100)
    print('TABLE — Reserve-aware evaluation: procured upward reserve vs. empirical shortage risk')
    print('=' * 100)
    pivot_reserve = reserve_df.pivot(index='Model', columns='eps_nominal (%)',
                                     values='Mean upward reserve (Wh)').reindex(MODEL_ORDER)
    pivot_shortage = reserve_df.pivot(index='Model', columns='eps_nominal (%)',
                                      values='Empirical shortage rate (%)').reindex(MODEL_ORDER)
    print('\nMean procured upward reserve (Wh), by nominal shortage-risk target (%):')
    print(pivot_reserve.round(1).to_string())
    print('\nEmpirical shortage rate (%), by nominal shortage-risk target (%):')
    print(pivot_shortage.round(2).to_string())

    reserve_df.to_csv(os.path.join(fig_dir, 'table_reserve_shortage.csv'), index=False)
    with open(os.path.join(fig_dir, 'table_reserve_shortage.tex'), 'w') as f:
        f.write(reserve_df.round(3).to_latex(
            index=False, escape=True,
            caption=('Reserve-aware evaluation: mean procured upward operating reserve and empirical '
                     'shortage (violation) rate, by nominal one-sided shortage-risk target, per model, '
                     f'test block ($n={len(Y_te)}$ days, $H=24$ hours).'),
            label='tab:reserve_shortage'))
    print(f"\n  CSV : {os.path.join(fig_dir, 'table_reserve_shortage.csv')}")
    print(f"  TeX : {os.path.join(fig_dir, 'table_reserve_shortage.tex')}")

    # Figure: procured reserve vs. empirical shortage rate, per model
    from .figures import fig_reserve_shortage
    fig_reserve_shortage(reserve_df, cfg)

    # Specific operational comparison: PCA-Mixed vs DCT-Gaussian at eps=5%
    r5 = reserve_df[reserve_df['eps_nominal (%)'] == 5.0].set_index('Model')
    pm_r = r5.loc['PCA-Mixed', 'Mean upward reserve (Wh)']
    gb_r = r5.loc['DCT-Gaussian', 'Mean upward reserve (Wh)']
    pm_v = r5.loc['PCA-Mixed', 'Empirical shortage rate (%)']
    gb_v = r5.loc['DCT-Gaussian', 'Empirical shortage rate (%)']
    print('\n' + '-' * 100)
    print("At nominal 5% shortage-risk target:")
    print(f"  PCA-Mixed        : reserve={pm_r:,.1f} Wh, empirical shortage={pm_v:.2f}%")
    print(f"  DCT-Gaussian: reserve={gb_r:,.1f} Wh, empirical shortage={gb_v:.2f}%")
    print(f"  Reserve reduction: {(1 - pm_r/gb_r)*100:.2f}%  |  Shortage rate delta: {pm_v - gb_v:+.2f} pp")
    print('-' * 100)
    return reserve_df, dict(pm_r=pm_r, gb_r=gb_r, pm_v=pm_v, gb_v=gb_v)


def day_level_reserve(fit, ev, reserve_df, r5vals, cfg=CFG):
    """Day-level reserve table (writes CSV/TeX), the basis-fixed comparison, and
    the iso-risk comparison. Returns day_reserve_df."""
    Y_te = fit['Y_te']
    fig_dir = cfg['figures_dir']
    eps_levels = cfg['eps_levels']
    ENSEMBLES, POINT_ALL = ev['ENSEMBLES'], ev['POINT_ALL']
    pm_r, pm_v = r5vals['pm_r'], r5vals['pm_v']

    print('\n' + '=' * 100)
    print('TABLE — Day-level joint-dependence reserve (uses full-curve Monte Carlo ensembles)')
    print('=' * 100)

    day_reserve_rows = []
    for m in MODEL_ORDER:
        S = ENSEMBLES[m]
        Yhat = POINT_ALL[m]
        diff = S - Yhat[:, None, :]
        peak_excess_samples = diff.max(axis=2)
        actual_peak_excess  = (Y_te - Yhat).max(axis=1)
        for eps in eps_levels:
            R_day = np.clip(np.quantile(peak_excess_samples, 1 - eps, axis=1), 0.0, None)
            violation_day = (actual_peak_excess > R_day).astype(float)
            day_reserve_rows.append({
                'Model': m,
                'eps_nominal (%)': eps * 100,
                'Mean day-level reserve (Wh)': float(R_day.mean()),
                'Empirical day-level shortage rate (%)': float(violation_day.mean() * 100),
            })
    day_reserve_df = pd.DataFrame(day_reserve_rows)

    pivot_day_reserve  = day_reserve_df.pivot(index='Model', columns='eps_nominal (%)',
                                              values='Mean day-level reserve (Wh)').reindex(MODEL_ORDER)
    pivot_day_shortage = day_reserve_df.pivot(index='Model', columns='eps_nominal (%)',
                                              values='Empirical day-level shortage rate (%)').reindex(MODEL_ORDER)
    print('\nMean day-level reserve (Wh), by nominal risk target (%):')
    print(pivot_day_reserve.round(1).to_string())
    print('\nEmpirical day-level shortage rate (%) — at least one hour exceeds the buffer:')
    print(pivot_day_shortage.round(2).to_string())

    day_reserve_df.to_csv(os.path.join(fig_dir, 'table_day_level_reserve.csv'), index=False)
    with open(os.path.join(fig_dir, 'table_day_level_reserve.tex'), 'w') as f:
        f.write(day_reserve_df.round(3).to_latex(
            index=False, escape=True,
            caption=('Day-level joint-dependence reserve: mean procured reserve and empirical rate of '
                     'at-least-one-hour shortage, computed from full-curve Monte Carlo ensembles '
                     f'(test block, $n={len(Y_te)}$ days).'),
            label='tab:day_level_reserve'))
    print(f"\n  CSV : {os.path.join(fig_dir, 'table_day_level_reserve.csv')}")
    print(f"  TeX : {os.path.join(fig_dir, 'table_day_level_reserve.tex')}")

    # Basis held fixed (PCA): mixed-likelihood effect at eps=5%
    r5_full = reserve_df[reserve_df['eps_nominal (%)'] == 5.0].set_index('Model')
    if 'PCA-Gaussian' in r5_full.index:
        pg_r = r5_full.loc['PCA-Gaussian', 'Mean upward reserve (Wh)']
        pg_v = r5_full.loc['PCA-Gaussian', 'Empirical shortage rate (%)']
        print('\n' + '-' * 100)
        print("Basis held fixed (PCA): isolating the mixed-likelihood effect at eps=5%")
        print(f"  PCA-Mixed    : reserve={pm_r:,.1f} Wh, empirical shortage={pm_v:.2f}%")
        print(f"  PCA-Gaussian : reserve={pg_r:,.1f} Wh, empirical shortage={pg_v:.2f}%")
        print(f"  Reserve reduction from mixed likelihood alone: {(1 - pm_r/pg_r)*100:.2f}%")
        print('-' * 100)

    matched_risk_table(fit, ev, reserve_df, day_reserve_df, cfg)
    return day_reserve_df
