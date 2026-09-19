"""
Data loading and preprocessing.

Loads the UCI ElectricityLoadDiagrams20112014 file in memory-efficient chunks,
converts kW-per-15-min readings to Wh, aggregates to hourly sums, reshapes into
24-hour daily curves, and drops incomplete days.
"""

import os
import sys

import numpy as np
import pandas as pd

from .config import CFG


def _missing_data_message(path):
    return (
        '\n[ERROR] Dataset not found: ' + path + '\n\n'
        'The raw UCI ElectricityLoadDiagrams20112014 file is NOT distributed with this\n'
        'repository (~250 MB compressed / ~700 MB raw). Please download it manually:\n\n'
        '  1. Go to  https://doi.org/10.24432/C58C86\n'
        '     (UCI Machine Learning Repository — ElectricityLoadDiagrams20112014)\n'
        '  2. Download and unzip the archive.\n'
        '  3. Place the raw file at  data/LD2011_2014.txt\n'
        '  4. Re-run:  python scripts/run_pipeline.py   (from the repository root)\n\n'
        'See data/README.md for details.\n')


def load_daily_curves(cfg=CFG):
    """Load and preprocess the dataset into daily 24-hour load curves.

    Returns
    -------
    Y : ndarray (N, 24)
        Daily load curves in Wh.
    dates : DatetimeIndex (N,)
        One timestamp (date) per curve.
    """
    path = cfg['data_path']
    if not os.path.isfile(path):
        sys.exit(_missing_data_message(path))

    print('Loading data (chunked, low-memory)...')
    chunk_means, chunk_idx, n_clients_seen = [], [], None
    CHUNKSIZE = 8000
    for chunk in pd.read_csv(
        path, sep=cfg['separator'],
        index_col=0, parse_dates=True, decimal=cfg['decimal'],
        chunksize=CHUNKSIZE, engine='c',
    ):
        if n_clients_seen is None:
            n_clients_seen = chunk.shape[1]
        row_mean = chunk.replace(0, np.nan).mean(axis=1)
        chunk_means.append(row_mean.astype('float64'))
        chunk_idx.append(chunk.index)
        del chunk

    # UCI documentation: values are in kW per 15-min reading; to convert to kWh,
    # divide by 4 (each 15-min kW reading represents 0.25h of energy). Previously
    # this factor was missing, so summing 4 readings per hour overstated all
    # absolute Wh figures by exactly 4x. This is a uniform linear rescaling: all
    # percentage/relative comparisons (CRPS %, CalErr, coverage, reserve-reduction
    # %) are unaffected; only absolute Wh/MWh figures change.
    series = pd.concat(chunk_means) * 1000 * 0.25
    series.index = pd.DatetimeIndex(pd.concat([pd.Series(i) for i in chunk_idx]).values)
    del chunk_means, chunk_idx

    print(f'  {series.shape[0]:,} records x {n_clients_seen} clients')
    print(f'  Period: {series.index[0].date()}  ->  {series.index[-1].date()}')

    series_h = series.resample('h').sum(min_count=4)

    agg_df         = series_h.to_frame('load')
    agg_df['date'] = agg_df.index.date
    daily          = agg_df.groupby('date')['load'].apply(list)
    daily          = daily[daily.apply(len) == 24]

    Y     = np.array(daily.tolist(), dtype=float)
    dates = pd.to_datetime(daily.index)

    mask     = ((Y == 0).sum(axis=1) >= 3) | np.isnan(Y).any(axis=1)
    Y, dates = Y[~mask], dates[~mask]
    print(f'  Y in R^({Y.shape[0]} x {Y.shape[1]})')
    return Y, dates


def temporal_partition(Y, dates, cfg=CFG):
    """Chronological Train / Calibration / Test split (test never used for
    selection).

    The first day of the raw training interval is retained only as lag history.
    Consequently every training response has a genuinely preceding daily curve;
    the first target is 2011-01-02, with 2011-01-01 supplying its lag features.
    """
    idx_tr = dates <= cfg['train_end']
    idx_ca = (dates > cfg['train_end']) & (dates <= cfg['cal_end'])
    idx_te = dates > cfg['cal_end']

    Y_tr_full, Y_ca, Y_te = Y[idx_tr], Y[idx_ca], Y[idx_te]
    d_tr_full, d_ca, d_te = dates[idx_tr], dates[idx_ca], dates[idx_te]
    if len(Y_tr_full) < 2:
        raise ValueError('Training interval needs at least two daily curves.')

    y_prev_tr = Y_tr_full[0].copy()
    Y_tr, d_tr = Y_tr_full[1:], d_tr_full[1:]

    print('Temporal partition:')
    print(f'  Train target: {Y_tr.shape[0]} days  ({d_tr[0].date()} -> {d_tr[-1].date()})')
    print(f'  Lag history : 1 day   ({d_tr_full[0].date()})')
    print(f'  Calibration : {Y_ca.shape[0]} days  ({d_ca[0].date()} -> {d_ca[-1].date()})')
    print(f'  Test        : {Y_te.shape[0]} days  ({d_te[0].date()} -> {d_te[-1].date()})')
    return dict(Y_tr=Y_tr, Y_ca=Y_ca, Y_te=Y_te,
                d_tr=d_tr, d_ca=d_ca, d_te=d_te,
                y_prev_tr=y_prev_tr)
