"""Configuration and shared constants for the SMLC pipeline."""

# ── Main configuration (edit this block to change paths / hyperparameters) ──
CFG = {
    # Data file (UCI ElectricityLoadDiagrams2011-2014)
    'data_path'      : 'data/LD2011_2014.txt',
    'separator'      : ';',
    'decimal'        : ',',

    # Temporal partition
    'train_end'      : '2013-06-30',
    'cal_end'        : '2013-12-31',

    # SMLC
    'lambda_ridge'   : 1.0,            # lambda
    'nu_student'     : 4,              # Student-t degrees of freedom nu
    'n_hours'        : 24,

    # QR-Direct
    'quantiles'      : [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975],
    'alpha_qr'       : 0.01,

    # FTS
    'fts_k_grid'          : [3, 5, 6, 8, 10, 12],
    'fts_k_criterion'     : 'CRPS',   # 'CRPS' | 'NLL_per_hour' | 'CalErr' | 'RMSE'
    'fts_candidate_orders': [(0,0,0),(1,0,0),(2,0,0),
                             (0,0,1),(1,0,1),(2,0,1),
                             (1,1,0),(0,1,1),(1,1,1)],
    'fts_var_calibration' : True,

    # Shared evaluation
    'alpha_levels'   : [0.50, 0.80, 0.90, 0.95],
    'alpha_map'      : {0.50:(0.25,0.75), 0.80:(0.10,0.90),
                        0.90:(0.05,0.95), 0.95:(0.025,0.975)},
    'n_mc_samples'   : 1000,           # M — same for every model
    'seed'           : 42,

    # Nominal one-sided shortage-risk targets
    'eps_levels'     : [0.10, 0.05, 0.025, 0.01],

    # Computational-cost benchmark: repeated timing runs (decrease if FTS is
    # too slow on your machine).
    'cost_n_repeats' : 5,

    # Output directory for all tables (.tex/.csv) and figures (.pdf)
    'figures_dir'    : 'figures',
}

# ── Distribution family labels ─────────────────────────────────────────────
DIST_NORMAL  = 'Normal'
DIST_LAPLACE = 'Laplace'
DIST_STUDENT = 'Student-t'

# ── Model set, canonical display order, and reference model ────────────────
MODEL_ORDER = ['DCT-Mixed', 'PCA-Mixed', 'DCT-Gaussian', 'PCA-Gaussian',
               'QR-Direct', 'FTS']
REFERENCE_MODEL = 'PCA-Mixed'

# ── Colors for publication figures ─────────────────────────────────────────
MODEL_COLORS = {
    'DCT-Mixed'   : '#6a6a6a',
    'PCA-Mixed'   : '#e8604c',
    'DCT-Gaussian': '#8c510a',
    'PCA-Gaussian': '#a1c9f4',
    'QR-Direct'   : '#2ea44f',
    'FTS'         : '#9467bd',
}
DIST_COLORS = {
    DIST_NORMAL : '#7fa9dc',
    DIST_LAPLACE: '#e2be5b',
    DIST_STUDENT: '#d889b5',
}
