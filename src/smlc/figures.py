"""Generate the data-dependent vector figures produced by the pipeline."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import CFG, MODEL_ORDER, MODEL_COLORS, DIST_COLORS


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})


def save_figure(name, cfg=CFG):
    """Save the active figure as vector PDF and close it."""
    path = os.path.join(cfg['figures_dir'], f'{name}.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'  [figure saved] {path}')


def fig_reserve_shortage(reserve_df, cfg=CFG):
    """Plot procured reserve against empirical shortage for all six models."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    markers = {'DCT-Mixed': 'o', 'PCA-Mixed': 's', 'DCT-Gaussian': '^',
               'PCA-Gaussian': 'P', 'QR-Direct': 'D', 'FTS': 'v'}
    for model in MODEL_ORDER:
        sub = reserve_df[reserve_df['Model'] == model]
        sub = sub.sort_values('eps_nominal (%)', ascending=False)
        ax.plot(sub['Mean upward reserve (Wh)'] / 1000,
                sub['Empirical shortage rate (%)'], marker=markers[model],
                color=MODEL_COLORS[model], label=model, linewidth=1.6,
                markersize=5.5)
    ax.set_xlabel('Mean procured upward reserve (kWh)')
    ax.set_ylabel('Empirical shortage rate (%)')
    ax.legend(fontsize=7.5, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure('fig_reserve_shortage', cfg)


def make_output_figures(fit, ev, cfg=CFG):
    """Generate the likelihood-family and coverage figures."""
    H = cfg['n_hours']
    alphas = cfg['alpha_levels']
    cal_dct, cal_pca = fit['cal_dct'], fit['cal_pca']
    cov_all = ev['cov_all']

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, cal, basis in [(axes[0], cal_dct, 'DCT'),
                           (axes[1], cal_pca, 'PCA')]:
        for j, family in enumerate(cal['distribs']):
            ax.bar(j + 1, 1, color=DIST_COLORS[family], edgecolor='white',
                   lw=0.4, width=0.8)
        ax.set_ylabel(basis)
        ax.set(yticks=[], xlim=(0.3, H + 0.7))
        ax.grid(False)
    legend_handles = [mpatches.Patch(color=color, label=name)
                      for name, color in DIST_COLORS.items()]
    axes[0].legend(handles=legend_handles, loc='upper right', ncol=3)
    axes[1].set_xlabel('Spectral mode $j$')
    axes[0].set_title('Selected likelihood family per mode')
    plt.tight_layout()
    save_figure('fig_dist_per_mode', cfg)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    nominal = [alpha * 100 for alpha in alphas]
    styles = [('-', 'o'), ('-', 's'), ('--', '^'), ('-.', 'P'),
              (':', 'D'), ('--', 'v')]
    ax.plot(nominal, nominal, 'k--', lw=1.2, label='Ideal', zorder=0)
    for model, (line_style, marker) in zip(MODEL_ORDER, styles):
        empirical = [cov_all[model][alpha] * 100 for alpha in alphas]
        ax.plot(nominal, empirical, color=MODEL_COLORS[model],
                ls=line_style, marker=marker, ms=6, lw=1.4, label=model)
    ax.set_xlabel('Nominal coverage (%)')
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_title('Coverage calibration')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure('fig_coverage_calibration', cfg)
