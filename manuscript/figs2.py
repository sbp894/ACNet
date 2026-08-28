"""
Manuscript Fig S2 -- do the model classifiers confuse the same categories as the
neural one?

(A) the four confusion matrices with the diagonal masked out. The diagonal runs to
    ~40 against off-diagonal cells of 0-13, so on the full matrix the confusion
    structure is invisible.
(B) each model's off-diagonal cell counts against the neural classifier's, cell by
    cell. A positive slope means the two classifiers make the same *errors*, not just
    the same number of them.
(C) the same for the diagonal -- the categories each classifier gets right.

Marker area is proportional to how many confusion-matrix cells land on that (x, y)
pair; the regression line and its 95 % band are fitted to the raw cells, not to the
unique ones.

    python figs2.py           # -> figures/figs2_raw.png and figures/figs2_log.png

The colour scale (CONFMAT_SCALES) affects row A only; rows B and C are scatter plots
of the same counts either way, so the two PNGs differ only in the top row.
"""

import gzip
import os
import pickle
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import esc50_lib as el                             # noqa: E402

PKL_PATH = os.path.join(HERE, 'data', 'fig3.pkl.gz')
OUT_PNG_FMT = os.path.join(HERE, 'figures', 'figs2_{scale}.png')

CONFMAT_SCALES = ('raw', 'log')
REFERENCE = 'Neural'           # the classifier every other one is scattered against

FONT_SIZE = 8
CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']
plt.rcParams.update({
    'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.prop_cycle': matplotlib.cycler(color=CB_COLOR_CYCLE),
    'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'font.size': FONT_SIZE,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
})

CM_PER_IN = 2.54
FIG_W_CM = 7.2 * CM_PER_IN
FIG_H_CM = 5.0 * CM_PER_IN
ROW_HEIGHTS = (1.0, 0.75, 0.75)


def load_cache():
    with gzip.open(PKL_PATH, 'rb') as fh:
        return pickle.load(fh)


def assert_equal_panels(fig, axes, label):
    fig.canvas.draw()
    sizes = np.array([[ax.get_window_extent().width, ax.get_window_extent().height]
                      for ax in axes])
    assert np.allclose(sizes, sizes[0], atol=1.0), (
        f"{label}: panels differ in size\n{sizes}")


def draw(cache, scale):
    model_names = cache['model_names']
    others = [n for n in model_names if n != REFERENCE]
    diag_mask = np.eye(el.N_CATEGORIES, dtype=bool)
    offdiag_mask = ~diag_mask
    vlim = (0, max(int(cache['confmats'][n][offdiag_mask].max()) for n in model_names))

    fig = plt.figure(layout='constrained',
                     figsize=(FIG_W_CM / CM_PER_IN, FIG_H_CM / CM_PER_IN))
    gs = fig.add_gridspec(3, 4, height_ratios=ROW_HEIGHTS)
    sf_a = fig.add_subfigure(gs[0, :])
    ax_a = sf_a.subplots(1, 4)
    ax_b = fig.add_subfigure(gs[1, 1:]).subplots(1, 3, sharex=True, sharey=True)
    ax_c = fig.add_subfigure(gs[2, 1:]).subplots(1, 3, sharex=True, sharey=True)

    # ---- row A: off-diagonal confusion matrices ----------------------------
    for idx, name in enumerate(model_names):
        img = el.plot_confmat(ax_a[idx], cache['confmats'][name], scale=scale,
                              vlim=vlim, hide_diagonal=True, title=name)
        ax_a[idx].set(xlabel='True class')
        if idx == 0:
            ax_a[idx].set(ylabel='Predicted class')
        else:
            ax_a[idx].tick_params(labelleft=False)

    # Attached to all four panels so the space it takes comes off each of them equally
    # and they stay identical -- and it is sized against the panels rather than the row
    # height, which a dedicated `cax` gridspec column cannot do.
    ticks = el.confmat_ticks(scale, *vlim)
    cbar = sf_a.colorbar(img, ax=list(ax_a), location='right', shrink=0.72, aspect=14,
                         pad=0.015)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:g}' for t in ticks])
    cbar.set_label(f'count ({scale})' if scale == 'log' else 'count')
    cbar.outline.set_linewidth(0.4)

    # ---- rows B and C: cell-by-cell against the neural classifier ----------
    ref_cm = cache['confmats'][REFERENCE]
    stats_rows = []
    for idx, name in enumerate(others):
        cm = cache['confmats'][name]
        r_off, p_off = el.scatter_sized_by_count(
            ax_b[idx], cm[offdiag_mask], ref_cm[offdiag_mask],
            line_color=el.COLOR_PALETTE[name], suppress_zero=False, base_s=50,
            color='k', alpha=0.6, linewidths=0)
        ax_b[idx].set(xlabel=f'Count ({name})')

        r_dia, p_dia = el.scatter_sized_by_count(
            ax_c[idx], cm[diag_mask], ref_cm[diag_mask],
            line_color=el.COLOR_PALETTE[name], base_s=10,
            color='k', alpha=0.6, linewidths=0)
        ax_c[idx].set(xlabel=f'Count ({name})')
        stats_rows.append((name, r_off, p_off, r_dia, p_dia))

    ax_b[0].set(ylabel=f'Count ({REFERENCE})')
    ax_c[0].set(ylabel=f'Count ({REFERENCE})')

    # ---- row labels in the empty first column ------------------------------
    for row, (letter, text) in enumerate([(None, None),
                                          ('B', 'Off-diagonal only'),
                                          ('C', 'Diagonal only')]):
        if letter is None:
            continue
        ax_lab = fig.add_subplot(gs[row, 0])
        ax_lab.axis('off')
        ax_lab.text(0.98, 0.6, text, transform=ax_lab.transAxes,
                    fontsize=FONT_SIZE + 1, va='center', ha='right')
        ax_lab.text(0.9, 0.9, letter, transform=ax_lab.transAxes,
                    fontsize=FONT_SIZE + 3, fontweight='bold', va='center', ha='right')
    ax_a[0].text(-0.3, 1.0, 'A', transform=ax_a[0].transAxes,
                 fontsize=FONT_SIZE + 3, fontweight='bold')

    assert_equal_panels(fig, list(ax_a), 'confusion matrices')
    assert_equal_panels(fig, list(ax_b), 'off-diagonal scatters')
    assert_equal_panels(fig, list(ax_c), 'diagonal scatters')

    out_png = OUT_PNG_FMT.format(scale=scale)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"wrote {out_png}")
    return stats_rows


def print_stats(stats_rows):
    print(f"\ncell-by-cell agreement with the {REFERENCE} classifier "
          f"(Pearson r on confusion-matrix counts):")
    for name, r_off, p_off, r_dia, p_dia in stats_rows:
        print(f"  {name:9s} off-diagonal r={r_off:.2f} (p={p_off:.2e}, n=2450)   "
              f"diagonal r={r_dia:.2f} (p={p_dia:.2e}, n=50)")


def main():
    cache = load_cache()
    print(f"cache built {cache['meta']['built']} from {cache['meta']['source_script']}")
    stats_rows = None
    for scale in CONFMAT_SCALES:
        stats_rows = draw(cache, scale)
    print_stats(stats_rows)


if __name__ == '__main__':
    main()
