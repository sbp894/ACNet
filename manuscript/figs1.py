"""
Manuscript Fig S1 -- ESC-50 confusion matrices with the diagonal removed.

The same four classifiers as Fig3 (Neural, Manifold, Shuffled, Stimulus), but only the
errors. The diagonal is two orders of magnitude larger than any off-diagonal cell, so
on the full matrix the confusion structure is invisible; masking it is what makes the
claim in Fig3's alignment panels -- that the manifold classifier makes the *same*
mistakes as the neural one -- something a reader can check by eye.

    python figs1.py           # -> figures/figs1_raw.png and figures/figs1_log.png

One PNG per confusion-matrix colour scale (CONFMAT_SCALES). Everything is read from
`data/fig3.pkl.gz`; the three ESC-50 figures share one cache because they share every
input.
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
OUT_PNG_FMT = os.path.join(HERE, 'figures', 'figs1_{scale}.png')

CONFMAT_SCALES = ('raw', 'log')

FONT_SIZE = 8
plt.rcParams.update({
    'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'font.size': FONT_SIZE,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
})

CM_PER_IN = 2.54
FIG_W_CM = 7.2 * CM_PER_IN
FIG_H_CM = 2.6 * CM_PER_IN


def load_cache():
    with gzip.open(PKL_PATH, 'rb') as fh:
        return pickle.load(fh)


def offdiag_vlim(cache):
    """Shared colour limits over every model's off-diagonal cells.

    The source script scaled to the Stimulus classifier's maximum, which happens to be
    the largest of the four; taking the max over all four makes that an observation
    rather than an assumption, and keeps the four panels comparable.
    """
    diag = np.eye(el.N_CATEGORIES, dtype=bool)
    vmax = max(int(cache['confmats'][n][~diag].max()) for n in cache['model_names'])
    return 0, vmax


def assert_equal_panels(fig, axes, label):
    fig.canvas.draw()
    sizes = np.array([[ax.get_window_extent().width, ax.get_window_extent().height]
                      for ax in axes])
    assert np.allclose(sizes, sizes[0], atol=1.0), (
        f"{label}: panels differ in size\n{sizes}")


def draw(cache, scale, vlim):
    fig = plt.figure(layout='constrained',
                     figsize=(FIG_W_CM / CM_PER_IN, FIG_H_CM / CM_PER_IN))
    axes = fig.subplots(1, 4)

    for idx, name in enumerate(cache['model_names']):
        img = el.plot_confmat(axes[idx], cache['confmats'][name], scale=scale,
                              vlim=vlim, hide_diagonal=True, title=name)
        axes[idx].set(xlabel='True class')
        if idx == 0:
            axes[idx].set(ylabel='Predicted class')
        else:
            axes[idx].tick_params(labelleft=False)

    # Attached to all four axes at once, so the space it takes comes off each of them
    # equally and they stay identical -- and it is sized against the panels rather than
    # the full figure height, which a dedicated `cax` column cannot do.
    ticks = el.confmat_ticks(scale, *vlim)
    cbar = fig.colorbar(img, ax=list(axes), location='right', shrink=0.72, aspect=14,
                        pad=0.015)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:g}' for t in ticks])
    cbar.set_label(f'count ({scale} scale)' if scale == 'log' else 'count')
    cbar.outline.set_linewidth(0.4)

    assert_equal_panels(fig, list(axes), 'confusion matrices')

    out_png = OUT_PNG_FMT.format(scale=scale)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"wrote {out_png}")


def print_stats(cache, vlim):
    diag = np.eye(el.N_CATEGORIES, dtype=bool)
    print(f"\noff-diagonal counts share the colour range 0..{vlim[1]}")
    for name in cache['model_names']:
        cm = cache['confmats'][name]
        off = cm[~diag]
        print(f"  {name:9s} diagonal {cm[diag].sum():5d}  off-diagonal {off.sum():5d}  "
              f"max off-diagonal cell {off.max():3d}  "
              f"non-zero off-diagonal cells {int((off > 0).sum())}/{off.size}")


def main():
    cache = load_cache()
    print(f"cache built {cache['meta']['built']} from {cache['meta']['source_script']}")
    vlim = offdiag_vlim(cache)
    for scale in CONFMAT_SCALES:
        draw(cache, scale, vlim)
    print_stats(cache, vlim)


if __name__ == '__main__':
    main()
