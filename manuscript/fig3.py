"""
Manuscript Fig3 -- ESC-50 categorisation from the auditory manifold.

Four classifiers are compared on the same 5-fold split of the same 2000 ESC-50 clips:

    Neural     -- recorded ferret A1/PEG firing rates (site REI084_087)
    Manifold   -- ACNet's shared backbone embeddings
    Shuffled   -- the same backbone with its weights shuffled (architecture-only null)
    Stimulus   -- the gammatonegram ACNet is given as input

Everything the figure plots is cached in `data/fig3.pkl.gz`: the per-fold predictions,
the confusion matrices, the across-layer accuracies and the UMAP embedding. ACNet is
still loaded and run on the stored ESC-50 waveforms on every run, as a check that the
cached numbers belong to this model -- it just is not plotted any more. Set
VERIFY_WITH_ACNET = False to skip it and avoid importing torch.

    python fig3.py            # -> figures/fig3_raw.png and figures/fig3_log.png

Two PNGs are written per run, one per confusion-matrix colour scale (see
CONFMAT_SCALES). The raw scale is what the counts are; the log scale is the only one
in which the off-diagonal structure -- which is what the alignment analysis is about
-- is visible at all.
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
sys.path.insert(0, os.path.dirname(HERE))          # acnet_model, gtgram, gtg_filters

import esc50_lib as el                             # noqa: E402

PKL_PATH = os.path.join(HERE, 'data', 'fig3.pkl.gz')
OUT_PNG_FMT = os.path.join(HERE, 'figures', 'fig3_{scale}.png')

# Draw the confusion matrices once per colour scale and save one figure each.
CONFMAT_SCALES = ('raw', 'log')
CONFMAT_VLIM = (0, 40)         # counts; the diagonal saturates, the off-diagonal is 0-6

VERIFY_WITH_ACNET = True       # False -> skip the live model check (and the torch import)
LIVE_TOL = 1e-3                # live-vs-cached relative tolerance (float32, GPU vs GPU)

PER_CATEGORY = True            # statistics unit: per category (n=50) or per fold (n=5)

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
FONT_SIZE = 8
CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']
plt.rcParams.update({
    'legend.fontsize': FONT_SIZE - 2,
    'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.prop_cycle': matplotlib.cycler(color=CB_COLOR_CYCLE),
    'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE,
    'font.size': FONT_SIZE,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
})

# Geometry in cm, converted at the end.
CM_PER_IN = 2.54
FIG_W_CM = 7.2 * CM_PER_IN
FIG_H_CM = 6.4 * CM_PER_IN
SCHEMATIC_TEXT = 'schematic of the\nfour classifiers'
# The confusion-matrix row holds four square panels across the figure width, so its
# height is fixed by the width (~4.4 cm each) no matter what it is given. Asking for
# more than that only adds white space.
ROW_HEIGHTS = (1.15, 1.0, 1.0)
BOT_WIDTHS = (3, 2.5, 2.5, 4)   # accuracy | align-all | align-incorrect | layer profile

BAR_LABELS_ACC = ['Neural', 'Manifold', 'Shuffled', 'Stimulus']
BAR_LABELS_ALIGN = ['Manifold', 'Shuffled', 'Stimulus']


# --------------------------------------------------------------------------- #
def load_cache():
    with gzip.open(PKL_PATH, 'rb') as fh:
        return pickle.load(fh)


def verify_against_acnet(cache):
    """Run ACNet on the stored ESC-50 clips and check it against the cached manifold.

    Nothing in the figure is drawn from this -- it is a provenance check. The cached
    predictions, confusion matrices and accuracies were all computed downstream of the
    manifold this reproduces, so if ACNet, the front end or the stored waveforms ever
    drift, the assertion below fires instead of the script quietly drawing a figure
    whose numbers belong to a different model.
    """
    demo = cache['demo']
    if not VERIFY_WITH_ACNET:
        print("VERIFY_WITH_ACNET = False -- the cached numbers are plotted unchecked")
        return None, None

    import torch                                   # noqa: PLC0415
    import acnet_model as sa                       # noqa: PLC0415

    device = sa.best_device()
    model, _ = sa.load_acnet()
    model.to(device).eval()
    model.update_audio_process(dict(demo['front_end']))

    mf_live, gtg_live = [], []
    with torch.no_grad():
        for wav in demo['wav']:
            mf, gtg = model.get_mf_embeddings(torch.from_numpy(np.asarray(wav)),
                                              fs=demo['wav_fs'])
            mf_live.append(mf.squeeze(0).cpu().numpy())
            gtg_live.append(gtg.cpu().numpy())
    mf_live = np.stack(mf_live)
    gtg_live = np.stack(gtg_live)

    err_mf = float(np.abs(mf_live - demo['mf_ref']).max()
                   / max(float(np.abs(demo['mf_ref']).max()), 1.0))
    err_gtg = float(np.abs(gtg_live - demo['gtg_ref']).max()
                    / max(float(np.abs(demo['gtg_ref']).max()), 1.0))
    print(f"live vs cached ({len(mf_live)} ESC-50 clips): "
          f"max rel |d manifold|={err_mf:.3e}  max rel |d gtg|={err_gtg:.3e}")
    assert err_mf < LIVE_TOL and err_gtg < LIVE_TOL, (
        f"live ACNet does not reproduce the published ESC-50 manifold "
        f"(manifold {err_mf:.3e}, gtg {err_gtg:.3e}, tol {LIVE_TOL:.0e})")
    return err_mf, err_gtg


def verify_confmats(cache):
    """Rebuild every confusion matrix from the cached per-fold predictions."""
    for name in cache['model_names']:
        p = cache['preds'][name]
        live = el.confusion_counts(p['pred_test'].ravel(), p['true_test'].ravel())
        assert np.array_equal(live, cache['confmats'][name]), (
            f"{name}: confusion matrix does not follow from the cached predictions")
    print("confusion matrices: all four rebuilt from the cached predictions, exact match")


def alignment_folds(cache):
    """Per-fold agreement with the neural classifier, for the three model classifiers.

    The neural classifier's own labels are the reference for `true`, exactly as in the
    source script -- so a sample that sits at a different within-fold position in the
    two runs is compared against the wrong partner. The builder measured how many do
    (4 of 2000, all in fold 4) and `print_stats` reports it.
    """
    neural = cache['preds']['Neural']
    out = {}
    for name in BAR_LABELS_ALIGN:
        full, inc = el.alignment_per_fold(cache['preds'][name]['pred_test'],
                                          neural['pred_test'], neural['true_test'])
        out[name] = {'full': full, 'incorrect': inc}
    return out


def assert_equal_panels(fig, axes, label):
    """Same-format axes must render at identical size (global plotting rule)."""
    fig.canvas.draw()
    sizes = np.array([[ax.get_window_extent().width, ax.get_window_extent().height]
                      for ax in axes])
    assert np.allclose(sizes, sizes[0], atol=1.0), (
        f"{label}: panels differ in size\n{sizes}")


# --------------------------------------------------------------------------- #
def draw(cache, scale):
    fig = plt.figure(layout='constrained',
                     figsize=(FIG_W_CM / CM_PER_IN, FIG_H_CM / CM_PER_IN))
    gs = fig.add_gridspec(3, 1, height_ratios=ROW_HEIGHTS)
    # Nested subfigures, not a raw subgridspec: an axes added from a subfigure's
    # subplotspec is positioned in FIGURE coordinates and constrained layout never
    # sees it, which is how the demo column ended up floating over the row.
    sf_top_l, sf_top_r = fig.add_subfigure(gs[0]).subfigures(1, 2, width_ratios=(2, 3))
    ax_umap = sf_top_l.subplots(1, 1)
    ax_schematic = sf_top_r.subplots(1, 1)
    sf_mid = fig.add_subfigure(gs[1])
    sp_mid = sf_mid.subplots(1, 4)
    sp_bot = fig.add_subfigure(gs[2]).subplots(1, 4, width_ratios=BOT_WIDTHS)

    # ---- top left: UMAP of the manifold, 10 best-clustering categories ------
    umap = cache['umap']
    el.plot_best10_panel(ax_umap, umap['X2d'], umap['y_labels'], umap['selection'])

    # ---- top right: reserved for the classifier schematic -------------------
    # The four-classifier schematic is drawn by hand; this keeps its slot at the size
    # the rest of the layout assumes, exactly as the source script did.
    ax_schematic.text(0.5, 0.5, SCHEMATIC_TEXT, ha='center', va='center',
                      transform=ax_schematic.transAxes)
    ax_schematic.axis('off')

    # ---- middle: confusion matrices ----------------------------------------
    for idx, name in enumerate(cache['model_names']):
        img = el.plot_confmat(sp_mid[idx], cache['confmats'][name], scale=scale,
                              vlim=CONFMAT_VLIM, title=name)
        if idx == 0:
            sp_mid[idx].set(xlabel='True class', ylabel='Predicted class')
        else:
            sp_mid[idx].tick_params(labelleft=False)

    # The colorbar takes its space from all four panels at once, so they shrink by the
    # same amount and stay identical -- which the assertion below checks.
    ticks = el.confmat_ticks(scale, *CONFMAT_VLIM)
    cbar = sf_mid.colorbar(img, ax=list(sp_mid), location='right', shrink=0.9,
                           aspect=18, pad=0.01)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:g}' for t in ticks])
    cbar.set_label(f'count ({scale} scale)' if scale == 'log' else 'count')
    cbar.outline.set_linewidth(0.4)

    # ---- bottom: accuracy, alignment, layer profile ------------------------
    acc = {n: cache['preds'][n]['r_test'] for n in BAR_LABELS_ACC}
    el.bar_strip(sp_bot[0], acc, BAR_LABELS_ACC, el.COLOR_PALETTE)
    sp_bot[0].axhline(1 / el.N_CATEGORIES, c='k', ls='--', lw=0.8)
    sp_bot[0].set(xlabel='', ylabel='Accuracy', title='Prediction accuracy')
    sp_bot[0].tick_params(axis='x', rotation=30)

    align = alignment_folds(cache)
    for ax, key, title in [(sp_bot[1], 'full', 'All predictions'),
                           (sp_bot[2], 'incorrect', 'Incorrect predictions only')]:
        el.bar_strip(ax, {n: align[n][key] for n in BAR_LABELS_ALIGN},
                     BAR_LABELS_ALIGN, el.COLOR_PALETTE)
        ax.axhline(1 / el.N_CATEGORIES, c='k', ls='--', lw=0.8)
        ax.set(title=title)
        ax.tick_params(axis='x', rotation=30)
    sp_bot[1].set(ylabel='Neural Alignment')

    xl = cache['xlayers']
    el.grouped_bar_strip(
        sp_bot[3],
        {'Data': list(xl['r_test_data']), 'Shuffle': list(xl['r_test_shuf'])},
        [str(i) for i in range(xl['r_test_data'].shape[0])],
        ['Data', 'Shuffle'],
        [el.COLOR_PALETTE['Manifold'], el.COLOR_PALETTE['Shuffled']])
    sp_bot[3].set(xlabel='Layer', ylabel='Accuracy', title='Accuracy across layers')

    # Same-format groups: the four confusion matrices, and the two alignment bars.
    assert_equal_panels(fig, list(sp_mid), 'confusion matrices')
    assert_equal_panels(fig, [sp_bot[1], sp_bot[2]], 'alignment bars')

    out_png = OUT_PNG_FMT.format(scale=scale)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"wrote {out_png}")


# --------------------------------------------------------------------------- #
def print_stats(cache):
    confmats = cache['confmats']
    preds = cache['preds']
    align = alignment_folds(cache)
    meta = cache['meta']

    n_bad = meta.get('fold_label_mismatch_total', 0)
    n_tot = meta.get('fold_label_n_total', preds['Neural']['true_test'].size)
    print(f"\nfold bookkeeping: the neural and model classifiers share the 5-fold split; "
          f"{n_bad}/{n_tot} samples ({100 * n_bad / n_tot:.2f} %) sit at a different "
          f"within-fold position, so their alignment entry pairs the wrong two clips.")

    if PER_CATEGORY:
        acc = {n: confmats[n].diagonal() / confmats[n].sum(axis=1)
               for n in cache['model_names']}
        true_cat = preds['Neural']['true_test'].ravel()
        pred_neural = preds['Neural']['pred_test'].ravel()
        cats = np.unique(true_cat)
        aln_full, aln_inc = {}, {}
        for name in BAR_LABELS_ALIGN:
            pm = preds[name]['pred_test'].ravel()
            aln_full[name] = el.category_alignment_full(pm, pred_neural, true_cat, cats)
            aln_inc[name] = el.category_alignment_incorrect(pm, pred_neural, true_cat, cats)
        valid_inc = ~np.any([np.isnan(aln_inc[n]) for n in BAR_LABELS_ALIGN], axis=0)
        unit = f'per category (n={len(cats)})'
        unit_inc = f'per category (n={valid_inc.sum()})'
    else:
        acc = {n: preds[n]['r_test'] for n in cache['model_names']}
        aln_full = {n: align[n]['full'] for n in BAR_LABELS_ALIGN}
        aln_inc = {n: align[n]['incorrect'] for n in BAR_LABELS_ALIGN}
        valid_inc = np.ones(len(aln_inc['Manifold']), dtype=bool)
        unit = f"per fold (n={preds['Manifold']['r_test'].size})"
        unit_inc = unit

    print(f"\n-- Accuracy -- {unit} --")
    for name in cache['model_names']:
        el.report_mean_sd(name, acc[name])
    print("  Key comparisons (Bonferroni-corrected within family, n=5):")
    el.wilcox_family([
        ('Neural', acc['Neural'], 'Manifold', acc['Manifold'], 'two-sided'),
        ('Manifold', acc['Manifold'], 'Shuffled', acc['Shuffled'], 'greater'),
        ('Manifold', acc['Manifold'], 'Stimulus', acc['Stimulus'], 'greater'),
        ('Neural', acc['Neural'], 'Shuffled', acc['Shuffled'], 'greater'),
        ('Neural', acc['Neural'], 'Stimulus', acc['Stimulus'], 'greater'),
    ])

    print(f"\n-- Neural alignment -- all predictions, {unit} --")
    for name in BAR_LABELS_ALIGN:
        el.report_mean_sd(name, aln_full[name])
    print("  Key comparisons (Bonferroni-corrected within family, n=2):")
    el.wilcox_family([
        ('Manifold', aln_full['Manifold'], 'Shuffled', aln_full['Shuffled'], 'greater'),
        ('Manifold', aln_full['Manifold'], 'Stimulus', aln_full['Stimulus'], 'greater'),
    ])

    print(f"\n-- Neural alignment -- incorrect only, {unit_inc} --")
    for name in BAR_LABELS_ALIGN:
        el.report_mean_sd(name, aln_inc[name][valid_inc])
    print("  Key comparisons (Bonferroni-corrected within family, n=2):")
    el.wilcox_family([
        ('Manifold', aln_inc['Manifold'][valid_inc],
         'Shuffled', aln_inc['Shuffled'][valid_inc], 'greater'),
        ('Manifold', aln_inc['Manifold'][valid_inc],
         'Stimulus', aln_inc['Stimulus'][valid_inc], 'greater'),
    ])

    print("\n-- Accuracy across layers (Spearman, layer index vs mean accuracy) --")
    xl = cache['xlayers']
    results, pvals = [], []
    for label, key in [('Manifold', 'r_test_data'), ('Shuffled', 'r_test_shuf')]:
        means = xl[key].mean(axis=1)
        r, p = el.layer_trend(means)
        results.append((label, r, p, means))
        pvals.append(p)
    for (label, r, p, means), pa in zip(results, el.bonferroni(pvals)):
        sig = '*' if pa < 0.05 else 'n.s.'
        print(f"  {label}: Spearman r={r:.2f}, p={p:.4g}, p_bonferroni={pa:.4g} {sig}  "
              f"(layer means: {np.round(means, 3).tolist()})")


# --------------------------------------------------------------------------- #
def main():
    cache = load_cache()
    print(f"cache built {cache['meta']['built']} from {cache['meta']['source_script']}")
    verify_confmats(cache)
    verify_against_acnet(cache)
    for scale in CONFMAT_SCALES:
        draw(cache, scale)
    print_stats(cache)


if __name__ == '__main__':
    main()
