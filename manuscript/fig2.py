"""
Manuscript Fig2 -- cross-animal similarity of the auditory manifold.

Four per-animal encoders (CLT, LMD, PRN, REI) are run on the same held-out stimulus.
For each animal three representations are compared across animals by RSA over
timepoints:

    MF     -- the shared backbone's manifold embeddings
    predR  -- that animal's predicted PSTHs
    trueR  -- that animal's recorded PSTHs

plus MF.GTG, each animal's manifold RDM against the stimulus gammatonegram's own RDM.

Everything except the model forward pass is cached in `data/fig2.pkl.gz`: the encoder
weights, the recorded PSTHs, the stimulus gammatonegram, the stimulus bootstrap, and
reference values for every number the figure prints.

    python fig2.py            # -> figures/fig2.png (300 dpi)

Set VERIFY_FULL_RSA = True to re-derive the plotted RSA numbers from full-resolution
RDMs instead of reading them from the cache (slow: ~13 distance matrices over 11400
timepoints, tens of minutes and a few GB). It is off by default because the cheap
decimated-grid check below already exercises the live models end to end.
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

import acnet_model as sa                            # noqa: E402
import rsa_lib as rl                                # noqa: E402

PKL_PATH = os.path.join(HERE, 'data', 'fig2.pkl.gz')
OUT_PNG = os.path.join(HERE, 'figures', 'fig2.png')

VERIFY_FULL_RSA = False        # see the module docstring
RSA_TOL = 1e-4                 # live-vs-cached tolerance (float32 RDMs, GPU reductions)

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
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
})

# RDM display (display only; does NOT touch the RDM triangles used for RSA)
RDM_CMAP = 'cividis'
RDM_GAMMA = 0.75               # PowerNorm gamma (<1 boosts low-dissimilarity contrast)
RDM_CLIP = (2, 98)             # percentile clip -> vmin/vmax, per panel
SERIATE = True                 # reorder timepoints (per row, from the MF RDM)

# Geometry in cm, converted at the end. The 3x3 panel block keeps a fixed area; the
# figure grows to the RIGHT by CBAR_STRIP_CM and the constrained-layout rect is
# restricted to the panel area, so the colorbar cannot shrink a panel.
CM_PER_IN = 2.54
PANEL_AREA_W_CM = 3.6 * CM_PER_IN
PANEL_AREA_H_CM = 3.5 * CM_PER_IN
CBAR_PAD_CM = 0.20
CBAR_W_CM = 0.22
CBAR_TEXT_CM = 1.05
CBAR_STRIP_CM = CBAR_PAD_CM + CBAR_W_CM + CBAR_TEXT_CM
FIG_W_CM = PANEL_AREA_W_CM + CBAR_STRIP_CM

RDM_ROW_MODELS = [1, 3]        # which animals get an RDM row (LMD, REI)
BAR_LABELS = ['MF', 'predR', 'trueR', 'MF.GTG']


# --------------------------------------------------------------------------- #
def load_cache():
    with gzip.open(PKL_PATH, 'rb') as fh:
        return pickle.load(fh)


def compute_live(cache):
    """The one part that actually runs the models.

    Rebuilds each per-animal encoder from its cached weights, runs it on the cached
    validation gammatonegram, and PCA-projects the manifold embeddings, the predicted
    PSTHs and the recorded PSTHs. Returns the three lists of projections plus the
    stimulus projection.
    """
    device = sa.best_device()
    stim_gtg = cache['stim']['gtg']
    resp = cache['resp']
    group_names = cache['group_names']
    ref = cache['reference']

    mf_pcproj, pred_pcproj, true_pcproj = [], [], []
    cumvar = {'mf': [], 'pred': [], 'true': []}
    nfeat = {'mf': [], 'pred': [], 'true': []}

    for m, animal in enumerate(group_names):
        entry = cache['models'][animal]
        model = rl.build_animal_model(entry, sa, device=device)
        mf, psth_pred = rl.model_signals(model, stim_gtg, device=device)

        # recorded PSTHs over the same cells, assembled head by head. The cell ORDER
        # differs from the model's readout order (np.isin returns ascending positions
        # in the stored cell list); PCA scores and Euclidean distances over timepoints
        # are invariant to permuting cell columns, so this does not matter here.
        cols = []
        for head_cells in entry['cell_names_per_head']:
            inds = np.where(np.isin(resp['cell_names'], head_cells))[0]
            cols.append(resp['psth_all'][inds, :].T)
        cat_true = np.concatenate(cols, axis=1)
        assert cat_true.shape[1] == psth_pred.shape[1], (
            f"{animal}: {cat_true.shape[1]} recorded cells vs {psth_pred.shape[1]} predicted")

        for key, data, var, store in [('mf', mf, rl.MODEL_VAREXP, mf_pcproj),
                                      ('pred', psth_pred, rl.MODEL_VAREXP, pred_pcproj),
                                      ('true', cat_true, rl.DATA_VAREXP, true_pcproj)]:
            p, c, nd, nf = rl.pca_project(data, var)
            store.append(p)
            cumvar[key].append(c)
            nfeat[key].append(nf)
            assert nd == ref[f'{key}_ndims'][m], (
                f"{animal} {key}: {nd} PCs reach {var:.0%} variance but the cache says "
                f"{ref[f'{key}_ndims'][m]} -- the live model is not the cached one")

    stim_pcproj, stim_cumvar, stim_nd, stim_nf = rl.pca_project(stim_gtg, rl.MODEL_VAREXP)
    assert stim_nd == ref['stim_ndims']
    cumvar['stim'] = stim_cumvar
    nfeat['stim'] = stim_nf
    return mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj, cumvar, nfeat


def verify_against_cache(cache, mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj):
    """Cheap end-to-end check: the bootstrap's centring statistic, recomputed live.

    The shipped bootstrap distribution is only meaningful for the projections it was
    built from. Its reference statistic -- every stimulus block drawn exactly once, on
    the decimated grid -- is one resample, so recomputing it costs seconds and ties the
    live models to the cached CIs.
    """
    stim = cache['stim']
    block_s = rl.BOOT_PRIMARY_BLOCK_S
    ref = rl.bootstrap_reference(block_s, mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj,
                                 stim['n_stim_val'], stim['n_t_per_stim'], stim['fs_gtg'])
    cached = cache['boot'][block_s]
    diffs = {k: float(np.nanmax(np.abs(cached[k] - r))) for k, r in
             zip(['ref_mf', 'ref_pred', 'ref_true', 'ref_stim'], ref)}
    worst = max(diffs.values())
    print(f"live vs cached (bootstrap reference, {block_s}s blocks): "
          + '  '.join(f"{k}={v:.2e}" for k, v in diffs.items()))
    assert worst < RSA_TOL, (
        f"live models do not reproduce the cached bootstrap reference (max diff "
        f"{worst:.2e} > {RSA_TOL}). The weights, the stimulus or the analysis "
        f"constants have changed; rebuild the cache rather than plotting this.")


def rsa_values(cache, mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj):
    """Full-resolution RSA matrices: from the cache, or recomputed if asked."""
    ref = cache['reference']
    if not VERIFY_FULL_RSA:
        return ref['mf_rsa'], ref['pred_rsa'], ref['true_rsa'], ref['stim_mf_rsa']

    print("VERIFY_FULL_RSA: recomputing full-resolution RDMs (slow)...", flush=True)
    n_models = len(cache['group_names'])
    stim_utri = rl.rdm_utri(stim_pcproj)
    mf_utri = [rl.rdm_utri(p) for p in mf_pcproj]
    mf_rsa = rl.rsa_matrix(mf_utri)
    stim_mf_rsa = np.array([np.corrcoef(mf_utri[m], stim_utri)[0, 1] for m in range(n_models)])
    del mf_utri, stim_utri
    pred_rsa = rl.rsa_matrix([rl.rdm_utri(p) for p in pred_pcproj])
    true_rsa = rl.rsa_matrix([rl.rdm_utri(p) for p in true_pcproj])
    for name, live, cached in [('MF', mf_rsa, ref['mf_rsa']), ('predR', pred_rsa, ref['pred_rsa']),
                               ('trueR', true_rsa, ref['true_rsa']),
                               ('MF.GTG', stim_mf_rsa, ref['stim_mf_rsa'])]:
        d = float(np.nanmax(np.abs(live - cached)))
        print(f"  {name:7s} max|live - cached| = {d:.2e}")
        assert d < RSA_TOL, f"{name} RSA does not reproduce the cached value"
    return mf_rsa, pred_rsa, true_rsa, stim_mf_rsa


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _resolve_cmap(name):
    return plt.get_cmap(name)


def _rdm_norm(M):
    vmin, vmax = np.percentile(M, RDM_CLIP)
    return matplotlib.colors.PowerNorm(gamma=RDM_GAMMA, vmin=vmin, vmax=max(vmax, vmin + 1e-12))


def _seriation_order(rdm_sub):
    if not SERIATE:
        return np.arange(rdm_sub.shape[0])
    _, order, _ = rl.compute_serial_matrix(rdm_sub, 'ward')
    return np.asarray(order)


def _show_rdm(ax, M, order):
    M_disp = M[np.ix_(order, order)]
    n = M_disp.shape[0]
    ax.imshow(M_disp, origin='lower', extent=(0, n, 0, n), aspect='equal',
              cmap=_resolve_cmap(RDM_CMAP), norm=_rdm_norm(M_disp), rasterized=True)
    ax.set(xticks=[], yticks=[])


def assert_equal_panels(fig, axes, label):
    """Same-format axes must render at identical size (global plotting rule)."""
    fig.canvas.draw()
    sizes = np.array([[ax.get_window_extent().width, ax.get_window_extent().height]
                      for ax in axes])
    assert np.allclose(sizes, sizes[0], atol=1.0), (
        f"{label}: panels differ in size\n{sizes}")


def main():
    cache = load_cache()
    group_names = cache['group_names']
    n_models = len(group_names)
    ref = cache['reference']

    mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj, cumvar, nfeat = compute_live(cache)
    verify_against_cache(cache, mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj)
    mf_rsa, pred_rsa, true_rsa, stim_mf_rsa = rsa_values(
        cache, mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj)

    upper_mask = ~np.isnan(mf_rsa)
    assert np.array_equal(upper_mask, ~np.isnan(pred_rsa)), "RSA nan patterns differ"
    assert np.array_equal(upper_mask, ~np.isnan(true_rsa)), "RSA nan patterns differ"
    bar_data = {'MF': mf_rsa[upper_mask], 'predR': pred_rsa[upper_mask],
                'trueR': true_rsa[upper_mask], 'MF.GTG': stim_mf_rsa}

    # ---- figure ----------------------------------------------------------
    fig = plt.figure(layout='constrained',
                     figsize=(FIG_W_CM / CM_PER_IN, PANEL_AREA_H_CM / CM_PER_IN))
    fig.get_layout_engine().set(rect=(0, 0, PANEL_AREA_W_CM / FIG_W_CM, 1))
    gs = fig.add_gridspec(3, 3)
    sp_top = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    ax_stim = fig.add_subplot(gs[2, 0])
    ax_bar = fig.add_subplot(gs[2, 1:])

    base_idx = np.arange(cache['stim']['gtg'].shape[0])[::rl.SUBSAMP_FACTOR]
    print(f"RDM display: {len(base_idx)} timepoints (every {rl.SUBSAMP_FACTOR}th)")

    for sp_idx, model_idx in enumerate(RDM_ROW_MODELS):
        mf_sub = rl.rdm_square(mf_pcproj[model_idx][base_idx], nfeat['mf'][model_idx])
        pred_sub = rl.rdm_square(pred_pcproj[model_idx][base_idx], nfeat['pred'][model_idx])
        true_sub = rl.rdm_square(true_pcproj[model_idx][base_idx], nfeat['true'][model_idx])
        order = _seriation_order(mf_sub)      # MF defines the row order; reused for pred & true
        _show_rdm(sp_top[sp_idx][0], mf_sub, order)
        _show_rdm(sp_top[sp_idx][1], pred_sub, order)
        _show_rdm(sp_top[sp_idx][2], true_sub, order)
        sp_top[sp_idx][0].set(ylabel=group_names[model_idx])

    # the three RDM columns are otherwise indistinguishable. Constrained layout takes
    # a title's height out of its own row, so a title on row 0 alone would make the
    # row-0 panels smaller than the row-1 panels -- a size difference that reads as a
    # data difference. Every row gets a title of the same height; rows 1-2 get a blank.
    for ax, title in zip(sp_top[0], ['Manifold', 'Predicted PSTH', 'Recorded PSTH']):
        ax.set_title(title, pad=2)
    for ax in sp_top[1] + [ax_stim, ax_bar]:
        ax.set_title(' ', pad=2)

    stim_sub = rl.rdm_square(stim_pcproj[base_idx], nfeat['stim'])
    _show_rdm(ax_stim, stim_sub, _seriation_order(stim_sub))   # its own structure
    ax_stim.set(ylabel='Stim (GTG)')

    # ---- bar + points ----------------------------------------------------
    x_pos = np.arange(len(BAR_LABELS))
    means = np.array([np.mean(bar_data[k]) for k in BAR_LABELS])
    ax_bar.bar(x_pos, means, color='gray', alpha=0.6, width=0.8, zorder=1)
    rng = np.random.default_rng(42)
    for i, k in enumerate(BAR_LABELS):
        v = np.asarray(bar_data[k])
        ax_bar.plot(i + rng.uniform(-0.12, 0.12, size=len(v)), v, linestyle='none',
                    marker='o', ms=2.6, color='black', alpha=0.75, mec='none', zorder=4)

    # two-sided 95% bootstrap CI on each group mean, from the primary (1 s) resample.
    # The 19 s whole-stimulus CI is reported in the printout only -- two sets of
    # whiskers on one bar would be unreadable.
    boot = cache['boot'][rl.BOOT_PRIMARY_BLOCK_S]
    boot_means = {'MF': np.nanmean(boot['mf'][:, upper_mask], axis=1),
                  'predR': np.nanmean(boot['pred'][:, upper_mask], axis=1),
                  'trueR': np.nanmean(boot['true'][:, upper_mask], axis=1),
                  'MF.GTG': np.nanmean(boot['stim'], axis=1)}
    boot_ref = {'MF': np.nanmean(boot['ref_mf'][upper_mask]),
                'predR': np.nanmean(boot['ref_pred'][upper_mask]),
                'trueR': np.nanmean(boot['ref_true'][upper_mask]),
                'MF.GTG': np.nanmean(boot['ref_stim'])}
    lo = np.empty(len(BAR_LABELS))
    hi = np.empty(len(BAR_LABELS))
    for i, k in enumerate(BAR_LABELS):
        v = boot_means[k][np.isfinite(boot_means[k])]
        lo[i], hi[i] = np.percentile(v, [2.5, 97.5]) + (means[i] - boot_ref[k])
    ax_bar.errorbar(x_pos, means, yerr=np.vstack([np.maximum(means - lo, 0),
                                                  np.maximum(hi - means, 0)]),
                    fmt='none', ecolor='black', elinewidth=1, capsize=2.5, zorder=5)

    # noise ceiling over the recorded-PSTH bar only: MF and predR are deterministic
    # functions of the stimulus, so their ceiling is 1.0. The bar stays at its measured
    # value -- the ceiling is shown, not applied.
    ceil = float(np.nanmean(ref['ceiling_pair']))
    x_true = BAR_LABELS.index('trueR')
    ax_bar.plot([x_true - 0.4, x_true + 0.4], [ceil, ceil], ls='--', lw=1,
                color='firebrick', zorder=6)
    ax_bar.annotate('ceiling', xy=(x_true, ceil), xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=FONT_SIZE - 3, color='firebrick')

    all_vals = np.concatenate([np.atleast_1d(bar_data[k]) for k in BAR_LABELS])
    ax_bar.set(xticks=x_pos, xticklabels=BAR_LABELS, ylabel='Similarity',
               ylim=(all_vals.min() * 0.7, max(all_vals.max(), ceil) * 1.05))

    # ---- shared colour scale, in its own strip right of the panel area ----
    # Each heatmap is normalised independently (per-panel percentile clip), so an
    # absolute dissimilarity axis would be correct for at most one panel. The bar shows
    # the ramp itself -- same cmap, same gamma -- on a relative low->high scale.
    fig.draw_without_rendering()          # constrained layout must settle first
    bb_top = sp_top[0][-1].get_position()
    bb_bot = sp_top[1][-1].get_position()
    cax = fig.add_axes([(PANEL_AREA_W_CM + CBAR_PAD_CM) / FIG_W_CM, bb_bot.y0,
                        CBAR_W_CM / FIG_W_CM, bb_top.y1 - bb_bot.y0])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(
        cmap=_resolve_cmap(RDM_CMAP),
        norm=matplotlib.colors.PowerNorm(gamma=RDM_GAMMA, vmin=0, vmax=1)), cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['low', 'high'])
    cb.set_label('Dissimilarity', labelpad=2)
    cb.outline.set_linewidth(0.5)
    cax.tick_params(length=0, pad=1.5, labelsize=FONT_SIZE - 2)

    assert_equal_panels(fig, [a for row in sp_top for a in row] + [ax_stim], 'RDM panels')
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"wrote {OUT_PNG}")

    print_stats(cache, bar_data, upper_mask, ref)


# --------------------------------------------------------------------------- #
def print_stats(cache, bar_data, upper_mask, ref):
    group_names = cache['group_names']
    n_models = len(group_names)
    mf_paired = bar_data['MF']
    pred_paired = bar_data['predR']
    true_paired = bar_data['trueR']
    stim_mf_rsa = bar_data['MF.GTG']

    print("\n" + "=" * 70)
    print("STATISTICS: Manifold (MF) vs rest")
    print("=" * 70)

    def _summ(x):
        x = np.asarray(x)
        return f"mean={np.mean(x):.3f}  median={np.median(x):.3f}  n={len(x)}"

    for name, v in [('MF', mf_paired), ('predR', pred_paired),
                    ('trueR', true_paired), ('MF.GTG', stim_mf_rsa)]:
        print(f"  {name:8s}: {_summ(v)}")
    print("-" * 70)

    # MF, predR, trueR are each the strict upper triangle of the same 4x4 RSA matrix,
    # so they are PAIRED by model-pair (n=6). MF.GTG is one value per model (n=4), so
    # MF-vs-MF.GTG is UNPAIRED. All p-values below are two-sided.
    pair_ij = [(i, j) for i in range(n_models) for j in range(i + 1, n_models)]
    assert len(pair_ij) == len(mf_paired), "pair_ij does not match the paired RSA vectors"

    for label, other in [('predR', pred_paired), ('trueR', true_paired)]:
        obs, p, pfloor, nperm = rl.signflip_perm_paired(mf_paired, other)
        n_win = int(np.sum(mf_paired > other))
        print(f"  MF vs {label:5s}: MF wins {n_win}/{len(mf_paired)} pairs, "
              f"d(MF-{label})={obs:+.3f}")
        print(f"      paired sign-flip perm ({nperm} perms, n={len(mf_paired)}): "
              f"p={p:.4f} (two-sided)  [p-floor={pfloor:.3f}]")
        obs_d, p_d, pfloor_d, nperm_d = rl.signflip_perm_dyadic(
            mf_paired, other, n_models, pair_ij)
        print(f"      animal-level dyadic sign-flip ({nperm_d} distinct patterns): "
              f"p={p_d:.4f} (two-sided)  [p-floor={pfloor_d:.3f}]")

    obs, p, pfloor, nperm = rl.label_perm_unpaired(mf_paired, stim_mf_rsa)
    n_win = int(np.sum(mf_paired[:, None] > stim_mf_rsa[None, :]))
    print(f"  MF vs MF.GTG: MF larger in {n_win}/{len(mf_paired) * len(stim_mf_rsa)} "
          f"cross-comparisons, d(MF-GTG)={obs:+.3f}")
    print(f"      label-shuffle perm ({nperm} perms, n={len(mf_paired)} vs "
          f"{len(stim_mf_rsa)}): p={p:.4f} (two-sided)  [p-floor={pfloor:.3f}]")
    print("-" * 70)
    print("  NOTE: n is governed by 4 animals -> a hard power ceiling. The paired")
    print("        sign-flip on 6 matched pairs can only reach p=0.031 (and only if")
    print("        MF beats the comparison on ALL 6 pairs); those 6 pairs are not")
    print("        independent (each animal sits in 3). Respecting that dependence")
    print("        (dyadic sign-flip) the floor is p=0.125, so NO animal-level test")
    print("        can reach p<0.05 here. A non-significant result is INCONCLUSIVE")
    print("        (n-limited), not evidence of no difference. The inferential claim")
    print("        comes from the stimulus bootstrap below.")
    print("=" * 70)

    print("\nNOISE CEILING (recorded PSTH only; MF and predR are deterministic -> 1.0)")
    for m, animal in enumerate(group_names):
        print(f"  {animal:4s}: RDM split-half r_half={ref['ceiling_half'][m]:.3f} -> "
              f"Spearman-Brown r_full={ref['ceiling_full'][m]:.3f}")
    print(f"  mean pairwise ceiling sqrt(r_AA*r_BB) = {np.nanmean(ref['ceiling_pair']):.3f}")
    print(f"  measured trueR mean                   = {np.mean(true_paired):.3f}")
    ratio = true_paired / ref['ceiling_pair'][upper_mask]
    print(f"  trueR / ceiling  mean={np.mean(ratio):.3f}  "
          f"range=[{ratio.min():.3f}, {ratio.max():.3f}]")
    if np.mean(ratio) > 0.9:
        print("  WARNING: trueR is close to its ceiling -> the MF-vs-trueR gap is largely")
        print("           measurement noise, not a representational difference.")
    print("=" * 70)

    print("\nSTIMULUS BOOTSTRAP (two-sided 95% percentile CI; animals are NOT resampled)")
    for block_s in rl.BOOT_BLOCK_S_LIST:
        if block_s not in cache['boot']:
            continue
        b = cache['boot'][block_s]
        tag = 'primary' if block_s == rl.BOOT_PRIMARY_BLOCK_S else 'conservative'
        print(f"  block_s={block_s}s ({b['n_blocks']} blocks, "
              f"{b['n_timepoints_per_resample']} timepoints/resample) [{tag}]")
        _mf = np.nanmean(b['mf'][:, upper_mask], axis=1)
        _pr = np.nanmean(b['pred'][:, upper_mask], axis=1)
        _tr = np.nanmean(b['true'][:, upper_mask], axis=1)
        _st = np.nanmean(b['stim'], axis=1)
        r_mf = np.nanmean(b['ref_mf'][upper_mask])
        r_pr = np.nanmean(b['ref_pred'][upper_mask])
        r_tr = np.nanmean(b['ref_true'][upper_mask])
        r_st = np.nanmean(b['ref_stim'])
        for nm, diff, obs, refv in [
                ('MF - predR ', _mf - _pr, np.mean(mf_paired) - np.mean(pred_paired), r_mf - r_pr),
                ('MF - trueR ', _mf - _tr, np.mean(mf_paired) - np.mean(true_paired), r_mf - r_tr),
                ('MF - MF.GTG', _mf - _st, np.mean(mf_paired) - np.mean(stim_mf_rsa), r_mf - r_st)]:
            lo, hi, p_b, is_floor, n_ok = rl.boot_ci(diff, shift=obs - refv)
            star = '' if (lo <= 0 <= hi) else '  *CI excludes 0'
            p_str = f"p<{2.0 / n_ok:.4f}" if is_floor else f"p={p_b:.4f}"
            print(f"      {nm}: obs={obs:+.3f} (decimated-grid ref {refv:+.3f})  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]  {p_str} (two-sided, n={n_ok}){star}")
    print("=" * 70)


if __name__ == '__main__':
    main()
