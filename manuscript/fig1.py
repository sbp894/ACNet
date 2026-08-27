"""Manuscript Figure 1 -- stimulus, response, prediction, and encoding accuracy.

Self-contained: needs only `data/fig1.pkl` and `../weights/acnet_v1.pt`. Every
number plotted was pre-extracted by `build/build_fig1_cache.py`; the ONE thing
computed live is the part that actually demonstrates the model --

    load ACNet  ->  predict every neuron's PSTH from the stored waveform.

    python fig1.py

Set USE_CACHED_PREDICTION = True to skip the model entirely (no torch needed)
and plot the cached reference prediction instead.
"""
import os
import pickle
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # so `import acnet_model` resolves

PKL_FNAME = os.path.join(HERE, 'data', 'fig1.pkl')
OUT_FNAME = os.path.join(HERE, 'figures', 'fig1.png')
USE_CACHED_PREDICTION = False

# --------------------------------------------------------------------------- #
# Style. CB_color_cycle is inlined from nems_lbhb.plots so this folder has no
# LBHB dependency; the hex values are identical.
# --------------------------------------------------------------------------- #
CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']
FONT_SIZE = 8
plt.rcParams.update({
    'legend.fontsize': FONT_SIZE - 2, 'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE, 'axes.spines.right': False, 'axes.spines.top': False,
    'axes.prop_cycle': mpl.cycler(color=CB_COLOR_CYCLE),
    'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
})
COL = {'neural': CB_COLOR_CYCLE[0], 'multisite': CB_COLOR_CYCLE[1],
       'cnnfull': CB_COLOR_CYCLE[2], 'ln': CB_COLOR_CYCLE[3]}
STYLE = {'neural': '-', 'multisite': '--', 'cnnfull': '-.', 'ln': ':'}

# ------------------------- geometry, in cm --------------------------------- #
# Same arrangement as the published figure: a 3-row left block (schematic /
# summary row / power-law row) beside a 3-row demo column. Panel sizes are
# derived from single constants so same-format axes are exactly equal; slack
# lives in the gaps and margins, never in a panel.
FIG_W_CM, FIG_H_CM = 19.5, 10.2
MARGIN_L_CM, MARGIN_R_CM = 1.35, 0.30
MARGIN_T_CM, MARGIN_B_CM = 0.30, 1.10
PANEL_W_CM, PANEL_H_CM = 1.95, 2.20      # the eight small left-block panels
GAP_X_CM, GAP_Y_CM = 1.15, 1.30           # sized for a y-label / an x-label + ticks
BLOCK_GAP_CM = 1.20                       # between the left block and the demo column
SCHEMA_H_CM = 1.90
DEMO_W_CM = FIG_W_CM - MARGIN_L_CM - MARGIN_R_CM - BLOCK_GAP_CM - (4 * PANEL_W_CM + 3 * GAP_X_CM)
DEMO_GAP_Y_CM = 0.45
DEMO_H_CM = (FIG_H_CM - MARGIN_T_CM - MARGIN_B_CM - 2 * DEMO_GAP_Y_CM) / 3
CM_PER_IN = 2.54


def add_axes_cm(fig, x_cm, y_cm, w_cm, h_cm):
    """Place an axes by its lower-left corner in cm from the figure's lower-left."""
    return fig.add_axes([x_cm / FIG_W_CM, y_cm / FIG_H_CM, w_cm / FIG_W_CM, h_cm / FIG_H_CM])


def assert_equal_panels(fig, axes, label):
    """Same-format axes must render at identical size (global plotting rule)."""
    fig.canvas.draw()
    sizes = [(round(ax.get_window_extent().width, 3), round(ax.get_window_extent().height, 3))
             for ax in axes]
    assert len(set(sizes)) == 1, f'{label} panels differ in size: {sizes}'


def pearson_r(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Load the cache
# --------------------------------------------------------------------------- #
assert os.path.exists(PKL_FNAME), f'{PKL_FNAME} missing -- run build/build_fig1_cache.py'
d = pickle.load(open(PKL_FNAME, 'rb'))
demo, rtest, cvpca, coverage, meta = d['demo'], d['rtest'], d['cvpca'], d['coverage'], d['meta']

print(f"fig1.pkl built {meta['built']} on {meta['host']}; ACNet v{meta['acnet_version']}")
print(f"  demo: {meta['demo']['sitename']} val stim {meta['demo']['demo_idx']} "
      f"= {demo['wav_name']}")

t_meanrate = demo['t_meanrate']
prestim_s = demo['prestim_s']
psth_true = demo['psth_true']
gtg_ref = demo['gtg_nems_sqrt']

# --------------------------------------------------------------------------- #
# THE live computation: ACNet -> predicted PSTHs for the stored waveform
# --------------------------------------------------------------------------- #
if USE_CACHED_PREDICTION:
    print('USE_CACHED_PREDICTION -- plotting the cached ACNet prediction')
    psth_pred, gtg_plot = demo['psth_pred_acnet'], gtg_ref
else:
    import torch
    from acnet_model import load_acnet

    model, cell_rtest = load_acnet()
    model.eval()

    fe = meta['frontend']
    model.set_bnt_mode(overall_db=fe['overall_db'], fixed_amp_scale=fe['fixed_amp_scale'],
                       nems_match=fe['nems_match'])
    model.update_audio_process({'keep_pre_s': fe['keep_pre_s'],
                                'stim_dur_after_onset': fe['stim_dur_after_onset'],
                                'stim_duration_s': fe['stim_duration_s']})

    waveform = torch.as_tensor(demo['wav_int16'].astype(np.float32) / demo['wav_int16_scale'],
                               dtype=torch.float32).unsqueeze(0)
    fs_wav = demo['wav_fs']

    # the panel is drawn in the sqrt-amplitude domain the published figure used
    model.update_audio_process({'compress': 'sqrt'})
    with torch.no_grad():
        gtg_plot = model.audio_process(waveform.to(model._device()), fs_wav).cpu().numpy().T

    model.update_audio_process({'compress': fe['compress']})
    with torch.no_grad():
        psth_all = model.predict_psth(waveform.to(model._device()), fs=fs_wav).cpu().numpy()

    assert psth_all.shape[1] == 3124, f'unexpected readout width {psth_all.shape[1]}'
    # `acnet_cols_plot` already carries the valid-cell mask and the cluster row
    # order, so no name lookup is needed here and rows cannot silently permute.
    psth_pred = psth_all[:, demo['acnet_cols_plot']].T.astype(np.float64)
    psth_pred[psth_pred < 0] = 0
    psth_pred = psth_pred / psth_pred.max(axis=1, keepdims=True)
    psth_pred = (psth_pred - psth_pred.min(axis=1, keepdims=True)) / np.ptp(psth_pred, axis=1, keepdims=True)

    # Loud verification: a drift in ACNet, the front end, or the wav must abort
    # rather than quietly produce a different figure.
    r_gtg = pearson_r(gtg_plot, gtg_ref)
    r_psth = pearson_r(psth_pred, demo['psth_pred_acnet'])
    max_dpsth = float(np.abs(psth_pred - demo['psth_pred_acnet']).max())
    print(f'  live vs cached: gtg r={r_gtg:.8f}  psth r={r_psth:.8f}  max|dPSTH|={max_dpsth:.2e}')
    assert r_gtg > 0.999, f'live gammatonegram != cached NEMS reference (r={r_gtg:.6f})'
    assert r_psth > 0.999 and max_dpsth < 1e-2, (
        f'live prediction != cached reference (r={r_psth:.6f}, max|d|={max_dpsth:.2e})')

print(f"  wav-path vs published y_est panel: median per-cell "
      f"r={demo['pred_acnet_vs_yest_median_r']:.4f}")

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(FIG_W_CM / CM_PER_IN, FIG_H_CM / CM_PER_IN))

row_pca_y = MARGIN_B_CM
row_summary_y = row_pca_y + PANEL_H_CM + GAP_Y_CM
row_schema_y = row_summary_y + PANEL_H_CM + GAP_Y_CM
left_x = [MARGIN_L_CM + i * (PANEL_W_CM + GAP_X_CM) for i in range(4)]
demo_x = left_x[3] + PANEL_W_CM + BLOCK_GAP_CM

ax_schema = add_axes_cm(fig, left_x[0], row_schema_y,
                        4 * PANEL_W_CM + 3 * GAP_X_CM, SCHEMA_H_CM)
ax_coverage, ax_rtest_cnn, ax_rtest_ln, ax_cumvar = [
    add_axes_cm(fig, x, row_summary_y, PANEL_W_CM, PANEL_H_CM) for x in left_x]
ax_pca_neural, ax_pca_multisite, ax_pca_cnn, ax_pca_ln = [
    add_axes_cm(fig, x, row_pca_y, PANEL_W_CM, PANEL_H_CM) for x in left_x]
ax_demo_gtg, ax_demo_true, ax_demo_pred = [
    add_axes_cm(fig, demo_x, FIG_H_CM - MARGIN_T_CM - (i + 1) * DEMO_H_CM - i * DEMO_GAP_Y_CM,
                DEMO_W_CM, DEMO_H_CM) for i in range(3)]

# ---- schematic placeholder (as in the published draft) -------------------- #
ax_schema.text(.5, .5, 'ACNet schematic', ha='center', va='center',
               transform=ax_schema.transAxes)
ax_schema.axis('off')

# ---- stimulus coverage per site ------------------------------------------- #
animal_name = coverage['animal_name']
uniq_animal = np.unique(animal_name)
gray_by_animal = {ani: f"{.25 + .5 * i / (len(uniq_animal) - 1)}"
                  for i, ani in enumerate(uniq_animal)}
for site in coverage['uniq_sites']:
    site_idx = coverage['fitstim_index_new'][coverage['fitstim_site'] == site]
    stim_hours = np.array([site_idx.min(), site_idx.max()]) * coverage['stim_dur_s'] / 3600
    ax_coverage.plot(stim_hours, [site, site], c=gray_by_animal[animal_name[site]], lw=.8)
site_yticks = [0, len(coverage['uniq_sites']) - 1]
ax_coverage.set(yticks=site_yticks, yticklabels=[v + 1 for v in site_yticks],
                xlabel='Stimulus (h)', ylabel='Site')

# ---- r_test: ACNet vs the two single-site baselines ------------------------ #
SKIP_N = rtest['skip_n']
MRK = 4
rtest_ticks, rtest_lim = [0, .5, 1], [-0.01, 1.01]
for ax, base, base_mu, resnet, resnet_mu, xlabel in [
        (ax_rtest_cnn, rtest['cnn'], rtest['cnn_site_mu'], rtest['cnn_matched_resnet'],
         rtest['cnn_site_mu_resnet'], 'Single-site CNN'),
        (ax_rtest_ln, rtest['ln'], rtest['ln_site_mu'], rtest['ln_matched_resnet'],
         rtest['ln_site_mu_resnet'], 'LN')]:
    ax.plot(base[::SKIP_N], resnet[::SKIP_N], alpha=.2, marker='.', linestyle='none',
            markersize=MRK, markeredgecolor='none', markerfacecolor='gray')
    ax.plot(base_mu, resnet_mu, alpha=.75, marker='.', linestyle='none',
            markersize=MRK + 1, markeredgecolor='none', markerfacecolor='k')
    ax.plot(rtest_lim, rtest_lim, 'k--', lw=.8)
    ax.set(xlim=rtest_lim, ylim=rtest_lim, xlabel=xlabel,
           xticks=rtest_ticks, yticks=rtest_ticks,
           xticklabels=['0', '0.5', '1'], yticklabels=['0', '0.5', '1'])
ax_rtest_cnn.set(ylabel='ACNet $r_{test}$')   # leftmost of the pair
ax_rtest_ln.set(yticklabels=[])

# ---- cumulative variance explained ---------------------------------------- #
ndims = cvpca['ndims']
XLIM_PCA = [5, 3000]
for tag, key, label in [('neural', 'neural', 'Data'), ('multisite', 'resnet', 'ACNet'),
                        ('ln', 'LNmodel', 'LN'), ('cnnfull', 'CNNfull', '1S:CNN')]:
    ax_cumvar.plot(ndims, cvpca[key]['cumvar'], c=COL[tag], linestyle=STYLE[tag],
                   label=label, lw=1.2)
ax_cumvar.set(xscale='log', xlim=XLIM_PCA, xlabel='PC dim', ylabel='Cum. var. exp.',
              yticks=[0.2, 0.4], yticklabels=['.2', '.4'])

# ---- power-law variance spectra ------------------------------------------- #
p_data_neural = cvpca['neural']['p_data']
ylim_ve = p_data_neural.max() * np.array([1e-3, 1.05])
for ax, tag, key, label in [(ax_pca_neural, 'neural', 'neural', 'PSTH'),
                            (ax_pca_multisite, 'multisite', 'resnet', 'ACNet'),
                            (ax_pca_cnn, 'cnnfull', 'CNNfull', '1S:CNN'),
                            (ax_pca_ln, 'ln', 'LNmodel', 'LN')]:
    ax.plot(ndims[::SKIP_N], cvpca[key]['p_data'][::SKIP_N], marker='.', linestyle='none',
            markersize=MRK, alpha=.25, markeredgecolor='none', markerfacecolor=COL[tag])
    ax.plot(ndims, cvpca[key]['p_fit'], c=COL[tag], linestyle='-', lw=1.5)
    ax.text(.05, .06, f"{label}\n$\\alpha$={cvpca[key]['alpha']:.2f}", transform=ax.transAxes,
            ha='left', va='bottom', fontsize=FONT_SIZE - 1)
    ax.set(xscale='log', yscale='log', xlim=XLIM_PCA, ylim=ylim_ve, xlabel='PC dim',
           yticks=[1e-4, 1e-2], yticklabels=['e-4', 'e-2'])
ax_pca_neural.set(ylabel='Var. exp.')
for ax in (ax_pca_multisite, ax_pca_ln, ax_pca_cnn):
    ax.set(yticklabels=[])

# ---- demo column: stimulus, response, prediction --------------------------- #
ax_demo_gtg.imshow(gtg_plot, origin='lower', aspect='auto', cmap='Blues', vmax=1,
                   extent=[-prestim_s, t_meanrate.max(), 0, gtg_plot.shape[0]])
ax_demo_gtg.set(yticks=demo['fticks'], yticklabels=demo['fticklabs'], ylabel='Freq (kHz)')

cell_yticks = [0, psth_true.shape[0] - 1]
ax_demo_true.pcolormesh(t_meanrate, np.arange(psth_true.shape[0]), psth_true,
                        cmap='gray_r', edgecolors='face', rasterized=True, shading='auto')
ax_demo_true.set(yticks=cell_yticks, yticklabels=[v + 1 for v in cell_yticks],
                 ylabel='Neuron (meas.)')

ax_demo_pred.pcolormesh(t_meanrate, np.arange(psth_pred.shape[0]), psth_pred,
                        cmap='gray_r', edgecolors='face', rasterized=True, vmax=1.5,
                        shading='auto')
ax_demo_pred.set(yticks=cell_yticks, yticklabels=[v + 1 for v in cell_yticks],
                 xticks=demo['tticks'], xlabel='Time (s)', ylabel='Neuron (ACNet)')

for ax in (ax_demo_gtg, ax_demo_true):        # tick labels only on the last row
    ax.set(xticks=demo['tticks'], xticklabels=[])
for ax in (ax_demo_gtg, ax_demo_true, ax_demo_pred):
    ax.set(xlim=[-prestim_s, t_meanrate.max()])

# ---- geometry check, then save -------------------------------------------- #
assert_equal_panels(fig, [ax_coverage, ax_rtest_ln, ax_rtest_cnn, ax_cumvar,
                          ax_pca_neural, ax_pca_multisite, ax_pca_ln, ax_pca_cnn],
                    'left-block')
assert_equal_panels(fig, [ax_demo_gtg, ax_demo_true, ax_demo_pred], 'demo-column')

os.makedirs(os.path.dirname(OUT_FNAME), exist_ok=True)
fig.savefig(OUT_FNAME, dpi=300)
print(f'wrote {OUT_FNAME}')

# --------------------------------------------------------------------------- #
# The statistical claims, recomputed from the cached vectors and checked against
# the values frozen at build time.
# --------------------------------------------------------------------------- #
from scipy.stats import wilcoxon  # noqa: E402

for tag, resnet_v, base_v, label in [
        ('ln', rtest['ln_matched_resnet'], rtest['ln'], 'LN'),
        ('cnn', rtest['cnn_matched_resnet'], rtest['cnn'], 'Single-site CNN')]:
    stat, pval = wilcoxon(resnet_v, base_v, alternative='greater')
    cached = rtest['wilcoxon'][tag]
    print(f'ACNet vs {label}: W={stat:.1f}, p={pval:.3e}, n={len(base_v)}, '
          f"median diff={np.median(resnet_v - base_v):.4f}")
    assert np.isclose(stat, cached['stat']) and len(base_v) == cached['n'], (
        f'{label} Wilcoxon does not reproduce the value frozen at build time')
