"""Build manuscript/data/fig1.pkl -- the one-time, LBHB-only cache step for Fig1.

Fig1 currently reads ~1.1 GB of intermediate pickles (a 758 MB cvPCA file for two
(2902,) vectors, a 160 MB predicted-PSTH file for one site, a 150 MB gammatonegram
file for one stimulus). This script reads them once and writes every number the
figure actually plots into a single ~2 MB pkl, so `manuscript/fig1.py` needs
nothing but that pkl and `weights/acnet_v1.pt`.

Reads only. Writes only into manuscript/data/.

    cd /auto/users/satya/code/projects_getting_started/ACNet_v1
    python -u manuscript/build/build_fig1_cache.py

Three things are RESOLVED here rather than assumed, because each is a silent
failure mode:
  * which wav `demo_idx=5` refers to (scanned over all val candidates);
  * `fixed_amp_scale`, which the site pkl stores as the literal string
    'nems_meta' -- it is fitted against the cached NEMS gammatonegram, and
    because correlation is scale-invariant the fit is scored on relative RMS
    error, not on r;
  * whether ACNet v1 was exported from the same run as the figure's model dir.
"""
import gc
import os
import pickle
import subprocess
import sys
from datetime import datetime

import numpy as np
import scipy
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import linregress, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ACNET_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ACNET_ROOT)
from acnet_model import _load_wav, load_acnet  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), 'data')
OUT_FNAME = os.path.join(OUT_DIR, 'fig1.pkl')
LOG_FNAME = os.path.join(HERE, 'build_fig1_cache.log')

# ------------------------------- sources ----------------------------------- #
PT_ROOT = '/auto/users/satya/code/projects_getting_started/pytorch_models/'
MDL_DIR = PT_ROOT + 'trained_MT_models/ResNetMT_site_nemsGT_3stage_log10xComp/MT_svd_init_6L_v2/nsites62_15/'
GTG_SITE_DIR = PT_ROOT + 'misc_output/BNTgtg_nems_fs100Hz_nCF32_sites_sqrt_amp/'
RTEST_LN_FNAME = PT_ROOT + 'misc_output/SVD_model_BNT_val_PSTHest/rtest_LNmodels.pkl'
RTEST_CNN_FNAME = PT_ROOT + 'misc_output/SVD_model_BNT_val_PSTHest/rtest_CNN_2L_full_fit_log10x.pkl'
STIM_STATS_FNAME = PT_ROOT + 'MS_AcxManifold/plot_data/fit_stim_stats.pkl'
EST_PSTH_FNAME = MDL_DIR + 'stage3_fine/estimated_test_PSTH_fit_nfloor1000.pkl'
EPOCH_DATA_FNAME = MDL_DIR + 'stage3_fine/C1D_75_100_125_150_175_200_data_epoch.pkl'
CVPCA_DIR = MDL_DIR + 'stage3_fine/cvPCA/'
SOUND_ROOT = '/auto/data/sounds/BigNat/v2/'

# --------------------------- figure parameters ----------------------------- #
# These reproduce MS_AcxManifold/Fig1_stim_resp_rtest.py exactly.
DEMO_SITENAME = 'PRN015a'
DEMO_STIM_TYPE = 'val'
DEMO_IDX = 5
FS_GTG = 100
PRESTIM_S = 0.5
STIM_DUR_AFTER_ONSET_S = 18.5     # save_nemsGTGram_BNT.py
TRIAL_DURATION_S = 18.0           # baphy ReferenceHandle Duration; NAT_stim zero-pads to this
POWERLAW_N1 = 5
POWERLAW_N2 = 400
CVPCA_MODEL_TYPE = 'fit'
CVPCA_POSTFIX = '_rfloor'
CVPCA_COMP_POSTFIX = '_log10x'

FAS_GRID = np.concatenate([np.array([1., 2., 5., 10., 25., 50., 100., 150., 200.,
                                     250., 300., 400., 500., 750., 1000.])])
MIN_GTG_R = 0.99          # live ACNet gtg vs cached NEMS gtg (shape/timing check)
MIN_WAV_MARGIN = 0.02     # winning wav must beat the runner-up by this much in r
MAX_GTG_RELERR = 0.05     # relative RMS error after fitting fixed_amp_scale (scale check)
MIN_LIVE_VS_YEST_R = 0.95  # wav-path prediction vs the published concatenated y_est


# ----------------------------- helpers ------------------------------------- #
def get_power_law_exp(n, p_cumsum, n1=POWERLAW_N1, n2=POWERLAW_N2):
    """Power-law fit to the cvPCA variance spectrum. Copied verbatim from
    MS_AcxManifold/Fig1_stim_resp_rtest.py:31 -- note this SHADOWS a different
    function of the same name in plot_helpers.py; do not substitute that one."""
    eps = 1e-10
    p_data = np.diff(p_cumsum, prepend=0)
    p_data[p_data < eps] = eps
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log(n[n1 - 1:n2]), np.log(p_data[n1 - 1:n2]))
    p_fit = np.exp(slope * np.log(n) + intercept)
    return slope, intercept, p_fit, p_data


def get_common_cell_rtest(model1_rtest, model1_cells, model2_rtest, model2_cells):
    """Align two models' r_test on their common cells, plus per-site means.
    Copied verbatim from MS_AcxManifold/Fig1_stim_resp_rtest.py:43. Handles both
    the ragged per-site-list layout (LN) and the flat (3124,) layout (CNN)."""
    if isinstance(model1_rtest, np.ndarray):
        m1_vals, m1_names = model1_rtest, model1_cells
    else:
        m1_vals = np.concatenate(model1_rtest)
        m1_names = np.concatenate(model1_cells)

    if isinstance(model2_rtest, np.ndarray):
        m2_vals, m2_names = model2_rtest, model2_cells
    else:
        m2_vals = np.concatenate(model2_rtest)
        m2_names = np.concatenate(model2_cells)

    common_names = np.intersect1d(m1_names, m2_names)

    mask1 = np.isin(m1_names, common_names)
    m1_vals_filtered, m1_names_filtered = m1_vals[mask1], m1_names[mask1]
    m1_aligned = m1_vals_filtered[np.argsort(m1_names_filtered)]

    mask2 = np.isin(m2_names, common_names)
    m2_vals_filtered, m2_names_filtered = m2_vals[mask2], m2_names[mask2]
    m2_aligned = m2_vals_filtered[np.argsort(m2_names_filtered)]

    sites_aligned = np.array([x.split('-')[0] for x in common_names])
    unique_sites = np.unique(sites_aligned)
    m1_site_means = np.array([m1_aligned[sites_aligned == s].mean() for s in unique_sites])
    m2_site_means = np.array([m2_aligned[sites_aligned == s].mean() for s in unique_sites])
    return m1_aligned, m2_aligned, m1_site_means, m2_site_means, common_names


def get_sorted_data(data_matrix, leaf_order=None):
    """Hierarchical-clustering row order. Copied from MS_AcxManifold/plot_helpers.py:33."""
    if leaf_order is None:
        linked = linkage(pdist(data_matrix, metric='correlation'), method='average')
        leaf_order = dendrogram(linked, no_plot=True)['leaves']
    return data_matrix[leaf_order], leaf_order


def normalise_psth(psth):
    """The published demo-panel chain: clip <0, drop dead cells, max-norm, ptp-norm."""
    psth = np.asarray(psth, dtype=np.float64).copy()
    psth[psth < 0] = 0
    valid = psth.max(axis=1) > 0
    psth = psth[valid] / psth[valid].max(axis=1, keepdims=True)
    psth = (psth - psth.min(axis=1, keepdims=True)) / np.ptp(psth, axis=1, keepdims=True)
    return psth, valid


def source_meta(path):
    st = os.stat(path)
    return {'path': path, 'size_bytes': st.st_size,
            'mtime': datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')}


def pearson_r(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def rel_rms_err(a, b):
    """Scale-SENSITIVE agreement, unlike r. Used to fit fixed_amp_scale."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def per_cell_r(a, b):
    return np.array([pearson_r(a[i], b[i]) if a[i].std() > 0 and b[i].std() > 0 else np.nan
                     for i in range(a.shape[0])])


def git_sha(repo_dir):
    try:
        return subprocess.check_output(['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def configure_frontend(model, overall_db, fixed_amp_scale, nems_match, compress):
    """Put ACNet's front end in the mode the BNT gammatonegrams were built in,
    then ASSERT it took -- update_audio_process silently ignores unknown keys."""
    model.set_bnt_mode(overall_db=overall_db, fixed_amp_scale=fixed_amp_scale,
                       nems_match=nems_match)
    model.update_audio_process({'keep_pre_s': PRESTIM_S,
                                'stim_dur_after_onset': STIM_DUR_AFTER_ONSET_S,
                                'stim_duration_s': TRIAL_DURATION_S,
                                'compress': compress})
    ap = model.audio_process
    assert ap.lbhb_mode is True and ap.level_mode == 'approx', 'set_bnt_mode did not take'
    assert ap.keep_pre_s == PRESTIM_S, f'keep_pre_s={ap.keep_pre_s}'
    assert ap.stim_dur_after_onset == STIM_DUR_AFTER_ONSET_S, f'stim_dur={ap.stim_dur_after_onset}'
    assert ap.stim_duration_s == TRIAL_DURATION_S, f'stim_duration_s={ap.stim_duration_s}'
    assert ap.compress == compress, f'compress={ap.compress}'
    assert ap.nems_match == nems_match, f'nems_match={ap.nems_match}'
    assert float(ap.fixed_amp_scale) == float(fixed_amp_scale)
    return model


def acnet_gtg(model, waveform, fs_wav):
    with torch.no_grad():
        g = model.audio_process(waveform.to(model._device()), fs_wav)
    return g.cpu().numpy().T          # (nCF, T)


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta_sources = {}

    # ---- 1. small pkls: LN / CNN r_test, stimulus coverage ---------------- #
    print('[1/8] small pkls (r_test, stim coverage)', flush=True)
    ln_data = pickle.load(open(RTEST_LN_FNAME, 'rb'))
    cnn_data = pickle.load(open(RTEST_CNN_FNAME, 'rb'))
    stim_stats = pickle.load(open(STIM_STATS_FNAME, 'rb'))
    for name, path in [('rtest_LN', RTEST_LN_FNAME), ('rtest_CNN', RTEST_CNN_FNAME),
                       ('fit_stim_stats', STIM_STATS_FNAME)]:
        meta_sources[name] = source_meta(path)
    assert isinstance(ln_data['r_test'], list), 'LN r_test expected ragged per-site list'
    assert isinstance(cnn_data['r_test'], np.ndarray), 'CNN r_test expected flat array'

    # ---- 2. demo site: NEMS gammatonegram + true PSTH (150 MB) ------------ #
    print('[2/8] demo site gammatonegram + true PSTH', flush=True)
    site_fnames = [f for f in os.listdir(GTG_SITE_DIR)
                   if f.endswith(f'_BNTgtg_nems_{DEMO_SITENAME}.pkl')]
    assert len(site_fnames) == 1, f'expected 1 site pkl for {DEMO_SITENAME}, got {site_fnames}'
    site_fname = GTG_SITE_DIR + site_fnames[0]
    meta_sources['gtg_site'] = source_meta(site_fname)
    site_data = pickle.load(open(site_fname, 'rb'))

    assert DEMO_STIM_TYPE == 'val', f'only the val path is cached; got {DEMO_STIM_TYPE}'
    gtg_nems = np.asarray(site_data['X_gtg_Val10Rep_nems'][DEMO_IDX].T, dtype=np.float64)
    psth_true_raw = np.asarray(site_data['Y_psth_val10Rep'][DEMO_IDX].T, dtype=np.float64)
    site_cell_names = np.asarray(site_data['cell_names'])
    site_overall_db = float(np.asarray(site_data['overall_db']).squeeze())
    n_bins = psth_true_raw.shape[1]
    n_val_stims = len(site_data['X_gtg_Val10Rep_nems'])
    assert gtg_nems.shape == (32, n_bins), f'gtg {gtg_nems.shape} vs psth {psth_true_raw.shape}'
    assert n_bins == int((PRESTIM_S + STIM_DUR_AFTER_ONSET_S) * FS_GTG), (
        f'{n_bins} bins != {(PRESTIM_S + STIM_DUR_AFTER_ONSET_S) * FS_GTG:.0f}; the front-end '
        f'constants no longer describe this file')
    assert len(site_cell_names) == psth_true_raw.shape[0]
    assert all(n.startswith(DEMO_SITENAME + '-') for n in site_cell_names)
    print(f'      site={DEMO_SITENAME} idx={DEMO_IDX}/{n_val_stims} gtg={gtg_nems.shape} '
          f'cells={len(site_cell_names)} overall_db={site_overall_db}', flush=True)
    del site_data
    gc.collect()

    # ---- 3. published predicted PSTH + ResNet r_test (160 MB) ------------- #
    print('[3/8] published predicted PSTH + ResNet r_test', flush=True)
    meta_sources['est_psth'] = source_meta(EST_PSTH_FNAME)
    est_data = pickle.load(open(EST_PSTH_FNAME, 'rb'))

    est_sites = [x[0].split('-')[0] for x in est_data['cell_names']]
    site_i = est_sites.index(DEMO_SITENAME)
    col_inds = np.arange(len(est_data['cell_names'][site_i])) + sum(
        len(est_data['cell_names'][i]) for i in range(site_i))
    row_inds = DEMO_IDX * n_bins + np.arange(n_bins)
    psth_yest_raw = np.asarray(est_data['y_est'][:, col_inds][row_inds, :].T, dtype=np.float64)
    demo_cell_names_all = np.asarray(est_data['cell_names'][site_i])
    # The published script index-aligns y_est columns to the site pkl's cell order
    # without checking. Check it: if these ever diverged the panel would be mis-rowed.
    assert np.array_equal(demo_cell_names_all, site_cell_names), (
        'cell order differs between the gtg site pkl and the estimated-PSTH pkl; '
        'the published demo panel mixes rows from two different orderings')

    resnet_rtest = [est_data['r_test'][i][m] for i, m in enumerate(est_data['RN_cells_above_floor'])]
    resnet_cells = [est_data['cell_names'][i][m] for i, m in enumerate(est_data['RN_cells_above_floor'])]
    del est_data
    gc.collect()

    psth_yest, valid_cells = normalise_psth(psth_yest_raw)
    psth_true = psth_true_raw[valid_cells] / psth_true_raw[valid_cells].max(axis=1, keepdims=True)
    psth_true = (psth_true - psth_true.min(axis=1, keepdims=True)) / np.ptp(psth_true, axis=1, keepdims=True)
    demo_cell_names = demo_cell_names_all[valid_cells]

    psth_true, leaf_order = get_sorted_data(psth_true)
    leaf_order = np.asarray(leaf_order)
    psth_yest = psth_yest[leaf_order, :]
    demo_cell_names = demo_cell_names[leaf_order]
    print(f'      demo cells kept {valid_cells.sum()}/{len(valid_cells)}', flush=True)

    # ---- 4. r_test comparisons + Wilcoxon --------------------------------- #
    print('[4/8] r_test comparisons + Wilcoxon', flush=True)
    ln_matched_resnet, rtest_ln, ln_mu_resnet, ln_mu, common_ln = get_common_cell_rtest(
        resnet_rtest, resnet_cells, ln_data['r_test'], ln_data['cell_names'])
    cnn_matched_resnet, rtest_cnn, cnn_mu_resnet, cnn_mu, common_cnn = get_common_cell_rtest(
        resnet_rtest, resnet_cells, cnn_data['r_test'], cnn_data['cell_names'])

    wilcoxon_out = {}
    for tag, resnet_v, base_v in [('ln', ln_matched_resnet, rtest_ln),
                                  ('cnn', cnn_matched_resnet, rtest_cnn)]:
        stat, pval = wilcoxon(resnet_v, base_v, alternative='greater')
        wilcoxon_out[tag] = {'stat': float(stat), 'p': float(pval), 'n': int(len(base_v)),
                             'median_diff': float(np.median(resnet_v - base_v)),
                             'alternative': 'greater'}
        print(f'      ResNet vs {tag.upper()}: W={stat:.1f}, p={pval:.3e}, n={len(base_v)}, '
              f'median diff={np.median(resnet_v - base_v):.4f}', flush=True)

    # ---- 5. cvPCA: the 758 MB read, for two vectors ----------------------- #
    print('[5/8] cvPCA (this loads the 758 MB neural file)', flush=True)

    def cvpca_path(tag):
        return CVPCA_DIR + f'cvPCA_{tag}_{CVPCA_MODEL_TYPE}{CVPCA_POSTFIX}{CVPCA_COMP_POSTFIX}.pkl'

    meta_sources['cvPCA_neural'] = source_meta(cvpca_path('neural'))
    with open(cvpca_path('neural'), 'rb') as fh:
        _neural = pickle.load(fh)
        cv_neural_odd = np.asarray(_neural['cv_oddFit_evenVal_VarExp'], dtype=np.float64).copy()
        cv_neural_even = np.asarray(_neural['cv_evenFit_oddVal_VarExp'], dtype=np.float64).copy()
    del _neural
    gc.collect()
    print(f'      neural cvPCA extracted {cv_neural_odd.shape}; 758 MB released', flush=True)

    cvpca_keys = {'resnet': ('cv_modelFit_oddVal_VarExp', 'cv_modelFit_evenVal_VarExp'),
                  'LNmodel': ('cv_1siteLNFit_oddVal_VarExp', 'cv_1siteLNFit_evenVal_VarExp'),
                  'CNNfull': ('cv_1siteCNNfullFit_oddVal_VarExp', 'cv_1siteCNNfullFit_evenVal_VarExp')}
    cv_odd = {'neural': cv_neural_odd}
    cv_even = {'neural': cv_neural_even}
    for tag, (k_odd, k_even) in cvpca_keys.items():
        meta_sources[f'cvPCA_{tag}'] = source_meta(cvpca_path(tag))
        d = pickle.load(open(cvpca_path(tag), 'rb'))
        cv_odd[tag] = np.asarray(d[k_odd], dtype=np.float64)
        cv_even[tag] = np.asarray(d[k_even], dtype=np.float64)

    ndims = 1 + np.arange(len(cv_neural_odd))
    cvpca = {'ndims': ndims, 'n1': POWERLAW_N1, 'n2': POWERLAW_N2}
    for tag in ['neural', 'resnet', 'LNmodel', 'CNNfull']:
        assert cv_odd[tag].shape == ndims.shape, f'{tag} cvPCA length mismatch'
        slope, intercept, p_fit, p_data = get_power_law_exp(ndims, cv_odd[tag] + cv_even[tag])
        cvpca[tag] = {'cv_odd': cv_odd[tag], 'cv_even': cv_even[tag],
                      'cumvar': (cv_odd[tag] + cv_even[tag]) / 2,
                      'p_data': p_data, 'p_fit': p_fit,
                      'alpha': float(-slope), 'intercept': float(intercept)}
        print(f'      {tag:8s} alpha={-slope:.3f}', flush=True)
    # quoted in the manuscript text (Fig1_stim_resp_rtest.py:229-232)
    cvpca['mu_resnet'] = float(np.mean((
        cv_odd['resnet'].max() / cv_even['neural'].max(),
        cv_even['resnet'].max() / cv_odd['neural'].max())))
    cvpca['mu_LN'] = float(np.mean((
        cv_odd['LNmodel'].max() / cv_even['neural'].max(),
        cv_even['LNmodel'].max() / cv_odd['neural'].max())))

    # ---- 6. ACNet: provenance assertion ----------------------------------- #
    print('[6/8] ACNet provenance', flush=True)
    model, acnet_cell_rtest = load_acnet()
    model.eval()
    ckpt = torch.load(os.path.join(ACNET_ROOT, 'weights', 'acnet_v1.pt'),
                      map_location='cpu', weights_only=False)
    acnet_cell_names = np.asarray(ckpt['cell_names'])
    assert len(acnet_cell_names) == 3124

    meta_sources['epoch_data'] = source_meta(EPOCH_DATA_FNAME)
    epoch_pkl = pickle.load(open(EPOCH_DATA_FNAME, 'rb'))
    # NB: Fig1_stim_resp_rtest.py:216 calls epoch_pkl['epoch_data'] "mt_rtest_cc",
    # but that key holds the training curves; the per-cell test r is
    # 'test_accuracy_cc'. The published script never uses the former numerically.
    mdl_rtest = np.concatenate(epoch_pkl['test_accuracy_cc'])
    mdl_cells = np.concatenate(epoch_pkl['cell_names'])
    del epoch_pkl
    order = np.argsort(mdl_cells)
    names_match = bool(np.array_equal(mdl_cells[order], acnet_cell_names))
    rtest_max_diff = float(np.abs(mdl_rtest[order] - acnet_cell_rtest).max()) if names_match else float('nan')
    print(f'      cell names match nsites62_15: {names_match}; '
          f'max |r_test diff|: {rtest_max_diff:.3e}', flush=True)
    assert names_match, ('ACNet checkpoint cell_names differ from nsites62_15 -- the cached '
                         'reference prediction does not describe this model')
    assert rtest_max_diff < 1e-5, (
        f'ACNet cell_rtest differs from nsites62_15 by {rtest_max_diff:.3e}: ACNet v1 was '
        f'exported from a DIFFERENT run than the figure model dir.')
    provenance = {'acnet_matches_nsites62_15': names_match, 'rtest_max_diff': rtest_max_diff}

    # ---- 7. resolve the demo wav AND fixed_amp_scale ---------------------- #
    print('[7/8] resolving demo wav + fixed_amp_scale', flush=True)
    candidates = sorted(f for f in os.listdir(SOUND_ROOT)
                        if f.startswith('00seq') and f.endswith('.wav'))
    assert candidates, f'no 00seq*.wav under {SOUND_ROOT}'
    print(f'      {len(candidates)} val-wav candidates: {candidates}', flush=True)

    # 7a. Which wav? r is scale-invariant, so a provisional fixed_amp_scale is
    #     fine for identification -- only the time/frequency pattern matters here.
    wav_r = {}
    waveforms = {}
    for cand in candidates:
        waveforms[cand] = _load_wav(SOUND_ROOT + cand, int16_scale=32768.0)
    configure_frontend(model, site_overall_db, 250.0, nems_match=False, compress='sqrt')
    for cand in candidates:
        w, fs_w = waveforms[cand]
        g = acnet_gtg(model, w, fs_w)
        wav_r[cand] = pearson_r(g, gtg_nems) if g.shape == gtg_nems.shape else -np.inf
        print(f'      {cand:20s} r={wav_r[cand]: .6f}  shape={g.shape}', flush=True)

    ranked = sorted(wav_r.items(), key=lambda kv: kv[1], reverse=True)
    wav_name, best_r = ranked[0]
    runner_up_name, runner_up_r = ranked[1]
    assert best_r > MIN_GTG_R, (
        f'best wav {wav_name} only reaches r={best_r:.4f} against the cached NEMS gtg; '
        f'the front-end constants or the demo index are wrong')
    assert best_r - runner_up_r > MIN_WAV_MARGIN, (
        f'{wav_name} (r={best_r:.4f}) does not clearly beat {runner_up_name} '
        f'(r={runner_up_r:.4f}); the wav<->demo_idx mapping is ambiguous')
    print(f'      -> demo_idx={DEMO_IDX} is {wav_name} '
          f'(r={best_r:.6f}, runner-up {runner_up_name} {runner_up_r:.6f})', flush=True)

    waveform, fs_wav = waveforms[wav_name]

    # 7b. fixed_amp_scale. The site pkl stores the string 'nems_meta', not the
    #     value (save_nemsGTGram_BNT.py:57). Fit it on relative RMS error --
    #     remove_clicks(w*fas, 15) is nonlinear in fas, so this is a search, not
    #     a closed form, and r cannot score it because r ignores scale.
    fas_scores = {}
    for nems_match in (False, True):
        for fas in FAS_GRID:
            configure_frontend(model, site_overall_db, float(fas), nems_match, compress='sqrt')
            g = acnet_gtg(model, waveform, fs_wav)
            fas_scores[(nems_match, float(fas))] = (rel_rms_err(g, gtg_nems), pearson_r(g, gtg_nems))
    best_key = min(fas_scores, key=lambda k: fas_scores[k][0])
    best_nems_match, best_fas = best_key
    best_relerr, best_fas_r = fas_scores[best_key]
    for (nm, fas), (err, r) in sorted(fas_scores.items(), key=lambda kv: kv[1][0])[:8]:
        print(f'      nems_match={str(nm):5s} fas={fas:7.1f}  relRMSE={err:.5f}  r={r:.6f}', flush=True)
    print(f'      -> fixed_amp_scale={best_fas:g}, nems_match={best_nems_match}, '
          f'relRMSE={best_relerr:.5f}, r={best_fas_r:.6f}', flush=True)
    assert best_relerr < MAX_GTG_RELERR, (
        f'best relative RMS error {best_relerr:.4f} exceeds {MAX_GTG_RELERR}: ACNet cannot '
        f'reproduce the cached NEMS gammatonegram at any fixed_amp_scale on the grid. The '
        f'stored gtg may not be sqrt-amplitude, or the level path differs.')
    assert best_fas not in (FAS_GRID[0], FAS_GRID[-1]), (
        f'fixed_amp_scale={best_fas:g} sits at the edge of the search grid; widen FAS_GRID')

    frontend = {'lbhb_mode': True, 'level_mode': 'approx', 'nems_match': best_nems_match,
                'overall_db': site_overall_db, 'fixed_amp_scale': float(best_fas),
                'keep_pre_s': PRESTIM_S, 'stim_dur_after_onset': STIM_DUR_AFTER_ONSET_S,
                'stim_duration_s': TRIAL_DURATION_S, 'num_cfs': 32, 'fs_gtg': FS_GTG,
                'compress': 'log10x', 'int16_scale': 32767.0 if best_nems_match else 32768.0}

    # store the waveform as int16 so the pkl stays ~1.5 MB
    wav_int16 = np.round(waveform.squeeze(0).numpy() * 32768.0).astype(np.int16)
    assert pearson_r(wav_int16.astype(np.float64) / 32768.0,
                     waveform.squeeze(0).numpy()) > 0.9999

    # ---- 8. live ACNet prediction from the wav ---------------------------- #
    print('[8/8] live ACNet predicted PSTH from the wav', flush=True)
    configure_frontend(model, site_overall_db, float(best_fas), best_nems_match,
                       compress='log10x')
    with torch.no_grad():
        psth_all = model.predict_psth(waveform.to(model._device()), fs=fs_wav)
    psth_all = psth_all.cpu().numpy()
    assert psth_all.shape == (n_bins, 3124), f'live psth {psth_all.shape}'

    name_to_col = {n: i for i, n in enumerate(acnet_cell_names)}
    missing = [n for n in demo_cell_names_all if n not in name_to_col]
    assert not missing, f'{len(missing)} demo cells absent from the ACNet checkpoint: {missing[:5]}'
    acnet_cols_all = np.array([name_to_col[n] for n in demo_cell_names_all])
    psth_live_raw = psth_all[:, acnet_cols_all].T

    psth_live, valid_live = normalise_psth(psth_live_raw)
    if not np.array_equal(valid_live, valid_cells):
        print(f'      WARNING: valid-cell mask differs between the wav path '
              f'({valid_live.sum()}) and the published y_est path ({valid_cells.sum()}); '
              f'using the y_est mask so both reference arrays stay row-comparable', flush=True)
    # re-derive on the published mask/order so every cached demo array lines up
    psth_live = psth_live_raw[valid_cells]
    psth_live[psth_live < 0] = 0
    psth_live = psth_live / psth_live.max(axis=1, keepdims=True)
    psth_live = (psth_live - psth_live.min(axis=1, keepdims=True)) / np.ptp(psth_live, axis=1, keepdims=True)
    psth_live = psth_live[leaf_order, :]

    acnet_cols_plot = acnet_cols_all[valid_cells][leaf_order]
    assert np.array_equal(acnet_cell_names[acnet_cols_plot], demo_cell_names)

    cellwise_r = per_cell_r(psth_live, psth_yest)
    median_r = float(np.nanmedian(cellwise_r))
    print(f'      wav-path vs published y_est: median per-cell r={median_r:.4f}, '
          f'overall r={pearson_r(psth_live, psth_yest):.4f}', flush=True)
    assert median_r > MIN_LIVE_VS_YEST_R, (
        f'wav-path prediction only reaches median per-cell r={median_r:.3f} against the '
        f'published y_est panel; the front-end configuration is still wrong')

    gtg_live = acnet_gtg(configure_frontend(model, site_overall_db, float(best_fas),
                                            best_nems_match, compress='sqrt'),
                         waveform, fs_wav)
    configure_frontend(model, site_overall_db, float(best_fas), best_nems_match, compress='log10x')

    # ---------------------------- assemble --------------------------------- #
    data_out = {
        'demo': {
            'wav_int16': wav_int16, 'wav_fs': int(fs_wav), 'wav_int16_scale': 32768.0,
            'wav_name': wav_name, 'wav_src_path': SOUND_ROOT + wav_name,
            'gtg_nems_sqrt': gtg_nems.astype(np.float32),
            'psth_true': psth_true.astype(np.float32),
            'psth_pred_acnet': psth_live.astype(np.float32),
            'psth_pred_yest': psth_yest.astype(np.float32),
            'cell_names': demo_cell_names,
            'acnet_cols_plot': acnet_cols_plot,
            'valid_cells': valid_cells, 'leaf_order': leaf_order,
            't_meanrate': np.arange(n_bins) / FS_GTG - PRESTIM_S,
            'prestim_s': PRESTIM_S,
            'fticks': [0, gtg_nems.shape[0] // 2, gtg_nems.shape[0]],
            'fticklabs': [0.2, 2, 20],
            'tticks': [0, 6, 12, 18],
            'gtg_live_vs_nems_r': best_fas_r,
            'gtg_live_vs_nems_relerr': best_relerr,
            'pred_acnet_vs_yest_median_r': median_r,
            'pred_acnet_vs_yest_cellwise_r': cellwise_r,
            'wav_scan_r': wav_r,
        },
        'rtest': {
            'ln_matched_resnet': ln_matched_resnet, 'ln': rtest_ln,
            'ln_site_mu_resnet': ln_mu_resnet, 'ln_site_mu': ln_mu,
            'ln_common_names': common_ln,
            'cnn_matched_resnet': cnn_matched_resnet, 'cnn': rtest_cnn,
            'cnn_site_mu_resnet': cnn_mu_resnet, 'cnn_site_mu': cnn_mu,
            'cnn_common_names': common_cnn,
            'wilcoxon': wilcoxon_out, 'skip_n': 7,
        },
        'cvpca': cvpca,
        'coverage': {
            'fitstim_site': stim_stats['fitstim_site'],
            'fitstim_index_new': stim_stats['fitstim_index_new'],
            'animal_name': stim_stats['animal_name'],
            'uniq_sites': np.unique(stim_stats['fitstim_site']),
            'stim_dur_s': TRIAL_DURATION_S,
        },
        'meta': {
            'built': datetime.now().isoformat(timespec='seconds'),
            'built_by': os.path.basename(__file__),
            'host': os.uname().nodename,
            'sources': meta_sources,
            'model_dir': MDL_DIR,
            'demo': {'sitename': DEMO_SITENAME, 'stim_type': DEMO_STIM_TYPE,
                     'demo_idx': DEMO_IDX, 'n_val_stims': n_val_stims},
            'frontend': frontend,
            'fas_search': {str(k): v for k, v in fas_scores.items()},
            'cvpca_variant': f'{CVPCA_MODEL_TYPE}{CVPCA_POSTFIX}{CVPCA_COMP_POSTFIX}',
            'provenance': provenance,
            'acnet_version': str(ckpt.get('version')),
            'versions': {'numpy': np.__version__, 'scipy': scipy.__version__,
                         'torch': torch.__version__, 'python': sys.version.split()[0]},
            'git_sha_acnet': git_sha(ACNET_ROOT), 'git_sha_pytorch_models': git_sha(PT_ROOT),
        },
    }

    pickle.dump(data_out, open(OUT_FNAME, 'wb'), protocol=4)
    size_mb = os.path.getsize(OUT_FNAME) / 1e6
    print(f'\nwrote {OUT_FNAME} ({size_mb:.2f} MB)', flush=True)
    assert size_mb < 10, f'fig1.pkl grew to {size_mb:.1f} MB; something large slipped in'


if __name__ == '__main__':
    log_fh = open(LOG_FNAME, 'w', buffering=1)
    sys.stdout = sys.stderr = log_fh
    try:
        main()
    finally:
        log_fh.flush()
