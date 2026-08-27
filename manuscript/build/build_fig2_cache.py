"""
One-time cache builder for manuscript Fig2 (cross-animal manifold RSA).

Ported from `pytorch_models/MS_AcxManifold/Fig_MF_RSA_xanimals.py`, which reads ~4.5 GB
of internal pickles (the per-repeat test PSTHs alone are 4.5 GB across 62 sites) plus
four internal model checkpoints and the `sppy` / `PT_EncMdl_helpers_v2` packages. This
script reads them once and writes a single portable `data/fig2.pkl.gz` holding:

  * the four per-animal encoder weights, converted to the released `acnet_model.ACNet`
    class (per-site heads concatenated into one readout, exactly as the ACNet v1 export
    did) -- so `fig2.py` needs no internal model code;
  * the recorded PSTHs (all repeats, and the odd/even halves for the noise ceiling)
    for the cells those four models were fitted to;
  * the validation-stimulus gammatonegram the models are run on;
  * the frozen stimulus bootstrap, and reference values for everything the figure
    computes live, so `fig2.py` can assert it reproduces them.

LBHB-internal: needs /auto/users/satya. Run once, on rhino:

    <ptn python> -u manuscript/build/build_fig2_cache.py
"""

import gc
import gzip
import os
import pickle
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../manuscript
ACNET_REPO = os.path.dirname(REPO)                                          # .../ACNet_v1
PT_ROOT = '/auto/users/satya/code/projects_getting_started/pytorch_models/'
FE_ROOT = '/auto/users/satya/code/projects_getting_started/FrontEnd_NeuroFit/'

sys.path.insert(0, REPO)
sys.path.insert(0, ACNET_REPO)
sys.path.insert(0, PT_ROOT)

OUT_PKL = os.path.join(REPO, 'data', 'fig2.pkl.gz')
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_fig2_cache.log')

# ---- sources ---------------------------------------------------------------
TEST_PSTH_DIR = FE_ROOT + 'misc_output/tests_psth_data_each_rep/'
ANIMAL_ROOT = PT_ROOT + 'ResNetMT_animal_nemsGT_3stage_log10xComp/'
DEMO_BNT_FILE = PT_ROOT + 'misc_output/BNTgtg_nems_fs100Hz_nCF32_sites_sqrt_amp/02_BNTgtg_nems_CLT029c.pkl'
BOOT_CACHE_DIR = PT_ROOT + 'MS_AcxManifold/plot_data/'

ALL_MODELS = [
    ANIMAL_ROOT + 'MT_svd_init_6L_v2_CLT/nsites19_2/stage3_fine/C1D_60_80_100_120_140_160_model.pt',
    ANIMAL_ROOT + 'MT_svd_init_6L_v2_LMD/nsites9_2/stage3_fine/C1D_60_80_100_120_140_160_model.pt',
    ANIMAL_ROOT + 'MT_svd_init_6L_v2_PRN/nsites30_2/stage3_fine/C1D_60_80_100_120_140_160_model.pt',
    ANIMAL_ROOT + 'MT_svd_init_6L_v2_REI/nsites7_3/stage3_fine/C1D_50_60_70_80_90_100_model.pt',
]
# SLJ (nsites4_2) is commented out in the source figure script -- 4 sites was judged too
# few for an animal-level manifold. Kept out here for the same reason.

# How closely a pre-existing bootstrap cache must reproduce the reference statistic
# recomputed from the models being cached. The measured gap is ~8e-7 -- float32 GEMM
# reduction order in the model forward pass, not a different model -- so the threshold
# sits an order of magnitude above that and well below any real drift.
BOOT_REF_TOL = 1e-5


def _open_log():
    log = open(LOG_PATH, 'w', buffering=1)
    sys.stdout = log
    sys.stderr = log
    return log


def src_meta(path):
    st = os.stat(path)
    return {'path': path, 'bytes': int(st.st_size), 'mtime': time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}


# --------------------------------------------------------------------------- #
def load_animal_models(sa, rl, device):
    """Load the four internal per-animal models and convert each to an ACNet checkpoint.

    Verifies on the validation stimulus that the concatenated-readout model reproduces
    the original's per-head outputs before the weights are cached; a silent mismatch
    here would put a different model behind every panel of the figure.
    """
    from PT_EncMdl_helpers_v2 import MT_ResNet_v2

    entries = {}
    group_names = []
    for model_path in ALL_MODELS:
        parts = model_path.split('/')
        animal = parts[-4].split('v2_')[-1]
        group_names.append(animal)
        prefit_path = ('/' + os.path.join(*parts[:-2]) + '/'
                       + parts[-1].split('.')[0].replace('_model', '_data_pre_fit') + '.pkl')
        prefit = pickle.load(open(prefit_path, 'rb'))
        model_params = prefit['model_params']
        data_params = prefit['data_params']
        prefit_cellnames = prefit['cell_names']
        print(f"  {animal}: prefit keys={sorted(prefit.keys())}")
        print(f"      heads={len(prefit_cellnames)} cells={sum(len(c) for c in prefit_cellnames)} "
              f"hidden={model_params['hidden_dim']} nl={model_params['nl_name']} "
              f"norm_1d={model_params['norm_1d']} compress={data_params.get('compress')}")
        assert model_params['nl_name'] == 'dexp', (
            f"{animal}: head nonlinearity is '{model_params['nl_name']}', but the released "
            f"ACNet readout is Linear->DEXP. Conversion would be wrong.")

        orig = MT_ResNet_v2(input_dim=model_params['input_dim'],
                            hidden_dim=model_params['hidden_dim'],
                            head_output_dims=model_params['head_output_dims'],
                            kernel_size=model_params['time_kernel_len'],
                            nl_name=model_params['nl_name'],
                            norm_1d=model_params['norm_1d'],
                            res_scale=model_params['res_scale'],
                            init_mode=model_params['init_mode'])
        ckpt = torch.load(model_path, map_location='cpu')
        orig.load_state_dict(ckpt['model_state_dict'])
        orig.to(device).eval()

        head_output_dims = [h[0].out_features for h in orig.layers_task_heads]
        assert head_output_dims == [len(c) for c in prefit_cellnames], (
            f"{animal}: head sizes {head_output_dims} != prefit cell counts "
            f"{[len(c) for c in prefit_cellnames]}")
        n_neurons = int(sum(head_output_dims))

        config = dict(sa.DEFAULT_CONFIG)
        config.update({
            'input_dim': int(orig.params['input_dim']),
            'hidden_dim': list(orig.params['hidden_dim']),
            'kernel_size': [int(k) for k in orig.params['kernel_size']],
            'n_neurons': n_neurons,
            'norm_1d': bool(orig.params['norm_1d']),
            'res_scale': float(orig.params['res_scale']),
            'skip': True,
            'fs_gtg': orig.audio_process.fs_gtg,
            'num_cfs': orig.audio_process.num_cfs,
            'f_min': orig.audio_process.f_min,
            'f_max': orig.audio_process.f_max,
            'overall_db': orig.audio_process.overall_db,
            'fixed_amp_scale': orig.audio_process.fixed_amp_scale,
            'compress': orig.audio_process.compress,
        })
        standalone = sa.ACNet(config).to(device).eval()
        standalone.layers_shared.load_state_dict(orig.layers_shared.state_dict(), strict=True)
        with torch.no_grad():
            standalone.readout_linear.weight.copy_(
                torch.cat([h[0].weight.data for h in orig.layers_task_heads], dim=0))
            standalone.readout_linear.bias.copy_(
                torch.cat([h[0].bias.data for h in orig.layers_task_heads], dim=0))
            standalone.readout_nl.base.copy_(
                torch.cat([h[1].base.data for h in orig.layers_task_heads], dim=0))
            standalone.readout_nl.amp.copy_(
                torch.cat([h[1].amp.data for h in orig.layers_task_heads], dim=0))
            standalone.readout_nl.kappa.copy_(
                torch.cat([h[1].kappa.data for h in orig.layers_task_heads], dim=0))

        entries[animal] = {
            'config': config,
            'state_dict': {k: v.detach().cpu().numpy() for k, v in standalone.state_dict().items()},
            'cell_names': np.asarray(np.concatenate(prefit_cellnames)),
            'cell_names_per_head': [np.asarray(c) for c in prefit_cellnames],
            'head_output_dims': head_output_dims,
            'source': src_meta(model_path),
            'prefit_source': src_meta(prefit_path),
            'fs_gtg': int(data_params['fs_gtg']),
            'data_params': {k: v for k, v in data_params.items()
                            if isinstance(v, (int, float, str, bool, type(None)))},
        }
        entries[animal]['_orig'] = orig          # dropped before writing; verification only
        entries[animal]['_standalone'] = standalone
    return entries, group_names


def verify_conversion(entries, stim_gtg, device):
    """Concatenated readout must reproduce the per-head originals on the real stimulus."""
    x = torch.as_tensor(stim_gtg[None, ...].astype(np.float32), device=device)
    for animal, e in entries.items():
        orig, sa_model = e['_orig'], e['_standalone']
        with torch.no_grad():
            mf_o = orig.layers_shared(x)
            mf_s = sa_model.layers_shared(x)
            psth_o = torch.cat([h(mf_o).squeeze(0) for h in orig.layers_task_heads], dim=1)
            psth_s = sa_model.readout_nl(sa_model.readout_linear(mf_s)).squeeze(0)
        d_mf = float((mf_o - mf_s).abs().max())
        d_psth = float((psth_o - psth_s).abs().max())
        scale = float(psth_o.abs().max())
        print(f"      {animal}: max|d embeddings|={d_mf:.3e}  max|d psth|={d_psth:.3e} "
              f"({d_psth / scale:.1e} of full scale)  psth shape={tuple(psth_s.shape)}")
        # The backbone is byte-identical (one shared module). The readout is not: the
        # concatenated Linear does a single GEMM where the originals did one per site,
        # so float32 accumulates in a different order. ~1e-7 on outputs of order 1 is
        # that reordering, not a different model -- anything larger is.
        assert d_mf == 0.0, f"{animal}: shared backbone differs after conversion"
        assert d_psth < 1e-5 * max(scale, 1.0), (
            f"{animal}: concatenated readout does not reproduce the per-head model "
            f"(max|d psth|={d_psth:.3e}, full scale {scale:.3f})")


# --------------------------------------------------------------------------- #
def load_recorded_psths(rl):
    """Average the per-repeat test PSTHs, keeping the same sites the source script keeps.

    Returns (psth_all, psth_odd, psth_even, cell_names, cell_snrs) with cells in the
    order the source script produces: sorted file order, cells within a file in the
    file's own order. `fig2.py` re-derives per-animal cell indices with the same
    `np.isin` call, which returns ascending positions, so subsetting the rows to the
    union of the four models' cells preserves every relative order that matters.
    """
    all_files = sorted(os.listdir(TEST_PSTH_DIR))
    psth_all, psth_odd, psth_even = [], [], []
    cell_names, cell_snrs = [], []
    n_skipped = 0
    for fidx, fname in enumerate(all_files):
        saved = pickle.load(open(TEST_PSTH_DIR + fname, 'rb'))
        psth_data_all = saved['Y_test_resp_all']
        if len(psth_data_all) != rl.N_TEST_STIMS:
            print(f"      [{fidx + 1}/{len(all_files)}] skip {fname}: "
                  f"{len(psth_data_all)} != {rl.N_TEST_STIMS} test stims")
            n_skipped += 1
            del saved, psth_data_all
            gc.collect()
            continue
        cat = np.concatenate(psth_data_all, axis=-1)       # (cells, reps, T)
        num_reps = cat.shape[1]
        if num_reps < rl.MIN_NREPS:
            print(f"      [{fidx + 1}/{len(all_files)}] skip {fname}: "
                  f"{num_reps} < {rl.MIN_NREPS} reps")
            n_skipped += 1
            del saved, psth_data_all, cat
            gc.collect()
            continue
        psth_odd.append(np.mean(cat[:, 0::2], axis=1).astype(np.float32))
        psth_even.append(np.mean(cat[:, 1::2], axis=1).astype(np.float32))
        psth_all.append(np.mean(cat, axis=1).astype(np.float32))
        cell_names.append(np.asarray(saved['site_meta']['unit_names']))
        cell_snrs.append(np.mean(saved['site_meta']['test_stim_cell_snr'], axis=0))
        del saved, psth_data_all, cat
        gc.collect()

    print(f"      kept {len(psth_all)} sites, skipped {n_skipped}")
    return (np.concatenate(psth_all, axis=0), np.concatenate(psth_odd, axis=0),
            np.concatenate(psth_even, axis=0), np.concatenate(cell_names),
            np.concatenate(cell_snrs))


# --------------------------------------------------------------------------- #
def main():
    import acnet_model as sa
    import rsa_lib as rl

    device = sa.best_device()
    print(f"device={device}  torch={torch.__version__}  numpy={np.__version__}")
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)

    # ---- [1/8] stimulus ---------------------------------------------------
    print("[1/8] validation-stimulus gammatonegram")
    demo = pickle.load(open(DEMO_BNT_FILE, 'rb'))
    val = np.asarray(demo['X_gtg_Val10Rep_nems'], dtype=np.float32)   # (n_stim, T, nCF)
    n_stim_val, n_t_per_stim, n_cf = val.shape
    stim_gtg = val.reshape(-1, n_cf)
    print(f"      {DEMO_BNT_FILE.split('/')[-1]}  val={val.shape} -> stim_gtg={stim_gtg.shape}")
    del demo
    gc.collect()

    # ---- [2/8] models -----------------------------------------------------
    print("[2/8] per-animal encoders -> concatenated-readout ACNet checkpoints")
    entries, group_names = load_animal_models(sa, rl, device)
    n_models = len(group_names)
    print(f"      groups={group_names}")

    print("[3/8] verifying the concatenated readout against the per-head originals")
    verify_conversion(entries, stim_gtg, device)

    # ---- [4/8] recorded PSTHs --------------------------------------------
    print("[4/8] recorded PSTHs (this reads the 4.5 GB per-repeat archive)")
    psth_all, psth_odd, psth_even, cat_cell_names, cat_cell_snrs = load_recorded_psths(rl)
    print(f"      psth_all={psth_all.shape} cells={len(cat_cell_names)}")
    assert psth_all.shape[0] == len(cat_cell_names)
    assert psth_all.shape[1] == stim_gtg.shape[0], (
        f"recorded PSTH has {psth_all.shape[1]} bins but the stimulus has {stim_gtg.shape[0]}")

    # restrict the shipped arrays to cells the four models were fitted to; the union is
    # taken in file order so np.isin positions stay in the same relative order
    model_cells = np.concatenate([entries[a]['cell_names'] for a in group_names])
    keep = np.where(np.isin(cat_cell_names, model_cells))[0]
    n_expected = sum(len(entries[a]['cell_names']) for a in group_names)
    print(f"      keeping {len(keep)}/{len(cat_cell_names)} cells "
          f"(models span {n_expected} cells across {n_models} animals)")
    psth_all, psth_odd, psth_even = psth_all[keep], psth_odd[keep], psth_even[keep]
    cat_cell_names, cat_cell_snrs = cat_cell_names[keep], cat_cell_snrs[keep]
    gc.collect()

    # ---- [5/8] live model signals + PCA ----------------------------------
    print("[5/8] manifold embeddings, predicted PSTHs, PCA projections")
    mf_pcproj, pred_pcproj, true_pcproj = [], [], []
    mf_cumvar, pred_cumvar, true_cumvar = [], [], []
    mf_nfeat, pred_nfeat, true_nfeat = [], [], []
    mf_ndims, pred_ndims, true_ndims = [], [], []
    n_cells_per_model = []

    for animal in group_names:
        e = entries[animal]
        model = e['_standalone']
        mf, psth_pred = rl.model_signals(model, stim_gtg, device=device)

        # true PSTHs for the same cell set, assembled head by head exactly as the source
        # script does. Cell ORDER differs from the model's readout order (np.isin returns
        # ascending file positions); PCA scores and Euclidean distances over timepoints
        # are invariant to permuting the cell columns, so this does not matter here --
        # but do not rely on the ordering elsewhere.
        true_cols = []
        for head_cells in e['cell_names_per_head']:
            inds = np.where(np.isin(cat_cell_names, head_cells))[0]
            true_cols.append(psth_all[inds, :].T)
        cat_true = np.concatenate(true_cols, axis=1)
        assert cat_true.shape[1] == psth_pred.shape[1], (
            f"{animal}: recorded PSTHs cover {cat_true.shape[1]} cells but the model "
            f"predicts {psth_pred.shape[1]} -- some fitted cells are missing from the "
            f"saved PSTHs")
        n_cells_per_model.append(int(cat_true.shape[1]))

        p, c, nd, nf = rl.pca_project(mf, rl.MODEL_VAREXP)
        mf_pcproj.append(p); mf_cumvar.append(c.astype(np.float32)); mf_ndims.append(nd); mf_nfeat.append(nf)
        p, c, nd, nf = rl.pca_project(psth_pred, rl.MODEL_VAREXP)
        pred_pcproj.append(p); pred_cumvar.append(c.astype(np.float32)); pred_ndims.append(nd); pred_nfeat.append(nf)
        p, c, nd, nf = rl.pca_project(cat_true, rl.DATA_VAREXP)
        true_pcproj.append(p); true_cumvar.append(c.astype(np.float32)); true_ndims.append(nd); true_nfeat.append(nf)
        print(f"      {animal}: cells={n_cells_per_model[-1]:5d}  "
              f"nPC mf={mf_ndims[-1]:4d} pred={pred_ndims[-1]:4d} true={true_ndims[-1]:4d}")
        del mf, psth_pred, cat_true, true_cols
        gc.collect()

    stim_pcproj, stim_cumvar, stim_ndims, stim_nfeat = rl.pca_project(stim_gtg, rl.MODEL_VAREXP)
    print(f"      stim (GTG): nPC={stim_ndims}")

    # ---- [6/8] RSA + noise ceiling ---------------------------------------
    print("[6/8] RDMs, RSA, noise ceiling")
    stim_utri = rl.rdm_utri(stim_pcproj)
    mf_utri = [rl.rdm_utri(p) for p in mf_pcproj]
    mf_rsa = rl.rsa_matrix(mf_utri)
    stim_mf_rsa = np.array([np.corrcoef(mf_utri[m], stim_utri)[0, 1] for m in range(n_models)])
    del mf_utri, stim_utri
    gc.collect()
    pred_utri = [rl.rdm_utri(p) for p in pred_pcproj]
    pred_rsa = rl.rsa_matrix(pred_utri)
    del pred_utri
    gc.collect()
    true_utri = [rl.rdm_utri(p) for p in true_pcproj]
    true_rsa = rl.rsa_matrix(true_utri)
    del true_utri
    gc.collect()
    print(f"      MF     mean={np.nanmean(mf_rsa):.3f}")
    print(f"      predR  mean={np.nanmean(pred_rsa):.3f}")
    print(f"      trueR  mean={np.nanmean(true_rsa):.3f}")
    print(f"      MF.GTG mean={np.mean(stim_mf_rsa):.3f}")

    # split-half reliability of the recorded-PSTH RDM, over the SAME cells the main path
    # uses -- MF and predR are deterministic functions of the stimulus so their ceiling
    # is exactly 1.0; only trueR is attenuated by trial noise.
    ceiling_half = np.full(n_models, np.nan)
    ceiling_full = np.full(n_models, np.nan)
    for m, animal in enumerate(group_names):
        head_cells = entries[animal]['cell_names']
        inds = np.where(np.isin(cat_cell_names, head_cells))[0]
        assert len(inds) == n_cells_per_model[m], (
            f"{animal}: ceiling would use {len(inds)} cells but the main trueR path used "
            f"{n_cells_per_model[m]} -- cell sets are misaligned")
        p_odd, _, _, _ = rl.pca_project(psth_odd[inds, :].T, rl.DATA_VAREXP)
        p_even, _, _, _ = rl.pca_project(psth_even[inds, :].T, rl.DATA_VAREXP)
        r_half = float(np.corrcoef(rl.rdm_utri(p_odd), rl.rdm_utri(p_even))[0, 1])
        ceiling_half[m] = r_half
        ceiling_full[m] = 2 * r_half / (1 + r_half)        # Spearman-Brown
        print(f"      {animal:4s} ({len(inds):5d} cells): r_half={r_half:.3f} -> "
              f"r_full={ceiling_full[m]:.3f}")
        del p_odd, p_even
        gc.collect()

    ceiling_pair = np.full((n_models, n_models), np.nan)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            ceiling_pair[i, j] = np.sqrt(ceiling_full[i] * ceiling_full[j])

    # ---- [7/8] stimulus bootstrap ----------------------------------------
    print("[7/8] stimulus bootstrap")
    fs_gtg = entries[group_names[0]]['fs_gtg']
    boot = {}
    for block_s in rl.BOOT_BLOCK_S_LIST:
        key = (f"mdlVE{rl.MODEL_VAREXP * 100:.0f}_dataVE{rl.DATA_VAREXP * 100:.0f}_{rl.SIM_MEASURE}"
               f"_nb{rl.BOOT_N_DEFAULT}_blk{block_s}_sub{rl.BOOT_SUBSAMP}_seed{rl.BOOT_SEED}")
        cache_path = os.path.join(BOOT_CACHE_DIR, f"MF_RSA_xanimals_bootstrap_{key}.pkl")
        ref = rl.bootstrap_reference(block_s, mf_pcproj, pred_pcproj, true_pcproj,
                                     stim_pcproj, n_stim_val, n_t_per_stim, fs_gtg)
        reused = False
        if os.path.exists(cache_path):
            cached = pickle.load(open(cache_path, 'rb'))
            diffs = [np.nanmax(np.abs(cached[k] - r)) for k, r in
                     zip(['ref_mf', 'ref_pred', 'ref_true', 'ref_stim'], ref)]
            print(f"      [{block_s}s] cache {os.path.basename(cache_path)}: "
                  f"max|d ref| = {max(diffs):.3e}")
            if max(diffs) < BOOT_REF_TOL:
                boot[block_s] = cached
                boot[block_s]['source'] = src_meta(cache_path)
                reused = True
            else:
                print(f"      [{block_s}s] cached reference does not match the models being "
                      f"cached -- recomputing")
        if not reused:
            boot[block_s] = rl.run_stim_bootstrap(
                block_s, rl.BOOT_N_DEFAULT, rl.BOOT_SEED, mf_pcproj, pred_pcproj,
                true_pcproj, stim_pcproj, n_stim_val, n_t_per_stim, fs_gtg)
            boot[block_s]['source'] = {'path': '(recomputed by build_fig2_cache.py)'}
        b = boot[block_s]
        print(f"      [{block_s}s] {b['n_blocks']} blocks, "
              f"{b['n_timepoints_per_resample']} timepoints/resample")

    # ---- [8/8] write ------------------------------------------------------
    print("[8/8] writing cache")
    for a in group_names:
        entries[a].pop('_orig', None)
        entries[a].pop('_standalone', None)

    out = {
        'group_names': group_names,
        'models': {a: entries[a] for a in group_names},
        'stim': {
            'gtg': stim_gtg,
            'n_stim_val': int(n_stim_val),
            'n_t_per_stim': int(n_t_per_stim),
            'n_cf': int(n_cf),
            'fs_gtg': int(fs_gtg),
            'source': src_meta(DEMO_BNT_FILE),
            'compress_note': (
                'NEMS sqrt-amplitude gammatonegram, stored by save_nemsGTGram_BNT.py and '
                'fed to the backbone AS STORED. MultiTask_BNTDataSet_Site_Nems applies no '
                'further compression, so this is the domain the encoders were fitted in.'),
        },
        'resp': {
            'psth_all': psth_all,
            'psth_odd': psth_odd,
            'psth_even': psth_even,
            'cell_names': cat_cell_names,
            'cell_snrs': np.asarray(cat_cell_snrs, dtype=np.float32),
            'min_nreps': rl.MIN_NREPS,
            'n_test_stims': rl.N_TEST_STIMS,
            'source_dir': TEST_PSTH_DIR,
        },
        'reference': {
            'mf_rsa': mf_rsa, 'pred_rsa': pred_rsa, 'true_rsa': true_rsa,
            'stim_mf_rsa': stim_mf_rsa,
            'mf_ndims': mf_ndims, 'pred_ndims': pred_ndims, 'true_ndims': true_ndims,
            'stim_ndims': stim_ndims,
            'mf_nfeat': mf_nfeat, 'pred_nfeat': pred_nfeat, 'true_nfeat': true_nfeat,
            'stim_nfeat': stim_nfeat,
            'mf_cumvar': mf_cumvar, 'pred_cumvar': pred_cumvar, 'true_cumvar': true_cumvar,
            'stim_cumvar': stim_cumvar.astype(np.float32),
            'ceiling_half': ceiling_half, 'ceiling_full': ceiling_full,
            'ceiling_pair': ceiling_pair,
            'n_cells_per_model': n_cells_per_model,
        },
        'boot': boot,
        'meta': {
            'built': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_figure': 'pytorch_models/MS_AcxManifold/Fig_MF_RSA_xanimals.py',
            'model_varexp': rl.MODEL_VAREXP, 'data_varexp': rl.DATA_VAREXP,
            'sim_measure': rl.SIM_MEASURE, 'subsamp_factor': rl.SUBSAMP_FACTOR,
            'boot_subsamp': rl.BOOT_SUBSAMP, 'boot_block_s_list': rl.BOOT_BLOCK_S_LIST,
            'boot_n': rl.BOOT_N_DEFAULT, 'boot_seed': rl.BOOT_SEED,
            'versions': {'numpy': np.__version__, 'torch': torch.__version__,
                         'scipy': __import__('scipy').__version__,
                         'sklearn': __import__('sklearn').__version__,
                         'python': sys.version.split()[0]},
        },
    }

    with gzip.open(OUT_PKL, 'wb', compresslevel=6) as fh:
        pickle.dump(out, fh, protocol=4)
    print(f"\nwrote {OUT_PKL} ({os.path.getsize(OUT_PKL) / 1e6:.2f} MB)")


if __name__ == '__main__':
    _log = _open_log()
    try:
        main()
    finally:
        _log.flush()
