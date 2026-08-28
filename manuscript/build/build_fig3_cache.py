"""
Build `manuscript/data/fig3.pkl.gz` -- the cache behind Fig3, FigS1 and FigS2.

LBHB-only, run once. It reads the ESC-50 classifier outputs, the across-layer
classifier outputs, the UMAP grid-search result and the 213 MB embedding dump that
`MS_AcxManifold/Fig_ESC50.py` needs, and writes one small gzipped pkl containing
every number those three figures plot -- plus a handful of 500 ms ESC-50 waveforms
and their reference manifold embeddings, so the figure scripts can run ACNet live
and check the result against the published one.

    python build/build_fig3_cache.py

The three source families:

  MFembed_gtg_2L_MLP_normIn1_60db__UR1rate_embeddings.pkl   Manifold / Shuffled / Stimulus
  neural_rate_MLP_classifier_REI084_087.pkl                  Neural
  Across_Layers/xLayers_..._layer{0..5}.pkl                  the layer profile

Nothing here is needed to draw a figure; it is kept as provenance.
"""

import gc
import gzip
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                     # ACNet_v1/manuscript
ACNET_ROOT = os.path.dirname(REPO)               # ACNet_v1
sys.path.insert(0, REPO)
sys.path.insert(0, ACNET_ROOT)

LOG_PATH = os.path.join(HERE, 'build_fig3_cache.log')
_log = open(LOG_PATH, 'w', buffering=1)
sys.stdout = _log
sys.stderr = _log

import torch                                     # noqa: E402
import torchaudio                                # noqa: E402

import esc50_lib as el                           # noqa: E402
import acnet_model as sa                         # noqa: E402

# --------------------------------------------------------------------------- #
# paths (LBHB)
# --------------------------------------------------------------------------- #
PT_ROOT = '/auto/users/satya/code/projects_getting_started/pytorch_models/'
ESC_ROOT = '/auto/users/satya/code/projects_getting_started/downloaded/dataset/ESC-50-master/'

MODEL_DIR = PT_ROOT + 'trained_MT_models/ResNetMT_site_nemsGT_3stage_log10xComp/MT_svd_init_6L_v2/nsites62_15/'
ESC50_DIR = MODEL_DIR + 'stage3_fine/ESC50/'
CLF_PKL = ESC50_DIR + 'MFembed_gtg_2L_MLP_normIn1_60db__UR1rate_embeddings.pkl'
EMBED_PKL = ESC50_DIR + 'ESC50_500ms_60db_RNembed_UR1rate.pkl'
XLAYERS_DIR = ESC50_DIR + 'Across_Layers/'
UMAP_PKL = ESC50_DIR + 'UMAP/data/umap_nn50_md0.50.pkl'
NEURAL_PKL = ESC_ROOT + 'misc_out/classifier_output/neural_rate_MLP_classifier_REI084_087.pkl'
WAV_DIR = ESC_ROOT + 'misc_out/ESC50_500ms_v0/'
META_CSV = ESC_ROOT + 'meta/esc50.csv'

OUT_PKL = os.path.join(REPO, 'data', 'fig3.pkl.gz')

# The embedding dump records dbspl=60 and the model's own compression, but not the
# rest of the front-end configuration; the generator script's defaults are two
# releases old. Rather than assume, run the demo clips through every plausible
# combination and keep the one that reproduces the stored embeddings.
# `lbhb_mode=False` forces `level_mode='exact'` (the front end asserts it), so the
# grid is three points, not four.
FRONT_END_GRID = [
    {'lbhb_mode': False, 'level_mode': 'exact'},
    {'lbhb_mode': True, 'level_mode': 'exact'},
    {'lbhb_mode': True, 'level_mode': 'approx'},
]
DBSPL = 60.0
N_DEMO_CLIPS = 6
DEMO_SEED = 0
# Same origin as Fig2's readout tolerance: one GEMM ordering vs another in float32.
FRONT_END_TOL = 1e-4


def _stamp(path):
    st = os.stat(path)
    return {'path': path, 'bytes': int(st.st_size),
            'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}


def _to_numpy_state_dict(sd):
    """Classifier state dicts were saved as CUDA tensors; the cache must be portable."""
    return {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v))
            for k, v in sd.items()}


def banner(text):
    print('\n' + '=' * 74)
    print(text)
    print('=' * 74)


# --------------------------------------------------------------------------- #
def main():
    print(f"build_fig3_cache -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"host={os.uname().nodename}  python={sys.version.split()[0]}  "
          f"numpy={np.__version__}  torch={torch.__version__}")

    cache = {}
    meta = {'sources': {}, 'model_dir': MODEL_DIR, 'dbspl': DBSPL}

    # ---------------------------------------------------------------- 1. preds
    banner('1. classifier outputs')
    clf = pickle.load(open(CLF_PKL, 'rb'))
    neural = pickle.load(open(NEURAL_PKL, 'rb'))
    meta['sources']['classifier'] = _stamp(CLF_PKL)
    meta['sources']['neural'] = _stamp(NEURAL_PKL)
    meta['neural_site'] = neural['site']

    src = {
        'Neural': (neural['accuracy_data'], 'r_test_MF', 'pred_test_MF', 'true_test_MF'),
        'Manifold': (clf['accuracy_data_acmf'], 'r_test', 'pred_test', 'true_test'),
        'Shuffled': (clf['accuracy_data_shuf'], 'r_test', 'pred_test', 'true_test'),
        'Stimulus': (clf['accuracy_data_gtg'], 'r_test', 'pred_test', 'true_test'),
    }

    preds, confmats, classifiers = {}, {}, {}
    for name, (ad, k_r, k_p, k_t) in src.items():
        pred = np.stack([np.asarray(p, dtype=np.int16) for p in ad[k_p]])
        true = np.stack([np.asarray(t, dtype=np.int16) for t in ad[k_t]])
        r_test = np.asarray(ad[k_r], dtype=np.float64)
        assert pred.shape == true.shape, f'{name}: pred/true shape mismatch'
        assert pred.shape[0] == r_test.size, f'{name}: fold count mismatch'
        preds[name] = {'pred_test': pred, 'true_test': true, 'r_test': r_test}
        confmats[name] = el.confusion_counts(pred.ravel(), true.ravel())
        print(f"  {name:9s} folds={pred.shape[0]} n_test={pred.shape[1]} "
              f"r_test={r_test.mean():.4f} +/- {r_test.std():.4f} "
              f"confmat sum={confmats[name].sum()}")
        if 'model_classifier' in ad:
            classifiers[name] = [_to_numpy_state_dict(sd) for sd in ad['model_classifier']]

    # The alignment analysis pairs each model fold against the SAME neural fold, so the
    # two classifiers must have seen the same 5-fold split in the same order. Measured:
    # they do, except that fold 4 of the neural run has a handful of samples in a
    # different order (same multiset, so it is an ordering difference in how the last
    # fold's index list was built, not a different split). Those positions compare a
    # model prediction on one clip against a neural prediction on another, so the count
    # is recorded here and printed by the figure rather than being waved away.
    mismatch = {}
    for name in ['Manifold', 'Shuffled', 'Stimulus']:
        a, b = preds[name]['true_test'], preds['Neural']['true_test']
        assert np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1)), (
            f"{name}: fold-wise label multisets differ from Neural -- these are "
            f"different splits, and the alignment numbers would be meaningless")
        mismatch[name] = [int((a[k] != b[k]).sum()) for k in range(a.shape[0])]
    n_bad = max(sum(v) for v in mismatch.values())
    n_tot = preds['Neural']['true_test'].size
    assert n_bad <= 0.01 * n_tot, (
        f"{n_bad}/{n_tot} samples sit at a different position in the model and neural "
        f"folds -- too many for the positional alignment analysis to mean anything")
    meta['fold_label_mismatch'] = mismatch
    meta['fold_label_mismatch_total'] = n_bad
    meta['fold_label_n_total'] = int(n_tot)
    print(f"  fold alignment: same 5-fold split; {n_bad}/{n_tot} samples "
          f"({100 * n_bad / n_tot:.2f} %) sit at a different within-fold position")
    for name, counts in mismatch.items():
        print(f"    {name:9s} per-fold mismatches: {counts}")

    cache['preds'] = preds
    cache['confmats'] = confmats
    cache['classifiers'] = classifiers
    cache['classifier_optim_params'] = clf['classifier_optim_params']
    cache['model_names'] = el.MODEL_NAMES

    del clf, neural
    gc.collect()

    # -------------------------------------------------------------- 2. xlayers
    banner('2. across-layer classifiers')
    layer_files = sorted(f for f in os.listdir(XLAYERS_DIR) if f.startswith('xLayers_'))
    r_data, r_shuf = [], []
    for fname in layer_files:
        d = pickle.load(open(XLAYERS_DIR + fname, 'rb'))
        r_data.append(np.asarray(d['accuracy_data_acmf']['r_test'], dtype=np.float64))
        r_shuf.append(np.asarray(d['accuracy_data_shuf']['r_test'], dtype=np.float64))
    cache['xlayers'] = {'files': layer_files,
                        'r_test_data': np.stack(r_data),
                        'r_test_shuf': np.stack(r_shuf)}
    meta['sources']['xlayers'] = [_stamp(XLAYERS_DIR + f) for f in layer_files]
    print(f"  {len(layer_files)} layers x {r_data[0].size} folds")
    for i, (a, b) in enumerate(zip(r_data, r_shuf)):
        print(f"    layer {i}: data {a.mean():.4f}  shuffled {b.mean():.4f}")

    # ----------------------------------------------------------------- 3. umap
    banner('3. UMAP embedding')
    umap = pickle.load(open(UMAP_PKL, 'rb'))
    cache['umap'] = {'X2d': np.asarray(umap['X2d'], dtype=np.float32),
                     'y_labels': np.asarray(umap['y_labels']),
                     'selection': el.V4_SELECTION,
                     'nn': umap['nn'], 'md': umap['md'], 'score': float(umap['score'])}
    meta['sources']['umap'] = _stamp(UMAP_PKL)
    missing = [c for c in el.V4_SELECTION if c not in set(cache['umap']['y_labels'])]
    assert not missing, f"categories absent from the UMAP labels: {missing}"
    print(f"  X2d {cache['umap']['X2d'].shape}  nn={umap['nn']} md={umap['md']:.2f} "
          f"score={umap['score']:.4f}")

    # ------------------------------------------------------- 4. category names
    banner('4. ESC-50 category names')
    names = None
    if os.path.isfile(META_CSV):
        import csv
        rows = list(csv.DictReader(open(META_CSV)))
        mapping = {int(r['target']): r['category'] for r in rows}
        assert set(mapping) == set(range(el.N_CATEGORIES)), 'target ids are not 0..49'
        names = np.array([mapping[i] for i in range(el.N_CATEGORIES)])
        meta['sources']['meta_csv'] = _stamp(META_CSV)
        print(f"  0..4: {names[:5].tolist()}")
    else:
        print(f"  {META_CSV} absent -- confusion matrices stay numeric")
    cache['category_names'] = names

    # ------------------------------------------------- 5. demo clips + ACNet
    banner('5. demo clips (waveform + reference manifold)')
    embed = pickle.load(open(EMBED_PKL, 'rb'))
    meta['sources']['embeddings'] = _stamp(EMBED_PKL)
    filenames = list(embed['filenames'])
    n_clips = len(filenames)

    # ESC-50 500 ms clip names are `{fold}_{clip}_{take}_{target}_seg{k}.wav`.
    targets = np.array([int(f.split('_')[3]) for f in filenames])
    assert targets.min() == 0 and targets.max() == el.N_CATEGORIES - 1

    # One clip from each of the first N_DEMO_CLIPS categories in the best-10 selection,
    # so the demo panel shows a sound the reader can name. Deterministic.
    rng = np.random.default_rng(DEMO_SEED)
    demo_idx = []
    if names is not None:
        name_to_target = {n: i for i, n in enumerate(names)}
        wanted = [name_to_target[c] for c in el.V4_SELECTION[:N_DEMO_CLIPS]]
    else:
        wanted = list(range(N_DEMO_CLIPS))
    for tgt in wanted:
        pool = np.flatnonzero(targets == tgt)
        demo_idx.append(int(rng.choice(pool)))
    demo_idx = np.array(demo_idx)

    demo_wavs, demo_names, demo_targets = [], [], []
    mf_ref, gtg_ref = [], []
    wav_fs = None
    for idx in demo_idx:
        fname = filenames[idx]
        wav, fs = torchaudio.load(WAV_DIR + fname)
        wav_fs = int(fs) if wav_fs is None else wav_fs
        assert int(fs) == wav_fs, 'demo clips differ in sample rate'
        demo_wavs.append(wav.numpy().astype(np.float32))
        demo_names.append(fname)
        demo_targets.append(int(targets[idx]))
        mf_ref.append(np.asarray(embed['embeddings'][idx], dtype=np.float32))
        gtg_ref.append(np.asarray(embed['gtg'][idx], dtype=np.float32))
        print(f"  {fname}  target={targets[idx]}"
              f"{'' if names is None else ' (' + names[targets[idx]] + ')'}  "
              f"wav {tuple(wav.shape)} @ {wav_fs} Hz  mf {mf_ref[-1].shape}")

    demo_wavs = np.stack(demo_wavs)
    mf_ref = np.stack(mf_ref)
    gtg_ref = np.stack(gtg_ref)

    # The manifold used by every classifier is the time-averaged backbone output; store
    # the full 2000-clip version too, so the figure can show where the demo clips sit.
    mf_all = np.stack([np.asarray(e, dtype=np.float32).mean(axis=0)
                       for e in embed['embeddings']])
    print(f"  time-averaged manifold over all clips: {mf_all.shape}")

    model_params = embed.get('model_params', {})
    data_params = embed.get('data_params', {})
    compress = data_params.get('compress', 'log10x')
    meta['embed_dbspl'] = int(embed['dbspl'])
    meta['embed_mdl_fname'] = embed.get('mdl_fname')
    meta['compress'] = compress
    del embed
    gc.collect()

    # --------------------------------------------- 6. resolve the front end
    banner('6. front-end configuration (measured, not assumed)')
    device = sa.best_device()
    model, cell_rtest = sa.load_acnet()
    model.to(device).eval()
    print(f"  ACNet v{model.version} on {device}; compress from the dump = {compress!r}")

    results = []
    for cfg in FRONT_END_GRID:
        params = dict(cfg)
        params.update({'overall_db': DBSPL, 'compress': compress,
                       'keep_pre_s': 0.0, 'stim_dur_after_onset': None,
                       'stim_duration_s': None, 'nems_match': False})
        model.update_audio_process(params)
        errs_mf, errs_gtg = [], []
        with torch.no_grad():
            for w, ref_mf, ref_gtg in zip(demo_wavs, mf_ref, gtg_ref):
                mf, gtg = model.get_mf_embeddings(torch.from_numpy(w), fs=wav_fs)
                mf = mf.squeeze(0).cpu().numpy()
                gtg = gtg.cpu().numpy()
                assert mf.shape == ref_mf.shape, f"manifold shape {mf.shape} != {ref_mf.shape}"
                scale = max(float(np.abs(ref_mf).max()), 1.0)
                errs_mf.append(float(np.abs(mf - ref_mf).max()) / scale)
                errs_gtg.append(float(np.abs(gtg - ref_gtg).max())
                                / max(float(np.abs(ref_gtg).max()), 1.0))
        results.append((cfg, float(np.max(errs_mf)), float(np.max(errs_gtg))))
        print(f"  {cfg}  max rel |d manifold| = {results[-1][1]:.3e}  "
              f"max rel |d gtg| = {results[-1][2]:.3e}")

    results.sort(key=lambda r: r[1])
    best_cfg, best_err, best_gtg_err = results[0]
    runner_err = results[1][1]
    assert best_err < FRONT_END_TOL, (
        f"no front-end configuration reproduces the stored embeddings "
        f"(best {best_err:.3e} with {best_cfg}). The published embeddings were made "
        f"with a front end this release cannot express; do not cache a mismatch.")
    assert runner_err > 10 * best_err, (
        f"front-end configuration is not identifiable: best {best_err:.3e} ({best_cfg}) "
        f"vs runner-up {runner_err:.3e} ({results[1][0]})")
    print(f"\n  -> {best_cfg}  (runner-up is {runner_err / best_err:.0f}x worse)")

    front_end = dict(best_cfg)
    front_end.update({'overall_db': DBSPL, 'compress': compress, 'keep_pre_s': 0.0,
                      'stim_dur_after_onset': None, 'stim_duration_s': None,
                      'nems_match': False})

    cache['demo'] = {
        'wav': demo_wavs, 'wav_fs': wav_fs, 'wav_names': demo_names,
        'targets': np.asarray(demo_targets, dtype=np.int16),
        'mf_ref': mf_ref, 'gtg_ref': gtg_ref,
        'front_end': front_end,
        'ref_max_rel_err_mf': best_err, 'ref_max_rel_err_gtg': best_gtg_err,
        'source_wav_dir': WAV_DIR,
    }
    cache['manifold_all'] = {'mf_mean': mf_all, 'filenames': filenames,
                             'targets': targets.astype(np.int16),
                             'demo_idx': demo_idx.astype(np.int32)}
    cache['model_params'] = model_params
    cache['data_params'] = {k: v for k, v in data_params.items()
                            if k in ('fs_gtg', 'GTGmode', 'compress', 'nCF', 'f_min', 'f_max')}

    # ---------------------------------------------------------------- 7. write
    banner('7. write')
    try:
        sha = subprocess.check_output(['git', '-C', ACNET_ROOT, 'rev-parse', 'HEAD'],
                                      text=True).strip()
    except Exception as exc:                                   # noqa: BLE001
        sha = f'unavailable ({exc})'
    meta.update({'built': time.strftime('%Y-%m-%d %H:%M:%S'),
                 'git_sha': sha, 'acnet_version': model.version,
                 'numpy': np.__version__, 'torch': torch.__version__,
                 'source_script': 'MS_AcxManifold/Fig_ESC50.py'})
    cache['meta'] = meta

    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with gzip.open(OUT_PKL, 'wb', compresslevel=6) as fh:
        pickle.dump(cache, fh, protocol=4)
    print(f"  wrote {OUT_PKL}  ({os.path.getsize(OUT_PKL) / 1e6:.2f} MB)")

    banner('done')


if __name__ == '__main__':
    main()
