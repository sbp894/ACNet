# Manuscript figure package — provenance

Last updated: 2026-08-27

**Scope:** how each `data/fig<N>.pkl` was built, what it was checked against, and
what would invalidate it. `README.md` covers how to *run* the figures; this file
covers whether to *trust* them.

## Why this folder exists

`pytorch_models/MS_AcxManifold/Fig1_stim_resp_rtest.py` reads ~1.1 GB of
intermediate pickles to draw one figure:

| source | size | what the figure uses |
|---|---|---|
| `cvPCA_neural_fit_rfloor_log10x.pkl` | 758 MB | two `(2902,)` vectors |
| `estimated_test_PSTH_fit_nfloor1000.pkl` | 160 MB | one site's predicted PSTH + per-site `r_test` |
| `37_BNTgtg_nems_PRN015a.pkl` | 150 MB | one stimulus' gammatonegram + measured PSTHs |
| 3× `cvPCA_*`, 2× `rtest_*`, `fit_stim_stats.pkl` | < 1 MB | all of it |

None of that is portable. `data/fig1.pkl` is 4.3 MB and holds every number the
figure plots, so the figure survives a machine move and can be shared with the
manuscript.

## Fig1

Built by `build/build_fig1_cache.py` on rhino, 2026-08-27. Full log in
`build/build_fig1_cache.log`.

**Model:** `trained_MT_models/ResNetMT_site_nemsGT_3stage_log10xComp/MT_svd_init_6L_v2/nsites62_15/`
**Demo:** site `PRN015a`, held-out ("val") stimulus index 5 of 6, 65 neurons.
**cvPCA variant:** `fit_rfloor_log10x`. **Power-law fit window:** dims 5–400.

### Three things the builder resolves rather than assumes

Each was a silent-failure risk; each is now asserted, with the measured value.

1. **Which wav is `demo_idx = 5`.** The val epochs match `^STIM_00seq` and there
   are six candidates; the index→wav mapping is not recorded anywhere. The
   builder scores all six against the cached NEMS gammatonegram:

   | wav | r |
   |---|---|
   | **00seq6_hand.wav** | **0.999953** |
   | 00seq5_hand.wav | 0.294367 |
   | 00seq1.wav | 0.164536 |
   | 00seq3.wav | 0.113794 |
   | 00seq2.wav | 0.015045 |
   | 00seq4.wav | 0.005427 |

   Unambiguous.

2. **`fixed_amp_scale`.** `save_nemsGTGram_BNT.py:57` stores the literal string
   `'nems_meta'`, not the value, so it cannot be read back from the site pkl.
   It is fitted on **relative RMS error, not on r** — `fixed_amp_scale` only
   scales the signal, and correlation is scale-invariant, so an r-based check
   would accept any value. Best fit: `fixed_amp_scale = 250`, `nems_match = True`
   → relative RMSE 0.00007, r = 1.000000. The runner-up (`nems_match=False`,
   same scale) gives relative RMSE 0.00811; every other grid point is ≥ 0.09,
   so the optimum is sharp and interior to the search grid.

3. **Is ACNet v1 the same run as `nsites62_15`?** If it were a different seed the
   cached reference prediction would describe a different model. Checked by
   name-sorting both cell lists and comparing per-cell `r_test`:
   **names identical, max |Δr_test| = 2.98e-08.** Same run.

### Front-end configuration

The BNT gammatonegrams were built with `keep_pre_s=0.5`,
`stim_dur_after_onset=18.5` (`save_nemsGTGram_BNT.py`) against an 18 s baphy
trial Duration, giving 50 + 1850 = 1900 bins at 100 Hz. `fig1.py` replays:

```python
model.set_bnt_mode(overall_db=65.0, fixed_amp_scale=250.0, nems_match=True)
model.update_audio_process({'keep_pre_s': 0.5, 'stim_dur_after_onset': 18.5,
                            'stim_duration_s': 18.0})
```

`update_audio_process` silently ignores unknown keys, so the builder asserts each
setting actually took effect after configuring.

### Verification record

| check | result |
|---|---|
| live ACNet gammatonegram vs cached NEMS reference | r = 1.00000000 |
| live predicted PSTH vs cached reference | r = 1.00000000, max abs diff 1.14e-07 |
| wav path vs the **published** `y_est` demo panel | median per-cell r = 0.9994 |
| ACNet provenance vs `nsites62_15` | max |Δr_test| = 2.98e-08 |
| Wilcoxon reproduced from cached vectors | exact match to build-time values |
| same-format panels equal in size | asserted in `fig1.py` via `get_window_extent` |

The 0.9994 rather than 1.0 in row 3 is expected and is the one number worth
knowing: the published panel came from `y_est`, computed over the six validation
stimuli **concatenated**, whereas the wav path predicts this stimulus in
isolation. The difference is confined to convolution edge effects at the
stimulus boundary. It is not a discrepancy in the model.

### What would invalidate this cache

- A new ACNet export (`weights/acnet_v1.pt`) from a different training run —
  `fig1.py`'s live-vs-cached assertions catch this.
- Regenerating the BNT gammatonegrams with different `keep_pre_s` /
  `stim_dur_after_onset` — the builder asserts the 1900-bin geometry.
- A scipy upgrade changing `scipy.cluster.hierarchy` tie-breaking, which would
  reorder the demo panel's rows. `leaf_order` is cached, so the shipped figure is
  stable; `meta['versions']['scipy']` records the version that produced it.

## Fig2

Built by `build/build_fig2_cache.py` on rhino, 2026-08-27. Full log in
`build/build_fig2_cache.log`.

**Ported from** `pytorch_models/MS_AcxManifold/Fig_MF_RSA_xanimals.py`.
**Models:** `ResNetMT_animal_nemsGT_3stage_log10xComp/MT_svd_init_6L_v2_<ANIMAL>/`
-- CLT `nsites19_2`, LMD `nsites9_2`, PRN `nsites30_2`, REI `nsites7_3`, all
`stage3_fine`. SLJ (`nsites4_2`) is excluded here as it is in the source script: 4
sites is too few for an animal-level manifold.
**Stimulus:** the 6 held-out BNT stimuli of `02_BNTgtg_nems_CLT029c.pkl`, concatenated
to 11400 bins at 100 Hz.
**Analysis:** PCA to 90% variance (model and data alike), RDM = squared Euclidean
distance between timepoints, RSA = Pearson r between RDM upper triangles.

### What replaces what

| source | size | what the figure uses |
|---|---|---|
| `tests_psth_data_each_rep/` (62 pickles) | **4.5 GB** | the rep-averaged PSTHs of 3528 cells |
| 4x `*_model.pt` + 4x `*_data_pre_fit.pkl` | 13 MB | the four encoders |
| `02_BNTgtg_nems_CLT029c.pkl` | 17 MB | one 11400 x 32 gammatonegram |
| 2x `MF_RSA_xanimals_bootstrap_*.pkl` | 0.8 MB | both bootstrap distributions |
| `PT_EncMdl_helpers_v2.py`, `sppy` | -- | model class, seriation, `dsearchnn` |

`data/fig2.pkl.gz` is **38 MB** (gzip level 6; the PSTHs are float32 and compress ~5x
because each cell's rep-averaged PSTH takes only a few dozen distinct values).

### Things the builder resolves rather than assumes

1. **The per-animal encoders are re-expressed in the released `ACNet` class.** Each was
   trained as `MT_ResNet_v2` with one head per recording site; the builder concatenates
   those heads into the single `Linear -> DEXP` readout the ACNet v1 export uses, so
   `fig2.py` needs no internal model code. The conversion is verified on the real
   stimulus before caching: shared-backbone embeddings are **bit-identical**, predicted
   PSTHs agree to **max 2.8e-07** (one GEMM instead of 19, so float32 accumulates in a
   different order).

2. **The stimulus is fed to the backbone as stored, NOT through `gtg_to_model_input`.**
   `save_nemsGTGram_BNT.py` stores the NEMS `stim` signal (sqrt-amplitude) and
   `MultiTask_BNTDataSet_Site_Nems` applies no further compression before training, so
   the stored signal *is* the domain these encoders were fitted in. Routing it through
   the front end's `compress` would put them in a domain they never saw. Note this is a
   different convention from Fig1's waveform path, which goes through the front end with
   `compress='log10x'`; the `compress` recorded in these models' `data_params` says
   `log10x` but nothing in the nemsGT data path ever applies it.

3. **The recorded PSTHs cover exactly the cells the models were fitted to.** The shipped
   arrays are restricted to the union of the four models' `prefit` cell names (3528 of
   the 3837 cells that survive the >=6-test-stim and >=5-repeat filters), in the source
   script's own file order, so `np.isin` returns the same relative ordering it did
   there. The builder asserts per animal that the recorded and predicted cell counts
   match -- a fitted cell missing from the saved PSTHs would otherwise leave the two
   sides of a comparison over different cells.

### Verification record

| check | result |
|---|---|
| concatenated readout vs per-head original, embeddings | max abs diff **0** |
| concatenated readout vs per-head original, predicted PSTH | max abs diff 2.8e-07 |
| live models vs cached bootstrap reference (recomputed every run) | max 8.2e-07 |
| live PC counts vs cached | exact, asserted per animal per signal |
| pre-existing bootstrap cache vs freshly computed reference | 8.2e-07, reused |
| same-format panels equal in size | asserted in `fig2.py` via `get_window_extent` |

Measured values (identical to the source script):

```
MF 0.819   predR 0.687   trueR 0.556   MF.GTG 0.219
noise ceiling (recorded PSTH, split-half + Spearman-Brown): CLT .954  LMD .906  PRN .939  REI .863
mean pairwise ceiling 0.915; trueR reaches 0.606 of it
nPC at 90% var -- MF 11-30, predicted PSTH 17-31, recorded PSTH 301-589, stimulus 8
```

### What the statistics can and cannot say

With 4 animals there are 6 animal pairs, and each animal sits in 3 of them. The exact
paired sign-flip test bottoms out at p=0.031 and the dependence-respecting dyadic
sign-flip at **p=0.125**, so no animal-level test here can reach p<0.05. A
non-significant animal-level result is n-limited, not evidence of no difference. The
inferential claim comes from the stimulus bootstrap, which resamples the stimulus
material (1 s blocks primary, 19 s whole stimuli as the conservative check) and leaves
the animals fixed. Both block sizes exclude 0 for all three contrasts.

### What would invalidate this cache

- Retraining any per-animal encoder -- `fig2.py`'s bootstrap-reference assertion catches
  it, as does the per-animal PC-count assertion.
- Regenerating the BNT gammatonegrams with different `keep_pre_s` /
  `stim_dur_after_onset`, which changes the 11400-bin timepoint axis.
- A scipy upgrade changing `scipy.cluster.hierarchy` tie-breaking, which would reorder
  the displayed RDMs. Unlike Fig1's `leaf_order`, the Fig2 seriation is recomputed at
  plot time (it depends on the live models), so the shipped PNG is the reference;
  `meta['versions']['scipy']` records the version that produced it.
- Changing `MODEL_VAREXP`, `DATA_VAREXP` or `SIM_MEASURE` in `rsa_lib.py`. These are the
  figure's identity, not free parameters; the cached bootstrap was built under them.

## Fig3 / FigS1 / FigS2 — ESC-50

Source: `pytorch_models/MS_AcxManifold/Fig_ESC50.py`, which draws three figures in one
run (the main summary, an off-diagonal confusion-matrix panel, and a three-row
confusion-structure supplement). Those become `fig3.py`, `figs1.py` and `figs2.py`.

### What replaces what

| the source needed | size | the cache stores |
|---|---|---|
| `ESC50_500ms_60db_RNembed_UR1rate.pkl` | 213 MB | 6 demo waveforms + their reference manifold, and the 2000×200 time-averaged manifold |
| `MFembed_gtg_2L_MLP_..._embeddings.pkl` | 3.1 MB | per-fold predictions, `r_test`, and the 3×5 classifier state dicts |
| `neural_rate_MLP_classifier_REI084_087.pkl` | 33 kB | the same, for the neural classifier |
| `Across_Layers/xLayers_..._layer{0..5}.pkl` | 6 × 66 kB | two 6×5 accuracy matrices |
| `UMAP/data/umap_nn50_md0.50.pkl` | 146 kB | `X2d`, `y_labels`, the v4 selection |
| `sppy`, `seaborn`, `sklearn`, `plot_helpers`, `ESC50_plot_best10` | — | `esc50_lib.py` |

Result: **4.42 MB** for all three figures, one cache, no LBHB mount. The three figures
share `data/fig3.pkl.gz` because they share every input; splitting it would duplicate
the confusion matrices three times.

### Three things resolved rather than assumed

1. **Which front end produced the published ESC-50 embeddings.** The 213 MB dump
   records `dbspl=60` and the model's compression, but not `lbhb_mode` / `level_mode`,
   and the generating script's defaults are two releases old. The builder runs the demo
   clips through every valid combination and keeps the winner:

   | config | max rel \|Δ manifold\| |
   |---|---|
   | `lbhb_mode=False, level_mode='exact'` | **8.530e-06** |
   | `lbhb_mode=True, level_mode='exact'` | 1.823e-01 |
   | `lbhb_mode=True, level_mode='approx'` | 7.425e-01 |

   Unambiguous — the runner-up is 21367× worse. **The published ESC-50 embeddings were
   made with `lbhb_mode=False`**, i.e. without the baphy peak limiter. That contradicts
   the assumption carried in the `acnet_frontend` notes that this generation shared the
   `lbhb_mode=True` bug of the 2026-08 MR/xLayers dumps; it does not.

2. **Whether the neural and model classifiers share a fold split.** They do — but not
   exactly. Folds 0-3 are identical element for element; fold 4 has the same 400 clips
   in a slightly different order, so **4 of 2000 samples (0.20 %)** sit at a different
   within-fold position. The source script pairs the two runs positionally, so those
   four alignment entries compare predictions on two different clips. The builder
   asserts the multisets match, counts the displaced samples, and `fig3.py` prints the
   count on every run. At 0.2 % it changes nothing, but it is now visible instead of
   assumed away.

3. **Confusion-matrix orientation.** The source calls sklearn's
   `confusion_matrix(pred, true)` — arguments in the opposite order to sklearn's
   `(y_true, y_pred)` signature. That makes rows *predicted* and columns *true*, which
   is what its axis labels say. `esc50_lib.confusion_counts(pred, true)` keeps that
   orientation deliberately and says so in its docstring.

### Verification record

| check | result |
|---|---|
| every confusion matrix rebuilt from the cached per-fold predictions | exact, all four |
| live ACNet manifold vs the published embeddings (6 clips) | max rel diff **8.528e-06** |
| live ACNet gammatonegram vs the published one | max rel diff **5.937e-08** |
| front-end configuration identifiable | winner 21367× better than runner-up |
| same-format panels equal in size | asserted via `get_window_extent` in all three scripts |

### Measured values (per category, n=50)

```
accuracy    Neural 0.45   Manifold 0.47   Shuffled 0.20   Stimulus 0.31
            Neural != Manifold  W=506    p=0.208     p_bonf=1          n.s.
            Manifold > Shuffled W=1275   p=8.9e-16   p_bonf=4.4e-15    *
            Manifold > Stimulus W=1190   p=1.2e-09   p_bonf=5.8e-09    *
alignment   all       Manifold 0.44   Shuffled 0.22   Stimulus 0.28
            errors    Manifold 0.22   Shuffled 0.09   Stimulus 0.12
            all four Manifold-vs-control tests p_bonf < 1.7e-08
layers      Manifold [0.308 0.343 0.374 0.426 0.439 0.454]  Spearman r=1.00
            Shuffled [0.231 0.249 0.253 0.244 0.239 0.220]  r=-0.37, p_bonf=0.94  n.s.
confusion   cell-by-cell Pearson r against the neural classifier (FigS2)
            off-diagonal (n=2450)  Manifold 0.56   Shuffled 0.34   Stimulus 0.34
            diagonal     (n=50)    Manifold 0.78   Shuffled 0.45   Stimulus 0.46
```

### What the statistics can and cannot say

The per-category unit (n=50) is the default because the per-fold unit gives n=5, where
the Wilcoxon p-floor is 0.031 and nothing survives Bonferroni. Set `PER_CATEGORY =
False` in `fig3.py` to switch. Categories are not independent of each other in the way
folds are, so the per-category p-values are anti-conservative; they are reported as
evidence of ordering, not as calibrated error rates. The ordering itself
(Manifold ≈ Neural > Stimulus > Shuffled) holds under both units.

`Neural != Manifold` being non-significant is the claim the figure is making, and a
non-significant two-sided test is not evidence of equivalence. What it supports is the
weaker, correct statement: **a classifier reading ACNet's manifold is not measurably
worse than one reading the recorded cortical population.**

### What would invalidate this cache

- Regenerating the ESC-50 embeddings with a different front end or dB SPL —
  `fig3.py`'s live check catches it.
- Retraining any of the four classifiers, which changes every prediction, confusion
  matrix and accuracy in the cache. Nothing in the figure would catch that; the cache's
  `meta['sources']` records each source file's size and mtime so it can be checked by
  hand.
- Re-running the UMAP grid search. `X2d` is cached, not recomputed, so the panel is
  frozen at `nn=50, md=0.50, score=3.5924`.
- Changing `CONFMAT_VLIM` in `fig3.py` (0..40, chosen so the diagonal saturates rather
  than compressing everything else) — that is a display decision, not a free parameter.

## Related

- `../README.md` — the ACNet model itself and its public API.
- `pytorch_models/MS_AcxManifold/Fig1_stim_resp_rtest.py`,
  `Fig_MF_RSA_xanimals.py`, `Fig_ESC50.py` — the original figure scripts these were
  ported from; still the source of truth for panel content.
