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

## Related

- `../README.md` — the ACNet model itself and its public API.
- `pytorch_models/MS_AcxManifold/Fig1_stim_resp_rtest.py` — the original figure
  script this was ported from; still the source of truth for panel content.
