# Manuscript figures

Reproduction package for the ACNet manuscript figures. Each figure is one script
plus one cached `.pkl`, and the only thing computed at figure time is the part
that actually exercises the model:

> **load ACNet → predict every neuron's PSTH from a stored waveform.**

Everything else — measured PSTHs, baseline model accuracies, cvPCA spectra,
recording coverage — is pre-extracted, so no LBHB data mount is needed.

## Run

```bash
cd manuscript
python fig1.py            # -> figures/fig1.png (300 dpi)
python fig2.py            # -> figures/fig2.png (300 dpi)
```

Requirements: `numpy`, `scipy`, `matplotlib`, plus `torch`/`torchaudio` for the
live prediction (same environment as the rest of the repo — see
`../requirements.txt`). Set `USE_CACHED_PREDICTION = True` at the top of
`fig1.py` to plot the cached prediction instead and skip torch entirely.

`fig1.py` verifies itself as it runs. It rebuilds the gammatonegram and the
predicted PSTHs from the stored waveform and asserts they match the cached
references; if ACNet, the front-end configuration, or the stimulus ever drifts
the script aborts rather than quietly drawing a different figure. Expected
output:

```
live vs cached: gtg r=1.00000000  psth r=1.00000000  max|dPSTH|=1.14e-07
```

## Layout

```
manuscript/
├── fig1.py                     # the figure script
├── fig2.py                     # the figure script
├── rsa_lib.py                  # Fig2 analysis primitives, shared by fig2.py and its builder
├── data/
│   ├── fig1.pkl                # 4.3 MB: every number Fig1 plots + the demo waveform
│   └── fig2.pkl.gz             # 38 MB: four encoders, the recorded PSTHs, the stimulus
├── figures/                    # output PNGs
└── build/
    ├── build_fig<N>_cache.py   # one-time cache builders -- LBHB paths, not needed to plot
    └── build_fig<N>_cache.log  # their output, kept as provenance
```

`build/` is the only part that touches LBHB storage. It ran once; you do not need
it to reproduce a figure. Every path in `fig1.py` is anchored on `__file__`, so
the folder works from any working directory.

## What Fig1 shows

| panel | content |
|---|---|
| schematic | ACNet architecture (placeholder) |
| coverage | natural-sound exposure per recording site, shaded by animal |
| r_test vs LN / vs single-site CNN | per-cell (grey) and per-site (black) test correlation, ACNet on y |
| cumulative variance | cvPCA cumulative variance explained, data vs three models |
| power-law row | cvPCA variance spectra with fitted exponents α |
| demo column | one held-out 18 s stimulus: gammatonegram, 65 measured PSTHs, 65 ACNet-predicted PSTHs (site PRN015a) |

Statistical claims printed by `fig1.py`, recomputed from the cached vectors and
checked against the values frozen at build time:

```
ACNet vs LN:              W=3989033.0, p=0.000e+00,  n=2902, median diff=0.0924
ACNet vs Single-site CNN: W=3098060.0, p=2.537e-107, n=2902, median diff=0.0219
```

## What Fig2 shows

Four per-animal encoders (CLT, LMD, PRN, REI) are run on the same held-out stimulus, and
their representations are compared **across animals** by RSA over timepoints.

| panel | content |
|---|---|
| RDM rows (LMD, REI) | timepoint x timepoint dissimilarity of the manifold, the predicted PSTHs and the recorded PSTHs, seriated by the manifold RDM |
| stim RDM | the same for the stimulus gammatonegram itself |
| bar | across-animal RSA: MF (manifold), predR (predicted PSTH), trueR (recorded PSTH), MF.GTG (manifold vs stimulus), with the stimulus-bootstrap 95% CI and the recorded-PSTH noise ceiling |

`fig2.py` rebuilds each encoder from its cached weights, runs it, and asserts the result
reproduces the cached bootstrap reference statistic before plotting anything. Expected
output:

```
live vs cached (bootstrap reference, 1s blocks): ref_mf=2.35e-07  ref_pred=8.18e-07  ref_true=2.91e-09  ref_stim=4.94e-08
```

The claim the figure makes, recomputed and printed on every run:

```
MF 0.819  >  predR 0.687  >  trueR 0.556  >  MF.GTG 0.219      (n=6 animal pairs, n=4 for MF.GTG)
MF - predR : +0.132  95% CI [+0.099, +0.153]   p<0.002 (stimulus bootstrap, 1 s blocks)
MF - trueR : +0.263  95% CI [+0.213, +0.317]   p<0.002
MF - MF.GTG: +0.600  95% CI [+0.563, +0.633]   p<0.002
recorded-PSTH noise ceiling 0.915; trueR reaches 0.606 of it
```

Set `VERIFY_FULL_RSA = True` in `fig2.py` to re-derive those RSA numbers from
full-resolution RDMs instead of reading them from the cache. It is off by default
because it builds 13 distance matrices over 11400 timepoints -- tens of minutes and a
few GB -- while the bootstrap-reference check above already exercises the live models
end to end.

See `WIKI.md` for provenance and the verification record.
