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
├── data/fig1.pkl               # 4.3 MB: every number Fig1 plots + the demo waveform
├── figures/fig1.png            # output
└── build/
    ├── build_fig1_cache.py     # one-time cache builder -- LBHB paths, not needed to plot
    └── build_fig1_cache.log    # its output, kept as provenance
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

See `WIKI.md` for provenance and the verification record.
