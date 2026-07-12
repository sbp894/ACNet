# ACNet

**ACNet** is a multi-task 1D-ResNet model of ferret (*Mustela putorius furo*)
auditory cortex. It maps a sound waveform to (a) a shared auditory "manifold"
embedding and (b) predicted peri-stimulus time histograms (PSTHs) for 3124
neurons recorded across 62 recording sites.

```
input waveform
  → gammatonegram (32 ERB channels, 100 Hz)
  → shared ResNet backbone (6 residual blocks, widths 75→200)  ── manifold embeddings
  → linear + double-exponential readout                         ── per-neuron PSTHs
```

## Install

```bash
conda create -n acnet python=3.11
conda activate acnet
pip install -r requirements.txt
```

That's all you need — `requirements.txt` pins loose versions and pip resolves a
working `torch`/`torchaudio` build for your platform.

ACNet probes the GPU at load time and automatically falls back to CPU if the installed
wheel has no kernels for it (`best_device()`)

**Optional — control CPU vs GPU.** If you want a specific build, install `torch`
and `torchaudio` from the matching [PyTorch index](https://pytorch.org) *before*
(or instead of) the requirements line:

```bash
# CPU-only (smallest, works everywhere):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Modern GPU (Ampere/Ada/Hopper/Blackwell), e.g. CUDA 12.6:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# Legacy GPU (Pascal, e.g. GTX 10xx): needs an older CUDA-11.8 build:
pip install "torch<2.8" torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick start

```python
from acnet_model import load_acnet

model, cell_rtest = load_acnet()          # cell_rtest: per-neuron test r (3124,)
model.update_audio_process({'overall_db': 60})   # set stimulus level in dB SPL

# Default: auditory embeddings
emb, gtg = model.get_mf_embeddings('examples/example_esc50_clip.wav')
#   emb -> (batch, time, 200)   manifold embeddings
#   gtg -> (time, 32)           gammatonegram

# Optional: predict neural PSTHs
psth = model.predict_psth('examples/example_esc50_clip.wav')
#   psth -> (time, 3124)        one column per recorded neuron
```

You can also pass a waveform directly: `model.get_mf_embeddings(waveform, fs=fs)`
where `waveform` is a `(1, n_samples)` tensor.

## Demo

```bash
python demo_acnet_embeddings.py
```

Edit the settings block at the top of the script to change the input
(`WAV_FILE`), the stimulus level (`DBSPL`), or whether to also predict neural
PSTHs (`PREDICT_PSTH`). The figure is saved next to the input wav with the same
base name (e.g. `examples/example_esc50_clip.png`, 300 dpi): gammatonegram, ACNet
embeddings, and (optionally) predicted neural PSTHs sorted by test-set prediction
quality.

## Files

| File | Purpose |
|------|---------|
| `acnet_model.py` | Model classes, audio front end, and `load_acnet()`. |
| `gtgram.py`, `gtg_filters.py` | Self-contained gammatonegram front end. |
| `demo_acnet_embeddings.py` | Runnable demo (figure above). |
| `weights/acnet_v1.pt` | Trained weights + config + per-neuron `cell_rtest`. |
| `examples/example_esc50_clip.wav` | Short example clip (ESC-50, CC BY-NC). |

## Notes

- **Embedding/PSTH time resolution** PSTH units are spikes/bin at 100 Hz (10 ms bins), matching the training
  targets.
- **Version.** This is **ACNet v1.0.0**. The version is stored in the checkpoint
  and exposed as `model.version` (and `model.model_name`) after `load_acnet()`.
  
## License

Copyright (c) 2026 Satyabrata Parida and Stephen V. David.

Code and weights are released under the **GNU General Public License v3.0**
(see `LICENSE`). The bundled example clip is from the
[ESC-50 dataset](https://github.com/karolpiczak/ESC-50) (CC BY-NC 3.0) and is
included for demonstration only.
