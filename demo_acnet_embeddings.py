"""
Demo: load ACNet, process a wav file, and plot the gammatonegram alongside the
ACNet manifold embeddings. Optionally also predict and plot the neural PSTHs.

Edit the settings block below and run:  python demo_acnet_embeddings.py
The figure is saved next to the wav file, with the same base name (300 dpi).
"""

import os

# Import torch first. In some MKL/OpenMP setups, importing numpy (or anything
# that pulls it in, like matplotlib) before torch deadlocks `import torch`.
import torch
import numpy as np
from acnet_model import load_acnet, best_device

import matplotlib
# matplotlib.use('Agg')  # uncomment for headless machines (no display); saves only
import matplotlib.pyplot as plt

# ---------------------------- settings ------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))

WAV_FILE = os.path.join(HERE, 'examples', 'example_esc50_clip.wav')  # audio to run
DBSPL = 60.0            # stimulus level in dB SPL (ACNet is level-dependent)
PREDICT_PSTH = True     # False -> embeddings only; True -> also predict neural PSTHs
WEIGHTS_PATH = None     # None -> weights/acnet_v1.pt
# --------------------------------------------------------------------------- #

# Figure is saved next to the wav file, with the same base name.
OUT_PNG = os.path.splitext(WAV_FILE)[0] + '.png'

device = best_device()  # 'cuda' only if a kernel can actually run on this GPU, else 'cpu'

# Load ACNet and set the stimulus level.
model, cell_rtest = load_acnet(WEIGHTS_PATH)   # already placed on best_device()
model.update_audio_process({'overall_db': DBSPL, 'level_mode': 'exact', 'lbhb_mode': False})
model.to(device)
model.eval()

# Run the model.
with torch.no_grad():
    if PREDICT_PSTH:
        psth, manifold_rep, gtg_rep = model.predict_psth(WAV_FILE, return_embeddings=True)
        psth = psth.cpu().numpy()
    else:
        manifold_rep, gtg_rep = model.get_mf_embeddings(WAV_FILE)

manifold_rep = manifold_rep.cpu().numpy().squeeze()  # (time, embedding_dim)
gtg_rep = gtg_rep.cpu().numpy()                       # (time, freq)
cf_kHz = model.audio_process.get_cfs()/1e3

fs_gtg = model.audio_process.fs_gtg
dur_ms = 1e3 * gtg_rep.shape[0] / fs_gtg
print(f"gammatonegram shape={gtg_rep.shape}, manifold shape={manifold_rep.shape} "
      f"[time x dim], fs={fs_gtg} Hz")

# Plot: gammatonegram, embeddings, and (optionally) predicted PSTHs.
n_panels = 3 if PREDICT_PSTH else 2
fig, ax = plt.subplots(n_panels, 1, figsize=(7, 2.2 * n_panels), sharex=True)

ax[0].imshow(gtg_rep.T, origin='lower', aspect='auto', extent=(0, dur_ms, 0, gtg_rep.shape[1]))
# Label the frequency axis at 200 Hz, 2 kHz, 20 kHz using the nearest CF channel.
cf_tick_kHz = np.array([0.2, 2, 20])
cf_tick_pos = [int(np.argmin(np.abs(cf_kHz - t))) + 0.5 for t in cf_tick_kHz]
ax[0].set_yticks(cf_tick_pos)
ax[0].set_yticklabels([f'{t:g}' for t in cf_tick_kHz])
ax[0].set(ylabel='CF (kHz)', title='Gammatonegram')

ax[1].imshow(manifold_rep.T, origin='lower', aspect='auto', extent=(0, dur_ms, 0, manifold_rep.shape[1]))
ax[1].set(ylabel='manifold dimension', title='ACNet embeddings')

if PREDICT_PSTH:
    # Order neurons by prediction quality so the raster reads cleanly.
    order = np.argsort(-cell_rtest) if cell_rtest is not None else np.arange(psth.shape[1])
    ax[2].imshow(psth[:, order].T, origin='lower', aspect='auto', extent=(0, dur_ms, 0, psth.shape[1]))
    ax[2].set(ylabel='neuron (sorted by r_test)', title='Predicted neural PSTHs')

ax[-1].set(xlabel='time (ms)')
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300)
fig.show()
print(f"Saved figure to {OUT_PNG}")
