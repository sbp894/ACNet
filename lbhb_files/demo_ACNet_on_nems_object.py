"""
Demo: run ACNet on a NEMS recording object, and verify that ACNet's own audio
front end reproduces the gammatonegram NEMS built for the same BNT stimuli.

Run in the `nems2026` env (it needs nems_lbhb + celldb; torch/torchaudio CPU are
installed there alongside):

    source /auto/users/satya/bin/miniconda3/etc/profile.d/conda.sh
    conda activate nems2026
    python -u lbhb_files/demo_ACNet_on_nems_object.py

What is being matched
---------------------
`nems_lbhb.runclass.NAT_stim` builds a BNT `stim` signal as:

    int16 / 32767
      -> pad to fs*Duration, scipy.signal.resample to fs0 = 2*f_max = 40 kHz
      -> truncate to floor(Duration*fs0), 5 ms symmetric np.hanning on/off ramp
      -> remove_clicks(w * FixedAmpScale, 15)          [BigNat branch]
      -> w /= 10**((80 - OveralldB)/20)
      -> gtgram, then **0.5                            [stim is SQRT-compressed]
      -> PreStimSilence / PostStimSilence zeros concatenated on

Three consequences, each of which this demo checks rather than assumes:

1. The level path is `lbhb_mode=True` + `level_mode='approx'`, NOT the
   `lbhb_mode=False` + 'exact' defaults that ACNet uses for arbitrary wavs.
   `OveralldB` and `FixedAmpScale` are read from exptparams, never hardcoded.
2. NEMS `stim` is sqrt(magnitude); ACNet's model input is log10x(magnitude).
   `model.gtg_to_model_input(x, 'sqrt')` squares first, then compresses.
3. The `-ld-norm.l1` suffix of the loadkey is DISCARDED by
   `baphy_io.parse_loadkey` (it splits on '-' and keeps only the first token),
   so no extra normalization is applied to the stim.

Two remaining differences are pure numerics, not modelling choices: NEMS uses
scipy's FFT resampler and a symmetric Hann ramp in float64, ACNet uses a
torchaudio polyphase resampler and a periodic Hann ramp in float32. ACNet's
choices are the ones its training gammatonegrams were built with, so they stay
the default; `nems_match=True` swaps them in only to demonstrate that the rest
of the algebra is identical. Hence both arms are run below.

Output: a 300 dpi PNG with NEMS vs ACNet gammatonegrams, the manifold
embeddings each produces, and per-channel / per-dimension correlations.
"""

import os

# Import torch before numpy -- in some MKL/OpenMP setups importing numpy (or
# anything that pulls it in) first deadlocks `import torch`.
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nems.tools.utils as nems_utils
from nems_lbhb.baphy_experiment import BAPHYExperiment

from acnet_model import GammatoneFilterbankProcessing, load_acnet, _load_wav

# ---------------------------- settings ------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))

PARMFILE = '/auto/data/daq/Reishi/REI003/REI003a07_p_BNT.m'
N_EPOCHS = 4                # epochs to score; the first one is plotted
PLOT_EPOCH = 0
WEIGHTS_PATH = None         # None -> weights/acnet_v1.pt
OUT_PNG = os.path.join(HERE, 'examples', 'acnet_vs_nems_gtgram.png')

# nems_lbhb points its joblib gammatone cache at /auto/data/tmp/tstim/, which is
# not writable here -- every load would silently recompute the whole set.
NEMS_JOBLIB_CACHE = '/auto/users/satya/code/nems2026/cache/nems_joblib/'
FALLBACK_SOUND_ROOT = '/auto/data/sounds/BigNat/v2/'
# --------------------------------------------------------------------------- #

nems_utils.cache_path = NEMS_JOBLIB_CACHE
os.makedirs(nems_utils.cache_path, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)


def mat_get(struct, key=1):
    """Matlab-struct field access: the wingert_merge2 branch keys these by str."""
    for k in (str(key), key):
        if hasattr(struct, 'keys') and k in struct:
            return struct[k]
    raise KeyError(f'{key} not in {list(struct.keys())}')


def sound_root_from(reference_handle):
    """Map the Windows SoundPath in exptparams to its path here (as NAT_stim does)."""
    root = reference_handle.get('SoundPath', '').replace('\\', '/')
    for old, new in [('C:/Users/lbhb/Desktop/v2', '/auto/data/sounds/BigNat/v2'),
                     ('E:/sounds/v2', '/auto/data/sounds/BigNat/v2'),
                     ('H:/', '/auto/data/'), ('h:/', '/auto/data/'), ('E:/', '/auto/data/')]:
        root = root.replace(old, new)
    if not root or not os.path.isdir(root):
        print(f'  SoundPath {root!r} not usable, falling back to {FALLBACK_SOUND_ROOT}')
        root = FALLBACK_SOUND_ROOT
    return root.rstrip('/') + '/'


def stats(a, b):
    """max|diff|, that as a fraction of full scale, and Pearson r."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    max_abs_diff = np.abs(a - b).max()
    return max_abs_diff, max_abs_diff / np.abs(a).max(), np.corrcoef(a, b)[0, 1]


def per_column_r(a, b):
    """Pearson r for each column of two (time, n) arrays."""
    return np.array([np.corrcoef(a[:, i], b[:, i])[0, 1] if a[:, i].std() > 0 else np.nan
                     for i in range(a.shape[1])])


# --------------------------------------------------------------------------- #
# 1. Load the NEMS object
# --------------------------------------------------------------------------- #
print(f'loading NEMS recording for {PARMFILE}')
manager = BAPHYExperiment(parmfile=[PARMFILE])

exptparams = manager.get_baphy_exptparams()[0]
trial_object = mat_get(exptparams['TrialObject'])
reference_handle = mat_get(trial_object['ReferenceHandle'])
overall_db = trial_object['OveralldB']
fixed_amp_scale = reference_handle['FixedAmpScale']
pre_stim_silence = reference_handle['PreStimSilence']
duration = reference_handle['Duration']
sound_root = sound_root_from(reference_handle)
print(f'  OveralldB={overall_db}  FixedAmpScale={fixed_amp_scale}  '
      f'PreStimSilence={pre_stim_silence}s  Duration={duration}s')
print(f'  sound_root={sound_root}')

rec = manager.get_recording(resp=False, stim=True, rasterfs=100,
                            loadkey='gtgram.fs100.ch32')
stim = rec['stim'].rasterize()
fs_gtg = stim.fs
n_cfs = stim.shape[0]
print(f'  stim: {n_cfs} channels @ {fs_gtg} Hz')
assert n_cfs == 32, (f'expected 32 gtgram channels, got {n_cfs}. A binaural BigNat '
                     f'site returns 2*32; use a mono site or the `mono` loadkey option.')

epoch_names = [e for e in stim.epochs['name'].unique() if e.startswith('STIM_')][:N_EPOCHS]
bad = [e for e in epoch_names if ('+' in e or ':' in e)]
assert not bad, (f'epochs {bad} name a wav combination; the wav->gtgram comparison '
                 f'is only defined for single-wav (mono) epochs.')
print(f'  scoring {len(epoch_names)} epochs: {epoch_names}')

# A NEMS epoch is PreStimSilence + Duration + PostStimSilence; keep the Duration
# window, which is what ACNet's own front end produces. BNT wavs are 17.79 s
# against an 18 s Duration -- NAT_stim zero-pads the waveform out to Duration
# before filtering, so `stim_duration_s=duration` below makes ACNet do the same
# and both sides come out at floor(Duration*fs_gtg) bins with no truncation.
start_bin = int(round(pre_stim_silence * fs_gtg))
n_bins_stim = int(np.floor(duration * fs_gtg))
stim_dict = stim.extract_epochs(epoch_names)


# --------------------------------------------------------------------------- #
# 2. Load ACNet and match its front end to this site's calibration
# --------------------------------------------------------------------------- #
model, cell_rtest = load_acnet(WEIGHTS_PATH)
model.eval()
device = model._device()
print(f'ACNet on {device}; model compress={model.audio_process.compress}')


def acnet_front_end(nems_match):
    """A sqrt-compressed copy of ACNet's front end, calibrated to this site.

    sqrt is the domain NEMS stores its stim in, so the two are directly
    comparable; `model.gtg_to_model_input` then squares and re-compresses either
    of them into the model's own input domain.
    """
    proc = GammatoneFilterbankProcessing(
        num_cfs=model.config['num_cfs'], f_min=model.config['f_min'], f_max=model.config['f_max'],
        fs_gtg=model.config['fs_gtg'], keep_pre_s=0., stim_dur_after_onset=None,
        compress='sqrt', lbhb_mode=True, level_mode='approx',
        overall_db=overall_db, fixed_amp_scale=fixed_amp_scale, nems_match=nems_match,
        stim_duration_s=duration)
    proc.device = device
    return proc


ARMS = {'ACNet default (training-faithful)': False, 'ACNet nems_match=True': True}
procs = {name: acnet_front_end(nems_match) for name, nems_match in ARMS.items()}


# --------------------------------------------------------------------------- #
# 3. Compare, epoch by epoch
# --------------------------------------------------------------------------- #
results = {name: {'gtg': [], 'model_input': [], 'manifold': [], 'psth': [],
                  'r_per_cf': [], 'r_per_dim': []} for name in ARMS}
plot_data = {}

for i_epoch, epoch in enumerate(epoch_names):
    wav_fname = sound_root + epoch.replace('STIM_', '')
    if not wav_fname.endswith('.wav'):
        wav_fname += '.wav'
    assert os.path.exists(wav_fname), wav_fname

    x_nems = stim_dict[epoch][0, :, start_bin:start_bin + n_bins_stim].T   # (time, n_cfs), sqrt
    with torch.no_grad():
        emb_nems, in_nems = model.get_mf_embeddings_from_gtg(x_nems, gtg_compress='sqrt')
        psth_nems = model.predict_psth_from_gtg(x_nems, gtg_compress='sqrt')
    emb_nems = emb_nems.cpu().numpy().squeeze()
    in_nems, psth_nems = in_nems.cpu().numpy(), psth_nems.cpu().numpy()

    print(f'\n{epoch}  wav={os.path.basename(wav_fname)}  nems gtg {x_nems.shape}')
    for name, nems_match in ARMS.items():
        int16_scale = 32767.0 if nems_match else 32768.0
        waveform, fs_wav = _load_wav(wav_fname, int16_scale=int16_scale)
        with torch.no_grad():
            x_acnet = procs[name](waveform.to(device), fs_wav)     # (time, n_cfs), sqrt domain
            emb_acnet, in_acnet = model.get_mf_embeddings_from_gtg(x_acnet, gtg_compress='sqrt')
            psth_acnet = model.predict_psth_from_gtg(x_acnet, gtg_compress='sqrt')
        x_acnet = x_acnet.cpu().numpy()
        emb_acnet = emb_acnet.cpu().numpy().squeeze()
        in_acnet, psth_acnet = in_acnet.cpu().numpy(), psth_acnet.cpu().numpy()

        n_t = min(x_nems.shape[0], x_acnet.shape[0])
        assert x_nems.shape[0] == x_acnet.shape[0], (
            f'length mismatch: nems {x_nems.shape[0]} vs ACNet {x_acnet.shape[0]} bins. '
            f'Both should be floor(Duration*fs_gtg)={n_bins_stim}; a mismatch means the '
            f'epoch window or stim_duration_s is wrong, not a numerical difference.')

        pairs = {'gtg': (x_nems[:n_t], x_acnet[:n_t]),
                 'model_input': (in_nems[:n_t], in_acnet[:n_t]),
                 'manifold': (emb_nems[:n_t], emb_acnet[:n_t]),
                 'psth': (psth_nems[:n_t], psth_acnet[:n_t])}
        print(f'  {name}')
        for key, (a, b) in pairs.items():
            max_abs_diff, rel, r = stats(a, b)
            results[name][key].append((max_abs_diff, rel, r))
            print(f'    {key:12s} max|d|={max_abs_diff:.3e}  rel={rel:.3e}  r={r:.8f}')
        results[name]['r_per_cf'].append(per_column_r(*pairs['gtg']))
        results[name]['r_per_dim'].append(per_column_r(*pairs['manifold']))

        if i_epoch == PLOT_EPOCH:
            plot_data[name] = {'x_nems': x_nems[:n_t], 'x_acnet': x_acnet[:n_t],
                               'emb_nems': emb_nems[:n_t], 'emb_acnet': emb_acnet[:n_t]}

print('\n' + '=' * 78)
print(f'summary over {len(epoch_names)} epochs (mean r, worst max|d| / full scale)')
print('=' * 78)
for name in ARMS:
    print(f'{name}')
    for key in ('gtg', 'model_input', 'manifold', 'psth'):
        arr = np.array(results[name][key])
        print(f'  {key:12s} mean r={arr[:, 2].mean():.8f}  min r={arr[:, 2].min():.8f}  '
              f'worst rel max|d|={arr[:, 1].max():.3e}')


# --------------------------------------------------------------------------- #
# 4. Figure
# --------------------------------------------------------------------------- #
epoch_plotted = epoch_names[PLOT_EPOCH]
cf_kHz = procs['ACNet default (training-faithful)'].get_cfs() / 1e3
cf_tick_kHz = np.array([0.2, 2, 20])
cf_tick_pos = [int(np.argmin(np.abs(cf_kHz - t))) + 0.5 for t in cf_tick_kHz]

ref = plot_data['ACNet nems_match=True']
dur_s = ref['x_nems'].shape[0] / fs_gtg
extent_gtg = (0, dur_s, 0, ref['x_nems'].shape[1])
extent_emb = (0, dur_s, 0, ref['emb_nems'].shape[1])

fig, ax = plt.subplots(3, 3, figsize=(13, 9.5))

# Row 0 -- gammatonegram, in the sqrt domain NEMS stores.
v_gtg = max(ref['x_nems'].max(), ref['x_acnet'].max())
for col, (title, img) in enumerate([('NEMS stim (sqrt magnitude)', ref['x_nems']),
                                    ('ACNet front end (nems_match)', ref['x_acnet'])]):
    ax[0, col].imshow(img.T, origin='lower', aspect='auto', extent=extent_gtg, vmin=0, vmax=v_gtg)
    ax[0, col].set(title=title, yticks=cf_tick_pos, yticklabels=[f'{t:g}' for t in cf_tick_kHz])
ax[0, 0].set(ylabel='CF (kHz)')
d_gtg = ref['x_acnet'] - ref['x_nems']
lim = np.abs(d_gtg).max() or 1.0
im = ax[0, 2].imshow(d_gtg.T, origin='lower', aspect='auto', extent=extent_gtg,
                     cmap='RdBu_r', vmin=-lim, vmax=lim)
ax[0, 2].set(title=f'difference (max |d| = {lim:.1e})',
             yticks=cf_tick_pos, yticklabels=[f'{t:g}' for t in cf_tick_kHz])
fig.colorbar(im, ax=ax[0, 2], fraction=0.046)

# Row 1 -- manifold embeddings from each gammatonegram.
v_emb = max(ref['emb_nems'].max(), ref['emb_acnet'].max())
for col, (title, img) in enumerate([('manifold from NEMS gtgram', ref['emb_nems']),
                                    ('manifold from ACNet gtgram', ref['emb_acnet'])]):
    ax[1, col].imshow(img.T, origin='lower', aspect='auto', extent=extent_emb, vmin=0, vmax=v_emb)
    ax[1, col].set(title=title)
ax[1, 0].set(ylabel='manifold dimension')
d_emb = ref['emb_acnet'] - ref['emb_nems']
lim = np.abs(d_emb).max() or 1.0
im = ax[1, 2].imshow(d_emb.T, origin='lower', aspect='auto', extent=extent_emb,
                     cmap='RdBu_r', vmin=-lim, vmax=lim)
ax[1, 2].set(title=f'difference (max |d| = {lim:.1e})')
fig.colorbar(im, ax=ax[1, 2], fraction=0.046)

# Row 2 -- how well each arm agrees with NEMS, channel by channel.
colors = {'ACNet default (training-faithful)': 'tab:orange', 'ACNet nems_match=True': 'tab:blue'}
for name in ARMS:
    # Dimensions that are constant (dead ReLU units) have no defined r; nanmean over
    # an all-nan column warns, so drop those columns instead of averaging them.
    r_cf = np.nanmean(np.array(results[name]['r_per_cf']), axis=0)
    r_dim_all = np.array(results[name]['r_per_dim'])
    live = ~np.all(np.isnan(r_dim_all), axis=0)
    r_dim = np.nanmean(r_dim_all[:, live], axis=0)
    ax[2, 0].plot(cf_kHz, r_cf, '.-', color=colors[name], label=name)
    ax[2, 1].plot(np.flatnonzero(live), r_dim, '.-', color=colors[name], ms=3, label=name)
    print(f'{name}: {live.sum()}/{live.size} manifold dims active, '
          f'min per-dim r={np.nanmin(r_dim):.8f}')
ax[2, 0].set(xscale='log', xlabel='CF (kHz)', ylabel='r vs NEMS',
             title=f'per-CF agreement (mean of {len(epoch_names)} epochs)')
ax[2, 0].legend(fontsize=7, loc='lower right')
ax[2, 1].set(xlabel='manifold dimension', ylabel='r vs NEMS', title='per-dimension agreement')

# Residual rather than a raw scatter: both arms sit on the identity line, so only
# the deviation from it distinguishes them.
sub = slice(None, None, 37)  # 1800*32 points is unreadable at full density
for name in reversed(list(ARMS)):   # draw the tighter arm last so it stays visible
    p = plot_data[name]
    ax[2, 2].plot(p['x_nems'].ravel()[sub],
                  (p['x_acnet'] - p['x_nems']).ravel()[sub], '.', ms=1.5,
                  alpha=0.3, color=colors[name], label=name)
ax[2, 2].axhline(0, color='k', lw=0.8)
ax[2, 2].set(xlabel='NEMS stim (sqrt magnitude)', ylabel='ACNet $-$ NEMS',
             title='per-bin residual')
ax[2, 2].legend(fontsize=7, loc='lower right', markerscale=6)

for a in ax[0, :].tolist() + ax[1, :].tolist():
    a.set(xlabel='time (s)')

siteid = os.path.basename(PARMFILE).split('_')[0]
fig.suptitle(f'ACNet front end vs NEMS stim -- {siteid}, {epoch_plotted}, '
             f'OveralldB={overall_db}, FixedAmpScale={fixed_amp_scale}', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT_PNG, dpi=300)
print(f'\nsaved figure to {OUT_PNG}')
