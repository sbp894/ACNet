'''
Script to use ACNet transfer learning for a new head. Let us first start with validation.
1. Load ACNet.
2. Choose a site (with decent number of neurons and rtest) from ACNet.
3. Initialize a new head for that site, and load the input (gtgram)/ output (PSTHs) for that site. This new model should be GTG -> ACNet_MF -> site_head.
4. Train with the normal three-step procedure. (Confirm that you know this - don't burn too many tokens, just ask me! )
5. Validate the new trained head with the same site head from ACNet.

--------------------------------------------------------------------------------
The three-step procedure (as implemented in PT_EncMdl_helpers_v2.fit_wrapper)
--------------------------------------------------------------------------------
  stage 1  'noNL'    head nonlinearity set to 'skip' -> the head is a bare Linear.
  stage 2  'coarse'  init_nonlinearity(val_dataset, activate=True): DEXP base/amp/
                     kappa are set from the VALIDATION TARGETS (base = mean-std,
                     amp = 4*std, kappa = 0), then the DEXP is switched on.
  stage 3  'fine'    same model, lower lr, plateau scheduler, long patience.
Each stage: AdamW (betas 0.8/0.999, eps 1e-7), per-group weight decay, linear lr
warmup, adaptive grad-norm clipping, ReduceLROnPlateau driven by the TRAIN loss,
early stop on the train loss, NMSELoss(axis=0, reduction='sum',
time_weight='mean', ref_pow='pow', scale=1/nCells). Stage N starts from the
weights stage N-1 ended on. Numbers below are read from
Claude/code/ACNet_nas_62site/run62_acnet.build_data_struct_v2ref, which is itself
read back from the shipped nsites62_15 run's own data_pre_fit.json -- i.e. they
are the settings that produced the head we are comparing against.

What transfer learning means here: `layers_shared` is frozen AND kept in eval()
mode, so the manifold is a deterministic function of the stimulus. That lets the
whole backbone be run ONCE up front; the three stages then train a
Linear(200, nCells) + DEXP head on cached embeddings, which is why this finishes
in minutes instead of hours.

--------------------------------------------------------------------------------
Two gates, in order
--------------------------------------------------------------------------------
  GATE 1 (pipeline): pull this site's columns out of ACNet's own readout, run
      them on the stimuli/PSTHs this script just loaded from NEMS, and compare
      the resulting r_test against the per-neuron r_test shipped in the
      checkpoint. This tests the DATA path -- epoch windows, minmax
      normalisation, compression domain, cell ordering -- before any training
      happens. If gate 1 fails, gate 2 is meaningless.
  GATE 2 (the actual question): retrained head vs ACNet's head -- per-neuron
      r_test, per-neuron correlation between the two predicted PSTHs, and
      cosine similarity of the 200-d readout weight vectors.

      Measured on PRN018a (2026-09-01): mean r_test 0.6462 retrained vs 0.6422
      original (+0.0040), per-neuron r_test correlation 0.9992, predicted PSTHs
      agreeing at median r = 0.9870, 40/40 neurons within 0.05 r_test.

      Read the weight cosine (median 0.694) with that in mind. The backbone is
      frozen, so the head is identifiable in principle -- there is no rotation
      freedom -- but the 200 manifold dimensions are far from orthogonal, so the
      least-squares problem is ill-conditioned and weight decay plus early
      stopping leave you anywhere in a long flat valley. Two quite different
      weight vectors therefore give the same prediction. The FUNCTIONAL numbers
      are the verification; the weight cosine is context, and a value well below
      1 is expected rather than a failure.
  GATE 3 (the output nonlinearity): the DEXP the retrained head learned vs the
      one ACNet ships, both as parameters and as a shape.

      Read this one carefully, because RAW KAPPA IS NOT COMPARABLE. The head
      computes base + amp*exp(-exp(-exp(kappa) * w.m)) with no shift term, so
      (w, kappa) and (c*w, kappa - ln c) are the *same function*: a head that
      happens to land on larger readout weights reports a smaller kappa for an
      identical nonlinearity. Three things are comparable and are what gate 3
      reports:

        base, amp   output units. Y is y_gain-normalised the same way for both
                    heads, so these live on a common scale.
        kappa_eff   kappa + ln(sd of that neuron's linear drive) -- kappa
                    measured against the drive it actually acts on, which is
                    exactly what the scale freedom leaves invariant.
        the curve   each head's DEXP evaluated at percentiles of its OWN drive.
                    Two heads implementing the same input-output relation give
                    the same curve here whatever their weight norms. This is the
                    'shape' test; it is scored as a per-neuron correlation and
                    as an RMSE in units of the range ACNet's curve covers.

      Two derived descriptors come along for free: `x50_sd`, where the DEXP's
      half-rise sits in SDs of the drive (is the neuron operating on the toe,
      the middle, or the saturated arm?), and `swing_used`, the fraction of the
      sigmoid's full 0..amp swing the test stimuli actually traverse. A neuron
      with swing_used near 0 is being used linearly, and its kappa is then only
      weakly constrained by the data -- which is the honest reason to expect
      some scatter rather than a defect in the fit.

--------------------------------------------------------------------------------
Where the data comes from
--------------------------------------------------------------------------------
By default: the per-site pickles the 62-site fits themselves were trained on,

    pytorch_models/misc_output/BNTgtg_nems_fs100Hz_nCF32_sites_sqrt_amp/
        <NN>_BNTgtg_nems_<SITE>.pkl

which is what `MultiTask_BNTDataSet_Site_Nems` reads. No celldb, no nems import.
Verified (Claude/claude_debug/debug_pkl_gtg_domain.py) that `X_gtg_*_nems` really
is **sqrt(magnitude)**, matching the directory name: rebuilding 00seq1/00seq2
with ACNet's own front end (nems_match, lbhb_mode, 65 dB, scale 250) gives
r = 1.000000, max|d| = 5e-5. It is NOT `_dlog`-compressed, despite the
`fn = lambda x: _dlog(x**2, -1)` line currently sitting in
sppy.read_neural_data_helper.read_ephys_data. `Y_psth_*` is already
minmax-normalised per neuron (that happens inside read_ephys_data).

Set DATA_SOURCE = 'nems' to go to the DB instead -- needed for a site that has no
pickle, e.g. one ACNet has never seen. That path caches to an npz so it is paid
once.

--------------------------------------------------------------------------------
Running it
--------------------------------------------------------------------------------
    cd /auto/users/satya/code/projects_getting_started/ACNet_v1
    /auto/users/satya/bin/miniconda3_25/envs/ptn/bin/python -u \
        lbhb_files/ACNet_new_head_txf_learning.py

`ptn` on potoroo has torch/torchaudio/numpy/matplotlib all working. Its GTX 1080
is sm_61, which this torch build ships no kernels for, so `best_device()` falls
back to CPU -- correct, just slower. DATA_SOURCE='nems' instead needs the
`nems2026` env (celldb + nems_lbhb).

Outputs (all under lbhb_files/transfer_out/):
    head_<SITE>.pt              trained head: state_dict + ctor args + metrics
                                (including the gate-3 DEXP profiles and curves)
    transfer_<SITE>.png         6-panel comparison figure, 300 dpi
    site_<SITE>.npz             only when DATA_SOURCE == 'nems'
'''

import os
import re
import sys
import copy
import time
import pickle

# Import torch before numpy -- in some MKL/OpenMP setups importing numpy (or
# anything that pulls it in) first deadlocks `import torch`.
import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # ACNet_v1/ -> acnet_model
from acnet_model import load_acnet, WEIGHTS_DEFAULT   # noqa: E402

# ---------------------------- settings ------------------------------------- #
SITEID = 'PRN018a'          # in ACNet: 40 neurons, shipped mean r_test = 0.639
WEIGHTS_PATH = None         # None -> weights/acnet_v1.pt

DATA_SOURCE = 'pkl'         # 'pkl' (the training pickles) or 'nems' (celldb)
PKL_DIR = ('/auto/users/satya/code/projects_getting_started/pytorch_models/misc_output/'
           'BNTgtg_nems_fs100Hz_nCF32_sites_sqrt_amp/')
BATCH_ID = 343              # BNT batch; DATA_SOURCE == 'nems' only

# Response/stimulus window. These are NOT free parameters: they are the values
# sppy.read_neural_data_helper.read_ephys_data was called with when the training
# pkls were built (keep_pre_s=0.5, stim_dur_after_onset=18.5), so the window is
# 0.5 s of PreStimSilence + 18.5 s = 1900 bins at 100 Hz.
RASTERFS = 100
KEEP_PRE_S = 0.5
STIM_DUR_AFTER_ONSET = 18.5

# The 6 '00seq' stimuli are the 10-repeat held-out set. Training used 2 of them
# for early stopping and scored r_test on the other 4 (helpers_v2 default
# val_stim_inds=[0, 4]); indices are into the sorted epoch names.
VAL_STIM_INDS = [0, 4]

STIMS_PER_BATCH = 25        # data_params['batch']
SEED = 0
QUICK = False               # True -> a few epochs per stage, for a smoke test

OUT_DIR = os.path.join(HERE, 'transfer_out')
NEMS_JOBLIB_CACHE = '/auto/users/satya/code/nems2026/cache/nems_joblib/'

# Shared across the three stages (run62_acnet.build_data_struct_v2ref).
# l2wd_conv / l2wd_lin_shared are listed for completeness: nothing in the shared
# backbone is trainable here, so only the head groups are ever used.
OPTIM_SHARED = {
    'l2wd_lin_head': 1e-3,      # the head's Linear
    'l2wd_nl': 0,               # the head's DEXP
    'adam_betas': (0.8, 0.999),
    'clip_grad_max_norm': 1,
    'clip_grad_adapt_norm': 1,
}

# Per-stage. num_epochs is the one place this differs from the 62-site fit,
# which used 25000 everywhere: with a frozen backbone and cached embeddings the
# head converges in the low hundreds, and every stage below still stops early on
# its own patience rule far short of these caps.
STAGES = [
    dict(tag='1_noNL',   num_epochs=400,  adam_lr=1e-3, min_lr=1e-3, factor_lr=0.5,
         patience_lr=10,  early_patience_fit=50,   loss_patience_fit=25,
         loss_patience_val=25,  warmup_epochs=20, loss_tol=1e-3),
    dict(tag='2_coarse', num_epochs=800,  adam_lr=1e-3, min_lr=2e-4, factor_lr=0.5,
         patience_lr=25,  early_patience_fit=200,  loss_patience_fit=50,
         loss_patience_val=50,  warmup_epochs=20, loss_tol=1e-3),
    dict(tag='3_fine',   num_epochs=2000, adam_lr=2e-4, min_lr=1e-5, factor_lr=0.5,
         patience_lr=100, early_patience_fit=1000, loss_patience_fit=100,
         loss_patience_val=100, warmup_epochs=25, loss_tol=1e-4),
]
# --------------------------------------------------------------------------- #

os.makedirs(OUT_DIR, exist_ok=True)
CACHE_NPZ = os.path.join(OUT_DIR, f'site_{SITEID}.npz')
OUT_PT = os.path.join(OUT_DIR, f'head_{SITEID}.pt')
OUT_PNG = os.path.join(OUT_DIR, f'transfer_{SITEID}.png')

if QUICK:
    for _s in STAGES:
        _s.update(num_epochs=8, early_patience_fit=2, loss_patience_fit=2,
                  loss_patience_val=2, warmup_epochs=1)


# --------------------------------------------------------------------------- #
# 1. Site data from NEMS  (the only part that needs celldb / nems_lbhb)
# --------------------------------------------------------------------------- #
def load_site_from_nems(siteid):
    """Stimuli and PSTHs for one BNT site, in exactly the form ACNet was fit on.

    Mirrors sppy.read_neural_data_helper.read_ephys_data:
      * resp is rasterized and `normalize('minmax')`d per neuron over the whole
        recording BEFORE any epoch extraction;
      * each epoch is cut to [PreStimSilence - 0.5 s, +18.5 s) = 1900 bins;
      * a stimulus's PSTH is the mean over its repeats (1 for the fit set, 10
        for the '00seq' set);
      * the stim signal is left as NEMS produces it -- sqrt(magnitude) -- and is
        converted to ACNet's input domain later by `gtg_to_model_input`.

    Returns a dict of plain arrays; also written to CACHE_NPZ.
    """
    import nems.tools.utils as nems_utils
    from nems_lbhb.baphy_experiment import BAPHYExperiment

    # nems_lbhb points its joblib gammatone cache at /auto/data/tmp/tstim/, which
    # is not writable here -- every load would silently recompute the whole set.
    nems_utils.cache_path = NEMS_JOBLIB_CACHE
    os.makedirs(nems_utils.cache_path, exist_ok=True)

    print(f'loading NEMS recording for {siteid} (batch {BATCH_ID})')
    manager = BAPHYExperiment(batch=BATCH_ID, cellid=siteid)
    rec = manager.get_recording(resp=True, stim=True, rasterfs=RASTERFS,
                                loadkey=f'gtgram.fs{RASTERFS}.ch32')
    stim = rec['stim'].rasterize()
    resp = rec['resp'].rasterize().normalize('minmax')

    n_cfs = stim.shape[0]
    assert n_cfs == 32, (f'expected 32 gtgram channels, got {n_cfs}. A binaural BigNat '
                         f'site returns 2*32; ACNet is a 32-channel model.')

    # Epoch window, from the recording's own PreStimSilence.
    ep_df = resp.epochs
    pre_s = np.unique((ep_df.loc[ep_df['name'] == 'PreStimSilence', 'end']
                       - ep_df.loc[ep_df['name'] == 'PreStimSilence', 'start']).round(8).values)
    assert pre_s.size == 1, f'multiple PreStimSilence durations: {pre_s}'
    start_bin = int((pre_s[0] - KEEP_PRE_S) * RASTERFS)
    n_bins = int((STIM_DUR_AFTER_ONSET + KEEP_PRE_S) * RASTERFS)
    assert start_bin >= 0, (f'PreStimSilence={pre_s[0]}s is shorter than KEEP_PRE_S='
                            f'{KEEP_PRE_S}s, so the training window does not exist here.')
    print(f'  PreStimSilence={pre_s[0]}s -> window [{start_bin}, {start_bin + n_bins}) '
          f'= {n_bins} bins @ {RASTERFS} Hz')

    # `sorted` reproduces nems.tools.epoch.epoch_names_matching, which sorts.
    names = resp.epochs['name'].unique()
    fit_epochs = sorted(n for n in names if re.match(r'^STIM_seq', n))
    test_epochs = sorted(n for n in names if re.match(r'^STIM_00seq', n))
    assert len(test_epochs) == 6, f'expected 6 10-rep stimuli, got {test_epochs}'
    print(f'  {len(fit_epochs)} fit stimuli, {len(test_epochs)} held-out stimuli')

    def cut_resp(epochs):
        d = resp.extract_epochs(epochs)
        # (reps, cells, T) -> mean over reps -> (T, cells)
        return np.stack([d[k][:, :, start_bin:start_bin + n_bins].mean(axis=0).T
                         for k in epochs]).astype(np.float32)

    def cut_stim(epochs):
        d = stim.extract_epochs(epochs)
        return np.stack([d[k][0, :, start_bin:start_bin + n_bins].T
                         for k in epochs]).astype(np.float32)

    out = {
        'X_est': cut_stim(fit_epochs), 'Y_est': cut_resp(fit_epochs),
        'X_val10': cut_stim(test_epochs), 'Y_val10': cut_resp(test_epochs),
        'cell_names': np.array(resp.chans), 'fit_epochs': np.array(fit_epochs),
        'test_epochs': np.array(test_epochs), 'siteid': np.array(siteid),
    }
    np.savez_compressed(CACHE_NPZ, **out)
    print(f'  cached to {CACHE_NPZ}')
    return out


def load_site_from_pkl(siteid):
    """The same per-site pickle `MultiTask_BNTDataSet_Site_Nems` trains from.

    Contents, straight out of read_ephys_data (so all the windowing and the
    minmax normalisation of resp already happened):
        X_gtg_EncMdl_nems   (n_fit, 1900, 32)  sqrt(magnitude)  -- verified, see
                                               the module docstring
        Y_psth_EncMdl       (n_fit, 1900, nCells)  minmax-normalised, rep-averaged
        X_gtg_Val10Rep_nems (6, 1900, 32)
        Y_psth_val10Rep     (6, 1900, nCells)
        cell_names, cell_respSNR, stimnames_EncMdl, overall_db, ...
    """
    matches = sorted(f for f in os.listdir(PKL_DIR)
                     if f.endswith(f'_{siteid}.pkl'))
    assert len(matches) == 1, f'expected exactly one pickle for {siteid}, found {matches}'
    path = os.path.join(PKL_DIR, matches[0])
    print(f'loading site data from {path}')
    with open(path, 'rb') as fh:
        p = pickle.load(fh)

    n_bins = int((STIM_DUR_AFTER_ONSET + KEEP_PRE_S) * RASTERFS)
    assert p['X_gtg_EncMdl_nems'].shape[1] == n_bins, (
        f"pickle has {p['X_gtg_EncMdl_nems'].shape[1]} bins per stimulus, this script "
        f"expects {n_bins} (keep_pre_s={KEEP_PRE_S}, stim_dur={STIM_DUR_AFTER_ONSET}).")

    # The 6 10-rep stimuli are stored in the order read_ephys_data got them from
    # epoch_names_matching, which sorts -- i.e. 00seq1..00seq4, 00seq5_hand,
    # 00seq6_hand. VAL_STIM_INDS indexes into that.
    #
    # One thing the dataset class does that is NOT reproduced here: if a site's
    # fit-stimulus count is not a multiple of 50 it copies stimuli from the front
    # of the list to pad it out (helpers_v2:6286). PRN018a has 250, so the
    # question does not arise; a site where it does will train on a slightly
    # different est set than ACNet did.
    return {
        'X_est': p['X_gtg_EncMdl_nems'].astype(np.float32),
        'Y_est': p['Y_psth_EncMdl'].astype(np.float32),
        'X_val10': p['X_gtg_Val10Rep_nems'].astype(np.float32),
        'Y_val10': p['Y_psth_val10Rep'].astype(np.float32),
        'cell_names': np.array(p['cell_names']),
        'fit_epochs': np.array(p['stimnames_EncMdl']),
        'test_epochs': np.array(['00seq1.wav', '00seq2.wav', '00seq3.wav',
                                 '00seq4.wav', '00seq5_hand.wav', '00seq6_hand.wav']),
        'siteid': np.array(siteid),
    }


def load_site(siteid):
    if DATA_SOURCE == 'pkl':
        return load_site_from_pkl(siteid)
    if os.path.exists(CACHE_NPZ):
        print(f'loading cached site data from {CACHE_NPZ}')
        z = np.load(CACHE_NPZ, allow_pickle=False)
        return {k: z[k] for k in z.files}
    return load_site_from_nems(siteid)


# --------------------------------------------------------------------------- #
# 2. The head
# --------------------------------------------------------------------------- #
class DEXP(nn.Module):
    """Double-exponential output nonlinearity, one set of parameters per neuron.

        y = base + amp * exp(-exp(-exp(kappa) * x))

    Same function as acnet_model.DEXP (so the released head drops straight in),
    plus the two training-time details from PT_EncMdl_helpers_v2.DEXP: the inner
    exponent is clamped (its gradient is 0*inf = NaN otherwise, for x very
    negative) and `nonlinearity='skip'` makes the head linear for stage 1.
    """

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.nonlinearity = 'dexp'
        self.base = nn.Parameter(torch.zeros(num_nodes))
        self.amp = nn.Parameter(1.5 * torch.ones(num_nodes))
        self.kappa = nn.Parameter(0.7 * torch.ones(num_nodes))

    def forward(self, x):
        if self.nonlinearity == 'skip':
            return x
        inner = torch.exp(torch.clamp(torch.exp(self.kappa) * (-x), max=20.0))
        return self.base + self.amp * torch.exp(-inner)

    def clamp_parameters(self, max_kappa=4.0):
        self.kappa.data.clamp_(max=max_kappa)


class SiteHead(nn.Module):
    """One site's readout: Linear(manifold_dim -> nCells) then DEXP."""

    def __init__(self, manifold_dim, n_cells):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.n_cells = n_cells
        self.linear = nn.Linear(manifold_dim, n_cells)
        self.nl = DEXP(n_cells)

    def forward(self, m):
        return self.nl(self.linear(m))


def acnet_reference_head(model, cols):
    """The site's existing head, sliced out of ACNet's concatenated readout.

    The release collapsed the 62 site-specific heads into one Linear(200, 3124)
    + DEXP(3124); `cols` are this site's neuron indices, so the slice IS the
    original head, bit for bit.
    """
    head = SiteHead(model.readout_linear.in_features, len(cols))
    with torch.no_grad():
        head.linear.weight.copy_(model.readout_linear.weight[cols])
        head.linear.bias.copy_(model.readout_linear.bias[cols])
        head.nl.base.copy_(model.readout_nl.base[cols])
        head.nl.amp.copy_(model.readout_nl.amp[cols])
        head.nl.kappa.copy_(model.readout_nl.kappa[cols])
    return head


def init_nonlinearity_from_targets(head, y_val):
    """PT_EncMdl_helpers_v2.get_dexp_init_vals, called the way fit_wrapper calls it.

    Note what it uses: the VALIDATION TARGETS, not the stage-1 head output. The
    DEXP is placed on the response's own scale (base one SD below the mean, amp
    4 SD) and kappa is reset to 0 -- so stage 2 does not start from stage 1's
    kappa=0.7 default.
    """
    with torch.no_grad():
        std = y_val.std(dim=0)
        head.nl.base.copy_(y_val.mean(dim=0) - std)
        head.nl.amp.copy_(4 * std)
        head.nl.kappa.zero_()
        head.nl.nonlinearity = 'dexp'


# --------------------------------------------------------------------------- #
# 3. Loss / metrics
# --------------------------------------------------------------------------- #
def nmse_loss(outputs, targets, scale):
    """PT_EncMdl_helpers_v2.NMSELoss(axis=0, reduction='sum', time_weight='mean',
    ref_pow='pow', scale=scale) -- the defaults get_optim_params sets and that
    the 62-site fits ran with (NOT the NMSELoss class defaults, which are
    time_weight='sum' / ref_pow='var').

    Upstream is called once per stimulus with a (T, nCells) tensor and averages
    over time, then sums over cells. A stacked (nStims, T, nCells) input is
    handled here by averaging over time and summing over BOTH remaining axes,
    which is identically the sum of the per-stimulus losses -- just one big
    matmul instead of 25 small ones, which matters on CPU. Getting the time axis
    wrong does not crash, it silently normalises across cells instead, so it is
    chosen from the rank rather than passed in.
    """
    time_axis = outputs.dim() - 2
    se = torch.mean((outputs - targets) ** 2, dim=time_axis)
    ref = torch.mean(targets ** 2, dim=time_axis) + 1e-8
    return torch.sum(se / ref) * scale


def per_cell_corr(pred, resp):
    """Pearson r per cell, ignoring non-finite bins. pred/resp: (T, nCells)."""
    out = np.full(pred.shape[-1], np.nan)
    for c in range(pred.shape[-1]):
        p, r = pred[:, c], resp[:, c]
        m = np.isfinite(p) & np.isfinite(r)
        if m.sum() < 3:
            continue
        pc, rc = p[m] - p[m].mean(), r[m] - r[m].mean()
        den = np.sqrt((pc ** 2).sum() * (rc ** 2).sum())
        if den > 0:
            out[c] = float((pc * rc).sum() / den)
    return out


@torch.no_grad()
def head_predict(head, m):
    head.eval()
    return head(m).float().cpu().numpy()


@torch.no_grad()
def dexp_profile(head, m, n_pts=101):
    """Describe a head's output nonlinearity in a gauge-invariant way (gate 3).

    The DEXP here has no shift term but does have a scale freedom:
    base + amp*exp(-exp(-exp(kappa) * z)) with z = w.m, so (w, kappa) and
    (c*w, kappa - ln c) are the same function. Raw kappa therefore cannot be
    compared between two independently fit heads. What can:

      base, amp   output units, common scale across heads (same y_gain).
      kappa_eff   kappa + ln(sd z) -- kappa against the drive it acts on.
      curve       the DEXP at percentiles of its OWN drive, so the weight norm
                  divides out and only the input-output shape is left.

    Also returned, as context for how much of the curve the data even visits:
      x50_sd      half-rise drive, in SDs of that neuron's drive.
      swing_used  (curve[99%] - curve[1%]) / amp, i.e. the fraction of the
                  sigmoid's full swing the test stimuli traverse. Near 0 means
                  the neuron is operating linearly and its kappa is only weakly
                  identified -- expect scatter there, and do not read it as a
                  failed fit.
    """
    head.eval()
    z = head.linear(m).float().cpu().numpy()               # (T, nCells) drive
    base = head.nl.base.detach().cpu().numpy()
    amp = head.nl.amp.detach().cpu().numpy()
    kappa = head.nl.kappa.detach().cpu().numpy()

    q = np.linspace(1., 99., n_pts)
    zq = np.percentile(z, q, axis=0)                       # (n_pts, nCells)
    # Same clamp as DEXP.forward, so the curve is the function the model runs.
    inner = np.exp(np.minimum(np.exp(kappa) * (-zq), 20.))
    yq = base + amp * np.exp(-inner)

    z_mean, z_sd = z.mean(axis=0), z.std(axis=0)
    x50 = -np.log(np.log(2.)) * np.exp(-kappa)             # exp(-exp(-e^k x))=0.5
    return {
        'base': base, 'amp': amp, 'kappa': kappa,
        'kappa_eff': kappa + np.log(z_sd + 1e-12),
        'z_mean': z_mean, 'z_sd': z_sd,
        'x50_sd': (x50 - z_mean) / (z_sd + 1e-12),
        'swing_used': (yq[-1] - yq[0]) / (amp + 1e-12),
        'q': q, 'curve': yq,
    }


# --------------------------------------------------------------------------- #
# 4. One fitting stage
# --------------------------------------------------------------------------- #
def fit_stage(head, data, params, scale, log=print):
    """One `fit_routine` call, reduced to what a frozen-backbone head needs.

    Kept from fit_routine: AdamW with per-group weight decay and betas, linear
    lr warmup over the first `warmup_epochs`, gradient accumulation with the
    adaptive clip-norm rule, DEXP kappa clamped after every optimizer step,
    ReduceLROnPlateau stepped on the TRAIN loss, early stopping on the train
    loss (`epoch_since_best_fit > loss_patience_fit` AND `epoch >
    early_patience_fit`), and a separately tracked best-validation snapshot.

    Dropped, with reasons: AMP/GradScaler and the `hostname=='rhino'` tf32
    branch (this is a 200 x nCells matmul -- fp32 costs nothing and removes the
    fp16-poisons-BatchNorm failure mode entirely), fused/capturable AdamW (CUDA
    only; nems2026 ships CPU torch), and all checkpoint bookkeeping.

    One upstream behaviour is deliberately preserved: fit_routine assigns
    `best_model_state_fit = self.state_dict()` WITHOUT a deepcopy, so those are
    live references and the "restore best fit" at the end is a no-op -- the
    model carried into the next stage, and the one that was shipped, is the LAST
    epoch's. So `final` below is the last epoch. `best_val` is deep-copied, as
    upstream does, and reported alongside.
    """
    m_est, y_est, m_val, y_val = data['M_est'], data['Y_est'], data['M_val'], data['Y_val']
    n_stims = m_est.shape[0]
    num_grad_acc = max(1, n_stims // STIMS_PER_BATCH)

    opt = torch.optim.AdamW([
        {'params': list(head.linear.parameters()),
         'weight_decay': OPTIM_SHARED['l2wd_lin_head'], 'betas': OPTIM_SHARED['adam_betas']},
        {'params': list(head.nl.parameters()),
         'weight_decay': OPTIM_SHARED['l2wd_nl'], 'betas': OPTIM_SHARED['adam_betas']},
    ], lr=params['adam_lr'], eps=1e-7)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=params['factor_lr'], patience=params['patience_lr'], min_lr=params['min_lr'])

    adapt_norm = float(OPTIM_SHARED['clip_grad_adapt_norm'])
    max_norm = float(OPTIM_SHARED['clip_grad_max_norm'])
    warmup = min(params['warmup_epochs'], max(1, params['num_epochs'] // 10))

    best_loss_fit, best_loss_val = float('inf'), float('inf')
    epoch_since_best_fit, epoch_since_best_val = 0, 0
    best_epoch_fit, best_epoch_val = 0, 0
    best_state_val = copy.deepcopy(head.state_dict())
    train_hist, val_hist = [], []
    t0 = time.time()

    def optimizer_step():
        nonlocal adapt_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=float('inf'))
        if grad_norm.item() > 2 * adapt_norm:
            adapt_norm *= 1.1
        elif grad_norm.item() < 0.5 * adapt_norm:
            adapt_norm *= 0.9
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=adapt_norm)
        adapt_norm = min(max_norm, adapt_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        head.nl.clamp_parameters()

    for epoch in range(params['num_epochs']):
        if epoch < warmup:                     # linear warmup, as in fit_routine
            for g in opt.param_groups:
                g['lr'] = params['adam_lr'] * min(1., float(epoch + 1) / warmup)

        head.train()
        opt.zero_grad(set_to_none=True)
        rng = np.random.RandomState(epoch)
        order = rng.permutation(n_stims)
        batches = [order[i:i + STIMS_PER_BATCH] for i in range(0, n_stims, STIMS_PER_BATCH)]

        batch_losses, ga = [], 0
        for b in batches:
            idx = torch.as_tensor(b, device=m_est.device)
            loss = nmse_loss(head(m_est[idx]), y_est[idx], scale) / STIMS_PER_BATCH / num_grad_acc
            loss.backward()
            batch_losses.append(loss.item() * num_grad_acc)
            assert np.isfinite(batch_losses[-1]), f'non-finite loss at epoch {epoch}'
            ga += 1
            if ga % num_grad_acc == 0:
                optimizer_step()
        if ga % num_grad_acc != 0:             # the leftover partial accumulation
            optimizer_step()

        total_train = float(np.sum(batch_losses) / len(batches))
        sched.step(total_train)

        head.eval()
        with torch.inference_mode():
            total_val = float(nmse_loss(head(m_val), y_val, scale).item())
        train_hist.append(total_train)
        val_hist.append(total_val)

        epoch_since_best_val += 1
        if (total_val - best_loss_val) < -params['loss_tol']:
            best_loss_val, best_epoch_val, epoch_since_best_val = total_val, epoch, 0
            best_state_val = copy.deepcopy(head.state_dict())

        epoch_since_best_fit += 1
        if (total_train - best_loss_fit) < -params['loss_tol']:
            best_loss_fit, best_epoch_fit, epoch_since_best_fit = total_train, epoch, 0

        if (epoch_since_best_fit > params['loss_patience_fit']
                and epoch > params['early_patience_fit']):
            log(f'    early stop at epoch {epoch + 1}')
            break

        if (epoch + 1) % 25 == 0:
            log(f'    ep {epoch + 1:5d}/{params["num_epochs"]}  train {total_train:.4f}  '
                f'val {total_val:.4f}  lr {opt.param_groups[0]["lr"]:.1e}  '
                f'since best fit={epoch_since_best_fit} val={epoch_since_best_val}')

    log(f'    stage {params["tag"]} done in {time.time() - t0:.1f}s over {len(train_hist)} epochs; '
        f'best train {best_loss_fit:.4f} @ep{best_epoch_fit + 1}, '
        f'best val {best_loss_val:.4f} @ep{best_epoch_val + 1}')
    return {'train_hist': train_hist, 'val_hist': val_hist,
            'best_state_val': best_state_val, 'best_epoch_val': best_epoch_val,
            'best_loss_val': best_loss_val, 'best_loss_fit': best_loss_fit}


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- ACNet ----------------------------------------------------------- #
    model, cell_rtest_published = load_acnet(WEIGHTS_PATH)
    model.eval()
    device = model._device()
    # `cell_names` sits in the checkpoint next to cell_rtest; load_acnet does not
    # return it, and it is what maps a site to its columns of the readout.
    ckpt = torch.load(WEIGHTS_PATH or WEIGHTS_DEFAULT, map_location='cpu', weights_only=False)
    ckpt_cell_names = list(ckpt['cell_names'])
    assert len(ckpt_cell_names) == model.readout_linear.out_features

    cols = [i for i, c in enumerate(ckpt_cell_names) if c.split('-')[0] == SITEID]
    assert cols, (f'{SITEID} is not one of ACNet\'s 62 training sites. For a site ACNet '
                  f'has never seen there is no reference head, so use the new-site script.')
    site_cells = [ckpt_cell_names[i] for i in cols]
    print(f'\n{SITEID}: {len(cols)} neurons in ACNet, shipped mean r_test = '
          f'{np.nanmean(cell_rtest_published[cols]):.4f}')

    # ---- site data ------------------------------------------------------- #
    d = load_site(SITEID)
    chans = list(d['cell_names'])
    missing = [c for c in site_cells if c not in chans]
    assert not missing, (f'{len(missing)} of ACNet\'s neurons for this site are not in the '
                         f'recording (e.g. {missing[:3]}). Sorting/spike ids have changed.')
    # Reorder the recording's neurons into the checkpoint's order, so column j of
    # everything below is the same neuron everywhere.
    rows = [chans.index(c) for c in site_cells]

    Y_est = d['Y_est'][..., rows]                        # (n_fit, T, nCells)
    Y_val10 = d['Y_val10'][..., rows]                    # (6, T, nCells)
    X_est, X_val10 = d['X_est'], d['X_val10']            # (n, T, 32), sqrt(magnitude)

    # Per-neuron output gain, exactly as MultiTask_BNTDataSet_Site_Nems does it:
    # 1 / (max over the FIT set), applied to fit and held-out alike. r_test does
    # not care (it is scale invariant), but the head's weights do -- without this
    # the retrained head cannot land on ACNet's.
    y_gain = 1.0 / Y_est.max(axis=(0, 1))
    Y_est, Y_val10 = Y_est * y_gain, Y_val10 * y_gain

    es_inds = list(VAL_STIM_INDS)
    test_inds = [i for i in range(6) if i not in es_inds]
    print(f'  early-stopping stimuli {[str(d["test_epochs"][i]) for i in es_inds]}')
    print(f'  r_test stimuli         {[str(d["test_epochs"][i]) for i in test_inds]}')

    # ---- frozen backbone -> manifold ------------------------------------- #
    # The stimuli are concatenated along time before being embedded, because that
    # is what get_fit_val_test_datasets does: the val/test tensors are one long
    # sequence, so the convolutions see the seams. Reproducing that keeps the
    # r_test numbers comparable to the published ones.
    for p in model.layers_shared.parameters():
        p.requires_grad_(False)
    model.layers_shared.eval()          # BatchNorm on running stats, not batch stats

    @torch.no_grad()
    def embed(x_np):
        """(T, 32) sqrt-magnitude gammatonegram -> (T, 200) manifold."""
        x = model.gtg_to_model_input(x_np, gtg_compress='sqrt')
        return model.layers_shared(x).squeeze(0).float()

    t0 = time.time()
    M_est = torch.stack([embed(x) for x in X_est])                       # (n_fit, T, 200)
    M_val = embed(np.concatenate([X_val10[i] for i in es_inds], axis=0))  # (2T, 200)
    M_test = embed(np.concatenate([X_val10[i] for i in test_inds], axis=0))
    print(f'  manifold: {tuple(M_est.shape)} fit, {tuple(M_val.shape)} early-stop, '
          f'{tuple(M_test.shape)} test  [{time.time() - t0:.1f}s on {device}]')

    Y_est_t = torch.tensor(Y_est, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(np.concatenate([Y_val10[i] for i in es_inds], axis=0),
                           dtype=torch.float32, device=device)
    Y_test_np = np.concatenate([Y_val10[i] for i in test_inds], axis=0)
    data = {'M_est': M_est, 'Y_est': Y_est_t, 'M_val': M_val, 'Y_val': Y_val_t}

    # ---- GATE 1: does ACNet's own head reproduce its published r_test? ---- #
    ref_head = acnet_reference_head(model, cols).to(device)
    pred_ref = head_predict(ref_head, M_test)
    r_ref = per_cell_corr(pred_ref, Y_test_np)
    r_pub = np.asarray(cell_rtest_published)[cols]
    gate1_r = float(np.corrcoef(r_ref, r_pub)[0, 1])
    print('\n' + '=' * 78)
    print('GATE 1 -- ACNet\'s own head, on the data this script just loaded')
    print('=' * 78)
    print(f'  recomputed mean r_test = {np.nanmean(r_ref):.4f}   '
          f'published mean r_test = {np.nanmean(r_pub):.4f}')
    print(f'  across-neuron correlation = {gate1_r:.4f}, '
          f'mean |difference| = {np.nanmean(np.abs(r_ref - r_pub)):.4f}')
    if gate1_r < 0.95 or np.nanmean(r_ref) < 0.8 * np.nanmean(r_pub):
        print('  *** GATE 1 FAILED. The data path does not reproduce the shipped model, so '
              'any\n      difference found below is a pipeline artefact, not a transfer-'
              'learning result.\n      Check, in order: the epoch window, the minmax '
              'normalisation of resp, the\n      compression domain (gtg_compress=\'sqrt\'), '
              'and the neuron ordering.')
    else:
        print('  gate 1 passed.')

    # ---- three-stage fit of a fresh head --------------------------------- #
    scale = 1.0 / len(cols)     # get_optim_params: 1 / max(head_output_dims)
    new_head = SiteHead(M_est.shape[-1], len(cols)).to(device)
    print('\n' + '=' * 78)
    print(f'three-stage fit of a NEW head for {SITEID} '
          f'({len(cols)} neurons, {M_est.shape[0]} fit stimuli, backbone frozen)')
    print('=' * 78)

    hist = {}
    for i_stage, params in enumerate(STAGES):
        print(f'\n  stage {params["tag"]}  lr={params["adam_lr"]:.1e}')
        if i_stage == 0:
            new_head.nl.nonlinearity = 'skip'          # stage 1: linear head
            print('    nonlinearity: skip')
        elif i_stage == 1:
            init_nonlinearity_from_targets(new_head, Y_val_t)
            print(f'    nonlinearity: dexp, initialised from the validation targets '
                  f'(base {new_head.nl.base.mean():.3f}, amp {new_head.nl.amp.mean():.3f}, '
                  f'kappa 0)')
        hist[params['tag']] = fit_stage(new_head, data, params, scale)

    # ---- GATE 2: retrained head vs ACNet's head --------------------------- #
    pred_new = head_predict(new_head, M_test)
    r_new = per_cell_corr(pred_new, Y_test_np)
    r_pred_agree = per_cell_corr(pred_new, pred_ref)

    w_new = new_head.linear.weight.detach().cpu().numpy()
    w_ref = ref_head.linear.weight.detach().cpu().numpy()
    cos_w = np.sum(w_new * w_ref, axis=1) / (np.linalg.norm(w_new, axis=1)
                                             * np.linalg.norm(w_ref, axis=1) + 1e-12)

    print('\n' + '=' * 78)
    print('GATE 2 -- retrained head vs ACNet\'s head (4 held-out stimuli)')
    print('=' * 78)
    print(f'  mean r_test:      ACNet {np.nanmean(r_ref):.4f}   retrained {np.nanmean(r_new):.4f}'
          f'   difference {np.nanmean(r_new - r_ref):+.4f}')
    print(f'  per-neuron r_test correlation between the two heads: '
          f'{np.corrcoef(r_ref, r_new)[0, 1]:.4f}')
    print(f'  agreement of the two predicted PSTHs, per neuron:    '
          f'median r = {np.nanmedian(r_pred_agree):.4f}  '
          f'(min {np.nanmin(r_pred_agree):.4f}, max {np.nanmax(r_pred_agree):.4f})')
    print(f'  readout weight cosine similarity, per neuron:        '
          f'median {np.median(cos_w):.4f}  (min {cos_w.min():.4f}, max {cos_w.max():.4f})')
    n_close = int(np.sum(np.abs(r_new - r_ref) < 0.05))
    print(f'  {n_close}/{len(cols)} neurons within 0.05 r_test of the original head')

    # ---- GATE 3: the output nonlinearity ---------------------------------- #
    prof_ref = dexp_profile(ref_head, M_test)
    prof_new = dexp_profile(new_head, M_test)
    c_ref, c_new = prof_ref['curve'], prof_new['curve']

    # Shape agreement per neuron: correlation over the curve, and RMSE as a
    # fraction of the range ACNet's own curve covers on these stimuli. The
    # correlation alone is a weak test (both curves are monotone), so it is the
    # NRMSE that carries the claim.
    curve_r = per_cell_corr(c_new, c_ref)
    curve_nrmse = (np.sqrt(np.mean((c_new - c_ref) ** 2, axis=0))
                   / (np.ptp(c_ref, axis=0) + 1e-12))

    print('\n' + '=' * 78)
    print('GATE 3 -- output nonlinearity: DEXP parameters and shape')
    print('=' * 78)
    print(f'  {"":<16}{"ACNet":>10}{"retrained":>11}{"r":>8}{"median |d|":>13}')
    for key, label in [('base', 'base'), ('amp', 'amp'),
                       ('kappa', 'kappa (raw)'), ('kappa_eff', 'kappa_eff'),
                       ('x50_sd', 'x50 (drive SD)'), ('swing_used', 'swing used')]:
        a, b = prof_ref[key], prof_new[key]
        print(f'  {label:<16}{np.mean(a):>10.3f}{np.mean(b):>11.3f}'
              f'{np.corrcoef(a, b)[0, 1]:>8.3f}{np.median(np.abs(b - a)):>13.3f}')
    print(f'\n  transfer curve (each head on its own drive percentiles):')
    print(f'    per-neuron correlation   median r = {np.nanmedian(curve_r):.4f}  '
          f'(min {np.nanmin(curve_r):.4f})')
    print(f'    normalised RMSE          median {np.median(curve_nrmse):.4f} of '
          f"ACNet's curve range  (max {curve_nrmse.max():.4f})")
    n_shape = int(np.sum((curve_r > 0.99) & (curve_nrmse < 0.10)))
    print(f'    {n_shape}/{len(cols)} neurons match in shape (r > 0.99 and NRMSE < 0.10)')
    print(f'  raw kappa is EXPECTED to disagree: (w, kappa) and (c*w, kappa - ln c) are the '
          f'same\n  function, so kappa only means anything beside the drive it acts on. Read '
          f'kappa_eff.\n  Median swing used: ACNet {np.median(prof_ref["swing_used"]):.3f}, '
          f'retrained {np.median(prof_new["swing_used"]):.3f} '
          f'-- neurons well below ~0.2 are\n  being driven through a near-linear stretch of '
          f'the DEXP, where kappa is barely identified.')

    # ---- save (weights, unconditionally, with the ctor args beside them) -- #
    torch.save({
        'state_dict': new_head.state_dict(),
        'ctor_args': {'manifold_dim': int(M_est.shape[-1]), 'n_cells': int(len(cols))},
        'siteid': SITEID, 'cell_names': site_cells, 'y_gain': y_gain,
        'val_stim_inds': es_inds, 'test_stim_inds': test_inds,
        'stages': STAGES, 'optim_shared': OPTIM_SHARED,
        'acnet_version': model.version,
        'r_test_new': r_new, 'r_test_acnet_head': r_ref, 'r_test_published': r_pub,
        'r_pred_agreement': r_pred_agree, 'weight_cosine': cos_w,
        'dexp_ref': prof_ref, 'dexp_new': prof_new,
        'dexp_curve_r': curve_r, 'dexp_curve_nrmse': curve_nrmse,
        'history': {k: {'train': v['train_hist'], 'val': v['val_hist']} for k, v in hist.items()},
        'best_state_val_stage3': hist['3_fine']['best_state_val'],
    }, OUT_PT)
    print(f'\nsaved head + metrics to {OUT_PT}')

    # ---- figure ----------------------------------------------------------- #
    # 2x3 on a plain grid, so every panel renders at the same size by
    # construction and no panel reads as more important than its neighbours.
    fig, ax = plt.subplots(2, 3, figsize=(14.0, 8.5))

    lim = [min(0., np.nanmin([r_pub, r_ref, r_new]) - 0.05),
           np.nanmax([r_pub, r_ref, r_new]) + 0.05]
    for a in (ax[0, 0], ax[0, 1]):
        a.plot(lim, lim, 'k-', lw=0.8, zorder=0)
        a.set(xlim=lim, ylim=lim)

    ax[0, 0].plot(r_pub, r_ref, 'o', ms=4, color='tab:gray')
    ax[0, 0].set(xlabel='published $r_{test}$ (checkpoint)', ylabel='recomputed, ACNet head',
                 title=f'gate 1: data path (r = {gate1_r:.3f})')

    ax[0, 1].plot(r_ref, r_new, 'o', ms=4, color='tab:blue')
    ax[0, 1].set(xlabel="ACNet's head $r_{test}$", ylabel='retrained head $r_{test}$',
                 title=f'gate 2: {np.nanmean(r_new - r_ref):+.3f} mean')

    ax[0, 2].plot(r_ref, r_pred_agree, 'o', ms=4, color='tab:green', label='predicted PSTH')
    ax[0, 2].plot(r_ref, cos_w, 's', ms=4, color='tab:orange', label='readout weights')
    ax[0, 2].axhline(1, color='k', lw=0.8)
    ax[0, 2].set(xlabel="ACNet's head $r_{test}$", ylabel='similarity to ACNet head',
                 ylim=(min(0., float(np.nanmin([r_pred_agree, cos_w])) - 0.05), 1.05),
                 title='gate 2: agreement, per neuron')
    ax[0, 2].legend(fontsize=8, loc='lower right')

    # gate 3, parameters. base / amp / kappa_eff live in different units, so each
    # is z-scored over neurons using ACNet's own mean and spread -- the same
    # transform applied to both heads, so the identity line still means
    # 'the retrained head recovered this parameter'.
    for key, label, color in [('base', 'base', 'tab:purple'),
                              ('amp', 'amp', 'tab:brown'),
                              ('kappa_eff', r'$\kappa_{eff}$', 'tab:cyan')]:
        a_ref, a_new = prof_ref[key], prof_new[key]
        mu, sd = float(a_ref.mean()), float(a_ref.std()) + 1e-12
        ax[1, 0].plot((a_ref - mu) / sd, (a_new - mu) / sd, 'o', ms=4, color=color,
                      label=f'{label}  (r = {np.corrcoef(a_ref, a_new)[0, 1]:.2f})')
    plim = [-3.5, 3.5]
    ax[1, 0].plot(plim, plim, 'k-', lw=0.8, zorder=0)
    ax[1, 0].set(xlim=plim, ylim=plim, xlabel="ACNet head (z over neurons)",
                 ylabel='retrained head', title='gate 3: DEXP parameters')
    ax[1, 0].legend(fontsize=8, loc='upper left')

    # gate 3, shape. One line per neuron per head: that neuron's DEXP evaluated
    # at percentiles of its OWN linear drive, which is what removes the
    # (w, kappa) scale freedom and leaves only the input-output relation.
    for c in range(len(cols)):
        ax[1, 1].plot(prof_ref['q'], c_ref[:, c], color='tab:blue', lw=0.6, alpha=0.35)
        ax[1, 1].plot(prof_new['q'], c_new[:, c], color='tab:red', lw=0.6, alpha=0.35,
                      ls='--')
    ax[1, 1].plot([], [], color='tab:blue', lw=1.2, label="ACNet head")
    ax[1, 1].plot([], [], color='tab:red', lw=1.2, ls='--', label='retrained')
    ax[1, 1].set(xlabel="percentile of that neuron's linear drive",
                 ylabel='DEXP output (normalized rate)',
                 title=f'gate 3: transfer curves '
                       f'(median NRMSE {np.median(curve_nrmse):.3f})')
    ax[1, 1].legend(fontsize=8, loc='upper left')

    # One example neuron -- the median-r_test one, so the trace is representative
    # rather than flattering.
    ex = int(np.argsort(r_ref)[len(r_ref) // 2])
    t_s = np.arange(min(600, Y_test_np.shape[0])) / RASTERFS
    n_t = t_s.size
    ax[1, 2].plot(t_s, Y_test_np[:n_t, ex], color='0.4', lw=1.0, label='neural PSTH')
    ax[1, 2].plot(t_s, pred_ref[:n_t, ex], color='tab:blue', lw=1.0,
                  label=f'ACNet head (r={r_ref[ex]:.2f})')
    ax[1, 2].plot(t_s, pred_new[:n_t, ex], color='tab:red', lw=1.0, ls='--',
                  label=f'retrained (r={r_new[ex]:.2f})')
    ax[1, 2].set(xlabel='time (s)', ylabel='normalized rate',
                 title=f'{site_cells[ex]} (median neuron)')
    ax[1, 2].legend(fontsize=8)

    fig.suptitle(f'ACNet transfer learning -- new head for {SITEID}, backbone frozen '
                 f'({len(cols)} neurons, {M_est.shape[0]} fit stimuli)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=300)
    print(f'saved figure to {OUT_PNG}')


if __name__ == '__main__':
    main()
