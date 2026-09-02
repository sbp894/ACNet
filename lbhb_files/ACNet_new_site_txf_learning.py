'''
ACNet transfer learning to a site ACNet has NEVER SEEN.

The companion script `ACNet_old_site_txf_learning.py` is the verification: on a
site that WAS in ACNet's 62, a fresh head trained on the frozen backbone
reproduces the head ACNet already has (PRN018a: r_test 0.6462 vs 0.6422,
per-neuron correlation 0.9992, predicted PSTHs agreeing at median r = 0.987).
That said the machinery works. This script asks the question it was built for:

    does a frozen ACNet backbone + one new head beat fitting that site from
    scratch?

--------------------------------------------------------------------------------
The two arms
--------------------------------------------------------------------------------
  ARM A  transfer   ACNet's `layers_shared` FROZEN and in eval() + a fresh
                    Linear(200, nCells) + DEXP head, fit with the same
                    three-stage procedure ACNet's own heads were fit with
                    (stage 1 nonlinearity skipped; stage 2 DEXP seeded from the
                    validation targets and switched on; stage 3 fine).
                    ~20k trainable parameters on top of a 211k frozen trunk.
  ARM B  baseline   `NemsSmallMatch` from
                    Claude/code/ACNet_nas_match_PT_nems_1site/nems_match_helpers.py
                    -- the PyTorch port of the nems single-site BNT fit
                    (arch_type='small': TWO conv layers, wc+fir+relu twice, then
                    a dense block and the wc->dexp readout; ResNet off), fit from
                    scratch on this site alone. That port was validated against
                    nems TF to within 0.0014 r_test per stage and 0.001 in
                    distribution (10 fresh inits, Mann-Whitney p = 0.14), so it
                    stands in for a real nems fit without needing TF or celldb.
                    ~39k trainable parameters. This is the GOLD-STANDARD
                    baseline: it is the fit these sites are actually produced
                    with, so "does transfer beat it" is the question that
                    matters operationally.
Available but OFF by default (RUN_RANDOM_TRUNK_CONTROL): an arm A' that is
identical to arm A except the frozen trunk is a randomly initialised ACNet. A vs
A' differs in exactly one thing -- the trunk weights -- so it is what would
license a claim about the ACNet representation specifically, as opposed to the
operational claim A vs B answers. Not run here by user decision; the question on
the table is whether transfer beats the gold-standard fit, not why.

Arm B runs the PRODUCTION protocol of `match_PT_nems_3stage.py` -- the nems
`no_rand` THREE-stage fit, L2 = 1e-5 (`l2:5`):

    stage 1  dexp skipped (skip_nonlinearity: base/amp/kappa frozen, only the
             dexp shift trains).                        lr 1e-3, tol 1e-3
    init     init_nl_lite -- dexp base/amp/kappa seeded from the RESPONSE stats
             (base = mean - std, amp = 4*std, kappa = 0), shift left as trained.
    stage 2  fit with the nonlinearity on.              lr 1e-3, tol 1e-3
    stage 3  fine fit.                                  lr 1e-4, tol 1e-4

with shuffle=True, KerasAdam, per-variable clipnorm 1.0, batch 25, max_iter
30000, and early stopping monitoring the VALIDATION loss on the same 2 held-out
stimuli arm A early-stops on. Not the 2-stage "effective" schedule in
`match_PT_nems_1site.py` -- that one reconstructs what one older driver actually
executed; this is the protocol the site fits are produced with.

The two arms therefore share a three-stage shape and the same
seed-the-nonlinearity-from-response-statistics step (ACNet's
`get_dexp_init_vals` and nems's `init_nl_lite` are the same formula; ACNet reads
it off the validation targets, nems off the training targets).

--------------------------------------------------------------------------------
What is and is not held fixed
--------------------------------------------------------------------------------
Held fixed, deliberately: the site, the neurons (SNR filter identical), the
stimuli, the est/early-stop/test split, the per-neuron output gain, the input
representation (log10x magnitude, so both arms see the same numbers), the seed,
the device, and the scoring (per-neuron Pearson r on the same 4 concatenated
held-out stimuli).

NOT held fixed, and this is the point rather than a flaw: each arm runs its OWN
published protocol end to end -- its own architecture, loss (NMSELoss mean/pow
vs nems `loss_se`), optimizer, stage schedule and stopping rule. This is a
protocol-vs-protocol comparison, not a one-variable ablation, and the result
should be read as "frozen-ACNet-plus-head vs the single-site fit we would
otherwise run", not as "feature X causes the difference". One seed each; the
62-site seed SD is ~0.005, so treat anything smaller than that as noise.

--------------------------------------------------------------------------------
Site choice
--------------------------------------------------------------------------------
REI003a -- Reishi, an animal that contributes NOTHING to ACNet (its 62 sites are
CLT/LMD/PRN/SLJ only), so this is cross-animal transfer, the strongest form of
"new site". It is a baphy `.m` recording, so it avoids the psi-branch loading
issues that affect every SQD site and REI058/071/083.

Measured on the pickles (2026-09-01), REI003a is also simply the best available
choice. Of the sites with a pickle that are NOT in ACNet's 62:
    REI003a  100 fit stims  100/100 cells pass SNR>0  mean SNR 0.198   <-- used
    REI007a  100            103/103                   0.162
    REI011a  100            118/118                   0.048
    REI042a  100             57/58                    0.118
    CLT050c  100              9/9                     0.155
    PRN056a  100              3/3                     0.093
    CLT027c    9              0/20   -- unusable, and why it is not in the 62
    PRN064a   50              0/32   -- unusable
    PRN065a   50              0/53   -- unusable
So the same-animal held-out sites are not a real option: they were left out of
ACNet because they have no cells above the SNR floor. AVOID the SQD sites too --
psi recordings with a known old-nems alignment problem.

Note REI003a has 100 fit stimuli, not the 250 PRN018a had. That is a property of
the site and is identical for both arms, but it puts this comparison in a
low-data regime, which is where transfer should help most if it helps at all.

--------------------------------------------------------------------------------
Running it
--------------------------------------------------------------------------------
    cd /auto/users/satya/code/projects_getting_started/ACNet_v1
    /auto/users/satya/bin/miniconda3_25/envs/ptn/bin/python -u \
        lbhb_files/ACNet_new_site_txf_learning.py

Data is the per-site pickle the 62-site fits themselves trained from
(misc_output/BNTgtg_nems_fs100Hz_nCF32_sites_sqrt_amp/), verified to hold
sqrt(magnitude); no celldb, no nems import. This file does reach into
pytorch_models/ for `nems_match_helpers` -- unlike the rest of ACNet_v1, which
is dependency-free. That is deliberate: arm B has to be the real comparator, not
a reimplementation of it.

SITES lists the sites to fit; they run in sequence in one process, sharing the
loaded (frozen) ACNet. The per-site .pt is the resume key -- a site whose .pt
already exists is skipped and its stored r_test is folded into the across-site
summary, so an interrupted sweep resumes where it stopped. Set FORCE_REDO=True
to refit anyway.

Outputs (lbhb_files/transfer_out/):
    newsite_<SITE>.pt       both fitted models + every metric below
    newsite_<SITE>.png      4-panel comparison, 300 dpi
plus an across-site table at the end, reported two ways: site as the replicate
(the conservative unit -- neurons within a site share a stimulus set and a
recording, so they are not independent samples of "does transfer help") and
pooled over neurons (more power, less conservative).
'''

import os
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
from acnet_model import load_acnet, ACNet          # noqa: E402

_NEMS_MATCH_DIR = ('/auto/users/satya/code/projects_getting_started/pytorch_models/'
                   'Claude/code/ACNet_nas_match_PT_nems_1site/')
sys.path.insert(0, _NEMS_MATCH_DIR)
from nems_match_helpers import (NemsSmallMatch, NormSELoss, fit_stage,      # noqa: E402
                                init_nl_lite,
                                per_cell_correlation)

# ---------------------------- settings ------------------------------------- #
# Sites to fit, in order. Every one must be OUTSIDE ACNet's 62 (asserted below).
# These four are the only new sites with enough neurons above the SNR floor to be
# worth fitting -- see the site-choice table in the docstring.
SITES = ['REI003a', 'REI007a', 'REI011a', 'REI042a']
FORCE_REDO = False          # False -> a site whose .pt already exists is skipped

# Command-line override, so one host can be pointed at a subset while another host
# works on the rest:  python ACNet_new_site_txf_learning.py REI011a REI042a
# (same convention as match_PT_nems_3stage.py). Sites already holding a .pt are
# skipped either way, so overlapping lists cost nothing once a site is finished.
# `--force` refits anyway -- used for the host-swap control, where the whole point
# is to redo a finished site on a different machine.
# NOTE: this block must stay BELOW the FORCE_REDO default, or the default wins.
if len(sys.argv) > 1:
    _args = sys.argv[1:]
    if '--force' in _args:
        FORCE_REDO = True
        _args = [a for a in _args if a != '--force']
    if _args:
        SITES = _args
WEIGHTS_PATH = None         # None -> weights/acnet_v1.pt

PKL_DIR = ('/auto/users/satya/code/projects_getting_started/pytorch_models/misc_output/'
           'BNTgtg_nems_fs100Hz_nCF32_sites_sqrt_amp/')

# Cell-inclusion floor: helpers_v2:6282 keeps a neuron if ANY held-out stimulus
# has cell_respSNR > SNR_THRESH. 0 is what the nems driver and the 62-site fits
# used. Both arms get the identical neuron set.
SNR_THRESH = 0.0

RASTERFS = 100
KEEP_PRE_S = 0.5
STIM_DUR_AFTER_ONSET = 18.5

# 2 of the 6 10-rep stimuli early-stop, r_test is scored on the other 4
# (helpers_v2 default val_stim_inds=[0, 4]). Both arms use the same split.
VAL_STIM_INDS = [0, 4]

STIMS_PER_BATCH = 25        # data_params['batch'] AND nems _batch_size -- same
SEED = 0

# Which loss arm A's early stopping watches.
#   'train'  what fit_wrapper/fit_routine actually does, and how ACNet's own heads
#            were selected. Val loss is tracked but never stops anything, so a
#            stage can run past its best validation point and end on worse weights.
#   'val'    stop on the 2 held-out early-stopping stimuli, which is what arm B
#            does (DelayedStopper on x_val/y_val). Makes the stopping rule MATCH
#            across arms and removes the largest non-representational difference
#            between them.
# Scoring is unaffected either way: r_test is always the 4 stimuli [1,2,3,5],
# which neither arm ever sees.
ES_ON = 'val'
REUSE_ARM_B = True          # load arm B from the existing .pt instead of refitting
# Optional control: arm A on a RANDOM frozen trunk instead of the trained one.
# Off -- the question here is A vs the gold-standard B. See the docstring.
RUN_RANDOM_TRUNK_CONTROL = False
QUICK = False               # True -> a few epochs per stage, for a smoke test

OUT_DIR = os.path.join(HERE, 'transfer_out')

# ---- ARM A: the three-stage head fit (run62_acnet.build_data_struct_v2ref) --- #
OPTIM_SHARED = {
    'l2wd_lin_head': 1e-3,
    'l2wd_nl': 0,
    'adam_betas': (0.8, 0.999),
    'clip_grad_max_norm': 1,
    'clip_grad_adapt_norm': 1,
}
STAGES_A = [
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

# ---- ARM B: the production nems single-site schedule (match_PT_nems_3stage) -- #
# Three stages, with init_nl_lite between 1 and 2. max_iter is the nems value;
# validation early stopping fires long before it.
STAGES_B = [
    dict(tag='nems_s1_noNL', lr=1e-3, max_iter=30000, early_tol=1e-3, skip_nl=True),
    dict(tag='nems_s2_NL',   lr=1e-3, max_iter=30000, early_tol=1e-3, skip_nl=False),
    dict(tag='nems_s3_fine', lr=1e-4, max_iter=30000, early_tol=1e-4, skip_nl=False),
]
NEMS_L2 = 1e-5              # 'l2:5', last three wc layers only
NEMS_EARLY_DELAY = 100
NEMS_EARLY_PATIENCE = 150
NEMS_CLIPNORM = 1.0
NEMS_SHUFFLE = True
# --------------------------------------------------------------------------- #

os.makedirs(OUT_DIR, exist_ok=True)


# Runs with a different arm-A stopping rule write to their own files, so a
# train-ES run and a val-ES run of the same site coexist and can be diffed.
OUT_SUFFIX = '' if ES_ON == 'train' else f'_{ES_ON}ES'


def out_pt_for(siteid, suffix=None):
    """Resume key: one .pt per (site, stopping rule)."""
    suffix = OUT_SUFFIX if suffix is None else suffix
    return os.path.join(OUT_DIR, f'newsite_{siteid}{suffix}.pt')


def out_png_for(siteid):
    return os.path.join(OUT_DIR, f'newsite_{siteid}{OUT_SUFFIX}.png')

if QUICK:
    for _s in STAGES_A:
        _s.update(num_epochs=8, early_patience_fit=2, loss_patience_fit=2,
                  loss_patience_val=2, warmup_epochs=1)
    for _s in STAGES_B:
        _s.update(max_iter=8)


# --------------------------------------------------------------------------- #
# 1. Site data
# --------------------------------------------------------------------------- #
def load_site_from_pkl(siteid):
    """The per-site pickle `MultiTask_BNTDataSet_Site_Nems` trains from.

    `X_gtg_*_nems` is sqrt(magnitude) -- verified against ACNet's own front end
    at r = 1.000000 (Claude/claude_debug/debug_pkl_gtg_domain.py). `Y_psth_*` is
    already minmax-normalised per neuron and rep-averaged, both done inside
    read_ephys_data.
    """
    matches = sorted(f for f in os.listdir(PKL_DIR) if f.endswith(f'_{siteid}.pkl'))
    assert len(matches) == 1, f'expected exactly one pickle for {siteid}, found {matches}'
    path = os.path.join(PKL_DIR, matches[0])
    print(f'loading site data from {path}')
    with open(path, 'rb') as fh:
        p = pickle.load(fh)

    n_bins = int((STIM_DUR_AFTER_ONSET + KEEP_PRE_S) * RASTERFS)
    assert p['X_gtg_EncMdl_nems'].shape[1] == n_bins, (
        f"pickle has {p['X_gtg_EncMdl_nems'].shape[1]} bins per stimulus, expected {n_bins}")

    # Cell inclusion, exactly as helpers_v2 does it. cell_respSNR is (n_val_stim,
    # nCells) for most sites but 1-D for the SQD ones; atleast_2d keeps the
    # reduction PER CELL either way.
    snr = np.atleast_2d(np.asarray(p['cell_respSNR']))
    keep = np.any(snr > SNR_THRESH, axis=0)
    print(f'  {int(keep.sum())}/{keep.size} neurons pass SNR > {SNR_THRESH}')
    assert keep.sum() > 0, f'no neuron in {siteid} passes SNR > {SNR_THRESH}'

    return {
        'X_est': p['X_gtg_EncMdl_nems'].astype(np.float32),
        'Y_est': p['Y_psth_EncMdl'].astype(np.float32)[..., keep],
        'X_val10': p['X_gtg_Val10Rep_nems'].astype(np.float32),
        'Y_val10': p['Y_psth_val10Rep'].astype(np.float32)[..., keep],
        'cell_names': np.array(p['cell_names'])[keep],
        'cell_snr': np.mean(snr, axis=0)[keep],
        'test_epochs': np.array(['00seq1.wav', '00seq2.wav', '00seq3.wav',
                                 '00seq4.wav', '00seq5_hand.wav', '00seq6_hand.wav']),
    }


# --------------------------------------------------------------------------- #
# 2. Arm A's head  (same definitions as ACNet_old_site_txf_learning.py; these
#    lbhb_files demos are deliberately standalone)
# --------------------------------------------------------------------------- #
class DEXP(nn.Module):
    """y = base + amp * exp(-exp(-exp(kappa) * x)), per neuron.

    Same function as acnet_model.DEXP, plus the two training-time details from
    PT_EncMdl_helpers_v2.DEXP: the inner exponent is clamped (its gradient is
    0*inf = NaN otherwise) and 'skip' makes the head linear for stage 1.
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


def init_nonlinearity_from_targets(head, y_val):
    """PT_EncMdl_helpers_v2.get_dexp_init_vals, called the way fit_wrapper calls
    it -- from the VALIDATION TARGETS, not the stage-1 head output. kappa is
    reset to 0, so stage 2 does not inherit stage 1's 0.7 default."""
    with torch.no_grad():
        std = y_val.std(dim=0)
        head.nl.base.copy_(y_val.mean(dim=0) - std)
        head.nl.amp.copy_(4 * std)
        head.nl.kappa.zero_()
        head.nl.nonlinearity = 'dexp'


def nmse_loss(outputs, targets, scale):
    """NMSELoss(axis=0, reduction='sum', time_weight='mean', ref_pow='pow') --
    the defaults get_optim_params sets, which is what the 62-site fits ran with
    (NOT the NMSELoss class defaults, 'sum'/'var').

    A stacked (nStims, T, nCells) input averages over time and sums over both
    remaining axes, identically the sum of the per-stimulus losses. The time
    axis is taken from the rank because getting it wrong does not crash -- it
    silently normalises across cells instead.
    """
    time_axis = outputs.dim() - 2
    se = torch.mean((outputs - targets) ** 2, dim=time_axis)
    ref = torch.mean(targets ** 2, dim=time_axis) + 1e-8
    return torch.sum(se / ref) * scale


def fit_stage_head(head, data, params, scale, log=print):
    """One `fit_routine` call, reduced to what a frozen-backbone head needs.

    Kept: AdamW with per-group weight decay and betas, linear lr warmup,
    gradient accumulation with the adaptive clip-norm rule, DEXP kappa clamped
    after every step, ReduceLROnPlateau on the TRAIN loss, early stop on the
    train loss, and a separately tracked best-validation snapshot.

    Preserved deliberately: fit_routine assigns `best_model_state_fit =
    self.state_dict()` with no deepcopy, so its "restore best fit" is a no-op and
    the model carried forward is the LAST epoch's. Same here.
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
    stopped_early = False
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
        if epoch < warmup:
            for g in opt.param_groups:
                g['lr'] = params['adam_lr'] * min(1., float(epoch + 1) / warmup)

        head.train()
        opt.zero_grad(set_to_none=True)
        order = np.random.RandomState(epoch).permutation(n_stims)
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
        if ga % num_grad_acc != 0:
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

        if ES_ON == 'val':
            stop = (epoch_since_best_val > params['loss_patience_val']
                    and epoch > params['early_patience_fit'])
        else:
            stop = (epoch_since_best_fit > params['loss_patience_fit']
                    and epoch > params['early_patience_fit'])
        if stop:
            stopped_early = True
            log(f'    early stop at epoch {epoch + 1} (on {ES_ON} loss)')
            break

        if (epoch + 1) % 25 == 0:
            log(f'    ep {epoch + 1:5d}/{params["num_epochs"]}  train {total_train:.4f}  '
                f'val {total_val:.4f}  lr {opt.param_groups[0]["lr"]:.1e}')

    # Restore the best-validation weights when val-ES actually fired. This matches
    # arm B exactly: keras (and so nems_match_helpers.fit_stage) restores
    # best_weights only on the patience trigger, never at plain max_iter -- see its
    # `if stopper.stopped` branch. On ES_ON='train' nothing is restored, which
    # reproduces fit_routine's own (deepcopy-less, therefore no-op) behaviour.
    if ES_ON == 'val' and stopped_early:
        head.load_state_dict(best_state_val)
        log(f'    restored best-val weights from epoch {best_epoch_val + 1}')

    log(f'    stage {params["tag"]} done in {time.time() - t0:.1f}s over {len(train_hist)} '
        f'epochs; best train {best_loss_fit:.4f} @ep{best_epoch_fit + 1}, '
        f'best val {best_loss_val:.4f} @ep{best_epoch_val + 1}'
        f'{" [restored]" if (ES_ON == "val" and stopped_early) else ""}')
    return {'train_hist': train_hist, 'val_hist': val_hist,
            'best_state_val': best_state_val, 'best_epoch_val': best_epoch_val,
            'best_loss_val': best_loss_val, 'best_loss_fit': best_loss_fit}


# --------------------------------------------------------------------------- #
# 3. Main
# --------------------------------------------------------------------------- #
def run_site(SITEID, model, acnet_sites):
    """Both arms for one site. `model` and `acnet_sites` are loaded once and shared:
    the trunk is frozen and in eval(), so reusing it across sites changes nothing."""
    OUT_PT, OUT_PNG = out_pt_for(SITEID), out_png_for(SITEID)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = model._device()

    assert SITEID not in acnet_sites, (
        f'{SITEID} IS one of ACNet\'s 62 training sites. This script is for a site ACNet '
        f'has never seen; use ACNet_old_site_txf_learning.py for an in-training site.')
    print('\n\n' + '#' * 78)
    print(f'# {SITEID} -- not in ACNet (its animals: '
          f'{sorted(set(x[:3] for x in acnet_sites))})')
    print('#' * 78)

    d = load_site_from_pkl(SITEID)
    cell_names, cell_snr = list(d['cell_names']), d['cell_snr']
    n_cells = len(cell_names)
    X_est, Y_est, X_val10, Y_val10 = d['X_est'], d['Y_est'], d['X_val10'], d['Y_val10']

    # Per-neuron output gain, as MultiTask_BNTDataSet_Site_Nems does it: 1 / (max
    # over the fit set). Applied before either arm sees the data, so both arms
    # have identical targets.
    y_gain = 1.0 / Y_est.max(axis=(0, 1))
    Y_est, Y_val10 = Y_est * y_gain, Y_val10 * y_gain

    es_inds = list(VAL_STIM_INDS)
    test_inds = [i for i in range(6) if i not in es_inds]
    print(f'  {n_cells} neurons, {X_est.shape[0]} fit stimuli')
    print(f'  early-stopping stimuli {[str(d["test_epochs"][i]) for i in es_inds]}')
    print(f'  r_test stimuli         {[str(d["test_epochs"][i]) for i in test_inds]}')

    # EVERY arm consumes the identical input tensor: log10x(magnitude).
    # `gtg_to_model_input(x, 'sqrt')` squares the stored sqrt back to magnitude
    # and applies the model's own compression, which is numerically the same
    # 0.5*ln(1 + 10*mag) the dataset class applies at helpers_v2:6380. The arms
    # then share the tensor objects below -- arm A/A' embed them through a frozen
    # trunk, arm B feeds them straight in -- so the input cannot silently differ.
    assert model.audio_process.compress == 'log10x', (
        f"ACNet's front end is set to compress='{model.audio_process.compress}'. The "
        f"62-site fits used log10x; both arms must see the same input domain.")
    with torch.no_grad():
        X_est_in = torch.stack([model.gtg_to_model_input(x, 'sqrt') for x in X_est])
        X_es_in = model.gtg_to_model_input(
            np.concatenate([X_val10[i] for i in es_inds], axis=0), 'sqrt')
        X_test_in = model.gtg_to_model_input(
            np.concatenate([X_val10[i] for i in test_inds], axis=0), 'sqrt')

    Y_est_t = torch.tensor(Y_est, dtype=torch.float32, device=device)
    Y_es_t = torch.tensor(np.concatenate([Y_val10[i] for i in es_inds], axis=0),
                          dtype=torch.float32, device=device)
    Y_test_np = np.concatenate([Y_val10[i] for i in test_inds], axis=0)

    scale = 1.0 / n_cells               # get_optim_params: 1 / max(head_output_dims)

    def run_head_arm(trunk, label):
        """Freeze `trunk`, embed every stimulus once, fit a head three-stage."""
        print('\n' + '=' * 78)
        print(f'{label}')
        print('=' * 78)
        for prm in trunk.parameters():
            prm.requires_grad_(False)
        trunk.eval()                    # BatchNorm on running stats, not batch stats

        t0 = time.time()
        with torch.no_grad():
            M_est = torch.stack([trunk(x).squeeze(0).float() for x in X_est_in])
            M_es = trunk(X_es_in).squeeze(0).float()
            M_test = trunk(X_test_in).squeeze(0).float()
        print(f'  manifold: {tuple(M_est.shape)} fit, {tuple(M_es.shape)} early-stop, '
              f'{tuple(M_test.shape)} test  [{time.time() - t0:.1f}s on {device}]')

        data = {'M_est': M_est, 'Y_est': Y_est_t, 'M_val': M_es, 'Y_val': Y_es_t}
        head = SiteHead(M_est.shape[-1], n_cells).to(device)
        hist = {}
        for i_stage, params in enumerate(STAGES_A):
            print(f'\n  stage {params["tag"]}  lr={params["adam_lr"]:.1e}')
            if i_stage == 0:
                head.nl.nonlinearity = 'skip'
                print('    nonlinearity: skip')
            elif i_stage == 1:
                init_nonlinearity_from_targets(head, Y_es_t)
                print('    nonlinearity: dexp, seeded from the validation targets')
            hist[params['tag']] = fit_stage_head(head, data, params, scale)

        head.eval()
        with torch.no_grad():
            pred = head(M_test).float().cpu().numpy()
        return head, pred, per_cell_correlation(pred, Y_test_np), hist

    # ---------------------- ARM A: frozen TRAINED ACNet + new head ---------- #
    head, pred_a, r_a, hist_a = run_head_arm(
        model.layers_shared,
        f'ARM A -- transfer: trained ACNet backbone FROZEN, new head for {SITEID}')
    n_par_a = sum(p.numel() for p in head.parameters())

    # ---------------------- ARM A' (optional control): RANDOM frozen trunk --- #
    head_r = rand_trunk = pred_ar = r_ar = hist_ar = None
    if RUN_RANDOM_TRUNK_CONTROL:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        rand_trunk = ACNet(model.config).layers_shared.to(device)
        # Recalibrate BatchNorm on the fit stimuli first. An untrained BN carries
        # running_mean=0 / running_var=1, so in eval() the random trunk would be
        # badly scaled and the control unfairly weak. One pass in train() mode
        # populates the statistics; no weight is updated.
        rand_trunk.train()
        with torch.no_grad():
            for x in X_est_in:
                rand_trunk(x)
        head_r, pred_ar, r_ar, hist_ar = run_head_arm(
            rand_trunk,
            "ARM A' -- control: RANDOM backbone frozen (BN recalibrated), same head "
            "+ same protocol")

    # ---------------------- ARM B: nems single-site from scratch ------------- #
    print('\n' + '=' * 78)
    print(f'ARM B -- gold standard: nems single-site 3-stage fit of {SITEID}')
    print('=' * 78)

    # Arm B does not depend on arm A's stopping rule, so when only arm A is being
    # re-run there is nothing to re-fit: load the stored fit. Its r_test is scored
    # on the same 4 stimuli, on the same neurons in the same order (asserted), so
    # the pairing is exact. Re-fitting would only add seed noise to the baseline.
    reuse_pt = out_pt_for(SITEID, suffix='')
    if REUSE_ARM_B and os.path.exists(reuse_pt):
        stored = torch.load(reuse_pt, map_location='cpu', weights_only=False)
        assert list(stored['cell_names']) == list(cell_names), (
            f'{reuse_pt} has a different neuron set/order than this run -- arm B '
            f'cannot be paired with arm A.')
        assert stored['test_stim_inds'] == test_inds, (
            f"{reuse_pt} scored stimuli {stored['test_stim_inds']}, this run uses "
            f'{test_inds}.')
        assert not stored.get('quick', False), f'{reuse_pt} is a QUICK smoke run.'
        nems_model = NemsSmallMatch(nCF=X_est_in.shape[-1], nCells=n_cells,
                                    l2=NEMS_L2).to(device)
        nems_model.load_state_dict(stored['arm_b']['state_dict'])
        n_par_b = stored['arm_b']['n_params']
        pred_b, r_b = stored['arm_b']['pred'], stored['arm_b']['r_test']
        hist_b = stored['arm_b']['history']
        print(f'  reused from {os.path.basename(reuse_pt)} '
              f'(mean r_test {np.nanmean(r_b):.4f}) -- not re-fit')
        arm_b_reused = True
    else:
        arm_b_reused = False
        torch.manual_seed(SEED)             # same seed, fresh init
        np.random.seed(SEED)
        nems_model = NemsSmallMatch(nCF=X_est_in.shape[-1], nCells=n_cells, l2=NEMS_L2).to(device)
        n_par_b = sum(p.numel() for p in nems_model.parameters())
        criterion = NormSELoss()

        hist_b = {}
        for i_stage, params in enumerate(STAGES_B):
            print(f'\n  {params["tag"]}  lr={params["lr"]:.1e}')
            if i_stage == 0:
                # nems: freeze dexp base/amp/kappa, keep shift trainable.
                nems_model.skip_nonlinearity()
                print('    nonlinearity: skipped (dexp shift still trains)')
            elif i_stage == 1:
                nems_model.unskip_nonlinearity()
                init_nl_lite(nems_model, Y_est_t.cpu().numpy())
                print(f'    init_nl_lite: base~{nems_model.dexp.base.mean():.3f} '
                      f'amp~{nems_model.dexp.amp.mean():.3f} '
                      f'kappa~{nems_model.dexp.kappa.mean():.3f}')
            hist_b[params['tag']] = fit_stage(
                nems_model, X_est_in.cpu(), Y_est_t.cpu(), criterion,
                lr=params['lr'], max_iter=params['max_iter'], batch_size=STIMS_PER_BATCH,
                early_delay=NEMS_EARLY_DELAY, early_patience=NEMS_EARLY_PATIENCE,
                early_tol=params['early_tol'], clipnorm=NEMS_CLIPNORM, shuffle=NEMS_SHUFFLE,
                device=device, tag=params['tag'],
                x_val=X_es_in.unsqueeze(0).cpu(), y_val=Y_es_t.unsqueeze(0).cpu())

        nems_model.eval()
        with torch.no_grad():
            pred_b = nems_model(X_test_in.unsqueeze(0).to(device)).squeeze(0).float().cpu().numpy()
        r_b = per_cell_correlation(pred_b, Y_test_np)

    # ---------------------- comparison -------------------------------------- #
    def paired_p(u, v):
        try:
            from scipy.stats import wilcoxon
            return float(wilcoxon(u, v).pvalue)
        except Exception:              # scipy is not an ACNet_v1 dependency
            return None

    d_ab = r_a - r_b
    p_ab = paired_p(r_a, r_b)
    n_ab = int(np.sum(d_ab > 0))
    has_ar = r_ar is not None
    d_aar = (r_a - r_ar) if has_ar else None
    p_aar = paired_p(r_a, r_ar) if has_ar else None
    n_aar = int(np.sum(d_aar > 0)) if has_ar else 0

    print('\n' + '=' * 78)
    print(f'{SITEID} -- {n_cells} neurons, {X_est.shape[0]} fit stimuli, 4 held-out stimuli')
    print('=' * 78)
    arms = [("A   trained ACNet frozen + head ", r_a, n_par_a),
            ("B   nems single-site (gold)    ", r_b, n_par_b)]
    if has_ar:
        arms.insert(1, ("A'  RANDOM trunk frozen + head ", r_ar, n_par_a))
    for tag, r, npar in arms:
        print(f'  {tag} mean r_test = {np.nanmean(r):.4f}   median {np.nanmedian(r):.4f}'
              f'   ({npar:,} trainable)')
    print(f"\n  A - B  (operational: beats the gold standard?)   "
          f"mean {np.nanmean(d_ab):+.4f}  median {np.nanmedian(d_ab):+.4f}  "
          f"A wins {n_ab}/{n_cells}" + (f"  p={p_ab:.4g}" if p_ab is not None else ""))
    if has_ar:
        print(f"  A - A' (attributable: is it the ACNet manifold?) "
              f"mean {np.nanmean(d_aar):+.4f}  median {np.nanmedian(d_aar):+.4f}  "
              f"A wins {n_aar}/{n_cells}" + (f"  p={p_aar:.4g}" if p_aar is not None else ""))
    print("\n  A vs B differs in many things at once (trunk, parameter count, loss,\n"
          "  optimizer, stopping rule), so it answers 'should I use this instead of the\n"
          "  usual single-site fit', NOT 'why'. Attributing a difference to the ACNet\n"
          "  representation would need the random-trunk control\n"
          "  (RUN_RANDOM_TRUNK_CONTROL). One seed each; the 62-site seed SD is ~0.005.")

    # ---------------------- save -------------------------------------------- #
    manifold_dim = int(head.manifold_dim)
    torch.save({
        'siteid': SITEID, 'cell_names': cell_names, 'cell_snr': cell_snr,
        'y_gain': y_gain, 'val_stim_inds': es_inds, 'test_stim_inds': test_inds,
        'snr_thresh': SNR_THRESH, 'seed': SEED, 'acnet_version': model.version,
        # Stamped so the resume key cannot mistake a smoke run for a finished fit.
        # It nearly did: a QUICK run wrote newsite_REI003a.pt with 8 epochs/stage,
        # and a later sweep would have skipped the site and reported those numbers.
        'quick': QUICK,
        'n_fit_stims': int(X_est.shape[0]), 'n_cells': n_cells,
        'arm_a': {'state_dict': head.state_dict(),
                  'ctor_args': {'manifold_dim': manifold_dim, 'n_cells': n_cells},
                  'stages': STAGES_A, 'optim_shared': OPTIM_SHARED, 'n_params': n_par_a,
                  'r_test': r_a, 'pred': pred_a,
                  'history': {k: {'train': v['train_hist'], 'val': v['val_hist']}
                              for k, v in hist_a.items()},
                  'best_state_val_stage3': hist_a['3_fine']['best_state_val']},
        'arm_a_rand': None if not has_ar else {
            'state_dict': head_r.state_dict(),
            'trunk_state_dict': rand_trunk.state_dict(),
            'ctor_args': {'manifold_dim': manifold_dim, 'n_cells': n_cells},
            'stages': STAGES_A, 'n_params': n_par_a,
            'r_test': r_ar, 'pred': pred_ar,
            'history': {k: {'train': v['train_hist'], 'val': v['val_hist']}
                        for k, v in hist_ar.items()}},
        'es_on': ES_ON, 'arm_b_reused': arm_b_reused,
        'arm_b': {'state_dict': nems_model.state_dict(),
                  'ctor_args': {'nCF': int(X_est_in.shape[-1]), 'nCells': n_cells,
                                'l2': NEMS_L2},
                  'stages': STAGES_B, 'n_params': n_par_b,
                  'r_test': r_b, 'pred': pred_b, 'history': hist_b},
        'p_wilcoxon': {'A_vs_B': p_ab, 'A_vs_Arand': p_aar},
    }, OUT_PT)
    print(f'\nsaved {"three" if has_ar else "both"} arms + metrics to {OUT_PT}')

    # ---------------------- figure ------------------------------------------ #
    # 2x2, so every panel renders at the same size by construction.
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 8.5))
    C_A, C_AR, C_B = 'tab:blue', 'tab:green', 'tab:orange'

    _all_r = [r_a, r_b] + ([r_ar] if has_ar else [])
    lim = [min(0., float(np.nanmin(_all_r)) - 0.05), float(np.nanmax(_all_r)) + 0.05]
    ax[0, 0].plot(lim, lim, 'k-', lw=0.8, zorder=0)
    ax[0, 0].plot(r_b, r_a, 'o', ms=4, color=C_B,
                  label=f'A wins {n_ab}/{n_cells}')
    _title = f'A - B  {np.nanmean(d_ab):+.3f} mean'
    if has_ar:
        ax[0, 0].plot(r_ar, r_a, 's', ms=4, color=C_AR,
                      label=f"vs A' random ({n_aar}/{n_cells})")
        _title += f"   A - A'  {np.nanmean(d_aar):+.3f}"
    ax[0, 0].set(xlim=lim, ylim=lim,
                 xlabel='nems-matched single-site PT fit $r_{test}$',
                 ylabel='A: frozen ACNet + new head $r_{test}$', title=_title)
    ax[0, 0].legend(fontsize=8, loc='upper left')

    ax[0, 1].plot(cell_snr, r_a, 'o', ms=4, color=C_A, label='A: transfer')
    ax[0, 1].plot(cell_snr, r_b, '^', ms=4, color=C_B,
                  label='B: nems-matched single-site PT')
    if has_ar:
        ax[0, 1].plot(cell_snr, r_ar, 's', ms=4, color=C_AR, label="A': random trunk")
    ax[0, 1].set(xlabel='neuron response SNR', ylabel='$r_{test}$',
                 title='all arms vs neuron quality')
    ax[0, 1].legend(fontsize=8, loc='lower right')

    for tag, h in hist_a.items():
        ax[1, 0].plot(h['val_hist'], lw=1.0, label=f'A {tag}')
    if has_ar:
        for tag, h in hist_ar.items():
            ax[1, 0].plot(h['val_hist'], lw=1.0, ls=':', label=f"A' {tag}")
    ax[1, 0].set(xlabel='epoch within stage', ylabel='validation loss (arm A)',
                 yscale='log',
                 title="three-stage schedule" + (", A solid / A' dotted" if has_ar else ''))
    ax[1, 0].legend(fontsize=7, ncol=2 if has_ar else 1)

    ex = int(np.argsort(r_a)[len(r_a) // 2])
    t_s = np.arange(min(600, Y_test_np.shape[0])) / RASTERFS
    n_t = t_s.size
    ax[1, 1].plot(t_s, Y_test_np[:n_t, ex], color='0.4', lw=1.0, label='neural PSTH')
    ax[1, 1].plot(t_s, pred_a[:n_t, ex], color=C_A, lw=1.0,
                  label=f'A transfer (r={r_a[ex]:.2f})')
    ax[1, 1].plot(t_s, pred_b[:n_t, ex], color=C_B, lw=1.0, ls='--',
                  label=f'B nems (r={r_b[ex]:.2f})')
    if has_ar:
        ax[1, 1].plot(t_s, pred_ar[:n_t, ex], color=C_AR, lw=0.8, ls=':',
                      label=f"A' random (r={r_ar[ex]:.2f})")
    ax[1, 1].set(xlabel='time (s)', ylabel='normalized rate',
                 title=f'{cell_names[ex]} (median neuron)')
    ax[1, 1].legend(fontsize=7)

    fig.suptitle(f'{SITEID}: a site ACNet never saw -- frozen ACNet + new head vs the '
                 f'nems-matched single-site PT fit\n({n_cells} neurons, '
                 f'{X_est.shape[0]} fit stimuli, same log10x input, same split, same '
                 f'targets, seed {SEED}; arm A early-stops on {ES_ON} loss)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f'saved figure to {OUT_PNG}')

    return {'siteid': SITEID, 'n_cells': n_cells, 'n_stims': int(X_est.shape[0]),
            'mean_snr': float(np.mean(cell_snr)), 'r_a': r_a, 'r_b': r_b,
            'mean_a': float(np.nanmean(r_a)), 'mean_b': float(np.nanmean(r_b)),
            'diff': float(np.nanmean(d_ab)), 'n_win': n_ab, 'p': p_ab}


def main():
    model, _ = load_acnet(WEIGHTS_PATH)
    model.eval()
    ckpt = torch.load(WEIGHTS_PATH or os.path.join(os.path.dirname(HERE), 'weights',
                                                   'acnet_v1.pt'),
                      map_location='cpu', weights_only=False)
    acnet_sites = sorted(set(c.split('-')[0] for c in ckpt['cell_names']))

    summaries = []
    for siteid in SITES:
        pt = out_pt_for(siteid)
        if os.path.exists(pt) and not FORCE_REDO:
            prev = torch.load(pt, map_location='cpu', weights_only=False)
            if prev.get('quick', False) and not QUICK:
                print(f'\n{siteid}: {pt} is from a QUICK smoke run -- refitting properly')
                prev = None
            else:
                print(f'\n{siteid}: {pt} exists, skipping (set FORCE_REDO=True to refit)')
        else:
            prev = None
        if prev is not None:
            r_a, r_b = prev['arm_a']['r_test'], prev['arm_b']['r_test']
            summaries.append({'siteid': siteid, 'n_cells': prev['n_cells'],
                              'n_stims': prev['n_fit_stims'],
                              'mean_snr': float(np.mean(prev['cell_snr'])),
                              'r_a': r_a, 'r_b': r_b,
                              'mean_a': float(np.nanmean(r_a)),
                              'mean_b': float(np.nanmean(r_b)),
                              'diff': float(np.nanmean(r_a - r_b)),
                              'n_win': int(np.sum(r_a - r_b > 0)),
                              'p': (prev.get('p_wilcoxon') or {}).get('A_vs_B')})
            continue
        t0 = time.time()
        summaries.append(run_site(siteid, model, acnet_sites))
        print(f'\n{siteid} done in {(time.time() - t0) / 60:.1f} min')

    # ------------------------- across-site summary -------------------------- #
    print('\n\n' + '=' * 92)
    print('ACROSS SITES -- frozen ACNet + new head (A) vs nems-matched single-site PT fit (B)')
    print('=' * 92)
    print(f'  {"site":9s} {"cells":>6s} {"stims":>6s} {"SNR":>6s} {"A":>8s} {"B":>8s} '
          f'{"A-B":>8s} {"A wins":>8s} {"p":>9s}')
    for r in summaries:
        pstr = f'{r["p"]:.4g}' if r['p'] is not None else 'n/a'
        print(f'  {r["siteid"]:9s} {r["n_cells"]:6d} {r["n_stims"]:6d} '
              f'{r["mean_snr"]:6.3f} {r["mean_a"]:8.4f} {r["mean_b"]:8.4f} '
              f'{r["diff"]:+8.4f} {r["n_win"]:4d}/{r["n_cells"]:<3d} {pstr:>9s}')

    if len(summaries) > 1:
        # Site is the replicate here: one number per site, paired across arms. That
        # is the honest unit -- neurons within a site are not independent samples of
        # "does transfer help", they share a stimulus set and a recording.
        a = np.array([r['mean_a'] for r in summaries])
        b = np.array([r['mean_b'] for r in summaries])
        print(f'\n  site-mean r_test: A {a.mean():.4f}  B {b.mean():.4f}  '
              f'A-B {a.mean() - b.mean():+.4f}, A better on {int((a > b).sum())}/{len(a)} sites')
        try:
            from scipy.stats import wilcoxon
            if len(a) >= 5:
                print(f'  Wilcoxon over sites: p = {wilcoxon(a, b).pvalue:.4g}')
            else:
                print(f'  (n={len(a)} sites is below the n=5 Wilcoxon floor -- '
                      f'report the per-site numbers, not a p-value over sites)')
        except Exception:
            pass
        # Pooling neurons across sites is the more powerful but less conservative
        # test; report both rather than picking the flattering one.
        ra = np.concatenate([r['r_a'] for r in summaries])
        rb = np.concatenate([r['r_b'] for r in summaries])
        print(f'  pooled over {ra.size} neurons: A {np.nanmean(ra):.4f}  B {np.nanmean(rb):.4f}'
              f'  A-B {np.nanmean(ra - rb):+.4f}, A better on {int(np.sum(ra > rb))}/{ra.size}')


if __name__ == '__main__':
    main()
