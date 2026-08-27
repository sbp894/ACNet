"""
Shared analysis primitives for the manuscript RSA figure (Fig2).

Imported by BOTH `build/build_fig2_cache.py` (which freezes reference values into
`data/fig2.pkl.gz`) and `fig2.py` (which recomputes them live and asserts they match).
Sharing the code is the point: an assertion between two copy-pasted implementations
tests nothing.

Everything here is dependency-light -- numpy, scipy, sklearn, torch. The two helpers
that used to come from the internal `sppy` package (`compute_serial_matrix`,
`dsearchnn`) are reimplemented below.
"""

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------- #
# analysis constants -- these define the figure and are asserted against the cache
# --------------------------------------------------------------------------- #
MODEL_VAREXP = 0.90        # variance kept for model-derived signals (MF, predicted PSTH)
DATA_VAREXP = 0.90         # variance kept for the recorded PSTH
SIM_MEASURE = 'mse'        # RDM metric: 'mse' (squared Euclidean) or 'corrcoef'
DIST_TOL = 1e-10
SUBSAMP_FACTOR = 10        # timepoint decimation for the displayed RDMs
MIN_NREPS = 5              # sites with fewer repeats are dropped from the recorded PSTHs
N_TEST_STIMS = 6           # sites without all 6 validation stimuli are dropped

GROUP_COLORS = {'CLT': 'blue', 'LMD': 'slateblue', 'PRN': 'navy',
                'SLJ': 'gray', 'REI': 'green'}


# --------------------------------------------------------------------------- #
# seriation (was sppy.clustering_helpers)
# --------------------------------------------------------------------------- #
def _seriation(Z, N, cur_index):
    """Leaf order implied by a scipy linkage tree, rooted at `cur_index`.

    Iterative rather than recursive: the original sppy version recurses once per
    merge, which overflows Python's default 1000-frame limit on the larger
    matrices this figure builds. The traversal order (left subtree then right) is
    identical, so the returned order is identical.
    """
    order = []
    stack = [cur_index]
    while stack:
        node = stack.pop()
        if node < N:
            order.append(int(node))
        else:
            left = int(Z[node - N, 0])
            right = int(Z[node - N, 1])
            stack.append(right)          # pushed first -> popped second
            stack.append(left)
    return order


def compute_serial_matrix(dist_mat, method='ward'):
    """Hierarchical-clustering seriation of a square distance matrix.

    Returns (seriated_dist, res_order, res_linkage), matching
    `sppy.clustering_helpers.compute_serial_matrix`.
    """
    N = len(dist_mat)
    # `linkage` may consume its condensed input in place, so pass a copy. (The sppy
    # original asked for `preserve_input=True`, which is a `pdist`-family keyword that
    # current scipy's `linkage` does not accept.)
    res_linkage = linkage(squareform(dist_mat, checks=False).copy(), method=method)
    res_order = _seriation(res_linkage, N, N + N - 2)
    seriated = dist_mat[np.ix_(res_order, res_order)]
    return seriated, np.asarray(res_order), res_linkage


def dsearchnn(x, v):
    """Index of the entry of `x` closest to `v` (was sppy.bookkeep.dsearchnn).

    `argmin` rather than the original `int(np.where(... == min))`, which raises when
    the minimum is attained more than once. Ties are broken by taking the first, which
    is what the caller (number of PCs reaching a variance target) wants anyway.
    """
    x = np.asarray(x)
    return int(np.argmin(np.abs(x - np.asarray(v))))


# --------------------------------------------------------------------------- #
# PCA -> RDM -> RSA
# --------------------------------------------------------------------------- #
def pca_project(in_data, var2explain, normalize=True):
    """Global z-score, full PCA, keep the PCs reaching `var2explain`.

    `in_data` is (n_timepoints, n_features). Returns (pcproj, cumvar, n_dims, n_features).
    Normalisation uses a single scalar mean/std over the whole array (not per-feature),
    matching the original `plot_apply_pc_time`.
    """
    in_data = np.asarray(in_data, dtype=np.float64)
    if normalize:
        in_data = (in_data - in_data.mean(keepdims=True)) / in_data.std(keepdims=True)

    n_features = in_data.shape[1]
    pca_full = PCA(n_components=None)
    pca_full.fit(in_data)
    cumvar = pca_full.explained_variance_ratio_.cumsum()
    n_dims = dsearchnn(cumvar, var2explain)
    pcproj = pca_full.transform(in_data)[:, :n_dims]
    return pcproj, cumvar, int(n_dims), int(n_features)


def rdm_utri(pcproj, sim_measure=SIM_MEASURE, dtype=np.float32):
    """Condensed upper-triangle RDM over timepoints.

    `pdist` returns exactly `np.triu_indices(n, k=1)` order, so no squareform is
    needed. The `/ n_features` scaling and the `DIST_TOL` flooring that the full-matrix
    version applies are a constant rescale and a change to entries that are already
    ~0; both cancel in a correlation between two RDMs, so they are omitted here. The
    original bootstrap helper (`_rdm_utri_boot`) made the same simplification.

    Returned as float32 by default: one full-resolution triangle over 11400 timepoints
    is 520 MB in float64, and up to five are held at once.
    """
    X = np.asarray(pcproj, dtype=np.float64)
    if sim_measure == 'corrcoef':
        rdm = 1 - np.corrcoef(X)
        v = rdm[np.triu_indices(X.shape[0], k=1)]
    elif sim_measure == 'mse':
        v = pdist(X, metric='sqeuclidean')
    else:
        raise ValueError(f"Unknown sim_measure '{sim_measure}'")
    return v.astype(dtype)


def rdm_square(pcproj, n_features, sim_measure=SIM_MEASURE):
    """Full square RDM, with the original scaling and flooring. Display use only.

    Only ever called on a decimated timepoint grid -- a full-resolution square RDM is
    1.04 GB and the figure needs 13 of them.
    """
    X = np.asarray(pcproj, dtype=np.float64)
    if sim_measure == 'corrcoef':
        rdm = 1 - np.corrcoef(X)
    elif sim_measure == 'mse':
        rdm = squareform(pdist(X, metric='sqeuclidean')) / n_features
    else:
        raise ValueError(f"Unknown sim_measure '{sim_measure}'")
    rdm = (rdm + rdm.T) / 2
    rdm[rdm < DIST_TOL] = 0
    return rdm


def rsa_matrix(utri_list):
    """Upper-triangular matrix of RSA correlations between a list of RDM triangles.

    Diagonal is left NaN (self-similarity is 1 by construction and carries no
    information); the lower triangle is NaN so the strict upper triangle can be
    selected with a single `~np.isnan` mask.
    """
    n = len(utri_list)
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = np.corrcoef(utri_list[i], utri_list[j])[0, 1]
    return out


# --------------------------------------------------------------------------- #
# per-animal model reconstruction
# --------------------------------------------------------------------------- #
def build_animal_model(entry, acnet_module, device=None):
    """Rebuild one per-animal encoder from its cached config + state dict.

    The per-animal models are the same architecture as the released ACNet -- a shared
    residual backbone plus a Linear->DEXP neural readout -- so they are instantiated
    with the released `acnet_model.ACNet` class rather than a private copy. The only
    difference is size: narrower hidden layers and one animal's neurons instead of
    3124. As in the ACNet export, the per-site heads were concatenated into a single
    readout; `entry['cell_names']` gives the resulting neuron order.
    """
    if device is None:
        device = acnet_module.best_device()
    model = acnet_module.ACNet(entry['config'])
    state = {k: torch.as_tensor(np.asarray(v)) for k, v in entry['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def model_signals(model, stim_gtg, device=None):
    """Manifold embeddings and predicted PSTHs for a pre-computed gammatonegram.

    NOTE: `stim_gtg` is fed to the backbone **as stored**, without passing through
    `gtg_to_model_input`. The training data for these encoders was the NEMS `stim`
    signal taken straight from the cached BNT gammatonegram files, with no further
    compression applied by the dataset class, so the model's input domain is that
    stored signal itself. Routing it through the front end's `compress` would put the
    model in a domain it was never fitted in.

    Returns (mf_embeddings (T, hidden[-1]), psth_pred (T, n_neurons)), both numpy.
    """
    if device is None:
        device = next(model.parameters()).device
    x = torch.as_tensor(np.asarray(stim_gtg, dtype=np.float32), device=device)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    mf = model.layers_shared(x)
    psth = model.readout_nl(model.readout_linear(mf))
    return mf.squeeze(0).cpu().numpy(), psth.squeeze(0).cpu().numpy()


# --------------------------------------------------------------------------- #
# stimulus block bootstrap
# --------------------------------------------------------------------------- #
# n is governed by 4 animals, which caps any animal-level test at a two-sided p-floor of
# 0.125 (see `signflip_perm_dyadic`). The stimulus dimension is where there is real n, so
# the inferential claim is built here: resample the stimulus material and ask whether the
# MF advantage survives.
#
# The SAME resample is applied to every model and every quantity within an iteration --
# all animals heard the same stimuli -- so the pairing is preserved and the difference
# MF - predR can be bootstrapped directly.
#
# Fixed PCA: only the RDM/RSA step is bootstrapped, reusing the projections computed
# once. This is the standard RSA stimulus bootstrap (Nili et al. 2014, PLoS Comput Biol):
# it resamples the conditions entering the RDM, not the feature extraction.
#
# Resolution: the bootstrap runs on a decimated timepoint grid (every BOOT_SUBSAMP-th
# sample). Full resolution is not an option -- one 11400^2 float64 RDM is 1.04 GB. The
# point estimates plotted as bars are still full resolution; only the CI is computed on
# the coarser grid, and `boot_ci`'s `shift` translates it onto the plotted value.
BOOT_SUBSAMP = 10
BOOT_BLOCK_S_LIST = [1, 19]        # 1 s = primary (114 blocks); 19 s = whole stimuli (6)
BOOT_PRIMARY_BLOCK_S = 1
BOOT_N_DEFAULT = 1000
BOOT_SEED = 0


def _rdm_utri_boot(pcproj, idx, keep_mask, sim_measure=SIM_MEASURE):
    """Upper-triangle RDM over a resampled timepoint index set."""
    X = np.asarray(pcproj, dtype=np.float64)[idx]
    if sim_measure == 'corrcoef':
        rdm = 1 - np.corrcoef(X)
        v = rdm[np.triu_indices(len(idx), k=1)]
    else:
        v = pdist(X, metric='sqeuclidean')
    return v[keep_mask]


def block_index_sets(block_s, n_stim_val, n_t_per_stim, fs_gtg, boot_subsamp=BOOT_SUBSAMP):
    """Decimated original-timepoint indices for each block; blocks never cross a stimulus."""
    block_len = int(round(block_s * fs_gtg))
    assert n_t_per_stim % block_len == 0, \
        f"block_s={block_s} ({block_len} samples) does not divide {n_t_per_stim} samples/stim"
    assert block_len % boot_subsamp == 0, \
        f"boot_subsamp={boot_subsamp} does not divide block length {block_len}"
    sets = []
    for i_stim in range(n_stim_val):
        for i_blk in range(n_t_per_stim // block_len):
            start = i_stim * n_t_per_stim + i_blk * block_len
            sets.append(np.arange(start, start + block_len, boot_subsamp))
    return np.asarray(sets)


def _rsa_for_index(idx, keep_mask, mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj,
                   sim_measure=SIM_MEASURE):
    n_models = len(mf_pcproj)
    mf_u = [_rdm_utri_boot(mf_pcproj[m], idx, keep_mask, sim_measure) for m in range(n_models)]
    pr_u = [_rdm_utri_boot(pred_pcproj[m], idx, keep_mask, sim_measure) for m in range(n_models)]
    tr_u = [_rdm_utri_boot(true_pcproj[m], idx, keep_mask, sim_measure) for m in range(n_models)]
    st_u = _rdm_utri_boot(stim_pcproj, idx, keep_mask, sim_measure)
    r_mf = np.full((n_models, n_models), np.nan)
    r_pr = np.full((n_models, n_models), np.nan)
    r_tr = np.full((n_models, n_models), np.nan)
    r_st = np.full(n_models, np.nan)
    for m1 in range(n_models):
        r_st[m1] = np.corrcoef(mf_u[m1], st_u)[0, 1]
        for m2 in range(m1 + 1, n_models):
            r_mf[m1, m2] = np.corrcoef(mf_u[m1], mf_u[m2])[0, 1]
            r_pr[m1, m2] = np.corrcoef(pr_u[m1], pr_u[m2])[0, 1]
            r_tr[m1, m2] = np.corrcoef(tr_u[m1], tr_u[m2])[0, 1]
    return r_mf, r_pr, r_tr, r_st


def bootstrap_reference(block_s, mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj,
                        n_stim_val, n_t_per_stim, fs_gtg, sim_measure=SIM_MEASURE):
    """The bootstrap's centring statistic: every block drawn exactly once.

    Cheap (one resample). `fig2.py` recomputes it from the live models and asserts it
    matches the cached value, which is what ties the shipped bootstrap distribution to
    the projections actually being plotted.
    """
    blocks = block_index_sets(block_s, n_stim_val, n_t_per_stim, fs_gtg)
    idx_ref = blocks.ravel()
    assert len(np.unique(idx_ref)) == len(idx_ref), "reference index set must have no duplicates"
    n_sub = blocks.size
    keep_all = np.ones(n_sub * (n_sub - 1) // 2, dtype=bool)
    return _rsa_for_index(idx_ref, keep_all, mf_pcproj, pred_pcproj, true_pcproj,
                          stim_pcproj, sim_measure)


def run_stim_bootstrap(block_s, n_boot, seed, mf_pcproj, pred_pcproj, true_pcproj,
                       stim_pcproj, n_stim_val, n_t_per_stim, fs_gtg,
                       sim_measure=SIM_MEASURE, verbose=True):
    import time
    blocks = block_index_sets(block_s, n_stim_val, n_t_per_stim, fs_gtg)
    n_blocks = len(blocks)
    n_models = len(mf_pcproj)
    rng = np.random.default_rng(seed)

    out = {'mf': np.full((n_boot, n_models, n_models), np.nan),
           'pred': np.full((n_boot, n_models, n_models), np.nan),
           'true': np.full((n_boot, n_models, n_models), np.nan),
           'stim': np.full((n_boot, n_models), np.nan)}

    n_sub = blocks.size
    iu0, iu1 = np.triu_indices(n_sub, k=1)

    ref = bootstrap_reference(block_s, mf_pcproj, pred_pcproj, true_pcproj, stim_pcproj,
                              n_stim_val, n_t_per_stim, fs_gtg, sim_measure)
    out['ref_mf'], out['ref_pred'], out['ref_true'], out['ref_stim'] = ref

    t0 = time.time()
    for i_boot in range(n_boot):
        draw = rng.integers(0, n_blocks, size=n_blocks)
        idx = blocks[draw].ravel()
        # a block drawn twice yields exactly-zero-distance cells that are not real data
        keep_mask = idx[iu0] != idx[iu1]
        r_mf, r_pr, r_tr, r_st = _rsa_for_index(idx, keep_mask, mf_pcproj, pred_pcproj,
                                                true_pcproj, stim_pcproj, sim_measure)
        out['mf'][i_boot], out['pred'][i_boot] = r_mf, r_pr
        out['true'][i_boot], out['stim'][i_boot] = r_tr, r_st
        if verbose and (i_boot == 19 or (i_boot + 1) % 100 == 0):
            el = time.time() - t0
            print(f"    [{block_s}s] boot {i_boot + 1}/{n_boot}  {el / (i_boot + 1):.2f} s/iter  "
                  f"eta {el / (i_boot + 1) * (n_boot - i_boot - 1) / 60:.1f} min", flush=True)

    out.update({'n_blocks': n_blocks, 'n_timepoints_per_resample': int(n_sub),
                'block_s': block_s, 'seed': seed})
    return out


def boot_ci(vals, shift=0.0, alpha=0.05):
    """Two-sided percentile CI and two-sided bootstrap p for H0: value == 0.

    `shift` translates the interval by (full-resolution estimate - decimated-grid
    reference), so the CI is centred on the value actually plotted. The bootstrap only
    ever estimates the WIDTH of the interval; decimation bias in its location is removed
    by the shift rather than silently carried into the figure. p is computed on the
    unshifted distribution.

    Returns (lo, hi, p, p_is_floor, n_finite). `p_is_floor` marks p as an upper bound:
    with n_boot resamples the smallest resolvable two-sided p is 2/n_boot, and 0 is not
    a p-value.
    """
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two = 2 * min(np.mean(v <= 0), np.mean(v >= 0))
    return float(lo + shift), float(hi + shift), float(min(max(p_two, 0.0), 1.0)), \
        p_two < 2.0 / len(v), len(v)


# --------------------------------------------------------------------------- #
# exact permutation tests
# --------------------------------------------------------------------------- #
def signflip_perm_paired(a, b):
    """Exact paired sign-flip permutation on d = a - b (the exact twin of Wilcoxon).

    Returns (observed mean diff, two-sided p, two-sided p-floor, n_perms).
    """
    import itertools
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    obs = d.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    null = (signs * d).mean(axis=1)
    return obs, float(np.mean(np.abs(null) >= np.abs(obs) - 1e-12)), 2.0 / 2 ** n, 2 ** n


def label_perm_unpaired(a, b):
    """Exact label-shuffle permutation on the difference of means. Two-sided."""
    import itertools
    from math import comb
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pooled = np.concatenate([a, b])
    na, n = len(a), len(a) + len(b)
    obs = a.mean() - b.mean()
    null = np.array([pooled[list(i)].mean() - np.delete(pooled, list(i)).mean()
                     for i in itertools.combinations(range(n), na)])
    return obs, float(np.mean(np.abs(null) >= np.abs(obs) - 1e-12)), 2.0 / comb(n, na), len(null)


def signflip_perm_dyadic(a, b, n_animals, pair_ij):
    """Exact sign-flip permutation with the ANIMAL as the exchangeable unit.

    The 6 pairs are not independent -- each animal sits in 3 of them -- so flipping pair
    signs freely (`signflip_perm_paired`) overstates the evidence. Here each animal a
    gets a sign s_a and pair (i,j) receives s_i*s_j (the QAP / network-permutation
    construction). s and -s give identical pair signs, so 2^n_animals assignments
    collapse to 2^(n_animals-1) distinct patterns -> with 4 animals only 8.

    Floor is 1/n_patterns, NOT 2/n_patterns. The ordinary sign-flip floor is 2/2^n
    because {+-1}^n is closed under global negation, so every |statistic| occurs at
    least twice. The dyadic pattern set is not: negating a pattern would need all four
    animals to carry mutually different signs, impossible with two sign values. So the
    identity pattern is alone at |obs| and the two-sided floor is 1/8 = 0.125.
    """
    import itertools
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = d.mean()
    seen, null = set(), []
    for s in itertools.product([1, -1], repeat=n_animals):
        pat = tuple(s[i] * s[j] for i, j in pair_ij)
        if pat in seen:
            continue
        seen.add(pat)
        null.append(float(np.mean(np.array(pat) * d)))
    null = np.array(null)
    return obs, float(np.mean(np.abs(null) >= np.abs(obs) - 1e-12)), 1.0 / len(null), len(null)
