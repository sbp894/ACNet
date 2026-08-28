"""
Analysis and plotting primitives shared by `fig3.py`, `figs1.py`, `figs2.py` and
`build/build_fig3_cache.py`.

Everything the ESC-50 figures need that is not matplotlib boilerplate lives here, so
the builder and the figures compute the same quantities from the same code. An
assertion between two copy-pasted implementations tests nothing.

Ported from `MS_AcxManifold/Fig_ESC50.py`, `ESC50_plot_best10.py` and
`MS_AcxManifold/plot_helpers.py`, with the seaborn and sklearn dependencies removed --
this package ships with numpy/scipy/matplotlib only.
"""

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, spearmanr

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
N_CATEGORIES = 50
MODEL_NAMES = ['Neural', 'Manifold', 'Shuffled', 'Stimulus']

# colour-blind-friendly palette (plot_helpers.get_cbf_colors, inlined)
CBF_COLORS = {
    'Blue': '#377eb8', 'Orange': '#ff7f00', 'Green': '#4daf4a', 'Pink': '#f781bf',
    'Brown': '#a65628', 'Purple': '#984ea3', 'Grey': '#999999', 'Red': '#e41a1c',
    'Yellow': '#dede00',
}
COLOR_PALETTE = {
    'Neural': CBF_COLORS['Blue'],
    'Manifold': CBF_COLORS['Red'],
    'Shuffled': CBF_COLORS['Grey'],
    'Stimulus': CBF_COLORS['Brown'],
}

# 10 saturated, distinct colours -- no yellows (ESC50_plot_best10._QUAL10)
QUAL10 = ListedColormap([
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#17becf', '#a65628', '#f781bf', '#1b7837', '#762a83',
])

# the 10 best-clustering ESC-50 categories (v4 max-min criterion, 2 per subgroup)
V4_SELECTION = ['sheep', 'cat', 'thunderstorm', 'crackling_fire',
                'brushing_teeth', 'snoring', 'can_opening', 'door_wood_knock',
                'chainsaw', 'train']

# ESC-50 ships its 50 categories in five blocks of ten, one per subgroup, ordered by
# `target`; that is the only place the subgroup of a category is recorded.
ESC50_GROUPS = ['Animals', 'Natural Soundscape', 'Human', 'Interior', 'Exterior']

ANNOT_FS = 6
N_STD = 1.0


# --------------------------------------------------------------------------- #
# confusion matrices and alignment
# --------------------------------------------------------------------------- #
def confusion_counts(pred, true, n_cat=N_CATEGORIES):
    """`M[p, t]` = number of samples with true class `t` predicted as `p`.

    NOTE the orientation: rows are PREDICTED, columns are TRUE. That is what the
    source script asked for -- it called sklearn's `confusion_matrix(pred, true)`
    with the arguments in the opposite order to sklearn's `(y_true, y_pred)`
    signature -- and it is what the axis labels say ('True class' on x).
    """
    pred = np.asarray(pred).ravel().astype(np.int64)
    true = np.asarray(true).ravel().astype(np.int64)
    assert pred.shape == true.shape, 'pred/true length mismatch'
    flat = np.bincount(pred * n_cat + true, minlength=n_cat * n_cat)
    return flat.reshape(n_cat, n_cat).astype(np.int64)


def get_alignment(pred_model, pred_reference, true, verbose=False):
    """Fraction of samples where the model and the reference (neural) classifier agree.

    Returns (overall, incorrect-only). The second number is over samples the model got
    wrong: two classifiers can only be said to share a confusion structure if they err
    the same way, and the overall number is dominated by the correct ones.
    """
    pred_model = np.asarray(pred_model).ravel()
    pred_reference = np.asarray(pred_reference).ravel()
    true = np.asarray(true).ravel()

    overall = np.sum(pred_model == pred_reference) / len(pred_model)
    wrong = pred_model != true
    if verbose:
        print(f"nwrong={wrong.sum()}: frac wrong {wrong.sum() / len(pred_model):.2f}")
    incorrect = np.sum(pred_model[wrong] == pred_reference[wrong]) / wrong.sum()
    return overall, incorrect


def alignment_per_fold(pred_folds_model, pred_folds_neural, true_folds_neural):
    """Per-fold (overall, incorrect) alignment, as arrays."""
    full, inc = zip(*[get_alignment(m, n, t) for m, n, t in
                      zip(pred_folds_model, pred_folds_neural, true_folds_neural)])
    return np.asarray(full), np.asarray(inc)


def category_alignment_full(pred_model, pred_neural, true, cats):
    """Per-category agreement with the neural classifier, over all samples."""
    return np.array([np.mean(pred_model[true == k] == pred_neural[true == k]) for k in cats])


def category_alignment_incorrect(pred_model, pred_neural, true, cats):
    """Per-category agreement, restricted to samples the model got wrong.

    NaN for a category the model never got wrong (never happens at these accuracies,
    but the caller masks on it rather than assuming).
    """
    out = np.full(len(cats), np.nan)
    for i, k in enumerate(cats):
        wrong = (true == k) & (pred_model != true)
        if wrong.sum() > 0:
            out[i] = np.mean(pred_model[wrong] == pred_neural[wrong])
    return out


# --------------------------------------------------------------------------- #
# confusion-matrix display
# --------------------------------------------------------------------------- #
def confmat_norm(scale, vmin, vmax):
    """Colour normalisation for a count matrix.

    'raw' -- linear in counts.
    'log' -- log10(1 + count), which is the only way the off-diagonal structure is
             visible at all: the diagonal runs to ~40 while the typical off-diagonal
             cell is 0-3. The data stay in counts, so the colorbar is still labelled
             in counts.
    """
    if scale == 'raw':
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    if scale == 'log':
        return mcolors.FuncNorm((lambda x: np.log10(1.0 + np.clip(x, 0, None)),
                                 lambda y: np.power(10.0, y) - 1.0),
                                vmin=vmin, vmax=vmax)
    raise ValueError(f"unknown confmat scale {scale!r}; expected 'raw' or 'log'")


def confmat_ticks(scale, vmin, vmax, n_raw=5):
    """Colorbar tick positions, in counts, spaced sensibly for the chosen scale."""
    if scale == 'raw':
        # Counts are integers, so 3.25 / 6.5 / 9.75 is a nonsense axis.
        return np.unique(np.round(np.linspace(vmin, vmax, n_raw)))
    ticks = [0, 1, 3, 10, 30, 100, 300, 1000]
    return np.array([t for t in ticks if vmin <= t <= vmax], dtype=float)


def set_border(ax, bw=0.25):
    """Draw all four spines -- an image panel needs a frame, unlike a line plot."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(bw)


def plot_confmat(ax, cm, scale='raw', vlim=None, hide_diagonal=False, cmap='Greys',
                 title=None, tickvals=None):
    """Draw one confusion matrix; returns the image handle (for a colorbar)."""
    cm_plot = np.asarray(cm, dtype=float)
    if hide_diagonal:
        cm_plot = cm_plot.copy()
        cm_plot[np.eye(cm_plot.shape[0], dtype=bool)] = np.nan

    if vlim is None:
        vlim = (0.0, float(np.nanmax(cm_plot)))
    img = ax.imshow(cm_plot, cmap=cmap, norm=confmat_norm(scale, vlim[0], vlim[1]),
                    origin='upper', interpolation='nearest', aspect='equal',
                    rasterized=True)

    if tickvals is None:
        tickvals = np.arange(0, N_CATEGORIES + 1, 10)
    ax.set(xticks=tickvals, xticklabels=tickvals, yticks=tickvals, yticklabels=tickvals)
    if title is not None:
        ax.set_title(title)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    set_border(ax, 0.1)
    return img


# --------------------------------------------------------------------------- #
# bar + strip (replaces seaborn's barplot/stripplot)
# --------------------------------------------------------------------------- #
def bar_strip(ax, values_by_label, labels, palette, bar_alpha=0.5, point_size=12,
              jitter=0.2, seed=0):
    """Mean bar per condition with the individual folds/categories drawn on top.

    seaborn's `barplot` + `stripplot` in ~15 lines, so this package needs no seaborn.
    The jitter is seeded: a figure that moves its points between renders cannot be
    compared to the one in the manuscript.
    """
    rng = np.random.default_rng(seed)
    x_pos = np.arange(len(labels))
    for x, lab in zip(x_pos, labels):
        vals = np.asarray(values_by_label[lab], dtype=float)
        vals = vals[~np.isnan(vals)]
        ax.bar(x, vals.mean(), width=0.8, color=palette[lab], alpha=bar_alpha,
               edgecolor='none', zorder=1)
        ax.plot(x + rng.uniform(-jitter, jitter, size=vals.size), vals, 'o',
                ms=np.sqrt(point_size), color=palette[lab], alpha=1.0,
                markeredgewidth=0, ls='none', zorder=3)
    ax.set(xticks=x_pos, xticklabels=labels)
    return x_pos


def grouped_bar_strip(ax, values, group_labels, series_labels, colors,
                      bar_alpha=0.5, point_size=10, jitter=0.12, seed=0):
    """Side-by-side bars for two series over a shared group axis (the layer profile).

    `values[series][group]` -> 1-D array of per-fold values.
    """
    rng = np.random.default_rng(seed)
    n_series = len(series_labels)
    width = 0.8 / n_series
    x_pos = np.arange(len(group_labels))
    for s_idx, series in enumerate(series_labels):
        offset = (s_idx - (n_series - 1) / 2) * width
        for g_idx in range(len(group_labels)):
            vals = np.asarray(values[series][g_idx], dtype=float)
            xc = x_pos[g_idx] + offset
            ax.bar(xc, vals.mean(), width=width * 0.9, color=colors[s_idx],
                   alpha=bar_alpha, edgecolor='none', zorder=1)
            ax.plot(xc + rng.uniform(-jitter, jitter, size=vals.size), vals, 'o',
                    ms=np.sqrt(point_size), color=colors[s_idx], alpha=1.0,
                    markeredgewidth=0, ls='none', zorder=3)
    ax.set(xticks=x_pos, xticklabels=group_labels)
    return x_pos


def scatter_sized_by_count(ax, x, y, base_s=10, line_color='r', suppress_zero=False,
                           **kwargs):
    """Scatter with marker area proportional to how many (x, y) pairs coincide.

    Confusion-matrix cells are small integers, so most points land on top of each
    other; a plain scatter shows a lattice and hides where the mass is. The regression
    line and its 95 % band are fitted to the *raw* pairs, not to the unique ones.

    `suppress_zero` scales against the second-largest count instead of the largest, so
    a single dominant pile-up (usually (0, 0)) does not shrink everything else away.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    unique_xy, counts = np.unique(np.column_stack([x, y]), axis=0, return_counts=True)
    if suppress_zero:
        scatter_size = np.clip(base_s * counts / np.sort(counts)[-2], None, base_s)
    else:
        scatter_size = base_s * counts / max(counts)

    ax.scatter(unique_xy[:, 0], unique_xy[:, 1], s=scatter_size, **kwargs)

    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    n = len(x)
    s_err = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci = t_crit * s_err * np.sqrt(1 / n + (x_line - x.mean()) ** 2
                                  / np.sum((x - x.mean()) ** 2))

    ax.plot(x_line, y_line, color=line_color, lw=2)
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.2, color=line_color)

    p_str = f'p={p_val:.1e}' if p_val >= 1e-4 else 'p<1e-4'
    ax.text(0.95, 0.05, f'r={r_val:.2f}\n{p_str}', transform=ax.transAxes,
            va='bottom', ha='right')
    return r_val, p_val


# --------------------------------------------------------------------------- #
# UMAP "best 10 categories" panel (ported from ESC50_plot_best10.py)
# --------------------------------------------------------------------------- #
def _draw_ellipse(ax, pts, color, n_std=N_STD, alpha_fill=0.18, lw=1.0):
    if len(pts) < 3:
        return
    cov = np.cov(pts.T)
    if np.linalg.matrix_rank(cov) < 2:
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    rgba = list(mcolors.to_rgba(color))
    rgba[3] = alpha_fill
    ell = Ellipse(
        xy=pts.mean(axis=0),
        width=2 * n_std * np.sqrt(eigvals[0]),
        height=2 * n_std * np.sqrt(eigvals[1]),
        angle=np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])),
        facecolor=rgba, edgecolor=None, linewidth=lw, zorder=2)
    ax.add_patch(ell)


def _draw_ellipse_outline(ax, pts, color, n_std=N_STD, lw=2.0):
    """Outline-only confidence ellipse -- used where filled ellipses would stack up."""
    if len(pts) < 3:
        return
    cov = np.cov(pts.T)
    if np.linalg.matrix_rank(cov) < 2:
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    ell = Ellipse(
        xy=pts.mean(axis=0),
        width=2 * n_std * np.sqrt(eigvals[0]),
        height=2 * n_std * np.sqrt(eigvals[1]),
        angle=np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])),
        facecolor='none', edgecolor=color, linewidth=lw, zorder=2)
    ax.add_patch(ell)


def _ellipse_boundary_point(pts, direction, n_std=N_STD):
    """The point on a cluster's confidence ellipse in *direction*, for label placement."""
    if len(pts) < 3:
        return pts.mean(axis=0)
    cov = np.cov(pts.T)
    if np.linalg.matrix_rank(cov) < 2:
        return pts.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    a = n_std * np.sqrt(eigvals[0])
    b = n_std * np.sqrt(eigvals[1])
    u_local = eigvecs.T @ direction                       # rotate into the ellipse frame
    denom = np.sqrt((u_local[0] / a) ** 2 + (u_local[1] / b) ** 2)
    if denom < 1e-10:
        return pts.mean(axis=0)
    return pts.mean(axis=0) + eigvecs @ (u_local / denom)  # and back


def _resolve_text_overlap(texts, cents, ax, n_iter=300, step_px=3.0, conn_thresh_px=6,
                          clamp_to_axes=True, draw_connectors=True):
    """Force-directed label repulsion using real rendered bboxes, not width guesses.

    `clamp_to_axes` pulls any label the repulsion pushed past the axes frame back
    inside it, which the repulsion alone will not do -- with ten labels in a crowded
    panel the outermost ones are always driven outwards.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_disp = ax.transData.transform
    to_data = ax.transData.inverted().transform

    for _ in range(n_iter):
        bbs = [t.get_window_extent(renderer).expanded(1.05, 1.10) for t in texts]
        moved = False
        for ii in range(len(texts)):
            for jj in range(ii + 1, len(texts)):
                if not bbs[ii].overlaps(bbs[jj]):
                    continue
                ci = bbs[ii].get_points().mean(axis=0)
                cj = bbs[jj].get_points().mean(axis=0)
                push = (ci - cj) / (np.linalg.norm(ci - cj) + 1e-6) * step_px
                texts[ii].set_position(
                    tuple(to_data(np.array(to_disp(texts[ii].get_position())) + push)))
                texts[jj].set_position(
                    tuple(to_data(np.array(to_disp(texts[jj].get_position())) - push)))
                moved = True
        if not moved:
            break

    if clamp_to_axes:
        fig.canvas.draw()
        ax_box = ax.get_window_extent(renderer)
        for txt in texts:
            bb = txt.get_window_extent(renderer)
            shift = np.zeros(2)
            if bb.x1 > ax_box.x1:
                shift[0] = ax_box.x1 - bb.x1
            if bb.x0 < ax_box.x0:
                shift[0] = ax_box.x0 - bb.x0
            if bb.y1 > ax_box.y1:
                shift[1] = ax_box.y1 - bb.y1
            if bb.y0 < ax_box.y0:
                shift[1] = ax_box.y0 - bb.y0
            if np.any(shift != 0):
                txt.set_position(
                    tuple(to_data(np.array(to_disp(txt.get_position())) + shift)))

    if not draw_connectors:
        return
    for txt, (cx, cy) in zip(texts, cents):
        tx, ty = txt.get_position()
        if np.linalg.norm(np.array(to_disp((tx, ty)))
                          - np.array(to_disp((cx, cy)))) > conn_thresh_px:
            ax.annotate('', xy=(cx, cy), xytext=(tx, ty),
                        arrowprops=dict(arrowstyle='-', color='#999999', lw=0.4, alpha=0.5),
                        zorder=9)


def plot_best10_panel(ax, x2d, y_labels, selection=V4_SELECTION):
    """UMAP scatter of every ESC-50 clip, with the 10 best-clustering categories drawn."""
    x2d = np.asarray(x2d)
    y_labels = np.asarray(y_labels)
    sel_colors = {cat: QUAL10(i) for i, cat in enumerate(selection)}

    ax.scatter(x2d[:, 0], x2d[:, 1], c='#e8e8e8', s=4, alpha=0.5, linewidths=0,
               rasterized=True)

    cents, pts_dict = {}, {}
    for cat in selection:
        pts = x2d[y_labels == cat]
        col = sel_colors[cat]
        ax.scatter(pts[:, 0], pts[:, 1], color=col, s=10, alpha=0.85, linewidths=0,
                   zorder=3, rasterized=True)
        _draw_ellipse(ax, pts, color=col)
        cents[cat] = pts.mean(axis=0)
        pts_dict[cat] = pts

    # Place each label just outside its ellipse, pointing away from the global centroid.
    cent_arr = np.array([cents[cat] for cat in selection])
    global_ctr = cent_arr.mean(axis=0)
    pad = 0.04 * (x2d.max(axis=0) - x2d.min(axis=0))

    text_pos = {}
    for cat in selection:
        direction = cents[cat] - global_ctr
        norm = np.linalg.norm(direction)
        unit = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        text_pos[cat] = _ellipse_boundary_point(pts_dict[cat], unit) + unit * pad

    texts = [ax.text(*text_pos[cat], cat, fontsize=ANNOT_FS, color=sel_colors[cat],
                     fontweight='bold', zorder=10,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
             for cat in selection]
    _resolve_text_overlap(texts, [cents[cat] for cat in selection], ax)
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2')


def category_groups(category_names):
    """Map every ESC-50 category name to its subgroup, from its `target` index."""
    return {name: ESC50_GROUPS[idx // 10] for idx, name in enumerate(category_names)}


def plot_subgroup_panel(ax, x2d, y_labels, group_categories, selection_colors,
                        selection_set, title, xlim, ylim, show_ylabel=False,
                        annot_fs=None):
    """One ESC-50 subgroup: every category in it, the two selected ones emphasised.

    Selected categories get a thick outline ellipse, full-alpha dots and a bold label;
    the rest get a thin ellipse, dim dots and a normal label. Both are drawn -- the
    point of the panel is that the selected pair separates *within* a subgroup whose
    other members overlap, which a panel showing only the pair could not make.
    """
    annot_fs = ANNOT_FS if annot_fs is None else annot_fs
    local_colors = {cat: QUAL10(i) for i, cat in enumerate(group_categories)}

    def _col(cat):
        return selection_colors[cat] if cat in selection_set else local_colors[cat]

    cents = {}
    # Non-selected first so the emphasised pair lands on top of it.
    for is_sel_pass in (False, True):
        for cat in group_categories:
            if (cat in selection_set) != is_sel_pass:
                continue
            pts = x2d[y_labels == cat]
            col = _col(cat)
            ax.scatter(pts[:, 0], pts[:, 1], color=col,
                       s=5 if is_sel_pass else 2,
                       alpha=0.85 if is_sel_pass else 0.30,
                       linewidths=0, zorder=3 if is_sel_pass else 1, rasterized=True)
            _draw_ellipse_outline(ax, pts, color=col, n_std=N_STD,
                                  lw=2.0 if is_sel_pass else 0.6)
            cents[cat] = pts.mean(axis=0)

    texts = [ax.text(*cents[cat], cat, fontsize=annot_fs, color=_col(cat),
                     fontweight='bold' if cat in selection_set else 'normal',
                     zorder=10, clip_on=False,
                     bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=0.5))
             for cat in group_categories]
    _resolve_text_overlap(texts, [cents[cat] for cat in group_categories], ax)

    ax.set(xlim=xlim, ylim=ylim, title=title, xlabel='UMAP 1',
           ylabel='UMAP 2' if show_ylabel else '')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(length=2, pad=3, labelleft=show_ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def bonferroni(pvals):
    return np.clip(np.asarray(pvals, dtype=float) * len(pvals), 0, 1)


def report_mean_sd(label, vals):
    vals = np.asarray(vals, dtype=float)
    print(f"  {label}: {np.nanmean(vals):.2f} +/- {np.nanstd(vals):.2f}  (n={vals.size})")


def wilcox_family(pairs):
    """Wilcoxon signed-rank over a family of paired comparisons, Bonferroni-corrected.

    `pairs` -> [(label_a, a, label_b, b, alternative), ...]. Correction is *within*
    the family, which is why the families are declared explicitly at the call site
    rather than every test being thrown into one pool.
    """
    raw = []
    for label_a, a, label_b, b, alt in pairs:
        stat, p = wilcoxon(a, b, alternative=alt)
        raw.append((label_a, label_b, alt, stat, p))
    p_adj = bonferroni([r[4] for r in raw])
    out = []
    for (label_a, label_b, alt, stat, p), pa in zip(raw, p_adj):
        sym = {'greater': '>', 'less': '<', 'two-sided': '!='}[alt]
        sig = '*' if pa < 0.05 else 'n.s.'
        print(f"  Wilcoxon {label_a} {sym} {label_b}: W={stat:.0f}, p={p:.4g}, "
              f"p_bonferroni={pa:.4g} {sig}")
        out.append((label_a, label_b, float(stat), float(p), float(pa)))
    return out


def layer_trend(layer_means):
    """Spearman correlation of layer index against mean accuracy."""
    r, p = spearmanr(np.arange(len(layer_means)), layer_means)
    return float(r), float(p)
