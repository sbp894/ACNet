"""
Manuscript Fig S1 -- the ESC-50 manifold, one panel per ESC-50 subgroup.

ESC-50 ships its 50 categories in five subgroups of ten (Animals, Natural Soundscape,
Human, Interior, Exterior). Each panel shows the UMAP embedding of ACNet's manifold
restricted to one subgroup: every category in it, with the two that separate best
drawn in bold -- thick outline ellipse, full-alpha points -- and the other eight left
thin and dim.

Drawing the whole subgroup rather than just the selected pair is the point. It is what
shows that the pair separates *within* a set of categories that otherwise overlap, and
it is the panel behind Fig3's ten-category summary scatter.

All five panels share the same axis limits, so the ellipses are comparable across
panels rather than each being auto-scaled to its own subgroup.

    python figs1.py           # -> figures/figs1.png (300 dpi)

Ported from `MS_AcxManifold/supFig_ESC50_best_clusters.py`. Everything is read from
`data/fig3.pkl.gz`; the three ESC-50 figures share one cache because they share every
input. This figure has no confusion matrices, so it writes a single PNG -- the
raw/log colour-scale option in `fig3.py` and `figs2.py` does not apply here.
"""

import gzip
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import esc50_lib as el                             # noqa: E402

PKL_PATH = os.path.join(HERE, 'data', 'fig3.pkl.gz')
OUT_PNG = os.path.join(HERE, 'figures', 'figs1.png')

AXIS_PAD_FRAC = 0.12           # room for the category labels inside the shared limits
ANNOT_FS = 7

FONT_SIZE = 8
plt.rcParams.update({
    'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1, 'ytick.labelsize': FONT_SIZE - 1,
    'font.size': FONT_SIZE,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
})

# Explicit cm geometry rather than constrained layout: the five panels must be exactly
# equal and square, and the bottom row of two has to sit centred under the top row of
# three. Every panel is derived from one width, and all the slack lives in the gaps.
CM_PER_IN = 2.54
FIG_W_CM = 9.0 * CM_PER_IN
MARGIN_L_CM = 0.55 * CM_PER_IN
MARGIN_R_CM = 0.12 * CM_PER_IN
MARGIN_T_CM = 0.15 * CM_PER_IN
MARGIN_B_CM = 0.55 * CM_PER_IN
GAP_W_CM = 0.10 * CM_PER_IN    # inner panels carry no y tick labels, so this can be tight
GAP_H_CM = 0.55 * CM_PER_IN    # title headroom + the x label below the top row


def load_cache():
    with gzip.open(PKL_PATH, 'rb') as fh:
        return pickle.load(fh)


def panel_geometry():
    """Axes rectangles in figure fractions, from the cm constants above."""
    panel_w = (FIG_W_CM - MARGIN_L_CM - MARGIN_R_CM - 2 * GAP_W_CM) / 3
    panel_h = panel_w                                        # square panels
    fig_h_cm = MARGIN_T_CM + 2 * panel_h + GAP_H_CM + MARGIN_B_CM

    def fx(x):
        return x / FIG_W_CM

    def fy(y):
        return y / fig_h_cm

    row_top_y = fy(MARGIN_B_CM + panel_h + GAP_H_CM)
    row_bot_y = fy(MARGIN_B_CM)
    top_xs = [fx(MARGIN_L_CM + i * (panel_w + GAP_W_CM)) for i in range(3)]
    # The bottom row holds two panels; centre them rather than left-aligning, so the
    # block reads as one figure instead of a ragged grid.
    bot_start = (FIG_W_CM - (2 * panel_w + GAP_W_CM)) / 2
    bot_xs = [fx(bot_start), fx(bot_start + panel_w + GAP_W_CM)]

    rects = [(x, row_top_y, fx(panel_w), fy(panel_h)) for x in top_xs]
    rects += [(x, row_bot_y, fx(panel_w), fy(panel_h)) for x in bot_xs]
    return rects, fig_h_cm


def assert_equal_panels(fig, axes, label):
    fig.canvas.draw()
    sizes = np.array([[ax.get_window_extent().width, ax.get_window_extent().height]
                      for ax in axes])
    assert np.allclose(sizes, sizes[0], atol=1.0), (
        f"{label}: panels differ in size\n{sizes}")


def draw(cache):
    umap = cache['umap']
    x2d = np.asarray(umap['X2d'])
    y_labels = np.asarray(umap['y_labels'])
    names = cache['category_names']
    assert names is not None, (
        "the cache has no ESC-50 category names, so categories cannot be assigned to "
        "subgroups -- rebuild with build/build_fig3_cache.py on a host that can read "
        "ESC-50-master/meta/esc50.csv")

    cat_to_group = el.category_groups(names)
    selection = list(umap['selection'])
    selection_set = set(selection)
    selection_colors = {cat: el.QUAL10(i) for i, cat in enumerate(selection)}

    # Shared limits, padded so the labels stay inside the frames.
    pad = AXIS_PAD_FRAC * (x2d.max(axis=0) - x2d.min(axis=0))
    xlim = (x2d[:, 0].min() - pad[0], x2d[:, 0].max() + pad[0])
    ylim = (x2d[:, 1].min() - pad[1], x2d[:, 1].max() + pad[1])

    rects, fig_h_cm = panel_geometry()
    fig = plt.figure(figsize=(FIG_W_CM / CM_PER_IN, fig_h_cm / CM_PER_IN))
    axes = [fig.add_axes(rect) for rect in rects]

    # y labels on the leftmost panel of each row; the rows are 3 and 2 panels wide.
    show_ylabel_idx = {0, 3}
    for idx, (ax, group) in enumerate(zip(axes, el.ESC50_GROUPS)):
        group_categories = [c for c in names if cat_to_group[c] == group]
        assert len(group_categories) == 10, (
            f"{group}: {len(group_categories)} categories, expected 10")
        el.plot_subgroup_panel(ax, x2d, y_labels, group_categories, selection_colors,
                               selection_set, group, xlim, ylim,
                               show_ylabel=(idx in show_ylabel_idx), annot_fs=ANNOT_FS)

    assert_equal_panels(fig, axes, 'subgroup panels')

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return cat_to_group


def print_stats(cache, cat_to_group):
    umap = cache['umap']
    print(f"\nUMAP: n_neighbors={umap['nn']}, min_dist={umap['md']:.2f}, "
          f"separation score={umap['score']:.4f}  (cached, not recomputed)")
    selection = list(umap['selection'])
    print("emphasised pair per subgroup (v4 max-min selection):")
    for group in el.ESC50_GROUPS:
        picked = [c for c in selection if cat_to_group[c] == group]
        print(f"  {group:20s} {', '.join(picked)}")


def main():
    cache = load_cache()
    print(f"cache built {cache['meta']['built']} from {cache['meta']['source_script']}")
    cat_to_group = draw(cache)
    print_stats(cache, cat_to_group)


if __name__ == '__main__':
    main()
