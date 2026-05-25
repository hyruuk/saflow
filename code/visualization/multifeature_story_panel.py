"""Composite multifeature classification story panel (single PNG, 5 tiers).

Tier A (top)    — Headline: joint AUC + significance, per-feature AUC bars
                   (family-colored, with significance markers).
Tier B          — Per-spatial map: brain surface (schaefer/atlas) or sensor
                   topomap of per-spatial multivariate AUC, with significance
                   contours.
Tier C          — Per-cell heatmap (parcels × features) with tmax/FDR overlay.
                   Rows sorted by per-spatial AUC (descending); columns
                   family-grouped.
Tier D          — Joint-axis importance heatmap (parcels × features) aligned
                   to the same row/column order as Tier C.
Tier E          — Per-feature mean importance bar (above Tier D's columns),
                   read as 'what the multivariate classifier leaned on per
                   feature, averaged across parcels'.

The panel reads a bundle file as produced by ``aggregate_multifeature.py``
(or by ``synthetic_mf_bundle.py`` for prototyping):

    feature-{label}_..._axis-bundle_..._mf_scores.npz

Usage (real bundle):
    python -m code.visualization.multifeature_story_panel \\
        --space schaefer_400 --trial-type alltrials

Usage (synthetic):
    python -m code.visualization.synthetic_mf_bundle --out /tmp/synth.npz
    python -m code.visualization.multifeature_story_panel \\
        --bundle /tmp/synth.npz --space schaefer_400 --trial-type alltrials
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec

from code.visualization.loaders import (
    BAND_ORDER, COMPLEXITY_ORDER, FOOOF_ORDER,
    family_sort_key, feature_family, short_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Family display labels and colors. Matches the project palette.
FAMILY_DISPLAY = {
    "fooof": "FOOOF",
    "psd": "PSD",
    "psd_corrected": "PSDc",
    "complexity": "Complexity",
}
FAMILY_COLORS = {
    "fooof": "#1f77b4",          # blue
    "psd": "#ff7f0e",            # orange
    "psd_corrected": "#d62728",  # red
    "complexity": "#2ca02c",     # green
    "other": "#7f7f7f",
}

CORRECTION_KEYS = {
    "tmax": "pvals_tmax",
    "fdr": "pvals_fdr_bh",
    "bonferroni": "pvals_bonferroni",
    "uncorrected": "pvals_uncorrected",
}


# ---------------------------------------------------------------------------
# Bundle discovery + loading
# ---------------------------------------------------------------------------

def _find_bundle(
    results_dir: Path,
    space: str,
    label: str,
    inout: str,
    clf: str,
    cv: str,
    importance: str,
    trial_type: str,
    analysis_level: str,
    inout_selection: str,
) -> Optional[Path]:
    """Locate a bundle npz by reconstructing its filename."""
    from code.classification.run_multifeature import build_mf_base_name

    base = build_mf_base_name(
        feature_label=label,
        space=space,
        inout_bounds=(int(inout[:2]), int(inout[2:])),
        clf_name=clf,
        cv_name=cv,
        axis="bundle",
        importance=importance,
        trial_type=trial_type,
        analysis_level=analysis_level,
        inout_selection=inout_selection,
    )
    p = results_dir / f"classification_{space}" / "group_mf" / f"{base}_scores.npz"
    return p if p.exists() else None


def _load_bundle(bundle_path: Path) -> Dict[str, np.ndarray]:
    with np.load(bundle_path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


# ---------------------------------------------------------------------------
# Sig helpers
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _joint_pvalue(bundle: Dict[str, np.ndarray]) -> float:
    """Recompute joint p-value from perm_scores (cheap, avoids reading sidecar)."""
    obs = float(bundle["joint/observed"])
    perm = bundle["joint/perm_scores"]
    return float((np.sum(perm >= obs) + 1) / (perm.size + 1))


# ---------------------------------------------------------------------------
# Tier A — headline + per-feature bars
# ---------------------------------------------------------------------------

def _render_headline(ax, bundle: Dict[str, np.ndarray]) -> None:
    """Joint AUC, n_sig cells, percent decoding effect-size summary."""
    joint_auc = float(bundle["joint/observed"])
    joint_p = _joint_pvalue(bundle)
    pc_pvals = bundle["per-cell/pvals_tmax"]
    n_sig_tmax = int(np.sum(pc_pvals < 0.05))
    n_total = pc_pvals.size
    pc_obs = bundle["per-cell/observed"]
    max_cell = float(np.nanmax(pc_obs))

    ax.axis("off")
    lines = [
        f"Joint AUC = {joint_auc:.3f} {_sig_stars(joint_p)}",
        f"p = {joint_p:.3g}  (n_perm = {bundle['joint/perm_scores'].size})",
        "",
        f"Per-cell tmax-significant: {n_sig_tmax}/{n_total} "
        f"({100 * n_sig_tmax / n_total:.1f}%)",
        f"Best per-cell AUC: {max_cell:.3f}",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=11, va="top", ha="left",
        family="monospace",
    )


def _feature_order(feature_names: List[str]) -> np.ndarray:
    """Return indices that sort features by family then in-family order."""
    return np.array(sorted(range(len(feature_names)),
                           key=lambda i: family_sort_key(feature_names[i])))


def _render_per_feature_bars(
    ax,
    bundle: Dict[str, np.ndarray],
    correction: str = "tmax",
    alpha: float = 0.05,
) -> None:
    """Per-feature multivariate AUC, family-colored, sorted family-grouped."""
    feature_names = list(bundle["feature_names"])
    pf_obs = bundle["per-feature/observed"]
    pf_p = bundle[f"per-feature/{CORRECTION_KEYS[correction]}"]
    order = _feature_order(feature_names)

    vals = pf_obs[order]
    labels = [short_label(feature_names[i]) for i in order]
    families = [feature_family(feature_names[i]) for i in order]
    colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"]) for f in families]
    sig = pf_p[order] < alpha

    xs = np.arange(len(order))
    bars = ax.bar(xs, vals - 0.5, bottom=0.5, color=colors, edgecolor="black",
                  linewidth=0.4, width=0.78)
    # Significance markers
    for i, s in enumerate(sig):
        if s:
            ax.text(i, vals[i] + 0.005, "*", ha="center", va="bottom",
                    fontsize=9, color="black")
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("per-feature AUC", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    pad = max(0.05, (np.nanmax(vals) - 0.5) * 1.4)
    ax.set_ylim(0.5 - pad * 0.2, 0.5 + pad)

    # Family legend swatches
    seen = []
    handles = []
    for f in families:
        if f in seen:
            continue
        seen.append(f)
        handles.append(mpatches.Patch(color=FAMILY_COLORS[f], label=FAMILY_DISPLAY.get(f, f)))
    ax.legend(handles=handles, fontsize=6, loc="upper right", frameon=False,
              ncol=2, columnspacing=0.6, handlelength=0.8)


# ---------------------------------------------------------------------------
# Tier B — per-spatial map
# ---------------------------------------------------------------------------

def _render_per_spatial_map(
    ax,
    bundle: Dict[str, np.ndarray],
    space: str,
    correction: str = "tmax",
    alpha: float = 0.05,
    fsaverage: Optional[dict] = None,
    sensor_info=None,
) -> None:
    """Brain surface (schaefer) or topomap (sensor) of per-spatial AUC."""
    ps_obs = np.asarray(bundle["per-spatial/observed"], dtype=float)
    ps_p = bundle[f"per-spatial/{CORRECTION_KEYS[correction]}"]
    spatial_names = list(bundle["spatial_names"])
    mask = ps_p < alpha
    vmax = float(np.nanmax(ps_obs))
    vmin = 0.5 - (vmax - 0.5)  # symmetric around 0.5
    cmap = "RdBu_r"
    n_sig = int(mask.sum())
    n_total = len(ps_obs)
    xlabel = (f"per-spatial AUC  |  {n_sig}/{n_total} {correction}-sig "
              f"(α={alpha})")

    if space == "schaefer_400" and fsaverage is not None:
        try:
            from code.visualization.stats_classif_panel import _plot_brain
            _plot_brain(
                ax, ps_obs, mask=mask, roi_names=spatial_names,
                atlas_name=space, fsaverage=fsaverage,
                vmin=vmin, vmax=vmax, cmap=cmap,
            )
            ax.set_xlabel(xlabel, fontsize=9)
            return
        except Exception as exc:
            logger.warning(f"Brain rendering failed ({exc}); falling back to ranked-strip view.")

    if space == "sensor" and sensor_info is not None:
        try:
            from code.visualization.stats_classif_panel import _plot_topomap
            _plot_topomap(
                ax, ps_obs, mask=mask, info=sensor_info,
                vmin=vmin, vmax=vmax, cmap=cmap,
            )
            ax.set_xlabel(xlabel, fontsize=9)
            return
        except Exception as exc:
            logger.warning(f"Topomap rendering failed ({exc}); falling back to ranked-strip view.")

    # Last-resort fallback: ranked horizontal strip when neither brain nor
    # topomap is available (e.g. fsaverage/info couldn't load).
    order = np.argsort(-ps_obs)
    ordered_vals = ps_obs[order]
    ordered_sig = mask[order] if mask is not None else None

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps[cmap]
    n = len(ordered_vals)
    xs = np.arange(n)
    ax.bar(xs, ordered_vals - 0.5, bottom=0.5,
           color=[cmap_obj(norm(v)) for v in ordered_vals], width=1.0,
           linewidth=0)
    if ordered_sig is not None:
        sig_xs = xs[ordered_sig]
        ax.scatter(sig_xs, np.full_like(sig_xs, vmax * 1.02, dtype=float),
                   marker="|", color="black", s=8)
    ax.axhline(0.5, color="gray", linewidth=0.4, linestyle=":")
    ax.set_xlim(-0.5, n - 0.5)
    pad = max(0.03, (vmax - 0.5) * 1.4)
    ax.set_ylim(0.5 - pad * 0.2, vmax * 1.05)
    ax.set_xlabel(
        f"spatial units sorted by AUC  |  {n_sig}/{n} {correction}-sig (α={alpha})",
        fontsize=9,
    )
    ax.set_ylabel("per-spatial AUC", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xticks([])


# ---------------------------------------------------------------------------
# Tiers C/D — per-cell + joint importance heatmaps
# ---------------------------------------------------------------------------

def _row_order(
    bundle: Dict[str, np.ndarray],
    strategy: str = "per_spatial_auc",
) -> np.ndarray:
    """Return spatial-row ordering. 'per_spatial_auc' = descending AUC."""
    if strategy == "per_spatial_auc":
        ps_obs = np.asarray(bundle["per-spatial/observed"], dtype=float)
        # Sort descending; NaNs go to bottom
        with np.errstate(invalid="ignore"):
            return np.argsort(np.where(np.isnan(ps_obs), -np.inf, ps_obs))[::-1]
    if strategy == "yeo":
        # Block-sort by Yeo network (parsed from name), falling back to original index
        names = list(bundle["spatial_names"])
        def _key(i):
            n = names[i]
            net = "ZZ"
            for token in ("Vis", "SomMot", "DorsAttn", "SalVentAttn",
                          "Limbic", "Cont", "Default"):
                if token in n:
                    net = token
                    break
            hemi = "LH" if "LH_" in n else ("RH" if "RH_" in n else "ZZ")
            return (net, hemi, n)
        return np.array(sorted(range(len(names)), key=_key))
    if strategy == "anatomical":
        return np.arange(len(bundle["spatial_names"]))
    raise ValueError(f"Unknown row-order strategy: {strategy}")


def _heatmap_ticks(
    ax,
    spatial_names: List[str],
    feature_names: List[str],
    row_order: np.ndarray,
    col_order: np.ndarray,
    show_row_labels: bool = False,
) -> None:
    """Cosmetic tick setup. Skip per-parcel labels in dense Schaefer mode."""
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(
        [short_label(feature_names[i]) for i in col_order],
        rotation=75, ha="right", fontsize=6,
    )
    if show_row_labels and len(row_order) <= 60:
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(
            [spatial_names[i] for i in row_order], fontsize=6,
        )
    else:
        ax.set_yticks([])


def _draw_family_separators(
    ax, feature_names: List[str], col_order: np.ndarray,
) -> None:
    """Vertical lines between feature-family blocks."""
    families = [feature_family(feature_names[i]) for i in col_order]
    for i in range(1, len(families)):
        if families[i] != families[i - 1]:
            ax.axvline(i - 0.5, color="white", linewidth=1.3)


def _render_per_cell_heatmap(
    ax,
    bundle: Dict[str, np.ndarray],
    row_order: np.ndarray,
    col_order: np.ndarray,
    correction: str = "tmax",
    alpha: float = 0.05,
) -> None:
    obs = bundle["per-cell/observed"]
    pvals = bundle[f"per-cell/{CORRECTION_KEYS[correction]}"]
    feature_names = list(bundle["feature_names"])
    spatial_names = list(bundle["spatial_names"])

    matrix = obs[np.ix_(row_order, col_order)]
    sig = (pvals[np.ix_(row_order, col_order)] < alpha)

    vmax = float(np.nanmax(matrix))
    vmin = 0.5 - (vmax - 0.5)
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdBu_r",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    # Significance overlay: black dots where sig.
    if sig.any():
        ys, xs = np.where(sig)
        # Show stippling only if reasonably sparse; otherwise skip dots and
        # rely on the colormap.
        if len(ys) < 0.25 * matrix.size:
            ax.scatter(xs, ys, s=3, color="black", marker=".", linewidths=0)
    _heatmap_ticks(ax, spatial_names, feature_names, row_order, col_order,
                   show_row_labels=False)
    _draw_family_separators(ax, feature_names, col_order)
    ax.set_ylabel(f"parcels (sorted by per-spatial AUC, n={len(row_order)})",
                  fontsize=8)
    return im


def _render_joint_importance(
    ax,
    bundle: Dict[str, np.ndarray],
    row_order: np.ndarray,
    col_order: np.ndarray,
) -> None:
    imp = np.asarray(bundle["joint/importances"], dtype=float)
    matrix = imp[np.ix_(row_order, col_order)]
    feature_names = list(bundle["feature_names"])
    spatial_names = list(bundle["spatial_names"])

    vmax = float(np.nanpercentile(np.abs(matrix), 99))
    vmin = -vmax
    im = ax.imshow(matrix, aspect="auto", cmap="PuOr_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    _heatmap_ticks(ax, spatial_names, feature_names, row_order, col_order,
                   show_row_labels=False)
    _draw_family_separators(ax, feature_names, col_order)
    ax.set_ylabel("parcels (same order)", fontsize=8)
    return im


def _render_per_feature_importance_bar(
    ax,
    bundle: Dict[str, np.ndarray],
    col_order: np.ndarray,
) -> None:
    imp = np.asarray(bundle["joint/importances"], dtype=float)
    feature_names = list(bundle["feature_names"])
    mean_imp = np.nanmean(imp, axis=0)[col_order]
    labels = [short_label(feature_names[i]) for i in col_order]
    families = [feature_family(feature_names[i]) for i in col_order]
    colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"]) for f in families]
    xs = np.arange(len(col_order))
    ax.bar(xs, mean_imp, color=colors, edgecolor="black", linewidth=0.4,
           width=0.78)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("mean joint importance\n(across parcels)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_panel(
    bundle: Dict[str, np.ndarray],
    space: str,
    trial_type: str,
    correction: str = "tmax",
    alpha: float = 0.05,
    row_order_strategy: str = "per_spatial_auc",
    title: Optional[str] = None,
    fsaverage: Optional[dict] = None,
    sensor_info=None,
) -> plt.Figure:
    feature_names = list(bundle["feature_names"])
    col_order = _feature_order(feature_names)
    row_order = _row_order(bundle, strategy=row_order_strategy)

    fig = plt.figure(figsize=(11.5, 20.5))
    # 6 rows: header strip, headline+bars, brain/topomap, per-cell, importance,
    # per-feature importance bar. Tier B gets ~3.6 ratio so the 2x2 brain
    # composite (LH/RH × lateral/medial) is not vertically crushed.
    gs = GridSpec(
        nrows=6, ncols=2,
        height_ratios=[0.18, 1.1, 3.6, 3.4, 3.4, 0.9],
        width_ratios=[1.0, 1.4],
        hspace=0.55, wspace=0.18,
        left=0.07, right=0.97, top=0.965, bottom=0.045,
    )

    # ---- title strip
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ttl = title or (
        f"Multifeature panel — space={space}  trial-type={trial_type}  "
        f"correction={correction}  α={alpha}"
    )
    ax_title.text(0.5, 0.5, ttl, ha="center", va="center",
                  fontsize=13, weight="bold", transform=ax_title.transAxes)

    # ---- Tier A: headline + per-feature bars
    ax_a1 = fig.add_subplot(gs[1, 0])
    _render_headline(ax_a1, bundle)
    ax_a1.text(-0.04, 1.05, "A", transform=ax_a1.transAxes,
               fontsize=14, weight="bold")

    ax_a2 = fig.add_subplot(gs[1, 1])
    _render_per_feature_bars(ax_a2, bundle, correction=correction, alpha=alpha)

    # ---- Tier B: per-spatial map
    ax_b = fig.add_subplot(gs[2, :])
    _render_per_spatial_map(ax_b, bundle, space=space,
                            correction=correction, alpha=alpha,
                            fsaverage=fsaverage, sensor_info=sensor_info)
    ax_b.text(-0.025, 1.02, "B", transform=ax_b.transAxes,
              fontsize=14, weight="bold")

    # ---- Tier C: per-cell heatmap
    ax_c = fig.add_subplot(gs[3, :])
    im_c = _render_per_cell_heatmap(
        ax_c, bundle, row_order, col_order,
        correction=correction, alpha=alpha,
    )
    ax_c.text(-0.025, 1.02, "C", transform=ax_c.transAxes,
              fontsize=14, weight="bold")
    cbar_c = fig.colorbar(im_c, ax=ax_c, shrink=0.7, pad=0.012, aspect=35)
    cbar_c.set_label("per-cell AUC", fontsize=8)
    cbar_c.ax.tick_params(labelsize=7)

    # ---- Tier D: joint importance heatmap
    ax_d = fig.add_subplot(gs[4, :])
    im_d = _render_joint_importance(ax_d, bundle, row_order, col_order)
    ax_d.text(-0.025, 1.02, "D", transform=ax_d.transAxes,
              fontsize=14, weight="bold")
    cbar_d = fig.colorbar(im_d, ax=ax_d, shrink=0.7, pad=0.012, aspect=35)
    cbar_d.set_label("joint importance", fontsize=8)
    cbar_d.ax.tick_params(labelsize=7)

    # ---- Tier E: per-feature mean importance bar
    ax_e = fig.add_subplot(gs[5, :])
    _render_per_feature_importance_bar(ax_e, bundle, col_order)
    ax_e.text(-0.025, 1.08, "E", transform=ax_e.transAxes,
              fontsize=14, weight="bold")

    return fig


def _load_fsaverage_if_needed(space: str):
    """Lazy-load fsaverage when we need brain surfaces."""
    if space != "schaefer_400":
        return None
    try:
        from code.visualization.plot_surface import _get_fsaverage_surfaces
        return _get_fsaverage_surfaces()
    except Exception as exc:
        logger.warning(f"Could not load fsaverage ({exc}); brain surface unavailable.")
        return None


def _load_sensor_info_if_needed(space: str, config_path: str):
    """Lazy-load a representative MEG info object for sensor topomaps."""
    if space != "sensor":
        return None
    try:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        data_root = Path(cfg["paths"]["data_root"])
        from code.visualization.render import _get_sensor_info
        return _get_sensor_info(cfg, data_root)
    except Exception as exc:
        logger.warning(f"Could not load sensor info ({exc}); topomap unavailable.")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bundle", default=None,
                        help="Explicit bundle npz (overrides discovery).")
    parser.add_argument("--space", required=True,
                        choices=["sensor", "schaefer_400"])
    parser.add_argument("--trial-type", default="alltrials")
    parser.add_argument("--label", default="combined-24",
                        help="Feature label baked into the bundle filename.")
    parser.add_argument("--inout", default="2575")
    parser.add_argument("--clf", default="logistic")
    parser.add_argument("--cv", default="logo")
    parser.add_argument("--importance", default="permutation")
    parser.add_argument("--analysis-level", default="epoch")
    parser.add_argument("--inout-selection", default="strict")
    parser.add_argument("--correction", default="tmax",
                        choices=list(CORRECTION_KEYS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--row-order", default="per_spatial_auc",
                        choices=["per_spatial_auc", "yeo", "anatomical"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-brain", action="store_true",
                        help="Skip fsaverage load; force fallback rendering for tier B.")
    args = parser.parse_args()

    if args.bundle:
        bundle_path = Path(args.bundle)
    else:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        data_root = Path(cfg["paths"]["data_root"])
        results_dir = data_root / cfg["paths"]["results"]
        bundle_path = _find_bundle(
            results_dir, args.space, args.label, args.inout,
            args.clf, args.cv, args.importance, args.trial_type,
            args.analysis_level, args.inout_selection,
        )
        if bundle_path is None:
            raise SystemExit(
                f"No bundle found for space={args.space} trial-type={args.trial_type} "
                f"label={args.label}. Run analysis.classify-multifeature first, or "
                f"pass --bundle <path>."
            )

    print(f"Loading bundle: {bundle_path}")
    bundle = _load_bundle(bundle_path)
    fsaverage = None if args.no_brain else _load_fsaverage_if_needed(args.space)
    sensor_info = None if args.no_brain else _load_sensor_info_if_needed(
        args.space, args.config)

    fig = render_panel(
        bundle=bundle,
        space=args.space,
        trial_type=args.trial_type,
        correction=args.correction,
        alpha=args.alpha,
        row_order_strategy=args.row_order,
        title=args.title,
        fsaverage=fsaverage,
        sensor_info=sensor_info,
    )

    if args.output is None:
        out_dir = Path("reports") / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = (
            out_dir
            / f"multifeature_story_space-{args.space}_type-{args.trial_type}.png"
        )
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote panel -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
