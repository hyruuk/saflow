"""Composite Yeo-network story panel (single PNG, 4 tiers).

Tier 1 — Whole-brain maps        (per-parcel t-values, AUC; Yeo outline overlay)
Tier 2 — Network-aggregated      (mean t per network + signed_count_sig)
Tier 3 — Network classification  (per-family bars + per-feature heatmap)
Tier 4 — Multivariate importance + coherence

Reads:
  - results/statistics_<space>/feature-*_..._type-<trial>_results.npz
  - results/statistics_<space>/group/networks/stats-networks_yeo{N}_type-*_correction-*.npz
  - results/statistics_<space>/group/networks/coherence_feature-*_yeo{N}_type-*.npz
  - results/classification_<space>/per-spatial mf score npz files
  - results/classification_<space>/group_mf/networks/classif-networks_yeo{N}_scope-*_type-*.npz
  - results/classification_<space>/group_mf/networks/importance-networks_yeo{N}_type-*.npz

Each tier is rendered independently and degrades to a "no data" placeholder if
its inputs are missing — useful while the upstream stages are still running.

Output:
    reports/figures/network_story_space-<space>_yeo{N}_type-<trial>.png

Usage:
    python -m code.visualization.network_story_panel \\
        --space schaefer_400 --trial-type correct --yeo 7
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from code.utils.yeo_networks import (
    YEO7_FULL_NAMES,
    network_order,
    network_palette,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Five features in the order they appear as columns in Tiers 1, 2, and the
# coherence row of Tier 4. Mirrors run_network_classification's
# DEFAULT_PER_FEATURE so the figures align across tiers.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "psd_corrected_alpha",
    "psd_corrected_theta",
    "fooof_exponent",
    "fooof_offset",
    "complexity_lzc_median",
)

FEATURE_DISPLAY: Dict[str, str] = {
    "psd_corrected_alpha":   r"PSDc $\alpha$ (8–12 Hz)",
    "psd_corrected_theta":   r"PSDc $\theta$ (4–8 Hz)",
    "fooof_exponent":        "FOOOF exponent",
    "fooof_offset":          "FOOOF offset",
    "complexity_lzc_median": "LZc (median)",
}

# Families for Tier-3 per-family bars (matches DEFAULT_FAMILIES).
FAMILY_COLUMNS: Tuple[str, ...] = ("psds_corrected", "fooof", "complexity")
FAMILY_DISPLAY: Dict[str, str] = {
    "psds_corrected": "PSDc (all bands)",
    "fooof":          "FOOOF (exp + off + R²)",
    "complexity":     "Complexity (LZc + entropy + fractal)",
}

# Per-correction → npz key fallbacks. Same as aggregate_networks.
PVAL_KEYS: Dict[str, Tuple[str, ...]] = {
    "fdr":         ("pvals_fdr_bh", "pvals_corrected_fdr_bh", "pvals_corrected_fdr"),
    "tmax":        ("pvals_tmax", "pvals_corrected_tmax"),
    "bonferroni":  ("pvals_bonferroni", "pvals_corrected_bonferroni"),
    "uncorrected": ("pvals_uncorrected", "pvals"),
}


# ---------------------------------------------------------------------------
# Loaders (all return None on missing inputs so panels degrade gracefully)
# ---------------------------------------------------------------------------

def _pick(npz: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> Optional[np.ndarray]:
    for k in candidates:
        if k in npz.files:
            return np.asarray(npz[k])
    return None


_STATS_LEVEL_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "average": (
        "feature-{feature}_inout-{inout}{sel}_test-paired_ttest"
        "_level-average_type-{trial}_results.npz",
        # legacy pre-`level` token
        "feature-{feature}_inout-{inout}{sel}_test-paired_ttest"
        "_path-subj-spectrum_type-{trial}_results.npz",
    ),
    "epoch": (
        "feature-{feature}_inout-{inout}{sel}_test-paired_ttest"
        "_level-epoch_type-{trial}_results.npz",
        # legacy pre-`level` token
        "feature-{feature}_inout-{inout}{sel}_test-paired_ttest"
        "_path-subj-trial-*_type-{trial}_results.npz",
    ),
}


def _find_stats_file(stats_dir: Path, feature: str, trial: str,
                     inout: str, level: str = "average",
                     inout_selection: str = "strict") -> Optional[Path]:
    """Locate one per-parcel stats results.npz for (feature, trial, level)."""
    sel_tok = "" if inout_selection == "strict" else f"_sel-{inout_selection}"
    pats = _STATS_LEVEL_PATTERNS.get(level, _STATS_LEVEL_PATTERNS["average"])
    for pat in pats:
        hits = sorted(stats_dir.glob(
            pat.format(feature=feature, inout=inout, sel=sel_tok, trial=trial)))
        if hits:
            return hits[0]
    return None


def _find_classif_file(clf_group_dir: Path, feature: str, trial: str,
                       inout: str, clf: str, cv: str, space: str,
                       level: str = "epoch",
                       inout_selection: str = "strict") -> Optional[Path]:
    """Locate one per-spatial univariate classification scores npz.

    ``clf_group_dir`` must already be ``classification_<space>/group/`` —
    per-spatial univariate scores live there, not in the parent dir.
    """
    sel_tok = "" if inout_selection == "strict" else f"_sel-{inout_selection}"
    pats = [
        f"feature-{feature}_space-{space}_inout-{inout}{sel_tok}"
        f"_clf-{clf}_cv-{cv}_mode-univariate_level-{level}"
        f"_type-{trial}_scores.npz",
        # legacy pre-`level` token
        f"feature-{feature}_space-{space}_inout-{inout}{sel_tok}"
        f"_clf-{clf}_cv-{cv}_mode-univariate_type-{trial}_scores.npz",
    ]
    for pat in pats:
        hits = sorted(clf_group_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _load_stats_per_parcel(path: Path, correction: str
                           ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=True) as npz:
        tvals = np.asarray(npz["tvals"]).flatten()
        pvals = _pick(npz, PVAL_KEYS[correction])
    pvals = pvals.flatten() if pvals is not None else None
    return tvals, pvals


def _load_classif_per_parcel(path: Path, correction: str
                             ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=True) as npz:
        scores = np.asarray(npz["observed"]).flatten()
        pvals = _pick(npz, PVAL_KEYS[correction])
    pvals = pvals.flatten() if pvals is not None else None
    return scores, pvals


def _load_network_stats(net_dir: Path, trial: str, correction: str,
                        n_networks: int) -> Optional[Dict[str, np.ndarray]]:
    path = (net_dir
            / f"stats-networks_yeo{n_networks}"
              f"_type-{trial}_correction-{correction}.npz")
    if not path.exists():
        logger.info(f"network-stats not found: {path}")
        return None
    with np.load(path, allow_pickle=True) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files if k != "meta"}


def _load_coherence(net_dir: Path, feature: str, trial: str,
                    n_networks: int) -> Optional[Dict[str, np.ndarray]]:
    path = (net_dir
            / f"coherence_feature-{feature}_"
              f"yeo{n_networks}_type-{trial}.npz")
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files if k != "meta"}


def _load_net_classif(net_dir: Path, scope: str, trial: str, clf: str,
                      cv: str, n_networks: int
                      ) -> Optional[Dict[str, np.ndarray]]:
    path = (net_dir
            / f"classif-networks_yeo{n_networks}_scope-{scope}_"
              f"type-{trial}_clf-{clf}_cv-{cv}.npz")
    if not path.exists():
        logger.info(f"net-classif not found: {path}")
        return None
    with np.load(path, allow_pickle=True) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files if k != "meta"}


def _load_net_importance(net_dir: Path, trial: str, clf: str, cv: str,
                         label: str, n_networks: int
                         ) -> Optional[Dict[str, np.ndarray]]:
    path = (net_dir
            / f"importance-networks_yeo{n_networks}_label-{label}_"
              f"clf-{clf}_cv-{cv}_type-{trial}.npz")
    if not path.exists():
        logger.info(f"net-importance not found: {path}")
        return None
    with np.load(path, allow_pickle=True) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files if k != "meta"}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _placeholder(ax, text: str = "no data") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=8, color="gray")
    ax.set_axis_off()


def _draw_network_bar(ax, values: np.ndarray, networks: Sequence[str],
                      palette: Dict[str, str], title: str,
                      sig_mask: Optional[np.ndarray] = None,
                      ylabel: str = "", axhline: Optional[float] = 0.0,
                      ylim: Optional[Tuple[float, float]] = None) -> None:
    if not np.isfinite(values).any():
        _placeholder(ax, "no data")
        return
    colors = [palette.get(n, "#888888") for n in networks]
    xs = np.arange(len(networks))
    ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.5)
    if axhline is not None:
        ax.axhline(axhline, color="black", linewidth=0.6, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(networks, rotation=35, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if sig_mask is not None:
        ymin, ymax = ax.get_ylim()
        offset = 0.04 * (ymax - ymin)
        for x, v, s in zip(xs, values, sig_mask):
            if s and np.isfinite(v):
                y = v + (offset if v >= 0 else -offset)
                ax.text(x, y, "*", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=10, fontweight="bold")


def _draw_grouped_within_between(ax, within: np.ndarray, between: np.ndarray,
                                 networks: Sequence[str],
                                 palette: Dict[str, str], title: str) -> None:
    if not (np.isfinite(within).any() or np.isfinite(between).any()):
        _placeholder(ax)
        return
    xs = np.arange(len(networks))
    w = 0.38
    colors = [palette.get(n, "#888888") for n in networks]
    ax.bar(xs - w / 2, within, width=w, color=colors,
           edgecolor="black", linewidth=0.5, label="within")
    ax.bar(xs + w / 2, between, width=w,
           color=[mcolors.to_rgba(c, alpha=0.45) for c in colors],
           edgecolor="black", linewidth=0.5, label="between")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(networks, rotation=35, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="best")


def _draw_heatmap(ax, M: np.ndarray, row_labels: Sequence[str],
                  col_labels: Sequence[str], title: str,
                  cmap: str = "magma", vmin: Optional[float] = None,
                  vmax: Optional[float] = None, cbar_label: str = "",
                  sig_mask: Optional[np.ndarray] = None) -> None:
    if not np.isfinite(M).any():
        _placeholder(ax)
        return
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    if sig_mask is not None:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if sig_mask[i, j]:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=10, fontweight="bold", color="white")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)


# ---------------------------------------------------------------------------
# Tier renderers
# ---------------------------------------------------------------------------

def _render_brain_into(ax, values: np.ndarray, mask: Optional[np.ndarray],
                       roi_names: List[str], atlas_name: str, fsaverage,
                       vmin: float, vmax: float, cmap: str,
                       yeo_overlay: Optional[int]) -> None:
    """Render the 2x2 brain composite for a single feature into ``ax``.

    If ``mask`` zeroes out every parcel (no significant ROIs), the inflated
    surface is still drawn — just with no colored overlay — so the panel
    keeps its slot instead of degrading to a "no data" placeholder. The
    placeholder only fires when ``values`` is truly missing (handled by
    the caller).
    """
    from PIL import Image

    from code.visualization.plot_surface import (
        render_inflated_view,
        roi_to_surface,
    )

    vals = np.asarray(values, dtype=float).copy()
    if mask is not None:
        vals[~np.asarray(mask, dtype=bool)] = np.nan
    # NB: previously we returned a placeholder here when nothing was
    # significant. Keep drawing the brain instead — `plot_surf_stat_map`
    # treats all-NaN as "no overlay" and renders just the sulci surface,
    # which is exactly what we want for "no parcels significant".

    lh, rh = roi_to_surface(vals, roi_names, atlas_name)
    images = {}
    for hemi, view in (("left", "lateral"), ("right", "lateral"),
                       ("left", "medial"),  ("right", "medial")):
        images[(hemi, view)] = render_inflated_view(
            lh, rh, hemi, view, fsaverage, cmap=cmap, vmin=vmin, vmax=vmax,
            yeo_overlay=yeo_overlay, yeo_atlas_name=atlas_name,
        )
    target_h = max(img.shape[0] for img in images.values())
    resized = {}
    for k, img in images.items():
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            nw = int(img.shape[1] * scale)
            resized[k] = np.asarray(
                Image.fromarray(img).resize((nw, target_h), Image.LANCZOS)
            )
        else:
            resized[k] = img
    top = np.concatenate(
        [resized[("left", "lateral")], resized[("right", "lateral")]], axis=1)
    bot = np.concatenate(
        [resized[("left", "medial")],  resized[("right", "medial")]],  axis=1)
    mw = max(top.shape[1], bot.shape[1])
    if top.shape[1] < mw:
        top = np.concatenate(
            [top, np.full((top.shape[0], mw - top.shape[1], 3), 255,
                          dtype=np.uint8)], axis=1)
    if bot.shape[1] < mw:
        bot = np.concatenate(
            [bot, np.full((bot.shape[0], mw - bot.shape[1], 3), 255,
                          dtype=np.uint8)], axis=1)
    composite = np.concatenate([top, bot], axis=0)
    # aspect="equal" preserves the natural 2:1-ish aspect of the 2x2 brain
    # composite so the cortical surface isn't squished vertically inside
    # near-square axes; matplotlib letterboxes the spare height/width.
    ax.imshow(composite, interpolation="bilinear", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _label_with_n_sig(label: str, mask: Optional[np.ndarray],
                      data_available: bool) -> str:
    """Mirror stats_classif_panel's topomap label: ``<label>\n(n=X sig)``.

    ``data_available=False`` is the "no input" case → just the label
    (the axes itself has been replaced by a placeholder).
    """
    if not data_available:
        return label
    if mask is None:
        return f"{label}\n(n/a sig)"
    return f"{label}\n(n={int(np.sum(mask))} sig)"


def _tier1_brains(gs, fig, stats_dir: Path, clf_group_dir: Path,
                  space: str, trial: str, inout: str, clf: str,
                  classif_cv: str, features: Sequence[str], alpha: float,
                  stats_level: str, stats_correction: str,
                  classif_level: str, classif_correction: str,
                  yeo_overlay: Optional[int],
                  inout_selection: str = "strict") -> None:
    """Tier 1 — two rows of 2x2 brain composites + colorbars.

    Row A (stats) and Row B (classif) use independent (level, correction)
    settings so the panel can show e.g. ``average / FDR`` per-parcel
    t-values together with ``single-epoch / tmax`` per-parcel AUCs.
    """
    from code.visualization.plot_surface import (
        _get_fsaverage_surfaces,
        _get_roi_names,
    )

    n_feat = len(features)
    # Inner grid: 2 rows (t, AUC) × (n_feat + 1) cols (features + colorbar).
    # Bias height toward the brain composites so they aren't squeezed
    # vertically by the xlabel + spacing budget.
    inner = GridSpecFromSubplotSpec(
        2, n_feat + 1, subplot_spec=gs,
        width_ratios=[1.0] * n_feat + [0.06],
        height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.04,
    )

    fsaverage = _get_fsaverage_surfaces()
    roi_names = _get_roi_names(space, [], [])

    t_collect: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
    auc_collect: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
    for feat in features:
        sp = _find_stats_file(stats_dir, feat, trial, inout,
                              level=stats_level,
                              inout_selection=inout_selection)
        if sp is None:
            logger.info(f"  tier1 stats not found for {feat} "
                        f"(level={stats_level})")
            t_collect.append((None, None))
        else:
            try:
                t_collect.append(_load_stats_per_parcel(sp, stats_correction))
            except Exception as exc:
                logger.warning(f"  tier1 t-load failed for {feat}: {exc}")
                t_collect.append((None, None))

        cp = _find_classif_file(clf_group_dir, feat, trial, inout, clf,
                                classif_cv, space, level=classif_level,
                                inout_selection=inout_selection)
        if cp is None:
            logger.info(f"  tier1 classif not found for {feat} "
                        f"(level={classif_level}, cv={classif_cv})")
            auc_collect.append((None, None))
        else:
            try:
                auc_collect.append(
                    _load_classif_per_parcel(cp, classif_correction))
            except Exception as exc:
                logger.warning(f"  tier1 auc-load failed for {feat}: {exc}")
                auc_collect.append((None, None))

    # Global vlims: symmetric for t (diverging), [0.5, max] for AUC.
    finite_t = [t[0] for t in t_collect if t[0] is not None]
    if finite_t:
        all_t = np.concatenate([t.flatten() for t in finite_t])
        tmax = float(np.nanpercentile(np.abs(all_t), 98)) if all_t.size else 3.0
    else:
        tmax = 3.0
    finite_auc = [a[0] for a in auc_collect if a[0] is not None]
    if finite_auc:
        all_auc = np.concatenate([a.flatten() for a in finite_auc])
        auc_lo = 0.5
        auc_hi = float(np.nanpercentile(all_auc, 98))
        if not np.isfinite(auc_hi) or auc_hi <= auc_lo:
            auc_hi = auc_lo + 0.1
    else:
        auc_lo, auc_hi = 0.5, 0.7

    row_a_label = (f"Tier 1A · per-parcel t (OUT − IN)\n"
                   f"{stats_level} / {stats_correction}")
    row_b_label = (f"Tier 1B · per-parcel AUC\n"
                   f"{classif_level} / {classif_correction}")

    for j, feat in enumerate(features):
        # ---- row 0: t-values (stats) ----
        ax_t = fig.add_subplot(inner[0, j])
        vals_t, pvals_t = t_collect[j]
        mask_t = ((pvals_t < alpha) if pvals_t is not None
                  else None) if vals_t is not None else None
        if vals_t is None:
            _placeholder(ax_t)
        else:
            _render_brain_into(
                ax_t, vals_t, mask_t, roi_names, space, fsaverage,
                vmin=-tmax, vmax=tmax, cmap="RdBu_r",
                yeo_overlay=yeo_overlay,
            )
        ax_t.set_xlabel(
            _label_with_n_sig(FEATURE_DISPLAY.get(feat, feat), mask_t,
                              data_available=vals_t is not None),
            fontsize=8.5, labelpad=2,
        )
        if j == 0:
            ax_t.set_ylabel(row_a_label, fontsize=9)

        # ---- row 1: AUC (classif) ----
        ax_a = fig.add_subplot(inner[1, j])
        vals_a, pvals_a = auc_collect[j]
        mask_a = ((pvals_a < alpha) if pvals_a is not None
                  else None) if vals_a is not None else None
        if vals_a is None:
            _placeholder(ax_a)
        else:
            _render_brain_into(
                ax_a, vals_a, mask_a, roi_names, space, fsaverage,
                vmin=auc_lo, vmax=auc_hi, cmap="magma",
                yeo_overlay=yeo_overlay,
            )
        ax_a.set_xlabel(
            _label_with_n_sig(FEATURE_DISPLAY.get(feat, feat), mask_a,
                              data_available=vals_a is not None),
            fontsize=8.5, labelpad=2,
        )
        if j == 0:
            ax_a.set_ylabel(row_b_label, fontsize=9)

    # Colorbars (one per row in the last column)
    cax_t = fig.add_subplot(inner[0, -1])
    norm_t = mcolors.Normalize(vmin=-tmax, vmax=tmax)
    sm_t = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_t)
    sm_t.set_array([])
    fig.colorbar(sm_t, cax=cax_t).set_label("t-value", fontsize=8)
    cax_t.tick_params(labelsize=7)

    cax_a = fig.add_subplot(inner[1, -1])
    norm_a = mcolors.Normalize(vmin=auc_lo, vmax=auc_hi)
    sm_a = plt.cm.ScalarMappable(cmap="magma", norm=norm_a)
    sm_a.set_array([])
    fig.colorbar(sm_a, cax=cax_a).set_label("AUC", fontsize=8)
    cax_a.tick_params(labelsize=7)


def _tier2_network_bars(gs, fig, net_dir: Path, trial: str,
                        correction: str, features: Sequence[str],
                        n_networks: int, alpha: float) -> None:
    """Tier 2 — mean-t bars (top row) + signed_count_sig bars (bottom row)."""
    bundle = _load_network_stats(net_dir, trial, correction, n_networks)
    n_feat = len(features)
    inner = GridSpecFromSubplotSpec(
        2, n_feat, subplot_spec=gs, hspace=0.65, wspace=0.30,
    )

    palette = network_palette(n_networks)
    nets = network_order(n_networks)

    if bundle is None:
        # Render n_feat × 2 placeholders so the slot keeps its space
        for i in range(2):
            for j in range(n_feat):
                ax = fig.add_subplot(inner[i, j])
                _placeholder(ax, "no network stats")
        return

    feat_idx = {f: i for i, f in enumerate(bundle["features"].tolist())}

    for j, feat in enumerate(features):
        col = feat_idx.get(feat)
        # Row 0 — mean t per network
        ax = fig.add_subplot(inner[0, j])
        if col is None:
            _placeholder(ax, "feature absent")
        else:
            t_per_net = bundle["tvals_mean"][col]
            p_per_net = bundle["pooled_p_stouffer"][col]
            sig = np.where(np.isfinite(p_per_net), p_per_net < alpha, False)
            title = FEATURE_DISPLAY.get(feat, feat)
            _draw_network_bar(ax, t_per_net, nets, palette,
                              title=title if j == 0 else FEATURE_DISPLAY.get(feat, feat),
                              ylabel="mean t" if j == 0 else "",
                              sig_mask=sig, axhline=0.0)
        # Row 1 — signed_count_sig (counts; can be negative)
        ax = fig.add_subplot(inner[1, j])
        if col is None:
            _placeholder(ax, "feature absent")
        else:
            sc = bundle["signed_count_sig"][col]
            _draw_network_bar(ax, sc, nets, palette,
                              title="",
                              ylabel="signed Σ sig parcels" if j == 0 else "",
                              axhline=0.0)


def _tier3_classif(gs, fig, net_dir: Path, trial: str, clf: str, cv: str,
                   n_networks: int, families: Sequence[str],
                   per_feature_subset: Sequence[str], alpha: float) -> None:
    """Tier 3 — per-family bars (row 0) + per-feature heatmap (row 1)."""
    pfam = _load_net_classif(net_dir, "per-family", trial, clf, cv, n_networks)
    pfeat = _load_net_classif(net_dir, "per-feature", trial, clf, cv, n_networks)
    palette = network_palette(n_networks)
    nets = network_order(n_networks)

    inner = GridSpecFromSubplotSpec(
        2, len(families), subplot_spec=gs,
        height_ratios=[1.0, 1.2], hspace=0.55, wspace=0.30,
    )

    # ---- Row 0: per-family bars
    if pfam is None:
        for j in range(len(families)):
            _placeholder(fig.add_subplot(inner[0, j]), "no per-family classif")
    else:
        fam_idx = {f: i for i, f in enumerate(pfam["families"].tolist())}
        for j, fam in enumerate(families):
            ax = fig.add_subplot(inner[0, j])
            col = fam_idx.get(fam)
            if col is None:
                _placeholder(ax, "family absent")
                continue
            scores = pfam["scores"][:, col]
            pvals = pfam["pvals"][:, col]
            sig = np.where(np.isfinite(pvals), pvals < alpha, False)
            _draw_network_bar(
                ax, scores, nets, palette,
                title=FAMILY_DISPLAY.get(fam, fam),
                ylabel="AUC" if j == 0 else "",
                sig_mask=sig, axhline=0.5,
                ylim=(0.4, max(0.7, float(np.nanmax(scores)) + 0.02)
                      if np.isfinite(scores).any() else (0.4, 0.7)),
            )

    # ---- Row 1: per-feature heatmap spanning the full width
    ax_hm = fig.add_subplot(inner[1, :])
    if pfeat is None:
        _placeholder(ax_hm, "no per-feature classif")
    else:
        feats = pfeat["features"].tolist()
        # Keep only the user-requested subset, in that order
        col_order = [feats.index(f) for f in per_feature_subset if f in feats]
        used_features = [feats[c] for c in col_order]
        M = pfeat["scores"][:, col_order] if col_order else pfeat["scores"]
        P = pfeat["pvals"][:, col_order] if col_order else pfeat["pvals"]
        sig = np.where(np.isfinite(P), P < alpha, False)
        col_labels = [FEATURE_DISPLAY.get(f, f) for f in used_features]
        vmax = (max(0.7, float(np.nanmax(M)))
                if np.isfinite(M).any() else 0.7)
        _draw_heatmap(
            ax_hm, M, row_labels=list(nets), col_labels=col_labels,
            title="per-feature × network AUC",
            cmap="magma", vmin=0.45, vmax=vmax, cbar_label="AUC",
            sig_mask=sig,
        )


def _tier4_importance_coherence(gs, fig, classif_net_dir: Path,
                                stats_net_dir: Path, trial: str, clf: str,
                                cv: str, label: str, n_networks: int,
                                features: Sequence[str]) -> None:
    """Tier 4 — joint-importance heatmap (row 0) + coherence bars (row 1)."""
    imp = _load_net_importance(classif_net_dir, trial, clf, cv, label, n_networks)
    inner = GridSpecFromSubplotSpec(
        2, len(features), subplot_spec=gs,
        height_ratios=[1.2, 1.0], hspace=0.55, wspace=0.30,
    )

    # ---- Row 0: importance heatmap spanning full width
    ax_hm = fig.add_subplot(inner[0, :])
    if imp is None:
        _placeholder(ax_hm, "no joint-importance bundle")
    else:
        feats_in_bundle = imp["features"].tolist()
        M = imp["importance_sum"]
        # Drop near-zero columns for readability if too many features
        col_keep = np.arange(M.shape[1])
        if M.shape[1] > 16:
            col_means = np.nanmean(np.abs(M), axis=0)
            keep_ix = np.argsort(col_means)[-16:]
            col_keep = np.sort(keep_ix)
        M_show = M[:, col_keep]
        col_labels = [feats_in_bundle[c] for c in col_keep]
        _draw_heatmap(
            ax_hm, M_show, row_labels=imp["network_names"].tolist(),
            col_labels=col_labels,
            title="joint classifier — per-network Σ permutation importance",
            cmap="magma",
            vmin=0.0,
            vmax=(float(np.nanpercentile(np.abs(M_show), 98))
                  if np.isfinite(M_show).any() else 1.0),
            cbar_label="Σ importance",
        )

    # ---- Row 1: coherence bars per feature
    palette = network_palette(n_networks)
    nets = network_order(n_networks)
    for j, feat in enumerate(features):
        ax = fig.add_subplot(inner[1, j])
        coh = _load_coherence(stats_net_dir, feat, trial, n_networks)
        if coh is None:
            _placeholder(ax, "no coherence")
            continue
        within = coh["within_r"]
        between = coh["between_r"]
        _draw_grouped_within_between(
            ax, within, between, nets, palette,
            title=FEATURE_DISPLAY.get(feat, feat),
        )


def _draw_yeo_legend(ax, n_networks: int) -> None:
    palette = network_palette(n_networks)
    nets = network_order(n_networks)
    patches = []
    for net in nets:
        display = YEO7_FULL_NAMES.get(net, net) if n_networks == 7 else net
        patches.append(mpatches.Patch(color=palette[net], label=display))
    ax.axis("off")
    ax.legend(handles=patches, loc="center", ncol=min(4, len(nets)),
              fontsize=8, frameon=False, title=f"Yeo-{n_networks} networks",
              title_fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--space", required=True,
                   help="Atlas space (e.g. schaefer_400).")
    p.add_argument("--trial-type", default="correct",
                   choices=["alltrials", "correct", "lapse"])
    p.add_argument("--yeo", type=int, default=7, choices=[7, 17])
    p.add_argument("--alpha", type=float, default=0.05)

    # Stats vs classif knobs are independent: default panel mixes
    # subject-level / FDR stats with single-epoch / tmax classification.
    p.add_argument("--stats-correction", default="fdr", choices=list(PVAL_KEYS),
                   help="Multiple-comparison correction for the stats row "
                        "(Tier 1A) + per-network aggregate (Tier 2). "
                        "Default: fdr.")
    p.add_argument("--stats-level", default="average",
                   choices=["average", "epoch"],
                   help="Granularity for the per-parcel stats input. "
                        "Default: average (pooled subject means).")
    p.add_argument("--classif-correction", default="tmax",
                   choices=list(PVAL_KEYS),
                   help="Multiple-comparison correction for Tier 1B "
                        "per-parcel AUC. Default: tmax.")
    p.add_argument("--classif-level", default="epoch",
                   choices=["epoch", "average"],
                   help="Granularity for the per-parcel classification "
                        "input. Default: epoch (single-trial).")
    p.add_argument("--classif-cv", default=None,
                   help="CV strategy token in classification filenames. "
                        "Defaults: 'logo' for level=epoch, 'group' for "
                        "level=average.")

    # Legacy aliases — set BOTH stats- and classif- variants when given.
    p.add_argument("--correction", default=None, choices=list(PVAL_KEYS),
                   help="(Deprecated) sets both --stats-correction and "
                        "--classif-correction.")
    p.add_argument("--cv", default=None,
                   help="(Deprecated) alias for --classif-cv.")

    p.add_argument("--clf", default="logistic")
    p.add_argument("--mf-label", default="all",
                   help="Label used when the joint mf run was launched "
                        "(filename: feature-<label>_..._axis-joint_...).")
    p.add_argument("--features", default=" ".join(FEATURE_COLUMNS),
                   help="Space-separated feature names for Tier-1/2/4 columns.")
    p.add_argument("--families", default=" ".join(FAMILY_COLUMNS),
                   help="Space-separated families for Tier-3 row 0.")
    p.add_argument("--per-feature-features", default=" ".join(FEATURE_COLUMNS),
                   help="Feature subset for Tier-3 row-1 heatmap "
                        "(must be a subset of --features for column alignment).")
    p.add_argument("--inout-selection", default=None,
                   choices=["strict", "lenient", "vtcfilt", "vtcraw"],
                   help="IN/OUT selection strategy whose outputs to read. "
                        "Defaults to config.analysis.inout_selection (or 'strict').")
    p.add_argument("--no-yeo-overlay", action="store_true",
                   help="Skip the Yeo network boundary overlay on Tier 1 "
                        "(faster, useful while iterating).")
    p.add_argument("--data-root", default=None,
                   help="Override config.paths.data_root (e.g. point at a "
                        "synthetic sandbox built by synthetic_network_bundle.py).")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.correction is not None:
        args.stats_correction = args.correction
        args.classif_correction = args.correction
    if args.cv is not None:
        args.classif_cv = args.cv
    if args.classif_cv is None:
        args.classif_cv = "logo" if args.classif_level == "epoch" else "group"
    return args


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(args.data_root) if args.data_root \
        else Path(config["paths"]["data_root"])
    inout_bounds = config["analysis"]["inout_bounds"]
    inout_str = f"{inout_bounds[0]}{inout_bounds[1]}"
    inout_selection = args.inout_selection or str(
        config.get("analysis", {}).get("inout_selection", "strict")
    )

    results_root = data_root / config["paths"]["results"]
    stats_dir = results_root / f"statistics_{args.space}"
    stats_net_dir = stats_dir / "group" / "networks"
    clf_dir = results_root / f"classification_{args.space}"
    clf_group_dir = clf_dir / "group"
    classif_net_dir = clf_dir / "group_mf" / "networks"

    features = args.features.split()
    families = args.families.split()
    per_feature_subset = args.per_feature_features.split()

    out_path = (Path(args.output) if args.output
                else Path("reports/figures")
                / (f"network_story_space-{args.space}_yeo{args.yeo}_"
                   f"type-{args.trial_type}.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Figure geometry: 4 tiers + header + legend
    # Heights are in arbitrary units; we tune to give brain rows the most space.
    fig_w = 24.0
    fig_h = 30.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110, facecolor="white")

    # Top: title + Yeo legend; then 4 tiers
    gs = GridSpec(
        nrows=6, ncols=1, figure=fig,
        height_ratios=[0.5, 0.45, 10.5, 5.0, 7.0, 7.0],
        hspace=0.25, left=0.04, right=0.98, top=0.97, bottom=0.03,
    )

    # ---- Title
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    title = (
        f"Network story panel — space={args.space} · "
        f"yeo-{args.yeo} · trial-type={args.trial_type}  |  "
        f"stats: {args.stats_level}/{args.stats_correction} · "
        f"classif: {args.classif_level}/{args.classif_correction} "
        f"(cv={args.classif_cv}, scoring=AUC)"
    )
    ax_title.text(0.5, 0.5, title, ha="center", va="center",
                  fontsize=14, fontweight="bold",
                  transform=ax_title.transAxes)

    # ---- Yeo legend
    _draw_yeo_legend(fig.add_subplot(gs[1]), args.yeo)

    # ---- Tier 1
    logger.info("Tier 1 — whole-brain maps")
    yeo_overlay = None if args.no_yeo_overlay else args.yeo
    try:
        _tier1_brains(
            gs[2], fig, stats_dir=stats_dir, clf_group_dir=clf_group_dir,
            space=args.space, trial=args.trial_type, inout=inout_str,
            clf=args.clf, classif_cv=args.classif_cv,
            features=features, alpha=args.alpha,
            stats_level=args.stats_level,
            stats_correction=args.stats_correction,
            classif_level=args.classif_level,
            classif_correction=args.classif_correction,
            yeo_overlay=yeo_overlay,
            inout_selection=inout_selection,
        )
    except Exception as exc:
        logger.exception(f"Tier 1 failed: {exc}")
        _placeholder(fig.add_subplot(gs[2]), f"Tier 1 failed: {exc}")

    # ---- Tier 2
    logger.info("Tier 2 — network-aggregated effects")
    try:
        _tier2_network_bars(
            gs[3], fig, net_dir=stats_net_dir,
            trial=args.trial_type, correction=args.stats_correction,
            features=features, n_networks=args.yeo, alpha=args.alpha,
        )
    except Exception as exc:
        logger.exception(f"Tier 2 failed: {exc}")
        _placeholder(fig.add_subplot(gs[3]), f"Tier 2 failed: {exc}")

    # ---- Tier 3
    logger.info("Tier 3 — network-restricted classification")
    try:
        _tier3_classif(
            gs[4], fig, net_dir=classif_net_dir,
            trial=args.trial_type, clf=args.clf, cv=args.classif_cv,
            n_networks=args.yeo, families=families,
            per_feature_subset=per_feature_subset, alpha=args.alpha,
        )
    except Exception as exc:
        logger.exception(f"Tier 3 failed: {exc}")
        _placeholder(fig.add_subplot(gs[4]), f"Tier 3 failed: {exc}")

    # ---- Tier 4
    logger.info("Tier 4 — multivariate importance + coherence")
    try:
        _tier4_importance_coherence(
            gs[5], fig, classif_net_dir=classif_net_dir,
            stats_net_dir=stats_net_dir, trial=args.trial_type,
            clf=args.clf, cv=args.classif_cv, label=args.mf_label,
            n_networks=args.yeo, features=features,
        )
    except Exception as exc:
        logger.exception(f"Tier 4 failed: {exc}")
        _placeholder(fig.add_subplot(gs[5]), f"Tier 4 failed: {exc}")

    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
