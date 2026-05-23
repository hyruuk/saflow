"""Paper-ready stats + classification multi-panel figure (Fig. 3 from cc_saflow).

A single composite figure with letter labels A-J and no per-axis titles.
For ``--space=sensor`` the spatial panels are MNE topomaps; for atlas spaces
(e.g. ``schaefer_400``, ``aparc.a2009s``) they become 2x2 inflated-brain
composites (left/right × lateral/medial):

    A  Per-band t-values for raw PSD          (7 spatial maps)
    B  Per-band AUC for raw PSD               (7 spatial maps)
    C  Raw spectrum (PSD)                     IN vs OUT line plot
    D  Aperiodic component                    IN vs OUT line plot
    E  Corrected spectrum (PSDc)              IN vs OUT line plot
    F  Periodic components                    IN vs OUT line plot
    G  FOOOF t-values (exponent, offset, R²)  (3 spatial maps)
    H  FOOOF AUC (exponent, offset, R²)       (3 spatial maps)
    I  Per-band t-values for corrected PSD    (7 spatial maps)
    J  Per-band AUC for corrected PSD         (7 spatial maps)

Significance is computed within each topomap's set of spatial units (e.g.
270 sensors), independently per feature — never pooled across bands or
metrics. The default FDR uses the ``pvals_fdr_bh`` field (Benjamini-Hochberg
over the 270 sensors of that feature). Old stats files used the key
``pvals_corrected_fdr`` for the same quantity; both are accepted.

Selection of the FOOOF spectra location: strongest exponent IN-vs-OUT
difference (i.e. largest |t-value| from the FOOOF-exponent stats map),
regardless of p-value significance.

Output: reports/figures/stats_classif_panel_space-<space>_type-<trial>_correction-<corr>.png
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Matplotlib's bundled math-text parser (used whenever a figure contains
# any mathtext, e.g. tick labels with $...$) calls older pyparsing APIs that
# emit DeprecationWarnings under pyparsing 3.x. The mismatch is between the
# pinned matplotlib and pyparsing in this env, not our code — silence it so
# the render output stays readable.
#
# Belt-and-braces: filter by class, by module regex, and by message text, so
# we still catch the warnings if any of (module-name resolution, category
# inheritance, MNE's catch_warnings juggling) fails us.
try:
    from pyparsing import PyparsingDeprecationWarning  # type: ignore
    warnings.simplefilter("ignore", PyparsingDeprecationWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r"(matplotlib|pyparsing)\..*")
for _msg in (r".*deprecated - use .*",
             r".*'parseAll' argument is deprecated.*"):
    warnings.filterwarnings("ignore", message=_msg)

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


BANDS = ["theta", "alpha", "lobeta", "hibeta", "gamma1", "gamma2", "gamma3"]
FOOOF_PARAMS = ["exponent", "offset", "r_squared"]

COLOR_IN = "#1f77b4"
COLOR_OUT = "#ff7f0e"
CMAP_T = "RdBu_r"
CMAP_AUC = "magma"

# correction-name -> ordered list of npz key candidates. Canonical names
# (current writers) come first; legacy stats keys are kept as fallbacks so
# pre-rename npz files still load.
PVAL_KEYS = {
    "fdr":         ["pvals_fdr_bh", "pvals_corrected_fdr_bh", "pvals_corrected_fdr"],
    "tmax":        ["pvals_tmax", "pvals_corrected_tmax"],
    "bonferroni":  ["pvals_bonferroni", "pvals_corrected_bonferroni"],
    "uncorrected": ["pvals_uncorrected", "pvals"],
}


# ---------------------------------------------------------------------------
# File discovery + loading
# ---------------------------------------------------------------------------

# Legacy filename tokens kept as fallbacks so pre-refactor stats files still load.
_STATS_LEGACY_TOKEN = {"average": "path-subj-spectrum", "epoch": "path-subj-trial-*"}


def _stats_file(stats_dir: Path, feature: str, inout: str, trial_type: str,
                level: str = "average") -> Path:
    cands = sorted(stats_dir.glob(
        f"feature-{feature}_inout-{inout}_test-paired_ttest"
        f"_level-{level}_type-{trial_type}_results.npz"
    ))
    if not cands:
        cands = sorted(stats_dir.glob(
            f"feature-{feature}_inout-{inout}_test-paired_ttest"
            f"_{_STATS_LEGACY_TOKEN.get(level, 'path-subj-spectrum')}"
            f"_type-{trial_type}_results.npz"
        ))
    if not cands:
        raise FileNotFoundError(
            f"No level-{level} stats result for feature={feature} "
            f"(inout={inout}, type={trial_type}) in {stats_dir}."
        )
    return cands[0]


def _classif_file(clf_dir: Path, feature: str, inout: str, trial_type: str,
                  clf: str, cv: str, space: str, level: str = "average") -> Path:
    cands = sorted(clf_dir.glob(
        f"feature-{feature}_space-{space}_inout-{inout}"
        f"_clf-{clf}_cv-{cv}_mode-univariate_level-{level}"
        f"_type-{trial_type}_scores.npz"
    ))
    if not cands:
        # legacy pre-`level` token
        cands = sorted(clf_dir.glob(
            f"feature-{feature}_space-{space}_inout-{inout}"
            f"_clf-{clf}_cv-{cv}_mode-univariate_type-{trial_type}_scores.npz"
        ))
    if not cands:
        raise FileNotFoundError(
            f"No classification scores for feature={feature} (clf={clf}, "
            f"cv={cv}, level={level}, type={trial_type}) in {clf_dir}."
        )
    return cands[0]


def _check_scoring_metadata(path: Path) -> None:
    """Warn if a classification scores npz wasn't computed with roc_auc.

    The panel labels every classification colorbar/axis "AUC", so a silent
    mismatch (e.g. someone re-ran with --scoring=balanced_accuracy) would
    mislead the reader. Read the sibling metadata.json and warn — don't
    fail — so missing/older metadata doesn't break the figure.
    """
    meta_path = path.with_name(path.stem.replace("_scores", "_metadata") + ".json")
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return
    scoring = meta.get("scoring") or meta.get("provenance", {}).get("scoring")
    if scoring and scoring != "roc_auc":
        logger.warning(
            "Classification file %s was scored with %r, not 'roc_auc' — "
            "panel labels it as AUC. Re-run with --scoring=roc_auc to "
            "match.", path.name, scoring,
        )


def _pick_pvals(npz, candidate_keys: List[str]) -> Optional[np.ndarray]:
    for key in candidate_keys:
        if key in npz.files:
            return np.asarray(npz[key]).flatten()
    return None


def _load_stats(path: Path, correction: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=True) as npz:
        tvals = np.asarray(npz["tvals"]).flatten()
        pvals = _pick_pvals(npz, PVAL_KEYS[correction])
    return tvals, pvals


def _load_classif(path: Path, correction: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=True) as npz:
        observed = np.asarray(npz["observed"]).flatten()
        pvals = _pick_pvals(npz, PVAL_KEYS[correction])
    return observed, pvals


# ---------------------------------------------------------------------------
# Topomap + spectrum helpers
# ---------------------------------------------------------------------------

def _plot_topomap(ax, values: np.ndarray, mask: Optional[np.ndarray],
                  info, vmin: float, vmax: float, cmap: str):
    import mne

    valid = ~np.isnan(values)
    if not valid.any():
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_axis_off()
        return None
    if not valid.all():
        vals = values[valid]
        m = mask[valid] if mask is not None else None
        info_p = mne.pick_info(info, np.where(valid)[0])
    else:
        vals, m, info_p = values, mask, info

    mask_params = dict(
        marker="o", markerfacecolor="w", markeredgecolor="k",
        linewidth=0, markersize=2.8,
    )
    im, _ = mne.viz.plot_topomap(
        vals, info_p, axes=ax, show=False, cmap=cmap,
        mask=m, mask_params=mask_params,
        vlim=(vmin, vmax), extrapolate="local",
        outlines="head", sphere=0.15, contours=0,
    )
    return im


def _plot_brain(ax, values: np.ndarray, mask: Optional[np.ndarray],
                roi_names: List[str], atlas_name: str, fsaverage: dict,
                vmin: float, vmax: float, cmap: str):
    """Render a 2x2 inflated-brain composite into an existing axes.

    Significance masking is applied by NaN-ing non-significant ROIs before
    surface mapping (those vertices are then transparent on the brain).
    When the mask zeroes out every ROI, the sulci surface still renders —
    nilearn's plot_surf_stat_map treats all-NaN as "no overlay", which is
    what we want for "no parcels significant". The placeholder is only
    used when ``values`` itself is unavailable.

    ``aspect="equal"`` preserves the natural ~4:3 aspect of the 2x2
    composite so it isn't stretched into near-square axes (matplotlib
    letterboxes spare height/width instead). The xlabel set by the caller
    stays visible because we turn off ticks/spines explicitly instead of
    calling ``set_axis_off`` (which would hide the xlabel as well).
    """
    from code.visualization.plot_surface import (
        render_inflated_view,
        roi_to_surface,
    )

    vals = np.asarray(values, dtype=float).copy()
    if mask is not None:
        vals[~np.asarray(mask, dtype=bool)] = np.nan

    lh_data, rh_data = roi_to_surface(vals, roi_names, atlas_name)

    views = [("left", "lateral"), ("right", "lateral"),
             ("left", "medial"),  ("right", "medial")]
    images = {}
    for hemi, view in views:
        images[(hemi, view)] = render_inflated_view(
            lh_data, rh_data, hemi, view, fsaverage,
            cmap=cmap, vmin=vmin, vmax=vmax,
        )

    target_h = max(img.shape[0] for img in images.values())
    resized = {}
    from PIL import Image
    for k, img in images.items():
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            pil = Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS)
            resized[k] = np.asarray(pil)
        else:
            resized[k] = img

    top = np.concatenate(
        [resized[("left", "lateral")], resized[("right", "lateral")]], axis=1)
    bot = np.concatenate(
        [resized[("left", "medial")],  resized[("right", "medial")]],  axis=1)
    max_w = max(top.shape[1], bot.shape[1])
    if top.shape[1] < max_w:
        pad = np.full((top.shape[0], max_w - top.shape[1], 3), 255, dtype=np.uint8)
        top = np.concatenate([top, pad], axis=1)
    if bot.shape[1] < max_w:
        pad = np.full((bot.shape[0], max_w - bot.shape[1], 3), 255, dtype=np.uint8)
        bot = np.concatenate([bot, pad], axis=1)
    composite = np.concatenate([top, bot], axis=0)

    ax.imshow(composite, interpolation="bilinear", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return None


def _mean_sem(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, sem


def _plot_spectrum(ax, freqs: np.ndarray, arr_in: np.ndarray,
                   arr_out: np.ndarray, show_legend: bool = False,
                   logx: bool = True, axhline_zero: bool = False):
    for arr, color, label in (
        (arr_in, COLOR_IN, "IN"),
        (arr_out, COLOR_OUT, "OUT"),
    ):
        mean, sem = _mean_sem(arr)
        ax.plot(freqs, mean, color=color, lw=1.6, label=label)
        ax.fill_between(freqs, mean - sem, mean + sem,
                        color=color, alpha=0.22, lw=0)
    if logx:
        ax.set_xscale("log")
    if axhline_zero:
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.grid(True, which="both", alpha=0.18)
    ax.tick_params(axis="both", labelsize=7)
    if show_legend:
        ax.legend(loc="best", fontsize=8, frameon=False,
                  handlelength=1.2, handletextpad=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--space", default="sensor",
                        help="Analysis space: 'sensor' (topomaps) or an atlas "
                             "name (inflated-brain panels). Vertex-level "
                             "'source' is not wired.")
    parser.add_argument("--trial-type", default="alltrials",
                        choices=["alltrials", "correct", "lapse"])
    parser.add_argument("--clf", default="logistic")
    parser.add_argument("--alpha", type=float, default=0.05)

    # Stats vs classif knobs are independent. Default panel mixes
    # subject-level / FDR stats with single-epoch / tmax classification.
    parser.add_argument(
        "--stats-correction", default="fdr", choices=list(PVAL_KEYS.keys()),
        help="P-value correction for stats topomaps/brains (rows A, G, I). "
             "Computed per spatial map (per-feature, never pooled across "
             "bands/metrics). Default: fdr.",
    )
    parser.add_argument(
        "--stats-level", default="average", choices=["average", "epoch"],
        help="Granularity for the per-spatial-unit stats input. "
             "Default: average (pooled subject means → t-test).",
    )
    parser.add_argument(
        "--classif-correction", default="tmax", choices=list(PVAL_KEYS.keys()),
        help="P-value correction for classification topomaps/brains "
             "(rows B, H, J). Default: tmax.",
    )
    parser.add_argument(
        "--classif-level", default="epoch", choices=["epoch", "average"],
        help="Granularity for the per-spatial-unit classification input. "
             "Default: epoch (single-trial).",
    )
    parser.add_argument(
        "--classif-cv", default=None,
        help="CV strategy token in classification filenames. Defaults: "
             "'logo' for level=epoch, 'group' for level=average.",
    )

    # Legacy aliases — set BOTH stats- and classif- variants when given.
    parser.add_argument(
        "--correction", default=None, choices=list(PVAL_KEYS.keys()),
        help="(Deprecated) sets both --stats-correction and "
             "--classif-correction.",
    )
    parser.add_argument("--cv", default=None,
                        help="(Deprecated) alias for --classif-cv.")

    parser.add_argument("--n-events-window", type=int, default=8)
    parser.add_argument("--output", default=None,
                        help="Output path. Default: "
                             "reports/figures/stats_classif_panel_space-<space>"
                             "_type-<trial-type>_stats-<lvl>-<corr>"
                             "_classif-<lvl>-<corr>.png")
    args = parser.parse_args()

    # Resolve legacy aliases / defaults that depend on other args.
    if args.correction is not None:
        args.stats_correction = args.correction
        args.classif_correction = args.correction
    if args.cv is not None:
        args.classif_cv = args.cv
    if args.classif_cv is None:
        args.classif_cv = "logo" if args.classif_level == "epoch" else "group"

    if args.space == "source":
        raise NotImplementedError(
            "space='source' (vertex-level) is not wired. Use 'sensor' or an "
            "atlas name (e.g. 'schaefer_400', 'aparc.a2009s')."
        )
    is_atlas = args.space != "sensor"

    config = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(config["paths"]["data_root"])
    inout_bounds = config["analysis"]["inout_bounds"]
    inout_str = f"{inout_bounds[0]}{inout_bounds[1]}"

    stats_dir = data_root / config["paths"]["results"] / f"statistics_{args.space}"
    clf_dir = (data_root / config["paths"]["results"]
               / f"classification_{args.space}" / "group")

    psd_features = [f"psd_{b}" for b in BANDS]
    psdc_features = [f"psd_corrected_{b}" for b in BANDS]
    fooof_features = [f"fooof_{p}" for p in FOOOF_PARAMS]

    # ---- Load results ----------------------------------------------------
    logger.info(
        "Loading stats (level=%s, correction=%s) + classification "
        "(level=%s, cv=%s, correction=%s)…",
        args.stats_level, args.stats_correction,
        args.classif_level, args.classif_cv, args.classif_correction,
    )
    rows: Dict[str, List[Tuple[np.ndarray, Optional[np.ndarray]]]] = {}
    aurows: Dict[str, List[Tuple[np.ndarray, Optional[np.ndarray]]]] = {}

    for tag, features in (("psd", psd_features),
                          ("fooof", fooof_features),
                          ("psdc", psdc_features)):
        rows[tag] = []
        aurows[tag] = []
        for feat in features:
            sfile = _stats_file(stats_dir, feat, inout_str, args.trial_type,
                                level=args.stats_level)
            cfile = _classif_file(clf_dir, feat, inout_str, args.trial_type,
                                  args.clf, args.classif_cv, args.space,
                                  level=args.classif_level)
            _check_scoring_metadata(cfile)
            t, pt = _load_stats(sfile, args.stats_correction)
            auc, pa = _load_classif(cfile, args.classif_correction)
            rows[tag].append((t, pt))
            aurows[tag].append((auc, pa))

    # ---- Shared color scales --------------------------------------------
    all_t = np.concatenate([v[0] for tag in rows for v in rows[tag]])
    all_auc = np.concatenate([v[0] for tag in aurows for v in aurows[tag]])
    tlim = float(np.nanpercentile(np.abs(all_t), 98))
    vmin_t, vmax_t = -tlim, tlim
    auc_upper = float(np.nanpercentile(all_auc, 98))
    vmin_a, vmax_a = 0.5, max(auc_upper, 0.55)

    # ---- Sensor info / atlas surface assets -----------------------------
    info = None
    roi_names: Optional[List[str]] = None
    fsaverage: Optional[dict] = None
    if is_atlas:
        from code.visualization.plot_surface import (
            _get_fsaverage_surfaces,
            _get_roi_names,
        )

        meta_any = _stats_file(stats_dir, "fooof_exponent",
                               inout_str, args.trial_type,
                               level=args.stats_level)
        meta_any = meta_any.with_name(
            meta_any.stem.replace("_results", "_metadata") + ".json"
        )
        all_metadata = [json.loads(meta_any.read_text())] if meta_any.exists() else []
        roi_names = _get_roi_names(args.space, [], all_metadata)
        if not roi_names:
            raise RuntimeError(
                f"Could not resolve ROI names for atlas '{args.space}' "
                f"(metadata at {meta_any} has no spatial_names/roi_names "
                "and the atlas annotation was not loadable)."
            )
        fsaverage = _get_fsaverage_surfaces()
    else:
        from code.visualization.render import _get_sensor_info
        info = _get_sensor_info(config, data_root)

    # ---- Channel spectra at sensor/region with strongest exponent diff --
    # Selection criterion: largest |t| on the FOOOF-exponent map (strongest
    # IN-vs-OUT difference), not the smallest p-value. We want the panel to
    # showcase the spatial unit where the effect is biggest, even if it
    # doesn't survive correction.
    logger.info("Loading channel spectra for panels C-F...")
    from code.statistics.subject_spectrum import load_channel_spectra
    exp_t, _exp_p = rows["fooof"][0]
    sel_idx = int(np.nanargmax(np.abs(exp_t)))
    spatial_unit_names = roi_names if is_atlas else list(info["ch_names"])
    sel_name = (spatial_unit_names[sel_idx]
                if sel_idx < len(spatial_unit_names) else f"unit-{sel_idx}")
    logger.info("Selected unit: %s (idx=%d, |t|=%.3f)",
                sel_name, sel_idx, float(np.abs(exp_t[sel_idx])))

    sfile_exp = _stats_file(stats_dir, "fooof_exponent", inout_str,
                            args.trial_type, level=args.stats_level)
    meta_path = sfile_exp.with_name(
        sfile_exp.stem.replace("_results", "_metadata") + ".json"
    )
    bad_rule, interp_thr = "ar2", 0
    if meta_path.exists():
        dm = json.loads(meta_path.read_text()).get("data_metadata", {})
        bad_rule = str(dm.get("bad_trial_rule", bad_rule))
        interp_thr = int(dm.get("interp_reject_threshold", interp_thr) or 0)

    spectra = load_channel_spectra(
        channel_index=sel_idx,
        space=args.space,
        inout_bounds=tuple(inout_bounds),
        config=config,
        bad_trial_rule=bad_rule,
        interp_reject_threshold=interp_thr,
        n_events_window=args.n_events_window,
    )
    freqs = np.asarray(spectra["freqs"], dtype=float)
    freqs_fit = np.asarray(spectra["freqs_fit"], dtype=float)
    fooof_cfg = config.get("features", {}).get("fooof", [{}])
    fooof_params = fooof_cfg[0] if isinstance(fooof_cfg, list) else fooof_cfg
    fit_lo, fit_hi = fooof_params.get("freq_range", [2.0, 120.0])
    full_mask = (freqs >= fit_lo) & (freqs <= fit_hi)
    f_full = freqs[full_mask]

    # ---- Figure layout ---------------------------------------------------
    # 8 columns: 7 topomap-slot columns + 1 narrow colorbar column.
    # All topomaps occupy exactly one column (same width across rows).
    # Middle-row line plots each span 2 topomap columns.
    #
    # Brain composites (atlas mode) are ~square (2×2 grid of ~square views).
    # Topomaps are also ~square. So a single spatial_row_h works for both.
    # Earlier sizing inflated atlas rows to 1.6 — that left ~35% letterbox
    # height in each brain cell. Use 1.0 across the board.
    #
    # hspace was 0.45 (≈45% inter-row gap relative to row height); reduced
    # to 0.10 so blank space between rows is ~10–15% of a row instead of
    # nearly an entire row.
    n_cols = 8
    col_widths = [1.0] * 7 + [0.10]
    spatial_row_h = 1.0
    row_heights = [spatial_row_h, spatial_row_h,
                   1.10, 1.10,
                   spatial_row_h, spatial_row_h]
    fig_w = 13.2
    fig_h = 13.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150, facecolor="white")
    gs = GridSpec(
        nrows=6, ncols=n_cols, figure=fig,
        width_ratios=col_widths, height_ratios=row_heights,
        left=0.045, right=0.965, top=0.97, bottom=0.045,
        hspace=0.10, wspace=0.12,
    )

    band_labels = ["Theta\n(4-8Hz)", "Alpha\n(8-12Hz)",
                   "Low Beta\n(12-20Hz)", "High Beta\n(20-30Hz)",
                   "Gamma 1\n(30-60Hz)", "Gamma 2\n(60-90Hz)",
                   "Gamma 3\n(90-120Hz)"]
    fooof_labels = ["Exponent", "Offset", "R²"]

    def _draw_spatial(ax, vals, mask, vmin, vmax, cmap):
        if is_atlas:
            _plot_brain(ax, vals, mask, roi_names, args.space, fsaverage,
                        vmin, vmax, cmap)
        else:
            _plot_topomap(ax, vals, mask, info, vmin, vmax, cmap)

    def _draw_selection_inset(ax_d):
        """Top-right inset on panel D: small topomap (or brain) showing the
        selected sensor/region, with the region name as a title.

        Rect is sized to keep the inset fully inside the parent axes
        (max x = 0.99, max y = 0.97 leaves room for the title) and to give
        the brain a 4:3-ish aspect so ``aspect="equal"`` doesn't waste
        space. ``set_axis_off`` would hide the title too, so we strip
        ticks/spines individually instead.
        """
        if is_atlas:
            # Wider than tall — fits a single lateral hemisphere with title.
            rect = [0.54, 0.54, 0.45, 0.42]
        else:
            rect = [0.74, 0.60, 0.26, 0.40]
        inset = ax_d.inset_axes(rect)
        if is_atlas:
            from code.visualization.plot_surface import (
                render_inflated_view,
                roi_to_surface,
            )
            roi_label = roi_names[sel_idx]
            hemi = "right" if roi_label.endswith("-rh") else "left"
            vals = np.full(len(roi_names), np.nan)
            vals[sel_idx] = 1.0
            lh, rh = roi_to_surface(vals, roi_names, args.space)
            img = render_inflated_view(
                lh, rh, hemi, "lateral", fsaverage,
                cmap="autumn", vmin=0.0, vmax=1.0,
            )
            inset.imshow(img, interpolation="bilinear", aspect="equal")
        else:
            import mne
            n_ch = len(info["ch_names"])
            vals = np.zeros(n_ch)
            mask = np.zeros(n_ch, dtype=bool)
            mask[sel_idx] = True
            mask_params = dict(
                marker="o", markerfacecolor="red", markeredgecolor="k",
                linewidth=0.6, markersize=5,
            )
            mne.viz.plot_topomap(
                vals, info, axes=inset, show=False, cmap="Greys",
                mask=mask, mask_params=mask_params,
                vlim=(-1, 1), extrapolate="local",
                outlines="head", sphere=0.15, contours=0,
            )
        # Strip ticks + spines (but keep the axes "on" so set_title renders).
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(False)
        inset.set_title(sel_name, fontsize=7, pad=1)

    def _label_with_n_sig(label: str, mask: Optional[np.ndarray]) -> str:
        if mask is None:
            return label
        return f"{label}\n(n={int(mask.sum())} sig)"

    def draw_band_row(row_idx, values_list, vmin, vmax, cmap, labels):
        for i, ((vals, pvals), label) in enumerate(zip(values_list, labels)):
            ax = fig.add_subplot(gs[row_idx, i])
            mask = pvals < args.alpha if pvals is not None else None
            _draw_spatial(ax, vals, mask, vmin, vmax, cmap)
            ax.set_xlabel(_label_with_n_sig(label, mask),
                          fontsize=8.5, labelpad=2)
        cax = fig.add_subplot(gs[row_idx, 7])
        return cax

    def draw_middle_row(row_idx, line_panels, topo_values, vmin, vmax,
                        cmap, labels):
        # ax_l2 shares Y with ax_l1 — the two line plots in a middle row
        # show the same log-power scale (C+D = raw PSD vs. aperiodic;
        # E+F = corrected vs. periodic). Sharing locks the scale and lets
        # us drop the right-panel y-tick labels + ylabel.
        ax_l1 = fig.add_subplot(gs[row_idx, 0:2])
        ax_l2 = fig.add_subplot(gs[row_idx, 2:4], sharey=ax_l1)
        plt.setp(ax_l2.get_yticklabels(), visible=False)
        for ax, panel in zip([ax_l1, ax_l2], line_panels):
            panel(ax)
        for j in range(3):
            ax = fig.add_subplot(gs[row_idx, 4 + j])
            vals, pvals = topo_values[j]
            mask = pvals < args.alpha if pvals is not None else None
            _draw_spatial(ax, vals, mask, vmin, vmax, cmap)
            ax.set_xlabel(_label_with_n_sig(labels[j], mask),
                          fontsize=8.5, labelpad=2)
        cax = fig.add_subplot(gs[row_idx, 7])
        return cax, (ax_l1, ax_l2)

    cax_A = draw_band_row(0, rows["psd"], vmin_t, vmax_t, CMAP_T, band_labels)
    cax_B = draw_band_row(1, aurows["psd"], vmin_a, vmax_a, CMAP_AUC, band_labels)

    # Panels C/D share Y (raw spectrum vs. aperiodic component) — both are
    # log power on the same scale. Panels E/F share Y (corrected spectrum
    # vs. periodic component) — both are residual/periodic log power.
    # The ylabel sits on the left panel only; the right panel inherits the
    # scale via sharey and hides its tick labels (see draw_middle_row).
    def panel_C(ax):
        _plot_spectrum(
            ax, f_full,
            spectra["IN"]["raw"][:, full_mask],
            spectra["OUT"]["raw"][:, full_mask],
            show_legend=True,
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=8.5)
        ax.set_ylabel("PSD (log$_{10}$ power) — raw / aperiodic",
                      fontsize=8.5)

    def panel_D(ax):
        _plot_spectrum(
            ax, f_full,
            spectra["IN"]["aperiodic"][:, full_mask],
            spectra["OUT"]["aperiodic"][:, full_mask],
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=8.5)

    def panel_E(ax):
        _plot_spectrum(
            ax, f_full,
            spectra["IN"]["corrected"][:, full_mask],
            spectra["OUT"]["corrected"][:, full_mask],
            axhline_zero=True,
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=8.5)
        ax.set_ylabel("Log$_{10}$ power — corrected / periodic",
                      fontsize=8.5)

    def panel_F(ax):
        _plot_spectrum(
            ax, freqs_fit,
            spectra["IN"]["periodic"],
            spectra["OUT"]["periodic"],
            axhline_zero=True,
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=8.5)

    cax_G, axes_CD = draw_middle_row(
        2, [panel_C, panel_D],
        rows["fooof"], vmin_t, vmax_t, CMAP_T, fooof_labels,
    )
    _draw_selection_inset(axes_CD[1])
    cax_H, axes_EF = draw_middle_row(
        3, [panel_E, panel_F],
        aurows["fooof"], vmin_a, vmax_a, CMAP_AUC, fooof_labels,
    )

    cax_I = draw_band_row(4, rows["psdc"], vmin_t, vmax_t, CMAP_T, band_labels)
    cax_J = draw_band_row(5, aurows["psdc"], vmin_a, vmax_a, CMAP_AUC, band_labels)

    def _add_cbar(cax, vmin, vmax, cmap, label, diverging=False):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        if diverging:
            cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([f"{vmin:.1f}", "0", f"{vmax:.1f}"])
            cbar.set_label(label, fontsize=8)
            cax.text(2.2, 1.02, "OUT > IN", transform=cax.transAxes,
                     fontsize=7, ha="center", va="bottom", color="#a40000")
            cax.text(2.2, -0.02, "IN > OUT", transform=cax.transAxes,
                     fontsize=7, ha="center", va="top", color="#00408a")
        else:
            mid = (vmin + vmax) / 2
            cbar.set_ticks([vmin, mid, vmax])
            cbar.set_ticklabels([f"{vmin:.2f}", f"{mid:.2f}", f"{vmax:.2f}"])
            cbar.set_label(label, fontsize=8)

    _add_cbar(cax_A, vmin_t, vmax_t, CMAP_T, "T-values", diverging=True)
    _add_cbar(cax_B, vmin_a, vmax_a, CMAP_AUC, "AUC")
    _add_cbar(cax_G, vmin_t, vmax_t, CMAP_T, "T-values", diverging=True)
    _add_cbar(cax_H, vmin_a, vmax_a, CMAP_AUC, "AUC")
    _add_cbar(cax_I, vmin_t, vmax_t, CMAP_T, "T-values", diverging=True)
    _add_cbar(cax_J, vmin_a, vmax_a, CMAP_AUC, "AUC")

    # ---- Letter labels --------------------------------------------------
    fig.canvas.draw()

    def _stamp(ax, letter, dx=-0.028, dy=0.020):
        bb = ax.get_position()
        fig.text(bb.x0 + dx, bb.y1 + dy, letter,
                 fontsize=14, fontweight="bold", ha="left", va="bottom")

    def _row_first_topo(row_idx):
        outer_ss = gs[row_idx, 0]
        bb = outer_ss.get_position(fig)
        target_yc = 0.5 * (bb.y0 + bb.y1)
        cands = [ax for ax in fig.axes
                 if abs(0.5 * (ax.get_position().y0 + ax.get_position().y1) - target_yc) < 0.01]
        return min(cands, key=lambda a: a.get_position().x0) if cands else None

    def _row_first_fooof(row_idx):
        outer_ss = gs[row_idx, 4]
        bb = outer_ss.get_position(fig)
        target_yc = 0.5 * (bb.y0 + bb.y1)
        cands = [ax for ax in fig.axes
                 if abs(0.5 * (ax.get_position().y0 + ax.get_position().y1) - target_yc) < 0.01]
        cands.sort(key=lambda a: a.get_position().x0)
        return cands[2] if len(cands) > 2 else None

    for row_idx, letter in [(0, "A"), (1, "B"), (4, "I"), (5, "J")]:
        ax = _row_first_topo(row_idx)
        if ax is not None:
            _stamp(ax, letter)
    for ax, letter in [(axes_CD[0], "C"), (axes_CD[1], "D")]:
        _stamp(ax, letter, dx=-0.024, dy=0.012)
    for ax, letter in [(axes_EF[0], "E"), (axes_EF[1], "F")]:
        _stamp(ax, letter, dx=-0.024, dy=0.012)
    for row_idx, letter in [(2, "G"), (3, "H")]:
        ax = _row_first_fooof(row_idx)
        if ax is not None:
            _stamp(ax, letter)

    out_path = (Path(args.output) if args.output
                else Path("reports") / "figures"
                / (f"stats_classif_panel_space-{args.space}"
                   f"_type-{args.trial_type}"
                   f"_stats-{args.stats_level}-{args.stats_correction}"
                   f"_classif-{args.classif_level}-"
                   f"{args.classif_correction}.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
