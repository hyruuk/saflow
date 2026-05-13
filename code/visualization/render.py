"""Pure renderers for sensor and atlas/source maps.

These take pre-loaded `MapResult` lists, not files. They are metric-agnostic:
the caller decides the colour scale and label.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code.visualization.loaders import MapResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour-scale helpers
# ---------------------------------------------------------------------------


def compute_vlim(maps: List[MapResult]) -> Tuple[float, float]:
    """Pick (vmin, vmax) for a row of maps based on the metric's color_mode."""
    if not maps:
        return (0.0, 1.0)
    metric = maps[0].metric  # assume homogeneous metric within one row
    all_vals = np.concatenate([m.values for m in maps])

    if metric.color_mode == "diverging":
        v = float(np.nanpercentile(np.abs(all_vals), 98))
        return (-v, v)

    if metric.color_mode == "sequential_above":
        chance = metric.chance_level
        # Anchor at chance, vmax = robust max above chance, clipped to a sane band.
        upper = float(np.nanpercentile(all_vals, 98))
        upper = max(upper, chance + metric.sequential_min_span)
        upper = min(upper, metric.sequential_max_vmax)
        return (chance, upper)

    raise ValueError(f"Unknown color_mode: {metric.color_mode}")


# ---------------------------------------------------------------------------
# Sensor row (MNE topomap)
# ---------------------------------------------------------------------------


def _get_sensor_info(config: Dict, data_root: Path):
    """Find a representative MEG file and return its mag info.

    Tries preprocessed FIFs first; falls back to raw CTF .ds directories under
    sourcedata when only results were synced (e.g. from HPC) and no
    preprocessed data is locally available.
    """
    import mne

    derivatives_dir = data_root / config["paths"]["derivatives"]
    sample_files = list(derivatives_dir.glob(
        "preprocessed/sub-*/meg/*_proc-clean_meg.fif"
    ))
    if not sample_files:
        sample_files = list(derivatives_dir.glob("**/sub-*/meg/*_meg.fif"))
    if sample_files:
        raw = mne.io.read_raw_fif(sample_files[0], preload=False, verbose=False)
        raw.pick("mag")
        return raw.info

    raw_dir = data_root / config["paths"]["raw"]
    ds_files = sorted(raw_dir.glob("**/SA*_SAflow*.ds"))
    if not ds_files:
        ds_files = sorted(raw_dir.glob("**/*.ds"))
    if ds_files:
        raw = mne.io.read_raw_ctf(str(ds_files[0]), preload=False, verbose=False)
        raw.pick("mag")
        return raw.info

    raise FileNotFoundError(
        "No preprocessed MEG (.fif) or raw CTF (.ds) files found to read "
        "sensor positions from. Searched: "
        f"{derivatives_dir}/preprocessed and {raw_dir}. "
        "Run preprocessing first (invoke pipeline.preprocess) or sync a raw "
        "CTF .ds under sourcedata/."
    )


def render_sensor_row(
    maps: List[MapResult],
    vmin: float,
    vmax: float,
    cmap: str,
    cbar_label: str,
    config: Dict,
    data_root: Path,
):
    """Single row of MEG topomaps with significance markers, shared colorbar."""
    import mne

    info = _get_sensor_info(config, data_root)
    n_maps = len(maps)
    fig, axes = plt.subplots(
        1, n_maps, figsize=(3 * n_maps + 1, 3.5), dpi=150,
    )
    if n_maps == 1:
        axes = [axes]

    mask_params = dict(
        marker="o", markerfacecolor="w", markeredgecolor="k",
        linewidth=0, markersize=5,
    )

    im = None
    for ax, mp in zip(axes, maps):
        title = f"{mp.label}\n(n={mp.n_significant} sig)" if mp.mask is not None else mp.label
        # MNE's topomap interpolator propagates NaN: a single NaN sensor turns
        # the entire surface white. Drop NaN channels from values + mask + info.
        valid = ~np.isnan(mp.values)
        if not valid.any():
            ax.text(0.5, 0.5, "no data\n(all NaN)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_axis_off()
            ax.set_title(title, fontsize=10)
            continue
        if not valid.all():
            vals_plot = mp.values[valid]
            mask_plot = mp.mask[valid] if mp.mask is not None else None
            info_plot = mne.pick_info(info, np.where(valid)[0])
        else:
            vals_plot = mp.values
            mask_plot = mp.mask
            info_plot = info
        im = mne.viz.plot_topomap(
            vals_plot, info_plot, axes=ax, show=False, cmap=cmap,
            mask=mask_plot, mask_params=mask_params,
            vlim=(vmin, vmax), extrapolate="local",
            outlines="head", sphere=0.15, contours=0,
        )
        ax.set_title(title, fontsize=10)

    if im is None:
        plt.close(fig)
        return None

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im[0], cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    midpoint = (vmin + vmax) / 2
    cbar.set_ticks([vmin, midpoint, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{midpoint:.2f}", f"{vmax:.2f}"])

    fig.subplots_adjust(left=0.02, right=0.88, top=0.80, bottom=0.05, wspace=0.1)
    return fig


# ---------------------------------------------------------------------------
# Atlas / source row (inflated brain panels stitched horizontally)
# ---------------------------------------------------------------------------


def render_atlas_row(
    maps: List[MapResult],
    space: str,
    vmin: float,
    vmax: float,
    cmap: str,
    cbar_label: str,
    threshold_below_chance: bool = False,
    chance_level: float = 0.0,
):
    """Single row of inflated brain panels (one 2x2 brain per map).

    For ``sequential_above`` metrics (e.g. ROC AUC), pass
    ``threshold_below_chance=True`` and ``chance_level=metric.chance_level`` to
    hide vertices at/below chance. For diverging metrics, leave the defaults.
    """
    from PIL import Image as PILImage

    from code.visualization.plot_surface import (
        VIEWS,
        _get_fsaverage_surfaces,
        _get_roi_names,
        render_inflated_view,
        roi_to_surface,
    )

    fsaverage = _get_fsaverage_surfaces()
    metadata_list = [m.metadata for m in maps]
    roi_names = _get_roi_names(space, [None] * len(maps), metadata_list)
    if roi_names is None:
        raise FileNotFoundError(
            f"Could not resolve ROI names for space '{space}'. "
            f"Check that the atlas timeseries exist under "
            f"'derivatives/atlas_timeseries_{space}/' or that fsaverage labels "
            f"are downloaded."
        )

    # Apply significance mask: hide non-sig vertices (set to NaN before painting).
    feature_images: List[np.ndarray] = []
    feature_titles: List[str] = []
    threshold = None
    if threshold_below_chance:
        # nilearn's `threshold` hides |x| < threshold; for AUC anchored at 0.5
        # this isn't quite right (we want to hide x ≤ chance), so we instead
        # mask values to NaN below chance.
        pass

    for mp in maps:
        vals = mp.values.astype(float).copy()
        if mp.mask is not None:
            vals = np.where(mp.mask, vals, np.nan)
        if threshold_below_chance:
            vals = np.where(vals > chance_level, vals, np.nan)

        if len(vals) != len(roi_names):
            logger.warning(
                f"{mp.feature}: values has {len(vals)} entries, "
                f"atlas has {len(roi_names)} ROIs — skipping"
            )
            continue

        lh_data, rh_data = roi_to_surface(vals, roi_names, space)

        images = {}
        for hemi, view in VIEWS:
            images[(hemi, view)] = render_inflated_view(
                lh_data, rh_data, hemi, view, fsaverage,
                cmap=cmap, vmin=vmin, vmax=vmax, threshold=threshold,
            )

        # Resize each view to a common height
        target_h = max(img.shape[0] for img in images.values())
        resized = {}
        for key, img in images.items():
            if img.shape[0] != target_h:
                scale = target_h / img.shape[0]
                new_w = int(img.shape[1] * scale)
                resized[key] = np.array(
                    PILImage.fromarray(img).resize(
                        (new_w, target_h), PILImage.LANCZOS
                    )
                )
            else:
                resized[key] = img

        top = np.concatenate(
            [resized[("left", "lateral")], resized[("right", "lateral")]], axis=1
        )
        bottom = np.concatenate(
            [resized[("left", "medial")], resized[("right", "medial")]], axis=1
        )
        max_w = max(top.shape[1], bottom.shape[1])
        for arr_name, arr in [("top", top), ("bottom", bottom)]:
            if arr.shape[1] < max_w:
                pad = np.full((arr.shape[0], max_w - arr.shape[1], 3), 255, dtype=np.uint8)
                if arr_name == "top":
                    top = np.concatenate([arr, pad], axis=1)
                else:
                    bottom = np.concatenate([arr, pad], axis=1)
        composite = np.concatenate([top, bottom], axis=0)
        feature_images.append(composite)

        title = f"{mp.label}\n(n={mp.n_significant} sig)" if mp.mask is not None else mp.label
        feature_titles.append(title)

    if not feature_images:
        return None

    # Pad all feature panels to a uniform size.
    max_h = max(img.shape[0] for img in feature_images)
    max_w = max(img.shape[1] for img in feature_images)
    padded = []
    for img in feature_images:
        ph, pw = max_h - img.shape[0], max_w - img.shape[1]
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=255)
        padded.append(img)

    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    n = len(padded)
    dpi = 150
    panel_w_in = max_w / dpi
    panel_h_in = max_h / dpi
    # Inch-based margins: suptitle + per-feature titles share the top band.
    top_margin_in = 1.10
    bottom_margin_in = 0.25
    left_margin_in = 0.15
    right_margin_in = 0.20
    cbar_slot_in = 0.85       # full slot reserved for colorbar (axis + ticks + label)
    wspace_frac = 0.04        # of average panel width

    fig_w = (
        left_margin_in + n * panel_w_in
        + (n - 1) * panel_w_in * wspace_frac
        + cbar_slot_in + right_margin_in
    )
    fig_h = top_margin_in + panel_h_in + bottom_margin_in
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")

    gs = GridSpec(
        1, n + 1,
        width_ratios=[panel_w_in] * n + [0.15],  # last column = colorbar
        left=left_margin_in / fig_w,
        right=1 - right_margin_in / fig_w,
        top=1 - top_margin_in / fig_h,
        bottom=bottom_margin_in / fig_h,
        wspace=wspace_frac,
    )

    for i, (img, t) in enumerate(zip(padded, feature_titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(t, fontsize=10, pad=4)

    cax = fig.add_subplot(gs[0, n])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    midpoint = (vmin + vmax) / 2
    cbar.set_ticks([vmin, midpoint, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{midpoint:.2f}", f"{vmax:.2f}"])

    return fig


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def render_row(
    maps: List[MapResult],
    space: str,
    config: Dict,
    data_root: Path,
    cmap_override: Optional[str] = None,
):
    """Pick the right renderer based on space, with the metric's colour mode."""
    if not maps:
        return None
    metric = maps[0].metric
    vmin, vmax = compute_vlim(maps)
    cmap = cmap_override or metric.cmap

    if space == "sensor":
        return render_sensor_row(
            maps, vmin, vmax, cmap, metric.cbar_label, config, data_root,
        )
    return render_atlas_row(
        maps, space, vmin, vmax, cmap, metric.cbar_label,
        threshold_below_chance=(metric.color_mode == "sequential_above"),
        chance_level=metric.chance_level,
    )
