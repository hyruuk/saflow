"""Render corrected feature-modulation bundles with the established Panel 1 layout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from code.analysis.contracts import CANONICAL_BANDS
from code.visualization import slide_style
from code.visualization.plot_surface import _get_fsaverage_surfaces
from code.visualization.stats_classif_panel import (
    CMAP_AUC,
    CMAP_T,
    _plot_brain,
    _plot_spectrum,
)

PANEL_NAMES = (
    "A_raw_PSD_modulation",
    "B_raw_PSD_decoding",
    "C_raw_spectrum",
    "D_aperiodic_spectrum",
    "E_corrected_spectrum",
    "F_periodic_spectrum",
    "G_FOOOF_modulation",
    "H_FOOOF_decoding",
    "I_corrected_PSD_modulation",
    "J_corrected_PSD_decoding",
)

PANEL1_CAPTION = """Figure X. Spectral feature modulation and decoding across attentional states. Participants were classified as in-the-zone (IN) or out-of-the-zone (OUT) using run-specific lower and upper quartiles of reflected-boundary filtered response-time variability. Neural observations were retained only when all eight contributing trials had the same state and none carried the final AR2 artifact flag. Unless explicitly overridden at rendering, state observations were pooled across retained windows within each participant (equal-window weighting); equal-run-weighted results were computed and retained as a sensitivity analysis. (A) Paired subject-level t statistics for OUT minus IN raw power spectral density (PSD) in seven canonical frequency bands. Significance was controlled independently within each feature across the 400 Schaefer parcels using 10,000 two-sided cluster-mass sign-flip permutations with surface-based parcel adjacency and an absolute cluster-forming t threshold of 2.0 (cluster-level family-wise p < 0.05). (B) Cross-validated area under the receiver-operating-characteristic curve (AUC) for parcel-wise decoding of IN versus OUT from raw PSD. Significance was assessed using 10,000 within-subject label permutations and the maximum parcel AUC within each feature (family-wise p < 0.05). (C-F) Group-mean IN and OUT spectra over parcels showing significant FOOOF-exponent modulation: raw PSD (C), fitted aperiodic component (D), aperiodic-corrected spectrum obtained by subtracting the fitted aperiodic component (E), and the modeled periodic component formed by the summed Gaussian peak fits (F). Ribbons show the subject-level standard error of the mean when subject spectra are available. Lower axes show the signed OUT-minus-IN difference, shaded orange where OUT > IN and blue where IN > OUT. (G) Paired OUT-minus-IN t statistics for FOOOF exponent and offset, with cluster-mass permutation correction across parcels. (H) Parcel-wise IN-versus-OUT decoding AUC for exponent and offset, with maximum-statistic permutation correction. FOOOF model R² remains available as a quality-control measure but is not displayed. (I) Paired OUT-minus-IN t statistics for aperiodic-corrected PSD across seven frequency bands, with cluster-mass permutation correction. (J) Parcel-wise decoding AUC for corrected PSD, with maximum-statistic permutation correction. Colored cortical parcels survive the stated primary correction at 0.05; gray parcels do not. Benjamini-Hochberg FDR maps are retained as sensitivity results. All analyses include 32 participants.\n"""


def render_bundle(
    bundle_directory: Path,
    reports_root: Path,
    *,
    output_name: str = "panel1_feature_modulation.png",
    weighting: str = "equal_window",
) -> list[Path]:
    """Render one real corrected bundle using the established A--J design."""
    arrays, metadata = _load_bundle(bundle_directory)
    arrays = _select_weighting(arrays, metadata, weighting)
    cache_directory = reports_root / ".cache" / "panel1_surface"
    figure, groups = _draw_composite(arrays, metadata, cache_directory)
    output = reports_root / "figures" / "manuscript" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    caption = output.with_suffix(".txt")
    is_legacy = metadata.get("summary", {}).get("available_weightings") is None
    caption_text = _caption_for_weighting(weighting, legacy=is_legacy)
    caption.write_text(caption_text)
    slide_caption = (
        reports_root / "figures" / "slides" / "panel1_feature_modulation"
        / "panel1_feature_modulation.txt"
    )
    slide_caption.parent.mkdir(parents=True, exist_ok=True)
    slide_caption.write_text(caption_text)
    sidecar = _metadata(metadata, bundle_directory, output, "composite", arrays)
    sidecar["weighting"] = weighting
    sidecar["caption"] = str(caption)
    _write_sidecar(output, sidecar)
    slides = _write_slides(arrays, groups, reports_root, sidecar)
    plt.close(figure)
    return [output, *slides]


def _caption_for_weighting(weighting: str, *, legacy: bool = False) -> str:
    """Describe the weighting variant actually rendered in the figure."""
    if weighting == "equal_window" and not legacy:
        return PANEL1_CAPTION
    caption = PANEL1_CAPTION.replace(
        "Unless explicitly overridden at rendering, state observations were pooled "
        "across retained windows within each participant (equal-window weighting); "
        "equal-run-weighted results were computed and retained as a sensitivity analysis.",
        "This sensitivity rendering averages state means equally across runs within "
        "each participant (equal-run weighting); pooled equal-window results were "
        "computed and retained as the primary analysis.",
    )
    if not legacy:
        return caption
    caption = caption.replace(
        "10,000 two-sided cluster-mass sign-flip permutations with surface-based "
        "parcel adjacency and an absolute cluster-forming t threshold of 2.0 "
        "(cluster-level family-wise p < 0.05)",
        "Benjamini-Hochberg FDR correction across parcels (q < 0.05)",
    )
    caption = caption.replace(
        "cluster-mass permutation correction across parcels",
        "Benjamini-Hochberg FDR correction across parcels",
    ).replace(
        "cluster-mass permutation correction",
        "Benjamini-Hochberg FDR correction",
    )
    return caption.replace(
        " Benjamini-Hochberg FDR maps are retained as sensitivity results.", ""
    )


def _load_bundle(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and minimally validate one real feature-modulation bundle."""
    metadata = json.loads((directory / "observed.json").read_text())
    if metadata.get("provenance", {}).get("data_mode") != "real":
        raise ValueError("Panel 1 rendering requires real data")
    with np.load(directory / "observed.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "raw_psd_modulation", "raw_psd_auc",
        "fooof_modulation", "fooof_auc",
        "corrected_psd_modulation", "corrected_psd_auc",
        "decoding_p_tmax", "parcel_order",
        "frequency", "spectrum_in", "spectrum_out",
        "aperiodic_spectrum_in", "aperiodic_spectrum_out",
        "corrected_spectrum_in", "corrected_spectrum_out",
        "periodic_spectrum_in", "periodic_spectrum_out",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"feature-modulation bundle lacks arrays: {missing}")
    for family in ("raw_psd", "fooof", "corrected_psd"):
        if not any(f"{family}_p_{method}" in arrays for method in ("cluster", "fdr")):
            raise ValueError(f"feature-modulation bundle lacks {family} corrected p-values")
    return arrays, metadata


def _select_weighting(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any], weighting: str
) -> dict[str, np.ndarray]:
    """Expose one stored weighting variant through the renderer's stable keys."""
    if weighting not in {"equal_window", "equal_run"}:
        raise ValueError(f"unknown Panel 1 weighting: {weighting}")
    available = metadata.get("summary", {}).get("available_weightings")
    if available is None:
        if weighting == "equal_run":
            return arrays
        raise ValueError(
            "legacy bundle contains only equal-run results; rerun the analysis for "
            "equal-window output or pass --weighting=equal_run"
        )
    if weighting not in available:
        raise ValueError(f"bundle does not contain {weighting} results")
    if weighting == "equal_window":
        return arrays
    selected = dict(arrays)
    suffix = "_equal_run"
    for name in tuple(arrays):
        if name.endswith(suffix):
            selected[name.removesuffix(suffix)] = arrays[name]
    return selected


def _draw_composite(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any],
    cache_directory: Path | None = None,
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
    """Draw the legacy six-row composition from corrected bundle arrays."""
    figure = plt.figure(figsize=(14.7, 13.0), dpi=150, facecolor="white")
    grid = GridSpec(
        6, 16, figure=figure, width_ratios=[1] * 14 + [0.125, 0.125],
        height_ratios=[1, 1, 0.78, 0.78, 1, 1],
        left=0.045, right=0.955, top=0.965, bottom=0.045,
        hspace=0.32, wspace=0.12,
    )
    fsaverage = _get_fsaverage_surfaces()
    parcel_order = arrays["parcel_order"].astype(str).tolist()
    t_values = np.concatenate([
        arrays["raw_psd_modulation"], arrays["fooof_modulation"][:2],
        arrays["corrected_psd_modulation"],
    ])
    auc_values = np.concatenate([
        arrays["raw_psd_auc"], arrays["fooof_auc"][:2], arrays["corrected_psd_auc"],
    ])
    t_limit = float(np.nanpercentile(np.abs(t_values), 98))
    auc_max = max(float(np.nanpercentile(auc_values, 98)), 0.55)
    groups: dict[str, list[plt.Axes]] = {}
    band_labels = [
        f"{band.display_name}\n({band.low_hz:g}-{band.high_hz:g}Hz)"
        for band in CANONICAL_BANDS
    ]
    fooof_labels = ["Exponent", "Offset"]

    def spatial_row(row: int, name: str, values: np.ndarray,
                    p_values: np.ndarray, labels: list[str], decoding: bool) -> None:
        axes = []
        first_column = 10 if len(labels) == 2 else 0
        value_min, value_max = ((0.5, auc_max) if decoding else (-t_limit, t_limit))
        color_map = CMAP_AUC if decoding else CMAP_T
        for index, label in enumerate(labels):
            column = first_column + 2 * index
            axis = figure.add_subplot(grid[row, column:column + 2])
            mask = p_values[index] < 0.05
            _plot_brain(axis, values[index], mask, parcel_order,
                        "schaefer_400", fsaverage, value_min, value_max, color_map,
                        cache_directory=cache_directory)
            axis.set_xlabel(f"{label}\n(n={int(mask.sum())} sig)", fontsize=8.5)
            axes.append(axis)
        color_axis = figure.add_subplot(grid[row, 14:16])
        _add_colorbar(figure, color_axis, value_min, value_max, color_map, decoding)
        axes.append(color_axis)
        groups[name] = axes

    spatial_row(0, PANEL_NAMES[0], arrays["raw_psd_modulation"],
                _primary_p(arrays, "raw_psd"), band_labels, False)
    spatial_row(1, PANEL_NAMES[1], arrays["raw_psd_auc"],
                arrays["decoding_p_tmax"][:7], band_labels, True)
    groups.update(_spectral_rows(figure, grid, arrays))
    spatial_row(2, PANEL_NAMES[6], arrays["fooof_modulation"][:2],
                _primary_p(arrays, "fooof")[:2], fooof_labels, False)
    spatial_row(3, PANEL_NAMES[7], arrays["fooof_auc"][:2],
                arrays["decoding_p_tmax"][7:9], fooof_labels, True)
    spatial_row(4, PANEL_NAMES[8], arrays["corrected_psd_modulation"],
                _primary_p(arrays, "corrected_psd"), band_labels, False)
    spatial_row(5, PANEL_NAMES[9], arrays["corrected_psd_auc"],
                arrays["decoding_p_tmax"][9:], band_labels, True)
    _stamp_groups(figure, groups)
    return figure, groups


def _primary_p(arrays: dict[str, np.ndarray], family: str) -> np.ndarray:
    """Prefer primary cluster-FWER maps while supporting legacy FDR bundles."""
    cluster_key = f"{family}_p_cluster"
    return arrays[cluster_key] if cluster_key in arrays else arrays[f"{family}_p_fdr"]


def _spectral_rows(
    figure: plt.Figure, grid: GridSpec, arrays: dict[str, np.ndarray]
) -> dict[str, list[plt.Axes]]:
    """Draw legacy spectrum and OUT-minus-IN subpanels."""
    frequency = np.asarray(arrays["frequency"], dtype=float)
    frequency_mask = np.isfinite(frequency) & (frequency >= 2.0) & (frequency <= 120.0)
    frequency = frequency[frequency_mask]
    specifications = (
        (2, 0, PANEL_NAMES[2], "raw", "PSD (log$_{10}$)"),
        (2, 5, PANEL_NAMES[3], "aperiodic", None),
        (3, 0, PANEL_NAMES[4], "corrected", "PSD (log$_{10}$)"),
        (3, 5, PANEL_NAMES[5], "periodic", None),
    )
    groups = {}
    for row, column, name, key, ylabel in specifications:
        subgrid = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=grid[row, column:column + 5],
            height_ratios=[7, 3], hspace=0.05,
        )
        paired_name = PANEL_NAMES[2] if name == PANEL_NAMES[3] else PANEL_NAMES[4]
        paired_axes = groups.get(paired_name)
        axis = figure.add_subplot(
            subgrid[0], sharey=paired_axes[0] if paired_axes else None
        )
        delta_axis = figure.add_subplot(
            subgrid[1],
            sharex=axis,
            sharey=paired_axes[1] if paired_axes else None,
        )
        inside = _subject_or_mean(arrays, key, "in")[:, frequency_mask]
        outside = _subject_or_mean(arrays, key, "out")[:, frequency_mask]
        _plot_spectrum(axis, frequency, inside, outside,
                       show_legend=key == "raw", axhline_zero=key in {"corrected", "periodic"},
                       ax_delta=delta_axis, in_linestyle="--")
        if ylabel:
            axis.set_ylabel(ylabel, fontsize=8.5)
        if paired_axes:
            plt.setp(axis.get_yticklabels(), visible=False)
            plt.setp(delta_axis.get_yticklabels(), visible=False)
        delta_axis.set_xlabel("Frequency (Hz)", fontsize=8.5)
        plt.setp(axis.get_xticklabels(), visible=False)
        groups[name] = [axis, delta_axis]
    return groups


def _subject_or_mean(arrays: dict[str, np.ndarray], key: str, state: str) -> np.ndarray:
    """Use subject curves when present and otherwise retain legacy mean lines."""
    prefix = "" if key == "raw" else f"{key}_"
    subject_key = f"subject_{prefix}spectrum_{state}"
    mean_key = f"{prefix}spectrum_{state}"
    values = np.asarray(arrays.get(subject_key, arrays[mean_key]), dtype=float)
    return values if values.ndim == 2 else values[None, :]


def _add_colorbar(figure: plt.Figure, axis: plt.Axes, value_min: float,
                  value_max: float, color_map: str, decoding: bool) -> None:
    """Add the established slim row colorbar."""
    position = axis.get_position()
    axis.set_position([position.x0 + position.width * 0.29,
                       position.y0 + position.height * 0.11,
                       position.width * 0.42, position.height * 0.78])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(value_min, value_max), cmap=color_map),
        cax=axis,
    )
    colorbar.set_label("AUC" if decoding else "T-values", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    if not decoding:
        axis.text(0.5, 1.06, "OUT > IN", transform=axis.transAxes,
                  fontsize=7, ha="center", color="#a40000")
        axis.text(0.5, -0.06, "IN > OUT", transform=axis.transAxes,
                  fontsize=7, ha="center", va="top", color="#00408a")


def _stamp_groups(figure: plt.Figure, groups: dict[str, list[plt.Axes]]) -> None:
    """Place A--J labels at their established component origins."""
    figure.canvas.draw()
    for name in PANEL_NAMES:
        axes = groups[name]
        x0 = min(axis.get_position().x0 for axis in axes)
        y1 = max(axis.get_position().y1 for axis in axes)
        figure.text(x0, y1 + 0.008, name[0], fontsize=14,
                    fontweight="bold", ha="left", va="bottom")


def _write_slides(
    arrays: dict[str, np.ndarray], groups: dict[str, list[plt.Axes]],
    reports_root: Path, common: dict[str, Any],
) -> list[Path]:
    """Render native, self-contained 16:9 slide figures for Panel 1."""
    _remove_obsolete_slide_exports(reports_root)
    bands = [
        f"{band.display_name}\n{band.low_hz:g}–{band.high_hz:g} Hz"
        for band in CANONICAL_BANDS
    ]
    t_limit = float(np.nanpercentile(np.abs(np.concatenate([
        arrays["raw_psd_modulation"], arrays["fooof_modulation"][:2],
        arrays["corrected_psd_modulation"],
    ])), 98))
    auc_max = max(float(np.nanpercentile(np.concatenate([
        arrays["raw_psd_auc"], arrays["fooof_auc"][:2],
        arrays["corrected_psd_auc"],
    ]), 98)), 0.55)
    spatial = (
        (1, PANEL_NAMES[0], "Raw PSD modulation: OUT − IN", arrays["raw_psd_modulation"],
         _primary_p(arrays, "raw_psd"), bands, False),
        (2, PANEL_NAMES[1], "Decoding attentional state from raw PSD", arrays["raw_psd_auc"],
         arrays["decoding_p_tmax"][:7], bands, True),
        (4, PANEL_NAMES[6], "Aperiodic parameter modulation: OUT − IN",
         arrays["fooof_modulation"][:2], _primary_p(arrays, "fooof")[:2],
         ["Exponent", "Offset"], False),
        (5, PANEL_NAMES[7], "Decoding attentional state from aperiodic parameters",
         arrays["fooof_auc"][:2], arrays["decoding_p_tmax"][7:9],
         ["Exponent", "Offset"], True),
        (6, PANEL_NAMES[8], "Aperiodic-corrected PSD modulation: OUT − IN",
         arrays["corrected_psd_modulation"], _primary_p(arrays, "corrected_psd"),
         bands, False),
        (7, PANEL_NAMES[9], "Decoding attentional state from corrected PSD",
         arrays["corrected_psd_auc"], arrays["decoding_p_tmax"][9:], bands, True),
    )
    outputs = [
        _write_spatial_slide(
            reports_root, common, _group_brain_images(groups, spec[1]),
            t_limit, auc_max, *spec
        )
        for spec in spatial
    ]
    outputs.insert(2, _write_spectral_progression_slide(arrays, reports_root, common))
    return outputs


def _remove_obsolete_slide_exports(reports_root: Path) -> None:
    """Remove only the superseded ten crop-style Panel 1 slide artifacts."""
    directory = reports_root / "figures" / "slides" / "panel1_feature_modulation"
    for index, name in enumerate(PANEL_NAMES, start=1):
        path = directory / f"{index:02d}_{name}.png"
        path.unlink(missing_ok=True)
        path.with_name(f"{path.name}.json").unlink(missing_ok=True)


def _write_spatial_slide(
    reports_root: Path, common: dict[str, Any], brain_images: list[np.ndarray],
    t_limit: float, auc_max: float, index: int,
    name: str, title: str, values: np.ndarray, p_values: np.ndarray,
    labels: list[str], decoding: bool,
) -> Path:
    """Render one large slide-native spatial-map family."""
    figure = plt.figure(
        figsize=slide_style.SLIDE_FIGSIZE, dpi=slide_style.SLIDE_DPI, facecolor="white"
    )
    figure.suptitle(title, fontsize=slide_style.TITLE_SIZE, fontweight="bold", y=0.965)
    figure.text(
        0.5, 0.905, _slide_subtitle(common, decoding),
        ha="center", va="center",
        fontsize=slide_style.SUBTITLE_SIZE, color=slide_style.SUBTITLE_COLOR,
    )
    rows, columns = ((2, 4) if len(labels) > 4 else (1, len(labels)))
    grid = figure.add_gridspec(
        rows, columns, left=0.035, right=0.90, top=0.84, bottom=0.12,
        hspace=0.24, wspace=0.08,
    )
    minimum, maximum = ((0.5, auc_max) if decoding else (-t_limit, t_limit))
    color_map = CMAP_AUC if decoding else CMAP_T
    for item, label in enumerate(labels):
        axis = figure.add_subplot(grid[item // columns, item % columns])
        mask = p_values[item] < 0.05
        axis.imshow(brain_images[item], interpolation="bilinear", aspect="equal")
        axis.set_axis_off()
        axis.set_title(
            f"{label}\n{int(mask.sum())} significant parcels",
            fontsize=(
                slide_style.CELL_TITLE_SIZE if len(labels) > 4
                else slide_style.LARGE_CELL_TITLE_SIZE
            ),
            pad=7, fontweight="semibold",
        )
    color_axis = figure.add_axes([0.925, 0.22, 0.018, 0.52])
    _add_slide_colorbar(figure, color_axis, minimum, maximum, color_map, decoding)
    figure.text(
        0.5, 0.045, _slide_footer(common, decoding),
        ha="center", fontsize=slide_style.FOOTER_SIZE, color=slide_style.FOOTER_COLOR,
    )
    return _save_slide(figure, reports_root, index, name, common)


def _group_brain_images(
    groups: dict[str, list[plt.Axes]], name: str
) -> list[np.ndarray]:
    """Reuse rendered brain rasters while rebuilding all slide typography/layout."""
    images = []
    for axis in groups[name]:
        if axis.images:
            images.append(np.asarray(axis.images[0].get_array()))
    if not images:
        raise ValueError(f"composite group contains no rendered brain maps: {name}")
    return images


def _write_spectral_progression_slide(
    arrays: dict[str, np.ndarray], reports_root: Path, common: dict[str, Any]
) -> Path:
    """Show how the original spectrum is decomposed and corrected."""
    frequency = np.asarray(arrays["frequency"], dtype=float)
    frequency_mask = np.isfinite(frequency) & (frequency >= 2.0) & (frequency <= 120.0)
    frequency = frequency[frequency_mask]
    figure = plt.figure(
        figsize=slide_style.SLIDE_FIGSIZE, dpi=slide_style.SLIDE_DPI, facecolor="white"
    )
    figure.suptitle(
        "Spectral decomposition of attentional-state effects",
        fontsize=slide_style.TITLE_SIZE, fontweight="bold", y=0.97,
    )
    figure.text(
        0.5, 0.915,
        f"IN vs OUT · {_weighting_label(common)} · {_uncertainty_label(common)}",
        ha="center", fontsize=slide_style.SUBTITLE_SIZE, color=slide_style.SUBTITLE_COLOR,
    )
    cells = (
        ("Original spectrum", "raw", (0.045, 0.23, 0.23, 0.56)),
        ("Aperiodic component", "aperiodic", (0.38, 0.55, 0.24, 0.28)),
        ("Aperiodic-corrected spectrum", "corrected", (0.725, 0.23, 0.24, 0.56)),
        ("Modeled periodic peaks", "periodic", (0.38, 0.13, 0.24, 0.25)),
    )
    for title, key, bounds in cells:
        axis, difference_axis = _add_slide_spectrum_cell(
            figure, bounds, frequency,
            _subject_or_mean(arrays, key, "in")[:, frequency_mask],
            _subject_or_mean(arrays, key, "out")[:, frequency_mask],
            zero_line=key in {"corrected", "periodic"},
            show_legend=key == "raw",
        )
        figure.text(
            bounds[0], bounds[1] + bounds[3] + 0.015,
            title, fontsize=slide_style.CELL_TITLE_SIZE, fontweight="bold", ha="left",
        )
        if key in {"raw", "corrected"}:
            axis.set_ylabel("PSD (log$_{10}$)", fontsize=slide_style.AXIS_LABEL_SIZE)
        difference_axis.set_xlabel("Frequency (Hz)", fontsize=slide_style.AXIS_LABEL_SIZE)
    overlay = figure.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_axis_off()
    _flow_arrow(
        overlay, (0.295, 0.61), (0.370, 0.69), "isolate 1/f",
        label_position=(0.327, 0.64),
    )
    _flow_arrow(
        overlay, (0.295, 0.40), (0.370, 0.28), "fit peaks", curved=True,
        label_position=(0.327, 0.38),
    )
    _flow_arrow(
        overlay, (0.630, 0.73), (0.715, 0.73), "original − aperiodic",
        label_position=(0.672, 0.76),
    )
    figure.text(
        0.5, 0.018,
        "Lower axes show OUT − IN; orange indicates OUT > IN and blue indicates IN > OUT",
        ha="center", fontsize=slide_style.FOOTER_SIZE, color=slide_style.FOOTER_COLOR,
    )
    return _save_slide(
        figure, reports_root, 3, "C-F_spectral_decomposition", common,
    )


def _add_slide_spectrum_cell(
    figure: plt.Figure, bounds: tuple[float, float, float, float],
    frequency: np.ndarray, inside: np.ndarray, outside: np.ndarray, *,
    zero_line: bool, show_legend: bool,
) -> tuple[plt.Axes, plt.Axes]:
    """Draw one spectrum-plus-difference cell inside explicit slide bounds."""
    left, bottom, width, height = bounds
    grid = GridSpec(
        2, 1, figure=figure, height_ratios=[7, 3], hspace=0.05,
        left=left, right=left + width, bottom=bottom, top=bottom + height,
    )
    axis = figure.add_subplot(grid[0])
    difference_axis = figure.add_subplot(grid[1], sharex=axis)
    _plot_spectrum(
        axis, frequency, inside, outside, axhline_zero=zero_line,
        ax_delta=difference_axis, show_legend=show_legend, in_linestyle="--",
    )
    axis.tick_params(labelsize=slide_style.TICK_LABEL_SIZE)
    difference_axis.tick_params(labelsize=slide_style.TICK_LABEL_SIZE - 2)
    difference_axis.set_ylabel("OUT−IN", fontsize=slide_style.TICK_LABEL_SIZE - 1)
    plt.setp(axis.get_xticklabels(), visible=False)
    for line in (*axis.lines, *difference_axis.lines):
        line.set_linewidth(max(line.get_linewidth(), 1.7))
    if show_legend:
        axis.legend(loc="best", fontsize=slide_style.LEGEND_SIZE, frameon=False)
    return axis, difference_axis


def _flow_arrow(
    axis: plt.Axes, start: tuple[float, float], end: tuple[float, float],
    label: str, *, curved: bool = False,
    label_position: tuple[float, float] | None = None,
) -> None:
    """Add one explanatory arrow to the spectral decomposition flow."""
    axis.annotate(
        "", xy=end, xytext=start, xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>", "lw": 3.0, "color": "#555555",
            "mutation_scale": 20,
            "connectionstyle": "arc3,rad=-0.25" if curved else "arc3,rad=0",
        },
    )
    label_x, label_y = label_position or start
    axis.text(
        label_x, label_y, label, transform=axis.transAxes,
        ha="center", va="center", fontsize=slide_style.ANNOTATION_SIZE, color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )


def _add_slide_colorbar(
    figure: plt.Figure, axis: plt.Axes, minimum: float, maximum: float,
    color_map: str, decoding: bool,
) -> None:
    """Add a presentation-scale colorbar with directional interpretation."""
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(
            norm=mcolors.Normalize(minimum, maximum), cmap=color_map
        )
        , cax=axis,
    )
    colorbar.set_label("Cross-validated ROC AUC" if decoding else "Paired t statistic",
                       fontsize=slide_style.AXIS_LABEL_SIZE, labelpad=10)
    colorbar.ax.tick_params(labelsize=slide_style.TICK_LABEL_SIZE)
    if not decoding:
        axis.text(0.5, 1.04, "OUT > IN", transform=axis.transAxes, ha="center",
                  fontsize=slide_style.ANNOTATION_SIZE, color=slide_style.POSITIVE_COLOR)
        axis.text(0.5, -0.04, "IN > OUT", transform=axis.transAxes, ha="center", va="top",
                  fontsize=slide_style.ANNOTATION_SIZE, color=slide_style.NEGATIVE_COLOR)


def _slide_subtitle(common: dict[str, Any], decoding: bool) -> str:
    """Return a concise self-contained subtitle for a spatial slide."""
    map_correction = str(common.get("map_correction") or "").lower()
    inference = (
        "10,000 within-subject permutations · maximum parcel AUC FWER p < 0.05"
        if decoding else (
            "Paired participant-level test · BH-FDR q < 0.05"
            if "benjamini" in map_correction or "fdr" in map_correction else
            "Paired participant-level test · cluster-mass permutation FWER p < 0.05"
        )
    )
    return f"{_weighting_label(common)} · {inference}"


def _slide_footer(common: dict[str, Any], decoding: bool) -> str:
    """Return the contrast and masking explanation printed on every slide."""
    if decoding:
        return "Colored parcels survive maximum-statistic correction; chance AUC = 0.50"
    map_correction = str(common.get("map_correction") or "").lower()
    if "benjamini" in map_correction or "fdr" in map_correction:
        return "Colored parcels survive BH-FDR correction; gray parcels are not significant"
    return "Colored parcels belong to significant spatial clusters; gray parcels are not significant"


def _weighting_label(common: dict[str, Any]) -> str:
    """Convert stored weighting metadata to a presentation label."""
    return str(common.get("weighting", "equal_window")).replace("_", "-") + " weighting"


def _uncertainty_label(common: dict[str, Any]) -> str:
    """Describe whether subject-level SEM ribbons are available."""
    uncertainty = str(common.get("spectral_uncertainty", ""))
    return (
        "mean ± SEM across participants"
        if "SEM" in uncertainty else "group mean; SEM unavailable in legacy bundle"
    )


def _save_slide(
    figure: plt.Figure, reports_root: Path, index: int, name: str,
    common: dict[str, Any],
) -> Path:
    """Save one exact 2560×1440 slide PNG and its provenance sidecar."""
    path = (
        reports_root / "figures" / "slides" / "panel1_feature_modulation"
        / f"{index:02d}_{name}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=slide_style.SLIDE_DPI, facecolor="white")
    plt.close(figure)
    _write_sidecar(path, {**common, "component": name, "path": str(path)})
    return path


def _metadata(metadata: dict[str, Any], directory: Path, output: Path,
              component: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Build a compact provenance sidecar for compatibility artifacts."""
    provenance = metadata["provenance"]
    has_subject_spectra = "subject_spectrum_in" in arrays
    return {
        "analysis_id": provenance["analysis_id"],
        "data_mode": "real",
        "panel": "panel1",
        "component": component,
        "path": str(output),
        "inputs": [str(directory / "observed.npz"), str(directory / "observed.json")],
        "git_commit": provenance.get("git", {}).get("commit"),
        "config_hash": provenance.get("config_hash"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": "canonical Panel 1 manuscript and native 16:9 slide renderer",
        "map_correction": metadata.get("summary", {}).get("map_correction"),
        "spectral_uncertainty": "subject SEM" if has_subject_spectra else "unavailable in compact bundle",
        "caption": str(
            output.parent / "panel1_feature_modulation.txt"
            if component == "composite"
            else Path("reports/figures/manuscript/panel1_feature_modulation.txt")
        ),
    }


def _write_sidecar(path: Path, metadata: dict[str, Any]) -> None:
    """Write one JSON artifact sidecar."""
    sidecar = (
        path.with_suffix(".json")
        if metadata.get("component") == "composite"
        else path.with_name(f"{path.name}.json")
    )
    sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    """Render corrected bundle data with the established Panel 1 design."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-directory", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--weighting", choices=("equal_window", "equal_run"), default="equal_window"
    )
    arguments = parser.parse_args()
    for output in render_bundle(
        arguments.bundle_directory, arguments.reports_root,
        weighting=arguments.weighting,
    ):
        print(output)


if __name__ == "__main__":
    main()
