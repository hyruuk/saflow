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
from PIL import Image

from code.analysis.contracts import CANONICAL_BANDS
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

PANEL1_CAPTION = """Figure X. Spectral feature modulation and decoding across attentional states. Participants were classified as in-the-zone (IN) or out-of-the-zone (OUT) using run-specific lower and upper quartiles of reflected-boundary filtered response-time variability. Neural observations were retained only when all eight contributing trials had the same state and none carried the final AR2 artifact flag. Unless explicitly overridden at rendering, state observations were pooled across retained windows within each participant (equal-window weighting); equal-run-weighted results were computed and retained as a sensitivity analysis. (A) Paired subject-level t statistics for OUT minus IN raw power spectral density (PSD) in seven canonical frequency bands. Significance was controlled independently within each feature across the 400 Schaefer parcels using 10,000 two-sided cluster-mass sign-flip permutations with surface-based parcel adjacency and an absolute cluster-forming t threshold of 2.0 (cluster-level family-wise p < 0.05). (B) Cross-validated area under the receiver-operating-characteristic curve (AUC) for parcel-wise decoding of IN versus OUT from raw PSD. Significance was assessed using 10,000 within-subject label permutations and the maximum parcel AUC within each feature (family-wise p < 0.05). (C-F) Group-mean IN and OUT spectra over parcels showing significant FOOOF-exponent modulation: raw PSD (C), fitted aperiodic component (D), aperiodic-corrected spectrum (E), and positive periodic component (F). Ribbons show the subject-level standard error of the mean when subject spectra are available. Lower axes show the signed OUT-minus-IN difference, shaded orange where OUT > IN and blue where IN > OUT. (G) Paired OUT-minus-IN t statistics for FOOOF exponent, offset, and model R², with cluster-mass permutation correction across parcels. (H) Parcel-wise IN-versus-OUT decoding AUC for the three FOOOF parameters, with maximum-statistic permutation correction. (I) Paired OUT-minus-IN t statistics for aperiodic-corrected PSD across seven frequency bands, with cluster-mass permutation correction. (J) Parcel-wise decoding AUC for corrected PSD, with maximum-statistic permutation correction. Colored cortical parcels survive the stated primary correction at 0.05; gray parcels do not. Benjamini-Hochberg FDR maps are retained as sensitivity results. All analyses include 32 participants.\n"""


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
    figure, groups = _draw_composite(arrays, metadata)
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
    slides = _write_slide_crops(figure, groups, reports_root, sidecar)
    _write_spectral_slides(arrays, reports_root, sidecar)
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
        raise ValueError("legacy Panel 1 compatibility rendering requires real data")
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
    arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
    """Draw the legacy six-row composition from corrected bundle arrays."""
    figure = plt.figure(figsize=(13.2, 13.0), dpi=150, facecolor="white")
    grid = GridSpec(
        6, 8, figure=figure, width_ratios=[1] * 7 + [0.25],
        height_ratios=[1, 1, 0.78, 0.78, 1, 1],
        left=0.045, right=0.955, top=0.965, bottom=0.045,
        hspace=0.32, wspace=0.12,
    )
    fsaverage = _get_fsaverage_surfaces()
    parcel_order = arrays["parcel_order"].astype(str).tolist()
    t_values = np.concatenate([
        arrays["raw_psd_modulation"], arrays["fooof_modulation"],
        arrays["corrected_psd_modulation"],
    ])
    auc_values = np.concatenate([
        arrays["raw_psd_auc"], arrays["fooof_auc"], arrays["corrected_psd_auc"],
    ])
    t_limit = float(np.nanpercentile(np.abs(t_values), 98))
    auc_max = max(float(np.nanpercentile(auc_values, 98)), 0.55)
    groups: dict[str, list[plt.Axes]] = {}
    band_labels = [
        f"{band.display_name}\n({band.low_hz:g}-{band.high_hz:g}Hz)"
        for band in CANONICAL_BANDS
    ]
    fooof_labels = ["Exponent", "Offset", "R²"]

    def spatial_row(row: int, name: str, values: np.ndarray,
                    p_values: np.ndarray, labels: list[str], decoding: bool) -> None:
        axes = []
        first_column = 4 if len(labels) == 3 else 0
        value_min, value_max = ((0.5, auc_max) if decoding else (-t_limit, t_limit))
        color_map = CMAP_AUC if decoding else CMAP_T
        for index, label in enumerate(labels):
            axis = figure.add_subplot(grid[row, first_column + index])
            mask = p_values[index] < 0.05
            _plot_brain(axis, values[index], mask, parcel_order,
                        "schaefer_400", fsaverage, value_min, value_max, color_map)
            axis.set_xlabel(f"{label}\n(n={int(mask.sum())} sig)", fontsize=8.5)
            axes.append(axis)
        color_axis = figure.add_subplot(grid[row, 7])
        _add_colorbar(figure, color_axis, value_min, value_max, color_map, decoding)
        axes.append(color_axis)
        groups[name] = axes

    spatial_row(0, PANEL_NAMES[0], arrays["raw_psd_modulation"],
                _primary_p(arrays, "raw_psd"), band_labels, False)
    spatial_row(1, PANEL_NAMES[1], arrays["raw_psd_auc"],
                arrays["decoding_p_tmax"][:7], band_labels, True)
    groups.update(_spectral_rows(figure, grid, arrays))
    spatial_row(2, PANEL_NAMES[6], arrays["fooof_modulation"],
                _primary_p(arrays, "fooof"), fooof_labels, False)
    spatial_row(3, PANEL_NAMES[7], arrays["fooof_auc"],
                arrays["decoding_p_tmax"][7:10], fooof_labels, True)
    spatial_row(4, PANEL_NAMES[8], arrays["corrected_psd_modulation"],
                _primary_p(arrays, "corrected_psd"), band_labels, False)
    spatial_row(5, PANEL_NAMES[9], arrays["corrected_psd_auc"],
                arrays["decoding_p_tmax"][10:], band_labels, True)
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
        (2, 2, PANEL_NAMES[3], "aperiodic", None),
        (3, 0, PANEL_NAMES[4], "corrected", "PSD (log$_{10}$)"),
        (3, 2, PANEL_NAMES[5], "periodic", None),
    )
    groups = {}
    for row, column, name, key, ylabel in specifications:
        subgrid = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=grid[row, column:column + 2],
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
                       ax_delta=delta_axis)
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


def _write_slide_crops(figure: plt.Figure, groups: dict[str, list[plt.Axes]],
                       reports_root: Path, common: dict[str, Any]) -> list[Path]:
    """Export exact legacy component crops as 2560x1440 slide PNGs."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    temporary = reports_root / "figures" / "manuscript" / ".panel1_slide_source.png"
    figure.savefig(temporary, dpi=300, facecolor="white")
    outputs = []
    with Image.open(temporary) as source:
        scale_x = source.width / figure.bbox.width
        scale_y = source.height / figure.bbox.height
        for index, name in enumerate(PANEL_NAMES, start=1):
            boxes = [axis.get_tightbbox(renderer) for axis in groups[name]]
            x0 = max(0, int((min(box.x0 for box in boxes) - 12) * scale_x))
            x1 = min(source.width, int((max(box.x1 for box in boxes) + 12) * scale_x))
            y0 = max(0, int(source.height - (max(box.y1 for box in boxes) + 18) * scale_y))
            y1 = min(source.height, int(source.height - (min(box.y0 for box in boxes) - 12) * scale_y))
            crop = source.crop((x0, y0, x1, y1))
            slide = Image.new("RGB", (2560, 1440), "white")
            crop.thumbnail((2460, 1340), Image.Resampling.LANCZOS)
            slide.paste(crop, ((2560 - crop.width) // 2, (1440 - crop.height) // 2))
            path = reports_root / "figures" / "slides" / "panel1_feature_modulation" / f"{index:02d}_{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            slide.save(path, dpi=(160, 160))
            _write_sidecar(path, {**common, "component": name, "path": str(path)})
            outputs.append(path)
    temporary.unlink()
    return outputs


def _write_spectral_slides(
    arrays: dict[str, np.ndarray],
    reports_root: Path,
    common: dict[str, Any],
) -> None:
    """Replace C--F crops with clean native 16:9 legacy spectrum slides."""
    frequency = np.asarray(arrays["frequency"], dtype=float)
    frequency_mask = np.isfinite(frequency) & (frequency >= 2.0) & (frequency <= 120.0)
    frequency = frequency[frequency_mask]
    specifications = (
        (3, PANEL_NAMES[2], "raw", "PSD (log$_{10}$)"),
        (4, PANEL_NAMES[3], "aperiodic", "PSD (log$_{10}$)"),
        (5, PANEL_NAMES[4], "corrected", "PSD (log$_{10}$)"),
        (6, PANEL_NAMES[5], "periodic", "PSD (log$_{10}$)"),
    )
    for index, name, key, ylabel in specifications:
        figure = plt.figure(figsize=(16, 9), dpi=160, facecolor="white")
        grid = figure.add_gridspec(
            2, 1, height_ratios=[7, 3], hspace=0.05,
            left=0.16, right=0.92, top=0.86, bottom=0.14,
        )
        axis = figure.add_subplot(grid[0])
        delta_axis = figure.add_subplot(grid[1], sharex=axis)
        inside = _subject_or_mean(arrays, key, "in")[:, frequency_mask]
        outside = _subject_or_mean(arrays, key, "out")[:, frequency_mask]
        _plot_spectrum(
            axis, frequency, inside, outside,
            show_legend=key == "raw",
            axhline_zero=key in {"corrected", "periodic"},
            ax_delta=delta_axis,
        )
        axis.set_ylabel(ylabel, fontsize=16)
        axis.tick_params(labelsize=13)
        delta_axis.set_xlabel("Frequency (Hz)", fontsize=16)
        delta_axis.set_ylabel("OUT−IN", fontsize=13)
        delta_axis.tick_params(labelsize=12)
        plt.setp(axis.get_xticklabels(), visible=False)
        figure.suptitle(name[2:].replace("_", " "), fontsize=22, fontweight="bold")
        path = (
            reports_root / "figures" / "slides" / "panel1_feature_modulation"
            / f"{index:02d}_{name}.png"
        )
        figure.savefig(path, dpi=160, facecolor="white")
        plt.close(figure)
        _write_sidecar(path, {**common, "component": name, "path": str(path)})


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
        "renderer": "legacy-layout compatibility adapter",
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
    path.with_name(f"{path.name}.json").write_text(
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
