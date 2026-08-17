"""Render Correct-versus-Lapse network and Schaefer-400 modulation maps."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from code.analysis.contracts import CORRECTED_FEATURES, FEATURE_DISPLAY_NAMES
from code.analysis.outcome_modulation import STATE_ORDER
from code.analysis.provenance import resolve_analysis_directory
from code.analysis.result_io import read_result_bundle
from code.utils.yeo_networks import network_display_name
from code.visualization.panel1_bundle import _add_colorbar
from code.visualization.plot_surface import _get_fsaverage_surfaces
from code.visualization.stats_classif_panel import CMAP_T, _plot_brain

LOGGER = logging.getLogger(__name__)
CAPTION = (
    "Supplementary Figure X. Correct-versus-lapse spectral modulation within "
    "attentional state. Commission-error (Lapse) minus correct-omission (Correct) "
    "effects are shown separately within IN (A, C) and OUT (B, D). A and B show "
    "complete Yeo-7 network-by-feature paired t-statistic matrices; dots mark "
    "synchronized maximum-|t| FWER p < 0.05 across all 63 cells within state. C "
    "and D show genuine Schaefer-400 parcel t maps for every feature; colored maps "
    "are unthresholded and titles report parcels surviving synchronized cluster-mass "
    "FWER correction across the nine-feature family. Each participant contributed "
    "one equal-window mean per state-outcome cell. Eligibility required at least "
    "{minimum_windows} windows in both outcome cells and included {in_n} participants "
    "for IN and {out_n} for OUT. Hierarchical-bootstrap sensitivity repeatedly "
    "matched Correct and Lapse counts within participant, state, and run and then "
    "resampled participants ({balanced_repetitions} repeats).\n"
)


def load_correct_lapse_bundle(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the immutable outcome-modulation result."""
    bundle = read_result_bundle(directory)
    result = bundle["result"]
    for state in STATE_ORDER:
        required = {
            "parcel_t_values",
            "parcel_p_cluster_fwer",
            "network_t_values",
            "network_p_fwer",
            "balanced",
            "subject_n",
        }
        missing = sorted(required - result.get(state, {}).keys())
        if missing:
            raise ValueError(f"{state} outcome-modulation result lacks: {missing}")
        if np.asarray(result[state]["parcel_t_values"]).shape != (9, 400):
            raise ValueError(f"{state} parcel maps must have shape (9, 400)")
    return result, bundle["provenance"]


def render_correct_lapse_panel(
    bundle_directory: Path,
    reports_root: Path,
    output_name: str = "supplement_correct_vs_lapse_modulation.png",
) -> Path:
    """Render network heatmaps and all parcel-level cortical maps."""
    result, provenance = load_correct_lapse_bundle(bundle_directory)
    output = reports_root / "figures" / "manuscript" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    limit = _shared_t_limit(result)
    figure = _draw_panel(result, reports_root, limit)
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    _write_artifacts(output, bundle_directory, result, provenance)
    return output


def _draw_panel(result: dict[str, Any], reports_root: Path, limit: float) -> plt.Figure:
    """Compose two network matrices and two nine-map cortical rows."""
    figure = plt.figure(figsize=(20, 10.5), facecolor="white")
    grid = GridSpec(
        3,
        19,
        figure=figure,
        height_ratios=(1.15, 1, 1),
        width_ratios=(1,) * 18 + (0.22,),
        left=0.035,
        right=0.97,
        bottom=0.045,
        top=0.93,
        hspace=0.35,
        wspace=0.12,
    )
    fsaverage = _get_fsaverage_surfaces()
    parcel_order = np.asarray(result["parcel_order"]).astype(str).tolist()
    for column, state in enumerate(STATE_ORDER):
        state_result = result[state]
        axis = figure.add_axes((0.065 + 0.515 * column, 0.69, 0.39, 0.22))
        _plot_network_heatmap(axis, state_result, f"{'AB'[column]}  {state}: Lapse−Correct", limit)
        _plot_parcel_row(
            figure,
            grid,
            column + 1,
            state_result,
            state,
            parcel_order,
            fsaverage,
            limit,
            reports_root / ".cache" / "correct_lapse_surface",
        )
    figure.suptitle("Correct-versus-lapse modulation within attentional state", fontsize=16)
    return figure


def _plot_network_heatmap(
    axis: plt.Axes, state_result: dict[str, Any], title: str, limit: float
) -> None:
    """Plot all corrected Yeo-network tests for one state."""
    values = np.asarray(state_result["network_t_values"])
    p_values = np.asarray(state_result["network_p_fwer"])
    image = axis.imshow(values, cmap=CMAP_T, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(9), _feature_labels(), rotation=45, ha="right", fontsize=7)
    axis.set_yticks(np.arange(7), _network_labels(), fontsize=7)
    axis.set_title(title, loc="left", fontsize=10)
    for row, column in np.argwhere(p_values < 0.05):
        axis.text(column, row, "•", ha="center", va="center", fontsize=10)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Paired t statistic", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def _plot_parcel_row(
    figure: plt.Figure,
    grid: GridSpec,
    row: int,
    state_result: dict[str, Any],
    state: str,
    parcel_order: list[str],
    fsaverage: dict[str, Any],
    limit: float,
    cache_directory: Path,
) -> None:
    """Render all nine genuine Schaefer-400 maps for one state."""
    t_values = np.asarray(state_result["parcel_t_values"])
    p_values = np.asarray(state_result["parcel_p_cluster_fwer"])
    stability = np.asarray(state_result["balanced"]["direction_stability"])
    for feature_index, feature in enumerate(CORRECTED_FEATURES):
        axis = figure.add_subplot(grid[row, feature_index * 2 : feature_index * 2 + 2])
        _plot_brain(
            axis,
            t_values[feature_index],
            None,
            parcel_order,
            "schaefer_400",
            fsaverage,
            -limit,
            limit,
            CMAP_T,
            cache_directory=cache_directory,
        )
        significant_n = int(np.sum(p_values[feature_index] < 0.05))
        stable_n = int(np.sum(stability[feature_index] >= 0.95))
        axis.set_title(
            f"{FEATURE_DISPLAY_NAMES[feature]}\n{significant_n} sig · {stable_n} ≥95% stable",
            fontsize=7,
            pad=1,
        )
        axis.set_axis_off()
    color_axis = figure.add_subplot(grid[row, 18])
    _add_colorbar(figure, color_axis, -limit, limit, CMAP_T, False)
    figure.text(
        0.006,
        0.49 if row == 1 else 0.185,
        f"{'CD'[row - 1]}  {state}",
        rotation=90,
        va="center",
        fontsize=10,
        fontweight="bold",
    )


def _shared_t_limit(result: dict[str, Any]) -> float:
    """Return a shared robust color limit for network and parcel maps."""
    values = np.concatenate(
        [
            np.asarray(result[state][key]).ravel()
            for state in STATE_ORDER
            for key in ("network_t_values", "parcel_t_values")
        ]
    )
    return max(float(np.nanpercentile(np.abs(values), 99)), 1e-6)


def _feature_labels() -> list[str]:
    """Return display labels in corrected feature order."""
    return [FEATURE_DISPLAY_NAMES[name] for name in CORRECTED_FEATURES]


def _network_labels() -> list[str]:
    """Return display labels in the analysis network order."""
    return [
        network_display_name(name)
        for name in ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
    ]


def _write_artifacts(
    output: Path,
    bundle_directory: Path,
    result: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    """Write caption, exhaustive statistics tables, and provenance."""
    caption = CAPTION.format(
        minimum_windows=result["minimum_windows_per_cell"],
        in_n=result["IN"]["subject_n"],
        out_n=result["OUT"]["subject_n"],
        balanced_repetitions=result["balanced_repetitions"],
    )
    output.with_suffix(".txt").write_text(caption)
    table = output.with_suffix(".csv")
    _write_statistics_table(table, result)
    sidecar = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Correct-versus-Lapse modulation within IN and OUT",
        "source_bundle": str(bundle_directory),
        "analysis_id": provenance.get("analysis_id"),
        "git_commit": provenance.get("git", {}).get("commit"),
        "parameters": provenance.get("parameters", {}),
        "statistics_table": str(table),
        "direction": "commission_error_minus_correct_omission",
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def _write_statistics_table(path: Path, result: dict[str, Any]) -> None:
    """Export every corrected network and parcel test."""
    fields = ("state", "resolution", "location", "feature", "t", "p_fwer", "significant")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for state in STATE_ORDER:
            _write_resolution_rows(writer, state, "network", _network_labels(), result[state])
            _write_resolution_rows(
                writer,
                state,
                "parcel",
                np.asarray(result["parcel_order"]).astype(str).tolist(),
                result[state],
            )


def _write_resolution_rows(
    writer: csv.DictWriter,
    state: str,
    resolution: str,
    locations: list[str],
    state_result: dict[str, Any],
) -> None:
    """Write one complete resolution-specific test family."""
    prefix = "network" if resolution == "network" else "parcel"
    t_values = np.asarray(state_result[f"{prefix}_t_values"])
    p_values = np.asarray(
        state_result[f"{prefix}_p_fwer" if prefix == "network" else "parcel_p_cluster_fwer"]
    )
    if resolution == "parcel":
        t_values, p_values = t_values.T, p_values.T
    for location_index, location in enumerate(locations):
        for feature_index, feature in enumerate(CORRECTED_FEATURES):
            p_value = float(p_values[location_index, feature_index])
            writer.writerow(
                {
                    "state": state,
                    "resolution": resolution,
                    "location": location,
                    "feature": feature,
                    "t": f"{t_values[location_index, feature_index]:.10g}",
                    "p_fwer": f"{p_value:.10g}",
                    "significant": p_value < 0.05,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the correct-versus-lapse renderer parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--analysis-id")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--output-name", default="supplement_correct_vs_lapse_modulation.png")
    return parser


def main() -> None:
    """Resolve the active analysis and render its outcome-modulation bundle."""
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    analysis = resolve_analysis_directory(args.analysis_root, args.analysis_id)
    output = render_correct_lapse_panel(
        analysis / "outcome_modulation", args.reports_root, args.output_name
    )
    LOGGER.info("Wrote Correct-versus-Lapse panel to %s", output)


if __name__ == "__main__":
    main()
