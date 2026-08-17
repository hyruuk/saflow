"""Render CO- and CE-specific IN-versus-OUT network modulation.

The existing network-dynamics bundle supplies all subject contrasts and
corrected tests. The sibling feature-modulation bundle supplies only the
canonical Schaefer parcel order required to paint Yeo networks on cortex.
"""

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
from code.analysis.provenance import resolve_analysis_directory
from code.utils.yeo_networks import (
    YEO7_NETWORKS,
    get_network_assignments,
    network_display_name,
)
from code.visualization.panel1_bundle import _add_colorbar
from code.visualization.plot_surface import _get_fsaverage_surfaces
from code.visualization.stats_classif_panel import CMAP_T, _plot_brain

LOGGER = logging.getLogger(__name__)
CONTRAST_INDICES = {"correct_omission": 3, "commission_error": 4}
OUTCOME_LABELS = {
    "correct_omission": "Correct omissions (CO)",
    "commission_error": "Commission errors (CE)",
}
CAPTION = (
    "Supplementary Figure X. Outcome-stratified attentional-state modulation. "
    "Yeo-7 network summaries compare OUT minus IN separately for windows anchored "
    "on correct omissions (CO; A, C) and commission errors (CE; B, D). A and B "
    "show complete network-by-feature paired t-statistic matrices. C and D show "
    "the same unthresholded statistics on cortical surfaces for every feature. "
    "Dots in A and B and starred surface-map titles denote synchronized maximum-|t| "
    "family-wise p < 0.05. FOOOF and corrected-PSD modulation were corrected as "
    "separate prespecified families across all five network-dynamics contrasts. "
    "State and outcome cells were pooled across retained windows within participant "
    "(equal-window weighting). Complete-case analysis included {subject_n} participants.\n"
)


def load_outcome_arrays(
    bundle_directory: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate arrays needed by the outcome-state panel."""
    archive_path = bundle_directory / "observed.npz"
    metadata_path = bundle_directory / "observed.json"
    if not archive_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"network-dynamics bundle is incomplete: {bundle_directory}")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["parcel_order"] = _load_parcel_order(bundle_directory.parent)
    metadata = json.loads(metadata_path.read_text())
    validate_outcome_arrays(arrays, metadata)
    return arrays, metadata


def _load_parcel_order(analysis_directory: Path) -> np.ndarray:
    """Load the Schaefer order from the sibling feature-modulation bundle."""
    archive_path = analysis_directory / "feature_modulation" / "observed.npz"
    if not archive_path.exists():
        raise FileNotFoundError(f"parcel order source is missing: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        parcel_order = np.asarray(archive["parcel_order"]).astype(str)
    if parcel_order.shape != (400,):
        raise ValueError("feature-modulation parcel_order must contain 400 parcels")
    return parcel_order


def validate_outcome_arrays(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Require the corrected nine-feature, five-contrast bundle contract."""
    required = {
        "modulation_contrasts",
        "fooof_t_values",
        "fooof_p_fwer",
        "corrected_psd_t_values",
        "corrected_psd_p_fwer",
        "parcel_order",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"network-dynamics bundle lacks arrays: {missing}")
    expected_shape = (5, 7, len(CORRECTED_FEATURES))
    if arrays["modulation_contrasts"].shape[1:] != expected_shape:
        raise ValueError(
            "modulation_contrasts must be subjects x 5 contrasts x 7 networks x 9 features"
        )
    order = metadata.get("summary", {}).get("contrast_order", [])
    if order[3:5] != ["OUT_vs_IN_within_correct", "OUT_vs_IN_within_lapse"]:
        raise ValueError("network-dynamics contrast order is incompatible")


def outcome_matrices(arrays: dict[str, np.ndarray], outcome: str) -> tuple[np.ndarray, np.ndarray]:
    """Return network-by-feature t and corrected-p matrices."""
    index = CONTRAST_INDICES[outcome]
    return _join_families(arrays, "t_values")[index], _join_families(arrays, "p_fwer")[index]


def _join_families(arrays: dict[str, np.ndarray], suffix: str) -> np.ndarray:
    """Restore the five-by-seven-by-nine inference tensor."""
    fooof = np.asarray(arrays[f"fooof_{suffix}"], dtype=float).reshape(5, 7, 2)
    corrected = np.asarray(arrays[f"corrected_psd_{suffix}"], dtype=float).reshape(5, 7, 7)
    return np.concatenate((fooof, corrected), axis=2)


def render_outcome_state_panel(
    bundle_directory: Path,
    reports_root: Path,
    output_name: str = "supplement_outcome_state_modulation.png",
) -> Path:
    """Render heatmap overviews and all 18 cortical feature maps."""
    arrays, metadata = load_outcome_arrays(bundle_directory)
    output = reports_root / "figures" / "manuscript" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    matrices = {outcome: outcome_matrices(arrays, outcome) for outcome in CONTRAST_INDICES}
    limit = _shared_limit([result[0] for result in matrices.values()])
    figure = _draw_panel(arrays, matrices, limit, reports_root)
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    _write_artifacts(output, bundle_directory, metadata, arrays)
    return output


def _draw_panel(
    arrays: dict[str, np.ndarray],
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    limit: float,
    reports_root: Path,
) -> plt.Figure:
    """Compose two overview matrices and two nine-map cortical rows."""
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
    for column, outcome in enumerate(CONTRAST_INDICES):
        values, p_values = matrices[outcome]
        axis = figure.add_axes((0.065 + 0.515 * column, 0.69, 0.39, 0.22))
        _plot_heatmap(
            axis, values, p_values, f"{'AB'[column]}  {OUTCOME_LABELS[outcome]}: OUT−IN", limit
        )
        _plot_brain_row(
            figure,
            grid,
            column + 1,
            arrays,
            values,
            p_values,
            outcome,
            limit,
            reports_root / ".cache" / "outcome_state_surface",
        )
    figure.suptitle("Outcome-stratified attentional-state modulation", fontsize=16)
    return figure


def _plot_heatmap(
    axis: plt.Axes, values: np.ndarray, p_values: np.ndarray, title: str, limit: float
) -> None:
    """Plot one complete Yeo-network by feature matrix."""
    image = axis.imshow(values, cmap=CMAP_T, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(9), _feature_labels(), rotation=45, ha="right", fontsize=7)
    axis.set_yticks(np.arange(7), _network_labels(), fontsize=7)
    axis.set_title(title, loc="left", fontsize=10)
    for row, column in np.argwhere(p_values < 0.05):
        axis.text(column, row, "•", ha="center", va="center", fontsize=10)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Paired t statistic", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def _plot_brain_row(
    figure: plt.Figure,
    grid: GridSpec,
    row: int,
    arrays: dict[str, np.ndarray],
    values: np.ndarray,
    p_values: np.ndarray,
    outcome: str,
    limit: float,
    cache_directory: Path,
) -> None:
    """Render all nine network-level effects on Schaefer cortical surfaces."""
    parcel_order = arrays["parcel_order"].astype(str).tolist()
    assignments = get_network_assignments(parcel_order, n_networks=7)
    fsaverage = _get_fsaverage_surfaces()
    for feature_index, feature in enumerate(CORRECTED_FEATURES):
        axis = figure.add_subplot(grid[row, feature_index * 2 : feature_index * 2 + 2])
        parcel_values = _network_to_parcels(values[:, feature_index], assignments)
        _plot_brain(
            axis,
            parcel_values,
            None,
            parcel_order,
            "schaefer_400",
            fsaverage,
            -limit,
            limit,
            CMAP_T,
            cache_directory=cache_directory,
        )
        significant_n = int(np.sum(p_values[:, feature_index] < 0.05))
        star = " *" if significant_n else ""
        axis.set_title(
            f"{FEATURE_DISPLAY_NAMES[feature]}{star}\n{significant_n}/7 FWER-significant",
            fontsize=7,
            pad=1,
        )
        axis.set_axis_off()
    color_axis = figure.add_subplot(grid[row, 18])
    _add_colorbar(figure, color_axis, -limit, limit, CMAP_T, False)
    figure.text(
        0.006,
        0.49 if row == 1 else 0.185,
        f"{'CD'[row - 1]}  {OUTCOME_LABELS[outcome]}",
        rotation=90,
        va="center",
        fontsize=10,
        fontweight="bold",
    )


def _network_to_parcels(network_values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Paint seven ordered network statistics onto Schaefer parcels."""
    lookup = dict(zip(YEO7_NETWORKS, np.asarray(network_values, dtype=float)))
    return np.asarray([lookup[name] for name in assignments], dtype=float)


def _feature_labels() -> list[str]:
    """Return display labels in the corrected feature order."""
    return [FEATURE_DISPLAY_NAMES[name] for name in CORRECTED_FEATURES]


def _network_labels() -> list[str]:
    """Return canonical Yeo-7 display labels."""
    return [network_display_name(name) for name in YEO7_NETWORKS]


def _shared_limit(matrices: list[np.ndarray]) -> float:
    """Return a nonzero symmetric limit shared across outcomes and maps."""
    return max(float(np.nanmax(np.abs(np.stack(matrices)))), 1e-6)


def _write_artifacts(
    output: Path,
    bundle_directory: Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    """Write caption, all-test table, and compact provenance sidecar."""
    summary = metadata.get("summary", {})
    provenance = metadata.get("provenance", metadata)
    subject_n = int(arrays["modulation_contrasts"].shape[0])
    output.with_suffix(".txt").write_text(CAPTION.format(subject_n=subject_n))
    table = output.with_suffix(".csv")
    _write_significance_table(table, arrays)
    sidecar = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "CO- and CE-specific OUT-minus-IN network modulation",
        "source_bundle": str(bundle_directory),
        "source_analysis_id": provenance.get("analysis_id"),
        "git_commit": provenance.get("git", {}).get("commit"),
        "subject_n": subject_n,
        "weighting": "equal-window pooled within participant and state-outcome cell",
        "contrast_order": summary.get("contrast_order"),
        "correction": summary.get("correction"),
        "significance_table": str(table),
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def _write_significance_table(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Export corrected tests for every outcome-network-feature cell."""
    fields = ("outcome", "network", "feature", "t", "p_fwer", "significant")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for outcome in CONTRAST_INDICES:
            t_values, p_values = outcome_matrices(arrays, outcome)
            for network_index, network in enumerate(_network_labels()):
                for feature_index, feature in enumerate(CORRECTED_FEATURES):
                    writer.writerow(
                        {
                            "outcome": outcome,
                            "network": network,
                            "feature": feature,
                            "t": f"{t_values[network_index, feature_index]:.10g}",
                            "p_fwer": f"{p_values[network_index, feature_index]:.10g}",
                            "significant": bool(p_values[network_index, feature_index] < 0.05),
                        }
                    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--analysis-id")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--output-name", default="supplement_outcome_state_modulation.png")
    return parser


def main() -> None:
    """Resolve the active analysis and render the panel."""
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    analysis = resolve_analysis_directory(args.analysis_root, args.analysis_id)
    output = render_outcome_state_panel(
        analysis / "network_dynamics", args.reports_root, args.output_name
    )
    LOGGER.info("Wrote outcome-state panel to %s", output)


if __name__ == "__main__":
    main()
