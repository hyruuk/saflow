"""Run independently schedulable observed Saflow analysis scientific cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from code.analysis.contracts import FEATURE_MODULATION_FEATURES, CORRECTED_FEATURES
from code.analysis.decoding import DecodingConfig
from code.analysis.real_inputs import AnalysisInputs, load_real_inputs
from code.analysis.result_io import write_result_bundle
from code.analysis.provenance import resolve_analysis_directory
from code.analysis.workers import (
    compute_feature_modulation_statistics,
    compute_multifeature_model,
    compute_network_coupling,
    compute_network_modulation,
)
from code.utils.config import load_config
from code.utils.yeo_networks import get_network_assignments
from code.statistics.run_group_statistics import build_atlas_adjacency


def run_observed_cell(args: argparse.Namespace) -> Path:
    """Load authoritative inputs and execute one independent observed cell."""
    config = load_config(args.config)
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    inputs = load_real_inputs(config, subjects, runs)
    analysis_dir = _analysis_directory(config, args.analysis_root, args.analysis_id)
    provenance = _provenance(analysis_dir, args.node, args.cell_index)
    provenance["inputs"] = list(inputs.input_inventory)
    result, directory = _dispatch(args, config, inputs, analysis_dir)
    write_result_bundle(directory, result, provenance)
    return directory


def _dispatch(
    args: argparse.Namespace,
    config: dict[str, Any],
    inputs: AnalysisInputs,
    analysis_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Dispatch a node name to its single-purpose scientific worker."""
    analysis_workflow = config.get("analysis_workflow", {})
    if args.node == "feature_modulation_statistics":
        feature = _require_member(args.feature, FEATURE_MODULATION_FEATURES, "feature")
        result = _run_feature_modulation(inputs, feature, analysis_workflow)
        return (
            result,
            analysis_dir / "feature_modulation" / "partials" / "statistics" / feature,
        )
    if args.node == "multifeature_decoding_models":
        model = _require_member(
            args.model,
            ("state", "lapse_within_IN", "lapse_within_OUT"),
            "model",
        )
        result = compute_multifeature_model(
            inputs.feature_tensor,
            inputs.states,
            inputs.outcomes,
            inputs.subjects,
            model=model,
            config=_decoding_config(analysis_workflow),
        )
        result.update(
            {
                "model": model,
                "feature_order": CORRECTED_FEATURES,
                "parcel_order": inputs.parcel_order,
            }
        )
        return (
            result,
            analysis_dir / "multifeature_decoding" / "partials" / "observed" / model,
        )
    if args.node in {"network_factorial_modulation", "network_coupling"}:
        feature = _require_member(args.feature, CORRECTED_FEATURES, "feature")
        index = CORRECTED_FEATURES.index(feature)
        values = inputs.feature_tensor[:, :, index : index + 1]
        networks = _network_assignments(inputs.parcel_order)
        common = {
            "minimum_windows": int(
                analysis_workflow.get(
                    "minimum_coupling_windows"
                    if args.node == "network_coupling"
                    else "minimum_modulation_windows",
                    5,
                )
            ),
            # Authoritative synchronized families are recomputed only after
            # all ten feature cells pass aggregation.
            "n_permutations": 1,
            "seed": int(analysis_workflow.get("random_seed", 42)) + index,
        }
        if args.node == "network_coupling":
            result = compute_network_coupling(
                values,
                inputs.coupling_cells,
                inputs.subjects,
                inputs.runs,
                networks,
                **common,
            )
            result["window_label_policy"] = "opposite_state_free_with_mid"
            result["minimum_windows_per_cell"] = common["minimum_windows"]
            branch = "coupling"
        else:
            result = compute_network_modulation(
                values,
                inputs.cells,
                inputs.subjects,
                networks,
                **common,
            )
            branch = "modulation"
        result["feature"] = feature
        return result, analysis_dir / "network_dynamics" / "partials" / branch / feature
    raise ValueError(f"unsupported observed node: {args.node}")


def _run_feature_modulation(
    inputs: AnalysisInputs, feature: str, workflow: dict[str, Any]
) -> dict[str, Any]:
    """Compute both prespecified run-weighting variants for one paired map."""
    index = FEATURE_MODULATION_FEATURES.index(feature)
    adjacency = _parcel_adjacency(inputs.parcel_order)
    common = {
        "feature_order": (feature,),
        "adjacency": adjacency,
        "n_permutations": int(workflow.get("cluster_permutations", 10_000)),
        "cluster_threshold": float(workflow.get("cluster_forming_threshold", 2.0)),
        "seed": int(workflow.get("random_seed", 42)) + index,
    }
    variants = {}
    included_by_weighting = {}
    for weighting in ("equal_window", "equal_run"):
        inside, outside, included = _subject_state_means(
            inputs.feature_modulation_tensor[:, :, index],
            inputs.states,
            inputs.subjects,
            inputs.runs,
            weighting=weighting,
        )
        variants[weighting] = compute_feature_modulation_statistics(
            inside[:, None, :], outside[:, None, :], **common
        )
        included_by_weighting[weighting] = included
    result = {
        **variants,
        "feature": feature,
        "primary_weighting": "equal_window",
        "subject_order_equal_window": included_by_weighting["equal_window"],
        "subject_order_equal_run": included_by_weighting["equal_run"],
        "window_counts": _subject_state_counts(
            inputs.states, inputs.subjects, included_by_weighting["equal_window"]
        ),
    }
    return result


def _parcel_adjacency(parcel_order: tuple[str, ...]) -> list[list[int]]:
    """Return Schaefer-400 surface neighbors in the input parcel order."""
    resolved = build_atlas_adjacency("schaefer_400", list(parcel_order), {})
    if resolved is None:
        raise RuntimeError("Schaefer-400 adjacency is required for primary cluster inference")
    matrix, _ = resolved
    return [matrix.getrow(index).indices.tolist() for index in range(matrix.shape[0])]


def _subject_state_means(
    values: np.ndarray,
    states: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    *,
    weighting: str = "equal_run",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average state windows with equal-run or equal-window weighting."""
    if weighting not in {"equal_run", "equal_window"}:
        raise ValueError(f"unknown feature-modulation weighting: {weighting}")
    inside = []
    outside = []
    included = []
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        in_runs, out_runs = [], []
        in_counts, out_counts = [], []
        for run in np.unique(runs[subject_mask]):
            run_mask = subject_mask & (runs == run)
            in_mask = run_mask & (states == "IN")
            out_mask = run_mask & (states == "OUT")
            if in_mask.any():
                in_runs.append(np.nanmean(values[in_mask], axis=0))
                in_counts.append(int(in_mask.sum()))
            if out_mask.any():
                out_runs.append(np.nanmean(values[out_mask], axis=0))
                out_counts.append(int(out_mask.sum()))
        if in_runs and out_runs:
            inside.append(_combine_run_means(in_runs, in_counts, weighting))
            outside.append(_combine_run_means(out_runs, out_counts, weighting))
            included.append(subject)
    if len(included) < 2:
        raise ValueError(
            "feature-modulation analysis paired inference requires at least two subjects"
        )
    return np.stack(inside), np.stack(outside), np.asarray(included)


def _combine_run_means(
    run_means: list[np.ndarray], counts: list[int], weighting: str
) -> np.ndarray:
    """Combine run means equally or in proportion to retained windows."""
    weights = None if weighting == "equal_run" else np.asarray(counts, dtype=float)
    return np.average(np.stack(run_means), axis=0, weights=weights)


def _subject_state_counts(
    states: np.ndarray, subjects: np.ndarray, included: np.ndarray
) -> np.ndarray:
    """Return paired IN/OUT window counts in included-subject order."""
    return np.asarray(
        [
            [
                np.sum((subjects == subject) & (states == "IN")),
                np.sum((subjects == subject) & (states == "OUT")),
            ]
            for subject in included
        ],
        dtype=int,
    )


def _network_assignments(parcel_order: tuple[str, ...]) -> np.ndarray:
    """Convert canonical Schaefer tokens to the network-dynamics analysis display contract."""
    short = get_network_assignments(parcel_order, n_networks=7)
    names = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default Mode",
    }
    assignments = np.asarray([names.get(value, value) for value in short])
    if len(assignments) != 400 or set(assignments) != set(names.values()):
        raise ValueError("Schaefer-400 does not provide the expected Yeo-7 assignments")
    return assignments


def _decoding_config(config: dict[str, Any]) -> DecodingConfig:
    """Resolve leakage-safe nested-decoding configuration."""
    return DecodingConfig(
        c_grid=tuple(
            float(value)
            for value in config.get("c_grid", (0.001, 0.01, 0.1, 1.0, 10.0, 100.0))
        ),
        inner_splits=int(config.get("inner_splits", 5)),
        seed=int(config.get("random_seed", 42)),
    )


def _analysis_directory(
    config: dict[str, Any], override: str | None, analysis_id: str
) -> Path:
    """Resolve an existing immutable analysis directory."""
    root = (
        Path(override)
        if override
        else Path(config["paths"]["data_root"])
        / "processed"
        / config.get("analysis_workflow", {}).get(
            "processed_directory", "analysis_workflow"
        )
    )
    directory = resolve_analysis_directory(root, analysis_id)
    if not (directory / "provenance.json").exists():
        raise FileNotFoundError(f"analysis provenance not found: {directory}")
    return directory


def _provenance(analysis_dir: Path, node: str, cell_index: int) -> dict[str, Any]:
    """Build a compact cell provenance record from immutable analysis state."""
    analysis = json.loads((analysis_dir / "provenance.json").read_text())
    return {
        "analysis_id": analysis["analysis_id"],
        "data_mode": "real",
        "node": node,
        "cell_index": cell_index,
        "git": analysis["git"],
        "config_hash": analysis["config_hash"],
        "inputs": analysis.get("inputs", []),
        "software": analysis.get("software", {}),
    }


def _require_member(value: str | None, allowed: tuple[str, ...], name: str) -> str:
    """Validate one scheduler-provided scientific cell identifier."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the observed-cell command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--node", required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--feature")
    parser.add_argument("--model")
    parser.add_argument("--subjects")
    parser.add_argument("--runs")
    return parser


def main() -> None:
    """Execute one observed cell."""
    run_observed_cell(build_parser().parse_args())


if __name__ == "__main__":
    main()
