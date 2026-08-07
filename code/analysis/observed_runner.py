"""Run independently schedulable observed Panel analysis scientific cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from code.analysis.contracts import PANEL1_FEATURES, PANEL23_FEATURES
from code.analysis.decoding import DecodingConfig
from code.analysis.real_inputs import RealFigure3Inputs, load_real_inputs
from code.analysis.result_io import write_result_bundle
from code.analysis.workers import (
    compute_panel1_statistics,
    compute_panel2_model,
    compute_panel3_coupling,
    compute_panel3_modulation,
)
from code.utils.config import load_config
from code.utils.yeo_networks import get_network_assignments


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
    inputs: RealFigure3Inputs,
    analysis_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Dispatch a node name to its single-purpose scientific worker."""
    panel_analysis = config.get("panel_analysis", {})
    if args.node == "panel1_statistics":
        feature = _require_member(args.feature, PANEL1_FEATURES, "feature")
        result = _run_panel1_feature(inputs, feature)
        return result, analysis_dir / "panel1" / "partials" / "statistics" / feature
    if args.node == "panel2_observed_models":
        model = _require_member(
            args.model,
            ("state", "lapse_within_IN", "lapse_within_OUT"),
            "model",
        )
        result = compute_panel2_model(
            inputs.feature_tensor,
            inputs.states,
            inputs.outcomes,
            inputs.subjects,
            model=model,
            config=_decoding_config(panel_analysis),
        )
        result.update(
            {
                "model": model,
                "feature_order": PANEL23_FEATURES,
                "parcel_order": inputs.parcel_order,
            }
        )
        return result, analysis_dir / "panel2" / "partials" / "observed" / model
    if args.node in {"panel3_factorial_maps", "panel3_coupling"}:
        feature = _require_member(args.feature, PANEL23_FEATURES, "feature")
        index = PANEL23_FEATURES.index(feature)
        values = inputs.feature_tensor[:, :, index : index + 1]
        networks = _network_assignments(inputs.parcel_order)
        common = {
            "minimum_windows": int(
                panel_analysis.get(
                    "minimum_coupling_windows"
                    if args.node == "panel3_coupling"
                    else "minimum_modulation_windows",
                    10 if args.node == "panel3_coupling" else 5,
                )
            ),
            # Authoritative synchronized families are recomputed only after
            # all ten feature cells pass aggregation.
            "n_permutations": 1,
            "seed": int(panel_analysis.get("random_seed", 42)) + index,
        }
        if args.node == "panel3_coupling":
            result = compute_panel3_coupling(
                values,
                inputs.cells,
                inputs.subjects,
                inputs.runs,
                networks,
                **common,
            )
            branch = "coupling"
        else:
            result = compute_panel3_modulation(
                values,
                inputs.cells,
                inputs.subjects,
                networks,
                **common,
            )
            branch = "modulation"
        result["feature"] = feature
        return result, analysis_dir / "panel3" / "partials" / branch / feature
    raise ValueError(f"unsupported observed node: {args.node}")


def _run_panel1_feature(
    inputs: RealFigure3Inputs, feature: str
) -> dict[str, Any]:
    """Compute one subject-level paired spatial map for Panel 1."""
    index = PANEL1_FEATURES.index(feature)
    inside, outside, included = _subject_state_means(
        inputs.panel1_tensor[:, :, index],
        inputs.states,
        inputs.subjects,
        inputs.runs,
    )
    result = compute_panel1_statistics(
        inside[:, None, :],
        outside[:, None, :],
        feature_order=(feature,),
    )
    result["feature"] = feature
    result["subject_order"] = included
    result["window_counts"] = _subject_state_counts(
        inputs.states, inputs.subjects, included
    )
    return result


def _subject_state_means(
    values: np.ndarray,
    states: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average windows within run, then runs within paired subjects."""
    inside = []
    outside = []
    included = []
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        in_runs = []
        out_runs = []
        for run in np.unique(runs[subject_mask]):
            run_mask = subject_mask & (runs == run)
            in_mask = run_mask & (states == "IN")
            out_mask = run_mask & (states == "OUT")
            if in_mask.any():
                in_runs.append(np.nanmean(values[in_mask], axis=0))
            if out_mask.any():
                out_runs.append(np.nanmean(values[out_mask], axis=0))
        if in_runs and out_runs:
            inside.append(np.nanmean(in_runs, axis=0))
            outside.append(np.nanmean(out_runs, axis=0))
            included.append(subject)
    if len(included) < 2:
        raise ValueError("Panel 1 paired inference requires at least two subjects")
    return np.stack(inside), np.stack(outside), np.asarray(included)


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
    """Convert canonical Schaefer tokens to the Panel 3 display contract."""
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
            for value in config.get(
                "c_grid", (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
            )
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
        / config.get("panel_analysis", {}).get("processed_directory", "panel_analysis")
    )
    directory = root / analysis_id
    if not (directory / "provenance.json").exists():
        raise FileNotFoundError(f"analysis provenance not found: {directory}")
    return directory


def _provenance(
    analysis_dir: Path, node: str, cell_index: int
) -> dict[str, Any]:
    """Build a compact cell provenance record from immutable analysis state."""
    analysis = json.loads((analysis_dir / "provenance.json").read_text())
    return {
        "analysis_id": analysis_dir.name,
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
