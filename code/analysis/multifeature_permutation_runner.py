"""Run one synchronized three-model multifeature-decoding analysis permutation chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.classification.multifeature_scientific import (
    NestedRidgeConfig,
    run_primary_analysis,
)
from code.analysis.chunks import derive_chunk_seed
from code.analysis.contracts import CORRECTED_FEATURES
from code.analysis.labels import (
    LABEL_IN,
    LABEL_OUT,
    OUTCOME_COMMISSION_ERROR,
    OUTCOME_CORRECT_OMISSION,
    permute_outcomes_within_run_state,
    reject_bad_windows,
    shift_and_rebuild_labels,
    valid_circular_offsets,
)
from code.analysis.real_inputs import AnalysisInputs, load_real_inputs
from code.analysis.result_io import write_result_bundle
from code.utils.config import load_config

MODEL_ORDER = ("state", "lapse_within_IN", "lapse_within_OUT")


def run_chunk(args: argparse.Namespace) -> Path:
    """Run one deterministic synchronized decoding permutation interval."""
    config = load_config(args.config)
    analysis_dir = _analysis_directory(config, args.analysis_root, args.analysis_id)
    directory = _chunk_directory(analysis_dir, args.chunk_index)
    if args.skip_valid and _compatible_chunk_exists(
        directory,
        args.analysis_id,
        args.chunk_index,
    ):
        return directory
    analysis_workflow = config.get("analysis_workflow", {})
    total = int(analysis_workflow.get("decoding_permutations", 1_000))
    size = int(analysis_workflow.get("decoding_chunk_size", 25))
    start = args.chunk_index * size
    stop = min(start + size, total)
    if start >= total:
        raise ValueError("chunk index lies outside decoding permutations")
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    inputs = load_real_inputs(config, subjects, runs)
    seed = derive_chunk_seed(
        args.analysis_id, "multifeature_decoding", "decoding_families", args.chunk_index
    )
    result = _run_permutations(
        inputs,
        stop - start,
        seed,
        int(analysis_workflow.get("minimum_circular_offset", 24)),
        _nested_config(analysis_workflow),
    )
    result.update(
        {
            "model_order": MODEL_ORDER,
            "feature_order": CORRECTED_FEATURES,
            "parcel_order": inputs.parcel_order,
            "permutation_interval": np.asarray([start, stop]),
            "chunk_index": args.chunk_index,
            "seed": seed,
        }
    )
    analysis = json.loads((analysis_dir / "provenance.json").read_text())
    provenance = {
        "analysis_id": args.analysis_id,
        "data_mode": "real",
        "node": "multifeature_decoding_permutations",
        "cell_index": args.cell_index,
        "git": analysis["git"],
        "config_hash": analysis["config_hash"],
        "inputs": list(inputs.input_inventory),
        "software": analysis.get("software", {}),
        "chunk_index": args.chunk_index,
        "permutation_interval": [start, stop],
        "seed": seed,
    }
    write_result_bundle(directory, result, provenance)
    return directory


def _chunk_directory(analysis_dir: Path, chunk_index: int) -> Path:
    """Return one immutable multifeature-decoding analysis permutation chunk directory."""
    return (
        analysis_dir
        / "multifeature_decoding"
        / "partials"
        / "permutations"
        / f"chunk-{chunk_index:04d}"
    )


def _compatible_chunk_exists(
    directory: Path,
    analysis_id: str,
    chunk_index: int,
) -> bool:
    """Return whether a completed chunk can be reused inside a batch."""
    archive = directory / "observed.npz"
    metadata = directory / "observed.json"
    if not archive.exists() and not metadata.exists():
        return False
    if not archive.exists() or not metadata.exists():
        raise ValueError(f"incomplete immutable result bundle: {directory}")
    provenance = json.loads(metadata.read_text()).get("provenance", {})
    expected = {"analysis_id": analysis_id, "chunk_index": chunk_index}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"incompatible immutable result bundle: {directory}")
    with np.load(archive, allow_pickle=False) as arrays:
        if not arrays.files:
            raise ValueError(f"empty immutable result bundle: {directory}")
    return True


def _run_permutations(
    inputs: AnalysisInputs,
    count: int,
    seed: int,
    minimum_offset: int,
    config: NestedRidgeConfig,
) -> dict[str, np.ndarray]:
    """Fit all three null models with synchronized permutation indices."""
    joint = np.full((count, 3), np.nan)
    standalone = np.full((count, 3, len(CORRECTED_FEATURES)), np.nan)
    feature = np.full_like(standalone, np.nan)
    parcel = np.full((count, 3, len(inputs.parcel_order)), np.nan)
    failures = np.zeros((count, 3), dtype=bool)
    generator = np.random.default_rng(seed)
    numeric_states = np.where(
        inputs.states == "IN", LABEL_IN, np.where(inputs.states == "OUT", LABEL_OUT, 0)
    )
    numeric_outcomes = np.where(
        inputs.outcomes == "correct_omission",
        OUTCOME_CORRECT_OMISSION,
        np.where(
            inputs.outcomes == "commission_error", OUTCOME_COMMISSION_ERROR, 0
        ),
    )
    for permutation in range(count):
        shifted_states = _shift_all_runs(inputs, minimum_offset, generator)
        permuted_outcomes = permute_outcomes_within_run_state(
            numeric_outcomes,
            inputs.subjects,
            inputs.runs,
            numeric_states,
            generator,
        )
        specifications = (
            ("state", shifted_states, numeric_outcomes),
            ("lapse_within_IN", numeric_states, permuted_outcomes),
            ("lapse_within_OUT", numeric_states, permuted_outcomes),
        )
        for model_index, (model, states, outcomes) in enumerate(specifications):
            try:
                result = _fit_null_model(inputs, states, outcomes, model, config)
            except ValueError:
                failures[permutation, model_index] = True
                continue
            joint[permutation, model_index] = result["joint"]["metrics"]["roc_auc"]
            standalone[permutation, model_index] = [
                item["metrics"]["roc_auc"] for item in result["standalone-feature"]
            ]
            feature[permutation, model_index] = result["feature-contribution"][
                "mean_delta_auc"
            ]
            parcel[permutation, model_index] = result["region-contribution"][
                "mean_delta_auc"
            ]
    return {
        "joint_auc": joint,
        "standalone_feature_auc": standalone,
        "feature_contribution": feature,
        "parcel_contribution": parcel,
        "class_failure": failures,
    }


def _shift_all_runs(
    inputs: AnalysisInputs,
    minimum_offset: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Rebuild strict labels once per run and permutation."""
    states = np.zeros(len(inputs.states), dtype=np.int8)
    for context in inputs.run_label_contexts:
        offset = int(
            generator.choice(valid_circular_offsets(len(context.vtc), minimum_offset))
        )
        shifted = shift_and_rebuild_labels(
            context.vtc, context.contributing_indices, offset
        )
        states[context.start : context.stop] = reject_bad_windows(
            shifted, context.contributing_bad_flags
        )
    return states


def _fit_null_model(
    inputs: AnalysisInputs,
    states: np.ndarray,
    outcomes: np.ndarray,
    model: str,
    config: NestedRidgeConfig,
) -> dict:
    """Select and fit one null model with subject-grouped nested LOSO."""
    rare = np.isin(
        outcomes, (OUTCOME_CORRECT_OMISSION, OUTCOME_COMMISSION_ERROR)
    )
    if model == "state":
        selector = np.isin(states, (LABEL_IN, LABEL_OUT))
        labels = (states[selector] == LABEL_OUT).astype(int)
    else:
        target = LABEL_IN if model.endswith("_IN") else LABEL_OUT
        selector = (states == target) & rare
        labels = (outcomes[selector] == OUTCOME_COMMISSION_ERROR).astype(int)
    if np.unique(labels).size < 2 or np.unique(inputs.subjects[selector]).size < 3:
        raise ValueError(f"{model} permutation lacks evaluable classes or subjects")
    return run_primary_analysis(
        inputs.feature_tensor[selector],
        labels,
        inputs.subjects[selector],
        config,
        n_permutations=0,
    )


def _nested_config(config: dict) -> NestedRidgeConfig:
    """Resolve the synchronized nested-ridge settings."""
    return NestedRidgeConfig(
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
    config: dict, override: str | None, analysis_id: str
) -> Path:
    """Resolve an initialized immutable analysis directory."""
    root = (
        Path(override)
        if override
        else Path(config["paths"]["data_root"])
        / "processed"
        / config.get("analysis_workflow", {}).get("processed_directory", "analysis_workflow")
    )
    directory = root / analysis_id
    if not (directory / "provenance.json").exists():
        raise FileNotFoundError(f"analysis provenance not found: {directory}")
    return directory


def main() -> None:
    """Run one synchronized multifeature-decoding analysis permutation chunk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--subjects")
    parser.add_argument("--runs")
    parser.add_argument("--skip-valid", action="store_true")
    run_chunk(parser.parse_args())


if __name__ == "__main__":
    main()
