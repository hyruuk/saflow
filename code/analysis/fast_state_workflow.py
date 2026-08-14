"""Prepare, run, and aggregate resumable fixed-ridge state permutations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from code.analysis.chunks import derive_chunk_seed
from code.analysis.contracts import MULTIFEATURE_FEATURES
from code.analysis.labels import LABEL_IN, LABEL_OUT
from code.analysis.multifeature_permutation_runner import _shift_all_runs
from code.analysis.real_inputs import load_real_inputs
from code.analysis.provenance import resolve_analysis_directory
from code.classification.fast_state_scientific import (
    FixedRidgeConfig,
    fit_held_out_subject,
    pool_held_out_folds,
)
from code.classification.multifeature_provenance import git_state
from code.utils.config import load_config

LOGGER = logging.getLogger(__name__)


def _active_analysis(analysis_root: str) -> tuple[Path, str, dict]:
    """Resolve the canonical main analysis and its internal provenance ID."""
    analysis_directory = resolve_analysis_directory(Path(analysis_root))
    provenance = json.loads((analysis_directory / "provenance.json").read_text())
    analysis_id = str(provenance.get("analysis_id", ""))
    if not analysis_id:
        raise ValueError("main analysis provenance lacks an internal analysis ID")
    return analysis_directory, analysis_id, provenance


def _save_array(path: Path, values: np.ndarray) -> None:
    """Atomically save one uncompressed memory-mappable NumPy array."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> Path:
    """Load source files once and materialize immutable state-decoding inputs."""
    config = load_config(args.config)
    analysis_directory, analysis_id, analysis_provenance = _active_analysis(
        args.analysis_root
    )
    directory = analysis_directory / "fast_state"
    if directory.exists():
        LOGGER.warning("Replacing unsaved active fast-state outputs in %s", directory)
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    LOGGER.info("Loading and aligning Schaefer-400 inputs")
    inputs = load_real_inputs(config, subjects, runs)
    indices = [inputs.feature_order.index(name) for name in MULTIFEATURE_FEATURES]
    tensor = np.asarray(inputs.feature_tensor[:, :, indices], dtype=np.float32)
    numeric_states = np.where(
        inputs.states == "IN", LABEL_IN, np.where(inputs.states == "OUT", LABEL_OUT, 0)
    ).astype(np.int8)
    permutations = np.empty((args.n_permutations, len(numeric_states)), dtype=np.int8)
    minimum_offset = int(
        config.get("analysis_workflow", {}).get("minimum_circular_offset", 24)
    )
    for index in range(args.n_permutations):
        seed = derive_chunk_seed(analysis_id, "fast_state", "labels", index)
        permutations[index] = _shift_all_runs(
            inputs, minimum_offset, np.random.default_rng(seed)
        )
    _save_array(directory / "features.npy", tensor)
    _save_array(directory / "subjects.npy", inputs.subjects.astype("U16"))
    _save_array(directory / "observed_states.npy", numeric_states)
    _save_array(directory / "permuted_states.npy", permutations)
    metadata = {
        "analysis_id": analysis_id,
        "space": "schaefer_400",
        "feature_order": list(MULTIFEATURE_FEATURES),
        "parcel_order": list(inputs.parcel_order),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "n_permutations": args.n_permutations,
        "minimum_circular_offset": minimum_offset,
        "alpha": args.alpha,
        "tolerance": args.tolerance,
        "git": git_state(Path.cwd()),
        "analysis_provenance": analysis_provenance,
        "input_inventory": list(inputs.input_inventory),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("Prepared %s with shape %s", directory, tensor.shape)
    return directory


def _load_prepared(directory: Path, stage_local: bool) -> tuple[dict, Path]:
    """Validate prepared metadata and optionally stage arrays to node-local storage."""
    metadata = json.loads((directory / "metadata.json").read_text())
    required = ("features.npy", "subjects.npy", "observed_states.npy", "permuted_states.npy")
    if not all((directory / name).exists() for name in required):
        raise FileNotFoundError(f"incomplete prepared fast-state inputs: {directory}")
    if not stage_local:
        return metadata, directory
    temporary_root = os.environ.get("SLURM_TMPDIR")
    if not temporary_root:
        raise RuntimeError("--stage-local requires SLURM_TMPDIR")
    staged = Path(temporary_root) / "saflow_fast_state"
    staged.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(directory / name, staged / name)
    return metadata, staged


def _fit_states(
    features: np.ndarray,
    states: np.ndarray,
    subjects: np.ndarray,
    ridge: FixedRidgeConfig,
) -> dict[str, object]:
    """Fit fixed-ridge LOSO using strict IN/OUT observations only."""
    selected = np.isin(states, (LABEL_IN, LABEL_OUT))
    flattened = features[selected].reshape(int(selected.sum()), -1)
    labels = (states[selected] == LABEL_OUT).astype(np.int8)
    groups = subjects[selected].astype(str)
    folds = [
        fit_held_out_subject(flattened, labels, groups, subject, ridge)
        for subject in np.unique(groups)
    ]
    result = pool_held_out_folds(folds)
    result["class_counts"] = np.bincount(labels, minlength=2)
    result["selector_count"] = int(selected.sum())
    return result


def run_batch(args: argparse.Namespace) -> Path:
    """Run observed state decoding or one checkpointed permutation batch."""
    analysis_directory, _, _ = _active_analysis(args.analysis_root)
    directory = analysis_directory / "fast_state"
    metadata, arrays = _load_prepared(directory, args.stage_local)
    features = np.load(arrays / "features.npy", mmap_mode="r")
    subjects = np.load(arrays / "subjects.npy", mmap_mode="r")
    ridge = FixedRidgeConfig(metadata["alpha"], metadata["tolerance"])
    if args.observed:
        output = directory / "observed.npz"
        if output.exists() and args.skip_valid:
            return output
        result = _fit_states(
            features, np.load(arrays / "observed_states.npy", mmap_mode="r"), subjects, ridge
        )
        _write_result(output, result)
        return output
    labels = np.load(arrays / "permuted_states.npy", mmap_mode="r")
    start = args.batch_index * args.permutations_per_job
    stop = min(start + args.permutations_per_job, metadata["n_permutations"])
    if start >= stop:
        raise ValueError("permutation batch lies outside prepared range")
    output_directory = directory / "permutations"
    output_directory.mkdir(exist_ok=True)
    for index in range(start, stop):
        output = output_directory / f"permutation-{index:04d}.npz"
        if output.exists() and args.skip_valid:
            continue
        LOGGER.info("Running state permutation %d", index)
        result = _fit_states(features, labels[index], subjects, ridge)
        result["permutation_index"] = index
        _write_result(output, result)
    return output_directory


def _write_result(path: Path, result: dict[str, object]) -> None:
    """Atomically checkpoint one observed or permuted LOSO result."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **result)
    temporary.replace(path)


def aggregate(args: argparse.Namespace) -> Path:
    """Validate complete permutation coverage and compute the primary p-value."""
    analysis_directory, _, _ = _active_analysis(args.analysis_root)
    directory = analysis_directory / "fast_state"
    metadata = json.loads((directory / "metadata.json").read_text())
    with np.load(directory / "observed.npz", allow_pickle=False) as observed:
        observed_auc = float(observed["roc_auc"])
    null = np.empty(metadata["n_permutations"], dtype=float)
    for index in range(len(null)):
        path = directory / "permutations" / f"permutation-{index:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing state permutation: {path}")
        with np.load(path, allow_pickle=False) as result:
            if int(result["permutation_index"]) != index:
                raise ValueError(f"wrong permutation index in {path}")
            null[index] = float(result["roc_auc"])
    p_value = (np.count_nonzero(null >= observed_auc) + 1) / (len(null) + 1)
    output = directory / "inference.npz"
    _write_result(
        output,
        {"observed_auc": observed_auc, "null_auc": null, "p_value": p_value},
    )
    LOGGER.info("Observed AUC %.4f; one-sided permutation p=%.6f", observed_auc, p_value)
    return output


def main() -> None:
    """Run one fast-state workflow stage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--analysis-root", required=True)
    common.add_argument("--config", default="config.yaml")
    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--subjects")
    prepare_parser.add_argument("--runs")
    prepare_parser.add_argument("--n-permutations", type=int, default=1000)
    prepare_parser.add_argument("--alpha", type=float, default=1.0)
    prepare_parser.add_argument("--tolerance", type=float, default=1e-4)
    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--observed", action="store_true")
    run_parser.add_argument(
        "--batch-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    run_parser.add_argument("--permutations-per-job", type=int, default=10)
    run_parser.add_argument("--stage-local", action="store_true")
    run_parser.add_argument("--skip-valid", action="store_true")
    subparsers.add_parser("aggregate", parents=[common])
    arguments = parser.parse_args()
    {"prepare": prepare, "run": run_batch, "aggregate": aggregate}[arguments.command](arguments)


if __name__ == "__main__":
    main()
