"""Scientific stages for population and within-subject state decoding."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np

from code.analysis.chunks import derive_chunk_seed
from code.analysis.contracts import MULTIFEATURE_FEATURES
from code.analysis.fast_state_workflow import _active_analysis, _save_array, _write_result
from code.analysis.labels import LABEL_IN, LABEL_OUT
from code.analysis.multifeature_permutation_runner import _shift_all_runs
from code.analysis.networks import synchronized_sign_flip_test
from code.analysis.real_inputs import load_real_inputs
from code.classification.fast_state_scientific import (
    FixedRidgeConfig,
    compute_grouped_reliance,
    fit_held_out_subject,
    fit_train_test,
    pool_held_out_folds,
)
from code.utils.config import load_config
from code.utils.yeo_networks import YEO7_NETWORKS, get_network_assignments

LOGGER = logging.getLogger(__name__)
REQUIRED_ARRAYS = (
    "features.npy",
    "subjects.npy",
    "runs.npy",
    "observed_states.npy",
    "permuted_states.npy",
    "network_assignments.npy",
)


def prepare(args: argparse.Namespace) -> Path:
    """Materialize one shared Schaefer-400 tensor and all shifted labels."""
    config = load_config(args.config)
    analysis_directory, analysis_id, provenance = _active_analysis(
        config, args.analysis_root
    )
    directory = analysis_directory / "multifeature_state"
    if directory.exists():
        LOGGER.warning("Replacing unsaved multifeature-state outputs in %s", directory)
        shutil.rmtree(directory)
    prepared = directory / "prepared"
    prepared.mkdir(parents=True)
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    LOGGER.info("Loading and aligning shared Schaefer-400 inputs")
    inputs = load_real_inputs(config, subjects, runs)
    feature_indices = [inputs.feature_order.index(name) for name in MULTIFEATURE_FEATURES]
    tensor = np.asarray(inputs.feature_tensor[:, :, feature_indices], dtype=np.float32)
    states = np.where(
        inputs.states == "IN", LABEL_IN, np.where(inputs.states == "OUT", LABEL_OUT, 0)
    ).astype(np.int8)
    shifted = np.empty((args.n_permutations, len(states)), dtype=np.int8)
    minimum_offset = int(
        config.get("analysis_workflow", {}).get("minimum_circular_offset", 24)
    )
    for index in range(args.n_permutations):
        seed = derive_chunk_seed(analysis_id, "state_multifeature", "labels", index)
        shifted[index] = _shift_all_runs(
            inputs, minimum_offset, np.random.default_rng(seed)
        )
    networks = get_network_assignments(inputs.parcel_order, n_networks=7)
    if set(networks) != set(YEO7_NETWORKS):
        raise ValueError("Schaefer-400 parcels do not cover the canonical Yeo-7 networks")
    arrays = {
        "features.npy": tensor,
        "subjects.npy": inputs.subjects.astype("U16"),
        "runs.npy": inputs.runs.astype("U16"),
        "observed_states.npy": states,
        "permuted_states.npy": shifted,
        "network_assignments.npy": networks.astype("U24"),
    }
    for name, values in arrays.items():
        _save_array(prepared / name, values)
    metadata = {
        "analysis_id": analysis_id,
        "analysis_provenance": provenance,
        "space": "schaefer_400",
        "shape": list(tensor.shape),
        "feature_order": list(MULTIFEATURE_FEATURES),
        "parcel_order": list(inputs.parcel_order),
        "network_order": list(YEO7_NETWORKS),
        "subject_order": np.unique(inputs.subjects.astype(str)).tolist(),
        "n_permutations": args.n_permutations,
        "minimum_circular_offset": minimum_offset,
        "alpha": args.alpha,
        "tolerance": args.tolerance,
        "reliance_repeats": args.reliance_repeats,
        "input_inventory": list(inputs.input_inventory),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return directory


def _load(args: argparse.Namespace) -> tuple[Path, dict, dict[str, np.ndarray]]:
    """Load or node-stage the shared arrays."""
    config = load_config(args.config)
    analysis_directory, _, _ = _active_analysis(config, args.analysis_root)
    directory = analysis_directory / "multifeature_state"
    metadata = json.loads((directory / "metadata.json").read_text())
    source = directory / "prepared"
    if not all((source / name).exists() for name in REQUIRED_ARRAYS):
        raise FileNotFoundError(f"incomplete multifeature-state preparation: {source}")
    if args.stage_local:
        temporary = os.environ.get("SLURM_TMPDIR")
        if not temporary:
            raise RuntimeError("--stage-local requires SLURM_TMPDIR")
        staged = Path(temporary) / "saflow_multifeature_state"
        staged.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_ARRAYS:
            shutil.copy2(source / name, staged / name)
        source = staged
    arrays = {name[:-4]: np.load(source / name, mmap_mode="r") for name in REQUIRED_ARRAYS}
    return directory, metadata, arrays


def _state_data(
    arrays: dict[str, np.ndarray], states: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten strict IN/OUT trials while retaining subject and run groups."""
    selected = np.isin(states, (LABEL_IN, LABEL_OUT))
    features = arrays["features"][selected].reshape(int(selected.sum()), -1)
    labels = (states[selected] == LABEL_OUT).astype(np.int8)
    return features, labels, arrays["subjects"][selected], arrays["runs"][selected]


def _ridge(metadata: dict) -> FixedRidgeConfig:
    return FixedRidgeConfig(metadata["alpha"], metadata["tolerance"])


def run_population(args: argparse.Namespace) -> Path:
    """Run observed population LOSO or a checkpointed permutation batch."""
    directory, metadata, arrays = _load(args)
    output_directory = directory / "population"
    output_directory.mkdir(exist_ok=True)
    if args.observed:
        output = output_directory / "observed.npz"
        states = arrays["observed_states"]
        indices = [None]
    else:
        output_directory = output_directory / "permutations"
        output_directory.mkdir(exist_ok=True)
        start = args.batch_index * args.permutations_per_job
        stop = min(start + args.permutations_per_job, metadata["n_permutations"])
        if start >= stop:
            raise ValueError("population permutation batch lies outside prepared range")
        indices = range(start, stop)
        states = arrays["permuted_states"]
    for index in indices:
        output = (
            output
            if index is None
            else output_directory / f"permutation-{index:04d}.npz"
        )
        if output.exists() and args.skip_valid:
            continue
        current_states = states if index is None else states[index]
        features, labels, subjects, _ = _state_data(arrays, current_states)
        folds = [
            fit_held_out_subject(features, labels, subjects, subject, _ridge(metadata))
            for subject in np.unique(subjects.astype(str))
        ]
        result = pool_held_out_folds(folds)
        result["mean_subject_auc"] = float(np.nanmean(result["subject_auc"]))
        if index is not None:
            result["permutation_index"] = index
        _write_result(output, result)
    return output


def _subject_from_cell(metadata: dict, cell_index: int) -> str:
    subjects = metadata["subject_order"]
    if cell_index < 0 or cell_index >= len(subjects):
        raise ValueError("subject cell index outside prepared subject order")
    return str(subjects[cell_index])


def _fit_within_subject(
    arrays: dict[str, np.ndarray], states: np.ndarray, subject: str, ridge: FixedRidgeConfig
) -> dict[str, object]:
    """Fit leave-one-run-out decoding for one subject."""
    features, labels, subjects, runs = _state_data(arrays, states)
    selected = subjects.astype(str) == subject
    features, labels, runs = features[selected], labels[selected], runs[selected].astype(str)
    folds = []
    for run in np.unique(runs):
        estimator, scores, auc = fit_train_test(
            features, labels, runs != run, runs == run, ridge
        )
        del estimator
        folds.append(
            {"held_out_subject": run, "labels": labels[runs == run], "scores": scores,
             "auc": auc, "elapsed_seconds": np.nan}
        )
    result = pool_held_out_folds(folds)
    result["mean_run_auc"] = float(np.nanmean(result["subject_auc"]))
    result["subject"] = subject
    return result


def run_within(args: argparse.Namespace) -> Path:
    """Run one subject's observed or permuted leave-one-run-out decoding."""
    directory, metadata, arrays = _load(args)
    batches_per_subject = math.ceil(
        metadata["n_permutations"] / args.permutations_per_job
    )
    if args.observed:
        subject_index = args.cell_index
        permutation_indices: list[int | None] = [None]
    else:
        subject_index = args.cell_index // batches_per_subject
        batch = args.cell_index % batches_per_subject
        start = batch * args.permutations_per_job
        permutation_indices = list(
            range(start, min(start + args.permutations_per_job, metadata["n_permutations"]))
        )
    subject = _subject_from_cell(metadata, subject_index)
    base = directory / "within_subject" / f"sub-{subject}"
    base.mkdir(parents=True, exist_ok=True)
    for index in permutation_indices:
        output = (
            base / "observed.npz"
            if index is None
            else base / "permutations" / f"permutation-{index:04d}.npz"
        )
        output.parent.mkdir(exist_ok=True)
        if output.exists() and args.skip_valid:
            continue
        states = arrays["observed_states"] if index is None else arrays["permuted_states"][index]
        result = _fit_within_subject(arrays, states, subject, _ridge(metadata))
        if index is not None:
            result["permutation_index"] = index
        _write_result(output, result)
    return base


def _reliance_blocks(metadata: dict, networks: np.ndarray) -> tuple[list[np.ndarray], ...]:
    """Build feature, network, and feature-by-network flattened blocks."""
    n_parcels, n_features = metadata["shape"][1:]
    feature = [np.arange(index, n_parcels * n_features, n_features) for index in range(n_features)]
    network = []
    cells = []
    for name in metadata["network_order"]:
        parcels = np.flatnonzero(networks.astype(str) == name)
        network.append(np.concatenate([np.arange(p * n_features, (p + 1) * n_features) for p in parcels]))
        cells.extend([parcels * n_features + index for index in range(n_features)])
    return feature, network, cells


def run_reliance(args: argparse.Namespace) -> Path:
    """Compute held-out grouped reliance for one subject and decoding regime."""
    directory, metadata, arrays = _load(args)
    subject = _subject_from_cell(metadata, args.cell_index)
    output_directory = directory / args.regime / "reliance"
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / f"sub-{subject}.npz"
    if output.exists() and args.skip_valid:
        return output
    features, labels, subjects, runs = _state_data(arrays, arrays["observed_states"])
    subjects = subjects.astype(str)
    runs = runs.astype(str)
    blocks = _reliance_blocks(metadata, arrays["network_assignments"])
    results = [[], [], []]
    if args.regime == "population":
        folds = [(subjects != subject, subjects == subject)]
    else:
        subject_rows = subjects == subject
        folds = [
            (subject_rows & (runs != run), subject_rows & (runs == run))
            for run in np.unique(runs[subject_rows])
        ]
    for fold_index, (train, test) in enumerate(folds):
        estimator, _, _ = fit_train_test(features, labels, train, test, _ridge(metadata))
        for family_index, family_blocks in enumerate(blocks):
            results[family_index].append(
                compute_grouped_reliance(
                    estimator,
                    features[test],
                    labels[test],
                    family_blocks,
                    runs[test],
                    repeats=metadata["reliance_repeats"],
                    seed=derive_chunk_seed(
                        metadata["analysis_id"], f"{args.regime}_reliance", subject, fold_index
                    ),
                )
            )
    _write_result(
        output,
        {
            "subject": subject,
            "feature_reliance": np.nanmean(np.stack(results[0]), axis=(0, 2)),
            "network_reliance": np.nanmean(np.stack(results[1]), axis=(0, 2)),
            "cell_reliance": np.nanmean(np.stack(results[2]), axis=(0, 2)),
        },
    )
    return output


def aggregate(args: argparse.Namespace) -> Path:
    """Aggregate population, individual, and reliance inference families."""
    directory, metadata, _ = _load(args)
    n_permutations = metadata["n_permutations"]
    with np.load(directory / "population" / "observed.npz") as observed:
        population_auc = float(observed["mean_subject_auc"])
    population_null = np.asarray([
        float(np.load(directory / "population" / "permutations" / f"permutation-{i:04d}.npz")["mean_subject_auc"])
        for i in range(n_permutations)
    ])
    subjects = metadata["subject_order"]
    within_auc = np.empty(len(subjects))
    within_p = np.empty(len(subjects))
    within_null = np.empty((n_permutations, len(subjects)))
    for subject_index, subject in enumerate(subjects):
        base = directory / "within_subject" / f"sub-{subject}"
        with np.load(base / "observed.npz") as observed:
            within_auc[subject_index] = float(observed["mean_run_auc"])
        null = np.asarray([
            float(np.load(base / "permutations" / f"permutation-{i:04d}.npz")["mean_run_auc"])
            for i in range(n_permutations)
        ])
        within_null[:, subject_index] = null
        within_p[subject_index] = (np.count_nonzero(null >= within_auc[subject_index]) + 1) / (n_permutations + 1)
    inference: dict[str, object] = {
        "population_auc": population_auc,
        "population_null": population_null,
        "population_p": (np.count_nonzero(population_null >= population_auc) + 1) / (n_permutations + 1),
        "within_subject_auc": within_auc,
        "within_subject_p_uncorrected": within_p,
        "within_group_mean_auc": float(np.nanmean(within_auc)),
        "within_group_null_mean_auc": np.nanmean(within_null, axis=1),
        "within_group_p": (
            np.count_nonzero(np.nanmean(within_null, axis=1) >= np.nanmean(within_auc))
            + 1
        )
        / (n_permutations + 1),
        "subject_order": np.asarray(subjects),
    }
    for regime in ("population", "within_subject"):
        values = {family: [] for family in ("feature", "network", "cell")}
        for subject in subjects:
            with np.load(directory / regime / "reliance" / f"sub-{subject}.npz") as result:
                for family in values:
                    values[family].append(result[f"{family}_reliance"])
        for family, rows in values.items():
            matrix = np.stack(rows)
            test = synchronized_sign_flip_test(
                {family: matrix}, args.sign_flip_permutations,
                derive_chunk_seed(metadata["analysis_id"], regime, family, 0),
            )
            inference[f"{regime}_{family}_reliance"] = matrix
            inference[f"{regime}_{family}_reliance_p_fwer"] = test["p_values_fwer"][0]
    output = directory / "results" / "inference.npz"
    output.parent.mkdir(exist_ok=True)
    _write_result(output, inference)
    return output


def main() -> None:
    """Run one state-multifeature stage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--analysis-root")
    common.add_argument("--config", default="config.yaml")
    common.add_argument("--stage-local", action="store_true")
    common.add_argument("--skip-valid", action="store_true")
    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--subjects")
    prepare_parser.add_argument("--runs")
    prepare_parser.add_argument("--n-permutations", type=int, default=1000)
    prepare_parser.add_argument("--alpha", type=float, default=1.0)
    prepare_parser.add_argument("--tolerance", type=float, default=1e-4)
    prepare_parser.add_argument("--reliance-repeats", type=int, default=20)
    for name in ("population", "within"):
        stage = subparsers.add_parser(name, parents=[common])
        stage.add_argument("--observed", action="store_true")
        stage.add_argument("--batch-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
        stage.add_argument("--cell-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
        stage.add_argument("--permutations-per-job", type=int, default=10)
    reliance = subparsers.add_parser("reliance", parents=[common])
    reliance.add_argument("--regime", choices=("population", "within_subject"), required=True)
    reliance.add_argument("--cell-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    aggregate_parser = subparsers.add_parser("aggregate", parents=[common])
    aggregate_parser.add_argument("--sign-flip-permutations", type=int, default=10000)
    args = parser.parse_args()
    dispatch = {
        "prepare": prepare,
        "population": run_population,
        "within": run_within,
        "reliance": run_reliance,
        "aggregate": aggregate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
