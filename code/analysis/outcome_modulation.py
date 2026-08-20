"""Analyze Correct-versus-Lapse modulation within IN and OUT states.

Three participant-weighting variants are computed over one shared eligible
cohort, so panels built from them differ only in weighting:

``equal_subject``
    Pool every eligible window into one mean per participant, state, and anchor
    outcome, then weight participants equally. This is the primary variant.
``equal_window``
    Pool windows identically, then weight each participant by the effective
    window count of their paired contrast, so every retained window carries
    comparable influence and imprecise participants stop dominating.
``equal_run``
    Average windows within run first and then across runs equally, matching the
    pre-refactor pipeline. Anchor outcomes are sparse per run, so a participant's
    two arms may rest on different runs; treat this variant as a sensitivity.

A run-stratified balanced-resampling sensitivity repeatedly downsamples each
within-run cell to the smaller outcome count, independently of the weighting.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import ttest_1samp

from code.analysis.contracts import CORRECTED_FEATURES
from code.analysis.inference import paired_effect_size, synchronized_cluster_mass_test
from code.analysis.networks import YEO7_ORDER, synchronized_sign_flip_test
from code.analysis.observed_runner import _network_assignments, _parcel_adjacency
from code.analysis.provenance import resolve_analysis_directory
from code.analysis.real_inputs import AnalysisInputs, load_real_inputs
from code.analysis.result_io import write_result_bundle
from code.utils.config import load_config

LOGGER = logging.getLogger(__name__)
STATE_ORDER = ("IN", "OUT")
OUTCOME_ORDER = ("correct_omission", "commission_error")
WEIGHTING_ORDER = ("equal_subject", "equal_window", "equal_run")
PRIMARY_WEIGHTING = "equal_subject"
WEIGHTING_POOLING = {
    "equal_subject": "window",
    "equal_window": "window",
    "equal_run": "run",
}
WEIGHTING_DESCRIPTIONS = {
    "equal_subject": (
        "pool all windows within participant-state-outcome; participants weighted equally"
    ),
    "equal_window": (
        "pool all windows within participant-state-outcome; participants weighted by the "
        "effective window count of their paired contrast"
    ),
    "equal_run": (
        "average windows within run then across runs equally; participants weighted equally"
    ),
}


@dataclass(frozen=True)
class ContrastData:
    """Participant-level Lapse-minus-Correct contrasts for one state."""

    differences: np.ndarray
    subjects: np.ndarray
    counts: np.ndarray
    run_counts: np.ndarray


def compute_subject_contrasts(
    values: np.ndarray,
    states: np.ndarray,
    outcomes: np.ndarray,
    subjects: np.ndarray,
    state: str,
    minimum_windows: int,
    *,
    runs: np.ndarray | None = None,
    pooling: str = "window",
) -> ContrastData:
    """Return paired participant contrasts under one within-participant pooling.

    Eligibility always uses total window counts, so every pooling shares one
    cohort. ``pooling="run"`` averages within run before averaging across runs.
    """
    if pooling not in {"window", "run"}:
        raise ValueError(f"unknown within-participant pooling: {pooling}")
    if pooling == "run" and runs is None:
        raise ValueError("run pooling requires run labels")
    if minimum_windows < 1:
        raise ValueError("minimum_windows must retain at least one window per cell")
    differences, included, counts, run_counts = [], [], [], []
    for subject in np.unique(subjects):
        masks = [
            (subjects == subject) & (states == state) & (outcomes == outcome)
            for outcome in OUTCOME_ORDER
        ]
        cell_counts = [int(mask.sum()) for mask in masks]
        if min(cell_counts) < minimum_windows:
            continue
        if pooling == "window":
            means = [np.nanmean(values[mask], axis=0) for mask in masks]
            contributing = [len(np.unique(runs[mask])) if runs is not None else 0 for mask in masks]
        else:
            means, contributing = [], []
            for mask in masks:
                run_means = [
                    np.nanmean(values[mask & (runs == run)], axis=0)
                    for run in np.unique(runs[mask])
                ]
                means.append(np.nanmean(np.stack(run_means), axis=0))
                contributing.append(len(run_means))
        differences.append(means[1] - means[0])
        included.append(subject)
        counts.append(cell_counts)
        run_counts.append(contributing)
    if len(included) < 2:
        raise ValueError(f"{state} Correct-versus-Lapse requires at least two participants")
    return ContrastData(
        np.stack(differences),
        np.asarray(included),
        np.asarray(counts),
        np.asarray(run_counts),
    )


def participant_weights(contrasts: ContrastData, weighting: str) -> np.ndarray | None:
    """Return fixed participant weights, or ``None`` when participants count equally."""
    if weighting not in WEIGHTING_ORDER:
        raise ValueError(f"unknown participant weighting: {weighting}")
    if weighting != "equal_window":
        return None
    correct, lapse = contrasts.counts[:, 0].astype(float), contrasts.counts[:, 1].astype(float)
    return correct * lapse / (correct + lapse)


def compute_balanced_contrasts(
    values: np.ndarray,
    states: np.ndarray,
    outcomes: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    state: str,
    minimum_windows: int,
    rng: np.random.Generator,
) -> ContrastData:
    """Downsample outcome cells equally within run before participant pooling."""
    differences, included, counts, run_counts = [], [], [], []
    for subject in np.unique(subjects):
        correct_parts, lapse_parts = [], []
        subject_runs = np.unique(runs[subjects == subject])
        for run in subject_runs:
            base = (subjects == subject) & (runs == run) & (states == state)
            indices = [np.flatnonzero(base & (outcomes == outcome)) for outcome in OUTCOME_ORDER]
            paired_n = min(map(len, indices))
            if paired_n:
                correct_parts.append(rng.choice(indices[0], paired_n, replace=False))
                lapse_parts.append(rng.choice(indices[1], paired_n, replace=False))
        correct = _concatenate_indices(correct_parts)
        lapse = _concatenate_indices(lapse_parts)
        if min(len(correct), len(lapse)) < minimum_windows:
            continue
        differences.append(np.nanmean(values[lapse], axis=0) - np.nanmean(values[correct], axis=0))
        included.append(subject)
        counts.append((len(correct), len(lapse)))
        run_counts.append((len(correct_parts), len(lapse_parts)))
    if len(included) < 2:
        raise ValueError(f"{state} balanced analysis requires at least two participants")
    return ContrastData(
        np.stack(differences),
        np.asarray(included),
        np.asarray(counts),
        np.asarray(run_counts),
    )


def _concatenate_indices(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate optional within-run index arrays."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def compute_balanced_sensitivity(
    inputs: AnalysisInputs,
    state: str,
    minimum_windows: int,
    repetitions: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Summarize balanced-window and participant-bootstrap sensitivity maps."""
    rng = np.random.default_rng(seed)
    t_maps, mean_maps, subject_ns = [], [], []
    for _ in range(repetitions):
        result = compute_balanced_contrasts(
            inputs.feature_tensor,
            inputs.states,
            inputs.outcomes,
            inputs.subjects,
            inputs.runs,
            state,
            minimum_windows,
            rng,
        )
        bootstrap = rng.integers(0, len(result.subjects), size=len(result.subjects))
        differences = np.moveaxis(result.differences[bootstrap], 2, 1)
        t_maps.append(ttest_1samp(differences, 0.0, axis=0, nan_policy="omit").statistic)
        mean_maps.append(np.nanmean(differences, axis=0))
        subject_ns.append(len(result.subjects))
    t_values = np.stack(t_maps)
    mean_values = np.stack(mean_maps)
    median_mean = np.nanmedian(mean_values, axis=0)
    return {
        "t_median": np.nanmedian(t_values, axis=0),
        "t_ci_low": np.nanpercentile(t_values, 2.5, axis=0),
        "t_ci_high": np.nanpercentile(t_values, 97.5, axis=0),
        "mean_difference_median": median_mean,
        "mean_difference_ci_low": np.nanpercentile(mean_values, 2.5, axis=0),
        "mean_difference_ci_high": np.nanpercentile(mean_values, 97.5, axis=0),
        "direction_stability": np.maximum(
            np.mean(mean_values > 0, axis=0), np.mean(mean_values < 0, axis=0)
        ),
        "subject_n": np.asarray(subject_ns, dtype=int),
    }


def compute_outcome_modulation(
    inputs: AnalysisInputs,
    config: dict[str, Any],
    *,
    minimum_windows: int,
    permutations: int,
    balanced_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Compute parcel/network inference and balanced-resampling sensitivity."""
    adjacency = _parcel_adjacency(inputs.parcel_order, config)
    assignments = _network_assignments(inputs.parcel_order)
    threshold = float(
        config.get("analysis_workflow", {}).get("cluster_forming_threshold", 2.0)
    )
    result: dict[str, Any] = {
        "state_order": STATE_ORDER,
        "feature_order": CORRECTED_FEATURES,
        "parcel_order": np.asarray(inputs.parcel_order),
        "network_order": YEO7_ORDER,
        "contrast": "commission_error_minus_correct_omission",
        "minimum_windows_per_cell": minimum_windows,
        "weighting_order": WEIGHTING_ORDER,
        "primary_weighting": PRIMARY_WEIGHTING,
        "weighting_descriptions": WEIGHTING_DESCRIPTIONS,
        "balanced_policy": ("equal counts within participant-state-run plus participant bootstrap"),
        "balanced_repetitions": balanced_repetitions,
    }
    for state_index, state in enumerate(STATE_ORDER):
        state_result: dict[str, Any] = {
            "balanced": compute_balanced_sensitivity(
                inputs,
                state,
                minimum_windows,
                balanced_repetitions,
                seed + 1_000 + state_index,
            )
        }
        for weighting in WEIGHTING_ORDER:
            contrasts = compute_subject_contrasts(
                inputs.feature_tensor,
                inputs.states,
                inputs.outcomes,
                inputs.subjects,
                state,
                minimum_windows,
                runs=inputs.runs,
                pooling=WEIGHTING_POOLING[weighting],
            )
            weights = participant_weights(contrasts, weighting)
            differences = np.moveaxis(contrasts.differences, 2, 1)
            # One permutation seed per state so variants share sign-flip draws and
            # any difference between them is attributable to weighting alone.
            parcel = synchronized_cluster_mass_test(
                differences,
                adjacency,
                n_permutations=permutations,
                cluster_threshold=threshold,
                seed=seed + state_index,
                weights=weights,
            )
            network_differences = _network_means(contrasts.differences, assignments)
            network = synchronized_sign_flip_test(
                {"lapse_minus_correct": network_differences.reshape(len(contrasts.subjects), -1)},
                n_permutations=permutations,
                seed=seed + 100 + state_index,
                weights=weights,
            )
            state_result[weighting] = {
                "subject_order": contrasts.subjects,
                "window_counts": contrasts.counts,
                "run_counts": contrasts.run_counts,
                "subject_n": len(contrasts.subjects),
                "participant_weights": (
                    np.ones(len(contrasts.subjects)) if weights is None else weights
                ),
                "effective_subject_n": _effective_subject_n(weights, len(contrasts.subjects)),
                "differences": differences,
                "parcel_t_values": parcel["t_values"],
                "parcel_effect_size_dz": paired_effect_size(differences, weights),
                "parcel_p_cluster_fwer": parcel["p_values_fwer"],
                "network_differences": network_differences,
                "network_t_values": network["t_values"][0].reshape(7, 9),
                "network_p_fwer": network["p_values_fwer"][0].reshape(7, 9),
            }
        result[state] = state_result
    return result


def _effective_subject_n(weights: np.ndarray | None, count: int) -> float:
    """Return Kish effective participant count for one weighting."""
    if weights is None:
        return float(count)
    return float(np.square(weights.sum()) / np.square(weights).sum())


def _network_means(values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Average parcel contrasts within each canonical Yeo-7 network."""
    return np.stack(
        [np.nanmean(values[:, assignments == network], axis=1) for network in YEO7_ORDER],
        axis=1,
    )


def run(args: argparse.Namespace) -> Path:
    """Load real inputs, compute the derivative, and write an immutable bundle."""
    config = load_config(args.config)
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    inputs = load_real_inputs(config, subjects, runs)
    root = Path(args.analysis_root) if args.analysis_root else _default_analysis_root(config)
    analysis_directory = resolve_analysis_directory(root, args.analysis_id)
    result = compute_outcome_modulation(
        inputs,
        config,
        minimum_windows=args.minimum_windows,
        permutations=args.permutations,
        balanced_repetitions=args.balanced_repetitions,
        seed=args.seed,
    )
    provenance = json.loads((analysis_directory / "provenance.json").read_text())
    provenance.update(
        {
            "node": "outcome_modulation",
            "script": "code.analysis.outcome_modulation",
            "parameters": {
                "minimum_windows": args.minimum_windows,
                "permutations": args.permutations,
                "balanced_repetitions": args.balanced_repetitions,
                "seed": args.seed,
                "weightings": list(WEIGHTING_ORDER),
            },
            "inputs": list(inputs.input_inventory),
        }
    )
    output = analysis_directory / "outcome_modulation"
    write_result_bundle(output, result, provenance)
    LOGGER.info("Wrote outcome-modulation bundle to %s", output)
    return output


def _default_analysis_root(config: dict[str, Any]) -> Path:
    """Resolve the configured corrected-analysis root."""
    return (
        Path(config["paths"]["data_root"])
        / "processed"
        / config.get("analysis_workflow", {}).get("processed_directory", "analysis_workflow")
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the outcome-modulation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis-root")
    parser.add_argument("--analysis-id")
    parser.add_argument("--subjects")
    parser.add_argument("--runs")
    parser.add_argument("--minimum-windows", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--balanced-repetitions", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Run the command-line outcome-modulation analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
