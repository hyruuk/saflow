"""Validate complete sensor or Schaefer feature barriers for Panel analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from code.analysis.preflight import inspect_inputs
from code.utils.config import load_config


def validate_feature_barrier(
    config_path: str,
    space: str,
    subjects_value: str | None,
    runs_value: str | None,
) -> None:
    """Fail unless every requested real feature recording is complete."""
    config = load_config(config_path)
    subjects = (
        subjects_value.split() if subjects_value else config["bids"]["subjects"]
    )
    runs = runs_value.split() if runs_value else config["bids"]["task_runs"]
    if space == "schaefer_400":
        report = inspect_inputs(config, subjects, runs)
        if report["status"] != "passed":
            failed = [
                f"sub-{item['subject']}/run-{item['run']}: {item['reason']}"
                for item in report["recordings"]
                if item["status"] != "passed"
            ]
            raise RuntimeError("Schaefer feature validation failed: " + "; ".join(failed))
        return
    if space != "sensor":
        raise ValueError("space must be sensor or schaefer_400")
    for subject in subjects:
        for run in runs:
            _validate_sensor_recording(config, subject, run)


def _validate_sensor_recording(
    config: dict, subject: str, run: str
) -> None:
    """Validate all three panel-analysis sensor feature bundles for one recording."""
    root = Path(config["paths"]["features"])
    directories = (
        ("welch_psds_sensor", {"psds", "freqs", "trial_metadata", "ch_names"}),
        (
            "fooof_sensor",
            {"exponent", "offset", "r_squared", "trial_metadata", "ch_names"},
        ),
        (
            "welch_psds_corrected_sensor",
            {"psds", "freqs", "trial_metadata", "ch_names"},
        ),
    )
    for directory, required in directories:
        candidates = sorted(
            (root / directory / f"sub-{subject}").glob(f"*run-{run}*w8*.npz")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected one {directory} bundle for sub-{subject} run-{run}"
            )
        with np.load(candidates[0], allow_pickle=True) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{candidates[0]} lacks fields {missing}")


def main() -> None:
    """Run a feature validation barrier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--space", required=True)
    parser.add_argument("--subjects")
    parser.add_argument("--runs")
    args = parser.parse_args()
    validate_feature_barrier(args.config, args.space, args.subjects, args.runs)


if __name__ == "__main__":
    main()
