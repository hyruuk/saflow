"""Benchmark prespecified fixed-ridge state decoding on real Schaefer-400 data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from code.analysis.contracts import MULTIFEATURE_FEATURES
from code.analysis.real_inputs import load_real_inputs
from code.classification.fast_state_scientific import (
    FixedRidgeConfig,
    fit_held_out_subject,
)
from code.utils.config import load_config

LOGGER = logging.getLogger(__name__)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Time prespecified real outer folds without writing analysis results."""
    config = load_config(args.config)
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    inputs = load_real_inputs(config, subjects, runs)
    selected = np.isin(inputs.states, ("IN", "OUT"))
    feature_indices = [inputs.feature_order.index(name) for name in MULTIFEATURE_FEATURES]
    features = inputs.feature_tensor[selected][:, :, feature_indices].reshape(
        int(selected.sum()), -1
    )
    labels = (inputs.states[selected] == "OUT").astype(int)
    groups = inputs.subjects[selected].astype(str)
    held_out = list(dict.fromkeys(groups))[: args.fold_count]
    ridge = FixedRidgeConfig(alpha=args.alpha, tolerance=args.tolerance)
    folds = [
        fit_held_out_subject(features, labels, groups, subject, ridge)
        for subject in held_out
    ]
    elapsed = np.asarray([fold["elapsed_seconds"] for fold in folds])
    report = {
        "space": "schaefer_400",
        "predictor_count": int(features.shape[1]),
        "feature_order": list(MULTIFEATURE_FEATURES),
        "trial_count": int(features.shape[0]),
        "fold_count": len(folds),
        "alpha": args.alpha,
        "tolerance": args.tolerance,
        "fold_seconds": elapsed.tolist(),
        "median_fold_seconds": float(np.median(elapsed)),
        "projected_loso_seconds": float(np.median(elapsed) * np.unique(groups).size),
    }
    LOGGER.info("Benchmark result:\n%s", json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    """Run the command-line benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--subjects")
    parser.add_argument("--runs")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
