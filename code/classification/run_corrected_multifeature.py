"""Run the prespecified leakage-safe multifeature analysis for one analysis ID."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from code.classification.multifeature_provenance import (
    canonical_config_hash,
    validate_analysis_id,
)
from code.classification.multifeature_scientific import NestedRidgeConfig, run_primary_analysis
from code.classification.run_classification import expand_feature_set, load_combined_features
from code.utils.logging_config import setup_logging

LOGGER = logging.getLogger(__name__)


def _flatten_arrays(value: Any, prefix: str = "") -> tuple[dict[str, np.ndarray], Any]:
    """Separate nested NumPy arrays for NPZ storage and JSON summaries."""
    arrays: dict[str, np.ndarray] = {}
    if isinstance(value, np.ndarray):
        arrays[prefix] = value
        return arrays, {"array": prefix, "shape": list(value.shape)}
    if isinstance(value, dict):
        summary = {}
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else key
            child_arrays, child_summary = _flatten_arrays(child, child_prefix)
            arrays.update(child_arrays)
            summary[key] = child_summary
        return arrays, summary
    if isinstance(value, list):
        summary_list = []
        for index, child in enumerate(value):
            child_prefix = f"{prefix}/{index}"
            child_arrays, child_summary = _flatten_arrays(child, child_prefix)
            arrays.update(child_arrays)
            summary_list.append(child_summary)
        return arrays, summary_list
    if isinstance(value, np.generic):
        return arrays, value.item()
    return arrays, value


def run(args: argparse.Namespace) -> Path:
    """Load strictly aligned inputs, execute all primary outputs, and persist them."""
    validate_analysis_id(args.analysis_id)
    config = yaml.safe_load(Path(args.config).read_text())
    features = expand_feature_set(args.features, config) if args.features in {
        "all", "psds", "psds_corrected", "fooof", "complexity"
    } else args.features.split()
    analysis_dir = Path(args.analysis_root) / args.analysis_id
    preflight = analysis_dir / "preflight_report.json"
    if not preflight.exists() or json.loads(preflight.read_text()).get("status") != "passed":
        raise RuntimeError("A passing preflight report is required before model fitting")
    tensor, labels, groups, metadata = load_combined_features(
        features, args.space, (args.inout_low, args.inout_high), config,
        subjects=args.subjects, trial_type=args.trial_type,
        n_events_window=args.n_events_window, strict_alignment=True,
    )
    scientific_config = NestedRidgeConfig(
        c_grid=tuple(config.get("analysis", {}).get("multifeature", {}).get(
            "c_grid", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        )),
        seed=args.seed,
        inner_splits=args.inner_splits,
    )
    result = run_primary_analysis(
        tensor, labels, groups, scientific_config, args.n_permutations
    )
    arrays, summary = _flatten_arrays(result)
    arrays["feature_names"] = np.asarray(features)
    arrays["spatial_names"] = np.asarray(metadata["spatial_names"])
    output_dir = analysis_dir / "results" / args.space / args.trial_type
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "primary_results.npz"
    np.savez_compressed(score_path, **arrays)
    result_metadata = {
        "analysis_id": args.analysis_id,
        "config_hash": canonical_config_hash(config),
        "seed": args.seed,
        "n_permutations": args.n_permutations,
        "space": args.space,
        "trial_type": args.trial_type,
        "features": features,
        "scientific_options": {
            "outer_cv": "leave-one-subject-out",
            "inner_cv": "subject-grouped",
            "classifier": "ridge-logistic",
            "class_weight": "balanced",
            "held_out_subject_standardization": False,
            "primary_metric": "roc_auc",
            "c_grid": list(scientific_config.c_grid),
        },
        "summary": summary,
    }
    (output_dir / "primary_results.json").write_text(
        json.dumps(result_metadata, indent=2, default=str) + "\n"
    )
    LOGGER.info("Corrected multifeature results saved to %s", score_path)
    return score_path


def build_parser() -> argparse.ArgumentParser:
    """Build the corrected analysis CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--features", default="all")
    parser.add_argument("--space", default="sensor")
    parser.add_argument("--trial-type", default="alltrials")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--n-events-window", type=int, default=8)
    parser.add_argument("--inout-low", type=int, default=25)
    parser.add_argument("--inout-high", type=int, default=75)
    parser.add_argument("--n-permutations", type=int, default=1_000)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Execute the corrected multifeature command."""
    args = build_parser().parse_args()
    setup_logging(
        __name__, log_file="corrected_multifeature.log",
        config={"paths": {"logs": "logs"}, "logging": {"level": "INFO"}},
    )
    run(args)


if __name__ == "__main__":
    main()
