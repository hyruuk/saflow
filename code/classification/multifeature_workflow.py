"""Command-line workflow utilities for corrected multifeature analyses."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from code.classification.multifeature_provenance import (
    canonical_config_hash,
    create_analysis_id,
    export_analysis,
    initialize_analysis_directory,
    inventory_legacy_results,
)
from code.classification.run_classification import (
    expand_feature_set,
    load_combined_features,
)
from code.utils.logging_config import setup_logging

LOGGER = logging.getLogger(__name__)


def run_preflight(args: argparse.Namespace) -> Path:
    """Strictly load and align all selected inputs, then write a report."""
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    features = expand_feature_set(args.features, config) if args.features in {
        "all", "psds", "psds_corrected", "fooof", "complexity"
    } else args.features.split()
    analysis_id = args.analysis_id or create_analysis_id(config, Path.cwd())
    root = Path(args.output_root or config["paths"]["data_root"]) / "processed" / "multifeature"
    analysis_dir = initialize_analysis_directory(
        root, analysis_id, config, vars(args), Path.cwd()
    )
    try:
        tensor, labels, groups, metadata = load_combined_features(
            features,
            args.space,
            (args.inout_low, args.inout_high),
            config,
            subjects=args.subjects,
            trial_type=args.trial_type,
            n_events_window=args.n_events_window,
            strict_alignment=True,
        )
        finite_fraction = np.isfinite(tensor).mean(axis=0)
        if np.any(finite_fraction == 0):
            raise ValueError("At least one feature column has no finite observations")
        report = {
            "status": "passed",
            "analysis_id": analysis_id,
            "config_hash": canonical_config_hash(config),
            "features": features,
            "shape": list(tensor.shape),
            "n_subjects": int(np.unique(groups).size),
            "class_counts": np.bincount(labels, minlength=2).tolist(),
            "finite_fraction_min": float(finite_fraction.min()),
            "input_metadata": metadata,
        }
    except Exception as error:
        report = {"status": "failed", "analysis_id": analysis_id, "error": str(error)}
        (analysis_dir / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    report_path = analysis_dir / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    LOGGER.info("Preflight passed: %s", report_path)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build the workflow command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", default="config.yaml")
    preflight.add_argument("--analysis-id")
    preflight.add_argument("--output-root")
    preflight.add_argument("--features", default="all")
    preflight.add_argument("--space", default="sensor")
    preflight.add_argument("--trial-type", default="alltrials")
    preflight.add_argument("--n-events-window", type=int, default=8)
    preflight.add_argument("--inout-low", type=int, default=25)
    preflight.add_argument("--inout-high", type=int, default=75)
    preflight.add_argument("--subjects", nargs="*")
    export = commands.add_parser("export")
    export.add_argument("--analysis-dir", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    legacy = commands.add_parser("legacy-inventory")
    legacy.add_argument("--source", type=Path, required=True)
    legacy.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> None:
    """Run a selected workflow utility."""
    args = build_parser().parse_args()
    setup_logging(
        "multifeature_workflow",
        log_file="multifeature_workflow.log",
        config={"paths": {"logs": "logs"}, "logging": {"level": "INFO"}},
    )
    if args.command == "preflight":
        run_preflight(args)
    elif args.command == "export":
        export_analysis(args.analysis_dir, args.destination)
    else:
        inventory_legacy_results(args.source, args.manifest)


if __name__ == "__main__":
    main()
