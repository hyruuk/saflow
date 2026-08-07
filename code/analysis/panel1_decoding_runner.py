"""Run one synchronized Panel 1 decoding permutation chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.classification.classifiers import get_classifier
from code.classification.run_classification import (
    get_cv_strategy,
    run_univariate_with_tmax,
)
from code.analysis.chunks import derive_chunk_seed
from code.analysis.contracts import PANEL1_FEATURES
from code.analysis.real_inputs import load_real_inputs
from code.analysis.result_io import write_result_bundle
from code.utils.config import load_config


def run_chunk(args: argparse.Namespace) -> Path:
    """Run one feature × permutation-interval decoding cell."""
    if args.feature not in PANEL1_FEATURES:
        raise ValueError(f"feature must be one of {PANEL1_FEATURES}")
    config = load_config(args.config)
    panel_analysis = config.get("panel_analysis", {})
    total = int(panel_analysis.get("map_permutations", 10_000))
    size = int(panel_analysis.get("map_chunk_size", 250))
    start = args.chunk_index * size
    stop = min(start + size, total)
    if start >= total:
        raise ValueError("chunk index lies outside configured map permutations")
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    inputs = load_real_inputs(config, subjects, runs)
    valid = np.isin(inputs.states, ("IN", "OUT"))
    labels = (inputs.states[valid] == "OUT").astype(int)
    groups = inputs.subjects[valid]
    if np.unique(groups).size < 3:
        raise ValueError("Panel 1 nested subject validation requires >= 3 subjects")
    index = PANEL1_FEATURES.index(args.feature)
    seed = derive_chunk_seed(
        args.analysis_id, "panel1", "decoding", args.chunk_index
    )
    result = run_univariate_with_tmax(
        inputs.panel1_tensor[valid, :, index],
        labels,
        groups,
        clf_factory=lambda: get_classifier("logistic"),
        cv=get_cv_strategy("logo"),
        n_permutations=stop - start,
        n_jobs=args.jobs,
        seed=seed,
        scoring="roc_auc",
    )
    result.update(
        {
            "feature": args.feature,
            "parcel_order": inputs.parcel_order,
            "permutation_interval": np.asarray([start, stop]),
            "chunk_index": args.chunk_index,
            "seed": seed,
        }
    )
    analysis_dir = _analysis_directory(config, args.analysis_root, args.analysis_id)
    analysis = json.loads((analysis_dir / "provenance.json").read_text())
    provenance = {
        "analysis_id": args.analysis_id,
        "data_mode": "real",
        "node": "panel1_decoding_permutations",
        "cell_index": args.cell_index,
        "git": analysis["git"],
        "config_hash": analysis["config_hash"],
        "inputs": list(inputs.input_inventory),
        "software": analysis.get("software", {}),
        "feature": args.feature,
        "chunk_index": args.chunk_index,
        "permutation_interval": [start, stop],
        "seed": seed,
    }
    directory = (
        analysis_dir
        / "panel1"
        / "partials"
        / "decoding"
        / args.feature
        / f"chunk-{args.chunk_index:04d}"
    )
    write_result_bundle(directory, result, provenance)
    return directory


def _analysis_directory(
    config: dict, override: str | None, analysis_id: str
) -> Path:
    """Resolve an initialized immutable analysis directory."""
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


def main() -> None:
    """Run one Panel 1 decoding chunk from scheduler arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--subjects")
    parser.add_argument("--runs")
    parser.add_argument("--jobs", type=int, default=1)
    run_chunk(parser.parse_args())


if __name__ == "__main__":
    main()
