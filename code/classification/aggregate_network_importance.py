"""Aggregate per-parcel permutation importance to per-Yeo-network summaries.

Reads the ``importances`` array from a multifeature ``axis=joint`` run
(shape ``n_spatial × n_features``) and produces a per-Yeo-network bundle of
both sum and mean importances per (network, feature). Sum is the most
honest aggregation for collinear parcels within a network (per the
multifeature memory note); mean stays useful when network sizes vary
wildly.

Output:
    results/classification_<space>/group_mf/networks/
        importance-networks_yeo{N}_label-<label>_clf-<clf>_cv-<cv>_type-<trial>.npz

Usage:
    python -m code.classification.aggregate_network_importance \\
        --space schaefer_400 --label all --clf logistic --cv logo \\
        --trial-type correct --yeo 7

    # or, point at a specific scores file directly
    python -m code.classification.aggregate_network_importance \\
        --input <path>/feature-all_..._axis-joint_..._scores.npz --yeo 7
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.classification.run_classification import inout_bounds_to_string
from code.classification.run_multifeature import build_mf_base_name
from code.utils.config import load_config
from code.utils.paths import get_results_root
from code.utils.yeo_networks import (
    UNKNOWN_NETWORK,
    aggregate_to_networks,
    get_network_assignments,
    network_order,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_input_path(
    output_dir: Path,
    label: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    importance: str,
    trial_type: str,
    analysis_level: Optional[str],
) -> Path:
    base = build_mf_base_name(
        label, space, inout_bounds, clf_name, cv_name, "joint",
        importance, trial_type, analysis_level=analysis_level,
    )
    return output_dir / f"{base}_scores.npz"


def aggregate_one(
    scores_path: Path,
    n_networks: int = 7,
) -> Dict[str, np.ndarray]:
    """Read a joint-axis scores npz and produce per-network importance bundle."""
    with np.load(scores_path, allow_pickle=True) as npz:
        if "importances" not in npz.files:
            raise KeyError(
                f"{scores_path.name} has no 'importances' key. The mf run was "
                f"likely launched with --importance=none. Re-run with "
                f"--importance=permutation."
            )
        imp = np.asarray(npz["importances"])  # (n_spatial, n_features)
        spatial_names = [str(x) for x in np.asarray(npz["spatial_names"])]
        feature_names = [str(x) for x in np.asarray(npz["feature_names"])]

    if imp.shape != (len(spatial_names), len(feature_names)):
        raise ValueError(
            f"Shape mismatch in {scores_path.name}: importances={imp.shape}, "
            f"spatial_names={len(spatial_names)}, feature_names={len(feature_names)}"
        )

    # Drop Unknown (medial-wall) parcels before aggregation.
    assignments = get_network_assignments(spatial_names, n_networks=n_networks)
    keep = assignments != UNKNOWN_NETWORK
    if not keep.all():
        logger.info(f"  dropping {(~keep).sum()} Unknown parcels")
        imp = imp[keep, :]
        spatial_names = [c for c, k in zip(spatial_names, keep) if k]

    # Aggregate over the parcel axis. Helper expects parcels as last axis.
    # imp is (n_parcels, n_features) -> transpose to (n_features, n_parcels)
    # so the parcel axis is last.
    imp_T = imp.T
    sum_T, nets = aggregate_to_networks(imp_T, spatial_names,
                                        n_networks=n_networks, agg="sum")
    mean_T, _ = aggregate_to_networks(imp_T, spatial_names,
                                      n_networks=n_networks, agg="mean")
    # Back to (n_networks, n_features)
    importance_sum = sum_T.T
    importance_mean = mean_T.T

    # Per-network parcel counts (after dropping Unknown)
    n_parc = np.zeros(len(nets), dtype=int)
    for k, net in enumerate(nets):
        n_parc[k] = int(np.sum(assignments[keep] == net))

    return {
        "network_names": np.asarray(nets),
        "features": np.asarray(feature_names),
        "importance_sum": importance_sum,
        "importance_mean": importance_mean,
        "n_parcels_per_network": n_parc,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate joint-axis permutation importance to Yeo networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=None,
                   help="Direct path to a joint-axis scores npz. Overrides "
                        "the auto-built path.")
    p.add_argument("--space", default=None,
                   help="Atlas space (required unless --input is given).")
    p.add_argument("--label", default="all",
                   help="Feature-set label used by the mf run (e.g. 'all').")
    p.add_argument("--clf", default="logistic")
    p.add_argument("--cv", default="logo")
    p.add_argument("--importance", default="permutation",
                   help="Importance backend used by the mf run.")
    p.add_argument("--analysis-level", default="epoch")
    p.add_argument("--trial-type", default="all",
                   help="alltrials | correct | lapse | all")
    p.add_argument("--yeo", type=int, default=7, choices=[7, 17])
    p.add_argument("--results-root", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    inout_bounds = tuple(config.get("analysis", {}).get("inout_bounds", [25, 75]))

    results_root = (Path(args.results_root) if args.results_root
                    else get_results_root(config))

    if args.input:
        inputs: List[Tuple[Path, str]] = [(Path(args.input), args.trial_type)]
    else:
        if args.space is None:
            raise SystemExit("Must pass either --input or --space.")
        mf_dir = (results_root / f"classification_{args.space}" / "group_mf")
        trial_types = (
            ["alltrials", "correct", "lapse"]
            if args.trial_type == "all" else [args.trial_type]
        )
        inputs = []
        for trial in trial_types:
            in_path = _build_input_path(
                mf_dir, args.label, args.space, inout_bounds, args.clf, args.cv,
                args.importance, trial, args.analysis_level,
            )
            inputs.append((in_path, trial))

    out_dir = None
    if args.space is not None:
        out_dir = (results_root / f"classification_{args.space}" /
                   "group_mf" / "networks")
        out_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "script": "code.classification.aggregate_network_importance",
        "timestamp": datetime.utcnow().isoformat(),
        "yeo": args.yeo,
        "label": args.label,
        "clf": args.clf,
        "cv": args.cv,
    }

    for in_path, trial in inputs:
        if not in_path.exists():
            logger.warning(f"skipping (missing): {in_path}")
            continue
        logger.info(f"=== {in_path.name} (trial={trial}) ===")
        payload = aggregate_one(in_path, n_networks=args.yeo)

        if out_dir is None:
            out_path = in_path.with_name(
                in_path.stem.replace("_scores", "")
                + f"_networks-yeo{args.yeo}.npz"
            )
        else:
            out_path = out_dir / (
                f"importance-networks_yeo{args.yeo}_label-{args.label}_"
                f"clf-{args.clf}_cv-{args.cv}_type-{trial}.npz"
            )
        np.savez(out_path, **payload,
                 meta=np.asarray(json.dumps(provenance | {
                     "trial_type": trial,
                     "input_path": str(in_path),
                 })))
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
