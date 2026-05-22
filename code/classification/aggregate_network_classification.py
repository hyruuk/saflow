"""Merge per-network partials from run_network_classification into combined files.

When ``run_network_classification.py`` is launched with ``--network=<name>``
(e.g. by a SLURM array task), it writes a partial npz with one network's row
filled in:

    classif-networks_yeo{N}_scope-{scope}_type-{trial}_clf-{clf}_cv-{cv}_net-{net}.npz

This script globs the partials for a given (scope, trial-type, …) cell,
stacks them in Yeo network order, and writes the combined file
``…_clf-{clf}_cv-{cv}.npz`` that the panel and downstream tools expect.

Usage:
    python -m code.classification.aggregate_network_classification \\
        --space schaefer_400 --scope per-family --trial-type correct \\
        --yeo 7 --clf logistic --cv logo

    # Aggregate every (scope × trial-type) combo present on disk
    python -m code.classification.aggregate_network_classification \\
        --space schaefer_400 --scope all --trial-type all --yeo 7
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.utils.config import load_config
from code.utils.paths import get_results_root
from code.utils.yeo_networks import network_order

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SCOPES: Tuple[str, ...] = ("per-family", "per-feature", "joint")
TRIAL_TYPES: Tuple[str, ...] = ("alltrials", "correct", "lapse")

_NET_FILE_RE = re.compile(r"_net-(?P<net>[A-Za-z]+)\.npz$")


def _scan_partials(net_dir: Path, scope: str, trial: str,
                   clf: str, cv: str, n_networks: int
                   ) -> Dict[str, Path]:
    """Return ``{network_name: partial_path}`` for the requested cell."""
    prefix = (f"classif-networks_yeo{n_networks}_scope-{scope}_"
              f"type-{trial}_clf-{clf}_cv-{cv}_net-")
    out: Dict[str, Path] = {}
    for path in sorted(net_dir.glob(f"{prefix}*.npz")):
        m = _NET_FILE_RE.search(path.name)
        if not m:
            continue
        out[m.group("net")] = path
    return out


def _stack_partials(partials: Dict[str, Path],
                    n_networks: int) -> Optional[Dict[str, np.ndarray]]:
    """Concatenate per-network partial arrays into a combined bundle.

    Networks not present in ``partials`` get NaN-filled rows so the combined
    file always has the canonical Yeo ordering. Returns None if no partials.
    """
    if not partials:
        return None
    nets = list(network_order(n_networks))

    sample = np.load(next(iter(partials.values())), allow_pickle=True)
    sample_files = list(sample.files)
    # Skip the meta sidecar key and the per-row network_names of each partial
    array_keys = [k for k in sample_files if k != "meta"]
    sample_arrays = {k: np.asarray(sample[k]) for k in array_keys}
    sample.close()

    # Build combined arrays: copy through 1-D / shape-(*,n_features,...) arrays,
    # stack rows by network for arrays whose leading axis is networks.
    combined: Dict[str, np.ndarray] = {}
    # Heuristic: any array whose first dim equals 1 in a partial corresponds
    # to a per-network row (1 = restricted to one net). Keys with other
    # leading dims (e.g. ``features``, ``families``, ``features_used``) are
    # invariants and copied verbatim from the first partial.
    for key, arr in sample_arrays.items():
        if arr.ndim >= 1 and arr.shape[0] == 1:
            tail_shape = arr.shape[1:]
            combined[key] = np.full((len(nets),) + tail_shape, np.nan,
                                    dtype=arr.dtype if arr.dtype.kind == "f"
                                    else float)
        else:
            combined[key] = arr.copy()

    # ``network_names`` should be the canonical ordering, not the
    # single-element partial value.
    combined["network_names"] = np.asarray(nets)

    # n_parcels_per_network needs to come from the partials, one per net.
    if "n_parcels_per_network" in combined:
        combined["n_parcels_per_network"] = np.zeros(len(nets), dtype=int)

    for i, net in enumerate(nets):
        if net not in partials:
            continue
        with np.load(partials[net], allow_pickle=True) as npz:
            for key in array_keys:
                arr = np.asarray(npz[key])
                if key in ("network_names",):
                    continue
                # Per-network row: leading dim is 1 in partial
                if arr.ndim >= 1 and arr.shape[0] == 1:
                    combined[key][i] = arr[0]
                elif key == "n_parcels_per_network" and arr.ndim == 1:
                    # Partial has shape (1,) — assign at index i
                    combined[key][i] = int(arr[0])
                # else: invariant key — already copied above

    return combined


def aggregate_one(net_dir: Path, scope: str, trial: str, clf: str, cv: str,
                  n_networks: int, delete_partials: bool = False
                  ) -> Optional[Path]:
    partials = _scan_partials(net_dir, scope, trial, clf, cv, n_networks)
    if not partials:
        logger.info(f"  no partials for scope={scope}, trial={trial}")
        return None
    expected = set(network_order(n_networks))
    missing = expected - set(partials)
    if missing:
        logger.warning(
            f"  scope={scope}, trial={trial}: missing partials for "
            f"{sorted(missing)} ({len(partials)}/{len(expected)} present); "
            f"missing rows will be NaN."
        )

    combined = _stack_partials(partials, n_networks)
    if combined is None:
        return None

    out_name = (f"classif-networks_yeo{n_networks}_scope-{scope}_"
                f"type-{trial}_clf-{clf}_cv-{cv}.npz")
    out_path = net_dir / out_name

    meta = {
        "script": "code.classification.aggregate_network_classification",
        "timestamp": datetime.utcnow().isoformat(),
        "scope": scope,
        "trial_type": trial,
        "clf": clf,
        "cv": cv,
        "yeo": n_networks,
        "n_partials_merged": len(partials),
        "networks_present": sorted(partials),
        "networks_missing": sorted(missing),
    }
    np.savez(out_path, **combined, meta=np.asarray(json.dumps(meta)))
    logger.info(f"  wrote {out_path} "
                f"({len(partials)}/{len(expected)} networks)")

    if delete_partials:
        for p in partials.values():
            p.unlink()
        logger.info(f"  deleted {len(partials)} partials")

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge per-network partial classif files into combined bundles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--space", required=True)
    p.add_argument("--scope", default="all",
                   choices=list(SCOPES) + ["all"])
    p.add_argument("--trial-type", default="all",
                   help="alltrials | correct | lapse | all")
    p.add_argument("--yeo", type=int, default=7, choices=[7, 17])
    p.add_argument("--clf", default="logistic")
    p.add_argument("--cv", default="logo")
    p.add_argument("--results-root", default=None)
    p.add_argument("--delete-partials", action="store_true",
                   help="Remove per-network partials after successful merge.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    results_root = (Path(args.results_root) if args.results_root
                    else get_results_root(config))
    net_dir = (results_root / f"classification_{args.space}" /
               "group_mf" / "networks")
    if not net_dir.exists():
        logger.error(f"Networks dir not found: {net_dir}")
        return

    scopes = list(SCOPES) if args.scope == "all" else [args.scope]
    trials = list(TRIAL_TYPES) if args.trial_type == "all" else [args.trial_type]

    for scope in scopes:
        for trial in trials:
            logger.info(f"=== scope={scope}  trial={trial} ===")
            aggregate_one(net_dir, scope, trial, args.clf, args.cv, args.yeo,
                          delete_partials=args.delete_partials)


if __name__ == "__main__":
    main()
