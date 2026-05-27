"""Yeo-network-restricted IN-vs-OUT classification.

Wraps the joint-axis classifier from ``run_multifeature`` but restricts the
spatial axis to one Yeo network at a time. Each call produces one balanced-
accuracy + permutation null per (network, feature_or_family) cell.

Three scopes (the user picks one or ``all``):
  - ``per-family``  : one classifier per (network, family); families default
                      to {psds_corrected, fooof, complexity}. Features within
                      the family are stacked. Feeds the Tier-3 bar plot.
  - ``per-feature`` : one classifier per (network, single_feature) using ALL
                      of the network's parcels as the input vector. Feeds the
                      Tier-3 heatmap.
  - ``joint``       : one classifier per network with ALL configured features
                      stacked across all of the network's parcels. Feeds the
                      Tier-4 importance comparison.

Outputs (one file per scope, indexed by network/family/feature):
    results/classification_<space>/group_mf/networks/
        classif-networks_yeo{N}_scope-<scope>_type-<trial>_clf-<clf>_cv-<cv>.npz

Usage:
    python -m code.classification.run_network_classification \\
        --space schaefer_400 --scope all --trial-type all --yeo 7
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.classification.run_classification import (
    balance_within_subject,
    expand_feature_set,
    get_cv_strategy,
    get_git_hash,
    inout_bounds_to_string,
    load_combined_features,
    standardize_within_subject,
)
from code.features.inout_selection import (
    DEFAULT_STRATEGY as DEFAULT_INOUT_STRATEGY,
    inout_selection_token,
)
from code.classification.run_multifeature import (
    make_pipeline_factory,
    run_joint,
)
from code.utils.config import load_config
from code.utils.paths import get_results_root
from code.utils.yeo_networks import (
    UNKNOWN_NETWORK,
    get_network_assignments,
    network_order,
    network_parcel_indices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SCOPES: Tuple[str, ...] = ("per-family", "per-feature", "joint")


def _output_is_complete(out_path: Path) -> bool:
    """Return True if a network-classif npz exists and loads cleanly.

    Mirrors ``run_multifeature._output_is_complete`` so a re-submitted
    SLURM array only re-runs the (scope × trial × network) cells that
    are actually missing or corrupted.
    """
    if not out_path.exists():
        return False
    try:
        with np.load(out_path, allow_pickle=True) as npz:
            if "scores" not in npz.files or "meta" not in npz.files:
                return False
    except (OSError, ValueError, KeyError) as e:
        logger.warning(
            f"Existing output looks invalid ({e}); will recompute: {out_path.name}"
        )
        return False
    return True


def _build_out_name(
    yeo: int, scope: str, trial: str, clf: str, cv: str,
    sel_tok: str, network: Optional[str],
) -> str:
    base = (f"classif-networks_yeo{yeo}_scope-{scope}_"
            f"type-{trial}_clf-{clf}_cv-{cv}{sel_tok}")
    return f"{base}_net-{network}.npz" if network else f"{base}.npz"

# Default families used by per-family scope. Each maps to the shortcut
# name that ``expand_feature_set`` understands.
DEFAULT_FAMILIES: Tuple[str, ...] = ("psds_corrected", "fooof", "complexity")

# Default per-feature scope feature list — matches the panel column choice
# so the heatmap rows align with what Tier 1/2 visualize. Override with
# --per-feature-features.
DEFAULT_PER_FEATURE: Tuple[str, ...] = (
    "psd_corrected_alpha",
    "psd_corrected_theta",
    "fooof_exponent",
    "fooof_offset",
    "complexity_lzc_median",
)


def _run_one_cell(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    parcel_idx: np.ndarray,
    feature_idx: np.ndarray,
    clf_factory,
    cv,
    n_permutations: int,
    scoring: str,
    seed: int,
    n_jobs: int,
) -> Dict[str, np.ndarray]:
    """Run one (network, feature-subset) joint classifier.

    ``X`` has shape (n_trials, n_spatial, n_features). The cell uses the
    subset ``X[:, parcel_idx, :][:, :, feature_idx]``.
    """
    if parcel_idx.size == 0:
        return {
            "score": float("nan"),
            "pvalue": float("nan"),
            "perm_scores": np.full(n_permutations, np.nan),
            "n_parcels": 0,
            "n_features": int(feature_idx.size),
        }
    X_cell = X[:, parcel_idx, :][:, :, feature_idx]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run_joint(
                X_cell, y, groups,
                clf_factory=clf_factory, cv=cv,
                n_permutations=n_permutations, scoring=scoring,
                importance_method="none",
                n_jobs=n_jobs, seed=seed,
            )
    except ValueError as exc:
        logger.warning(f"  cell skipped: {exc}")
        return {
            "score": float("nan"),
            "pvalue": float("nan"),
            "perm_scores": np.full(n_permutations, np.nan),
            "n_parcels": int(parcel_idx.size),
            "n_features": int(feature_idx.size),
        }
    return {
        "score": float(res["observed"]),
        "pvalue": float(res["pvalue"]),
        "perm_scores": np.asarray(res["perm_scores"]),
        "n_parcels": int(parcel_idx.size),
        "n_features": int(feature_idx.size),
    }


def _run_scope_per_family(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    network_to_parcels: Dict[str, np.ndarray],
    networks: Tuple[str, ...],
    families: List[str],
    family_to_features: Dict[str, List[str]],
    clf_factory,
    cv,
    n_permutations: int,
    scoring: str,
    seed: int,
    n_jobs: int,
) -> Dict[str, np.ndarray]:
    feat_to_col = {f: i for i, f in enumerate(feature_names)}
    n_nets, n_fams = len(networks), len(families)

    scores = np.full((n_nets, n_fams), np.nan)
    pvals = np.full((n_nets, n_fams), np.nan)
    perm = np.full((n_nets, n_fams, n_permutations), np.nan)
    n_parc = np.zeros(n_nets, dtype=int)
    n_feat = np.zeros(n_fams, dtype=int)

    for j, fam in enumerate(families):
        feats = family_to_features[fam]
        idx_feat = np.asarray([feat_to_col[f] for f in feats if f in feat_to_col],
                              dtype=int)
        n_feat[j] = idx_feat.size
        if idx_feat.size == 0:
            logger.warning(f"family {fam!r}: no features loaded; skipping")
            continue
        for i, net in enumerate(networks):
            idx_par = network_to_parcels[net]
            if j == 0:
                n_parc[i] = idx_par.size
            logger.info(f"  [{net} × {fam}] n_parcels={idx_par.size}, "
                        f"n_features={idx_feat.size}")
            res = _run_one_cell(X, y, groups, idx_par, idx_feat,
                                clf_factory, cv, n_permutations, scoring,
                                seed, n_jobs)
            scores[i, j] = res["score"]
            pvals[i, j] = res["pvalue"]
            perm[i, j, :] = res["perm_scores"]

    return {
        "network_names": np.asarray(networks),
        "families": np.asarray(families),
        "scores": scores,
        "pvals": pvals,
        "perm_scores": perm,
        "n_parcels_per_network": n_parc,
        "n_features_per_family": n_feat,
    }


def _run_scope_per_feature(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    network_to_parcels: Dict[str, np.ndarray],
    networks: Tuple[str, ...],
    features_subset: List[str],
    clf_factory,
    cv,
    n_permutations: int,
    scoring: str,
    seed: int,
    n_jobs: int,
) -> Dict[str, np.ndarray]:
    feat_to_col = {f: i for i, f in enumerate(feature_names)}
    n_nets, n_feats = len(networks), len(features_subset)

    scores = np.full((n_nets, n_feats), np.nan)
    pvals = np.full((n_nets, n_feats), np.nan)
    perm = np.full((n_nets, n_feats, n_permutations), np.nan)
    n_parc = np.zeros(n_nets, dtype=int)

    for j, feat in enumerate(features_subset):
        if feat not in feat_to_col:
            logger.warning(f"feature {feat!r}: not loaded; skipping")
            continue
        idx_feat = np.asarray([feat_to_col[feat]], dtype=int)
        for i, net in enumerate(networks):
            idx_par = network_to_parcels[net]
            if j == 0:
                n_parc[i] = idx_par.size
            logger.info(f"  [{net} × {feat}] n_parcels={idx_par.size}")
            res = _run_one_cell(X, y, groups, idx_par, idx_feat,
                                clf_factory, cv, n_permutations, scoring,
                                seed, n_jobs)
            scores[i, j] = res["score"]
            pvals[i, j] = res["pvalue"]
            perm[i, j, :] = res["perm_scores"]

    return {
        "network_names": np.asarray(networks),
        "features": np.asarray(features_subset),
        "scores": scores,
        "pvals": pvals,
        "perm_scores": perm,
        "n_parcels_per_network": n_parc,
    }


def _run_scope_joint(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    network_to_parcels: Dict[str, np.ndarray],
    networks: Tuple[str, ...],
    clf_factory,
    cv,
    n_permutations: int,
    scoring: str,
    seed: int,
    n_jobs: int,
) -> Dict[str, np.ndarray]:
    n_nets = len(networks)
    idx_feat_all = np.arange(len(feature_names))

    scores = np.full(n_nets, np.nan)
    pvals = np.full(n_nets, np.nan)
    perm = np.full((n_nets, n_permutations), np.nan)
    n_parc = np.zeros(n_nets, dtype=int)

    for i, net in enumerate(networks):
        idx_par = network_to_parcels[net]
        n_parc[i] = idx_par.size
        logger.info(f"  [{net} × ALL features] n_parcels={idx_par.size}, "
                    f"n_features={idx_feat_all.size}")
        res = _run_one_cell(X, y, groups, idx_par, idx_feat_all,
                            clf_factory, cv, n_permutations, scoring,
                            seed, n_jobs)
        scores[i] = res["score"]
        pvals[i] = res["pvalue"]
        perm[i, :] = res["perm_scores"]

    return {
        "network_names": np.asarray(networks),
        "features_used": np.asarray(feature_names),
        "scores": scores,
        "pvals": pvals,
        "perm_scores": perm,
        "n_parcels_per_network": n_parc,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Yeo-network-restricted IN-vs-OUT classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--space", required=True,
                   help="Atlas space (e.g. schaefer_400).")
    p.add_argument("--scope", default="all",
                   choices=list(SCOPES) + ["all"],
                   help="Which scope to run. 'all' runs the three in sequence.")
    p.add_argument("--trial-type", default="all",
                   help="alltrials | correct | lapse | all")
    p.add_argument("--yeo", type=int, default=7, choices=[7, 17])

    # Per-family configuration
    p.add_argument("--families", default=" ".join(DEFAULT_FAMILIES),
                   help="Space-separated family shortcuts for per-family scope.")
    p.add_argument("--per-feature-features", default=" ".join(DEFAULT_PER_FEATURE),
                   help="Space-separated feature names for per-feature scope.")

    # Standard classification knobs (mirror run_multifeature defaults)
    p.add_argument("--clf", default="logistic",
                   choices=["lda", "svm", "rf", "logistic"])
    p.add_argument("--cv", default="logo")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--scoring", default="roc_auc")
    p.add_argument("--standardize", default="per-subject",
                   choices=["per-subject", "none"])
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--no-per-feature-scale", action="store_true")
    p.add_argument("--n-events-window", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subjects", type=str, default=None,
                   help="Optional space-separated subject IDs (default: config).")
    p.add_argument("--results-root", default=None)
    p.add_argument("--keep-bad-trials", action="store_true")
    p.add_argument("--network", default=None,
                   help="Restrict to a single Yeo network (e.g. 'Default'). "
                        "Writes a partial *_net-<network>.npz; combine via "
                        "code.classification.aggregate_network_classification. "
                        "Used by SLURM array sharding.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even when valid output already exists. "
                        "Default skips (trial × scope) cells whose npz is "
                        "present and loads cleanly — lets a re-submitted "
                        "SLURM array fill only the missing pieces.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    inout_bounds = tuple(config.get("analysis", {}).get("inout_bounds", [25, 75]))
    inout_selection = str(
        config.get("analysis", {}).get("inout_selection", DEFAULT_INOUT_STRATEGY)
    )
    subjects = args.subjects.split() if args.subjects else None

    results_root = (Path(args.results_root) if args.results_root
                    else get_results_root(config))
    out_dir = (results_root / f"classification_{args.space}" /
               "group_mf" / "networks")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # Expand families now so we can union them into a single feature load.
    families = args.families.split()
    family_to_features: Dict[str, List[str]] = {
        fam: expand_feature_set(fam, config) for fam in families
    }
    per_feature_subset = args.per_feature_features.split()

    # Union of all features we'll need across scopes — load once.
    feature_union: List[str] = []
    for fam in families:
        feature_union.extend(family_to_features[fam])
    for f in per_feature_subset:
        if f not in feature_union:
            feature_union.append(f)
    # Stable ordering
    feature_union = list(dict.fromkeys(feature_union))
    logger.info(f"Will load {len(feature_union)} feature(s): {feature_union}")

    scopes = list(SCOPES) if args.scope == "all" else [args.scope]
    trial_types = (
        ["alltrials", "correct", "lapse"]
        if args.trial_type == "all" else [args.trial_type]
    )

    provenance_base = {
        "script": "code.classification.run_network_classification",
        "timestamp": datetime.utcnow().isoformat(),
        "git_hash": get_git_hash(),
        "space": args.space,
        "yeo": args.yeo,
        "clf": args.clf,
        "cv": args.cv,
        "n_permutations": args.n_permutations,
        "scoring": args.scoring,
        "standardize": args.standardize,
        "no_balance": args.no_balance,
        "families": families,
        "per_feature_features": per_feature_subset,
        "inout_bounds": list(inout_bounds),
        "inout_selection": inout_selection,
    }

    sel_tok = inout_selection_token(inout_selection)
    for trial in trial_types:
        logger.info(f"=== trial-type={trial} ===")

        # Skip-existing: figure out which scopes still need work for this
        # trial-type BEFORE loading features. Filenames depend only on args,
        # so we can short-circuit the expensive load when everything is done.
        scopes_pending: List[str] = []
        for scope in scopes:
            out_name = _build_out_name(
                args.yeo, scope, trial, args.clf, args.cv, sel_tok,
                args.network,
            )
            out_path = out_dir / out_name
            if not args.force and _output_is_complete(out_path):
                logger.info(f"  SKIP scope={scope}: already complete -> {out_name}")
                continue
            scopes_pending.append(scope)
        if not scopes_pending:
            logger.info(f"  trial={trial}: all scopes already complete; skipping load")
            continue

        try:
            X, y, groups, metadata = load_combined_features(
                features=feature_union,
                space=args.space,
                inout_bounds=inout_bounds,
                config=config,
                subjects=subjects,
                drop_bad_trials=not args.keep_bad_trials,
                trial_type=trial,
                n_events_window=args.n_events_window,
                inout_selection=inout_selection,
            )
        except FileNotFoundError as exc:
            logger.warning(f"  trial={trial}: feature load failed: {exc}")
            continue

        X = np.asarray(X)  # (n_trials, n_spatial, n_features)
        logger.info(f"  loaded X={X.shape}, y={y.shape}, groups={groups.shape}")

        spatial_names = metadata.get("spatial_names")
        if spatial_names is None:
            raise RuntimeError(
                "load_combined_features returned no spatial_names; cannot map "
                "parcels to networks. Backfill via code.utils.backfill_ch_names."
            )
        ch_names = [str(x) for x in spatial_names]

        # Drop Unknown (medial-wall) parcels.
        assignments = get_network_assignments(ch_names, n_networks=args.yeo)
        keep = assignments != UNKNOWN_NETWORK
        if not keep.all():
            logger.info(f"  dropping {(~keep).sum()} Unknown parcels")
            X = X[:, keep, :]
            ch_names = [c for c, k in zip(ch_names, keep) if k]

        # Per-subject z-score (matches run_multifeature default).
        if args.standardize == "per-subject":
            X = standardize_within_subject(X, groups)

        # Within-subject balance.
        if not args.no_balance:
            X, y, groups = balance_within_subject(X, y, groups, seed=args.seed)
            logger.info(f"  balanced: n_trials={len(y)}  "
                        f"IN={int((y == 0).sum())}  OUT={int((y == 1).sum())}")

        # Build the per-network parcel index map ONCE per trial-type.
        net_to_parcels = network_parcel_indices(ch_names, n_networks=args.yeo)
        all_networks = network_order(args.yeo)
        if args.network is not None:
            if args.network not in all_networks:
                raise SystemExit(
                    f"--network={args.network!r} not in Yeo-{args.yeo} order "
                    f"{list(all_networks)}"
                )
            networks = (args.network,)
            logger.info(f"  restricted to single network: {args.network}")
        else:
            networks = all_networks
        for net in networks:
            logger.info(f"    network {net}: {net_to_parcels[net].size} parcels")

        cv = get_cv_strategy(args.cv, n_splits=args.n_splits)
        clf_factory = make_pipeline_factory(
            args.clf,
            per_feature_scale=not args.no_per_feature_scale,
            seed=args.seed,
        )

        for scope in scopes_pending:
            logger.info(f"--- scope={scope} ---")
            if scope == "per-family":
                payload = _run_scope_per_family(
                    X, y, groups, feature_union, net_to_parcels, networks,
                    families, family_to_features, clf_factory, cv,
                    n_permutations=args.n_permutations, scoring=args.scoring,
                    seed=args.seed, n_jobs=args.n_jobs,
                )
            elif scope == "per-feature":
                payload = _run_scope_per_feature(
                    X, y, groups, feature_union, net_to_parcels, networks,
                    per_feature_subset, clf_factory, cv,
                    n_permutations=args.n_permutations, scoring=args.scoring,
                    seed=args.seed, n_jobs=args.n_jobs,
                )
            elif scope == "joint":
                payload = _run_scope_joint(
                    X, y, groups, feature_union, net_to_parcels, networks,
                    clf_factory, cv,
                    n_permutations=args.n_permutations, scoring=args.scoring,
                    seed=args.seed, n_jobs=args.n_jobs,
                )
            else:
                raise ValueError(f"Unknown scope {scope!r}")

            out_name = _build_out_name(
                args.yeo, scope, trial, args.clf, args.cv, sel_tok,
                args.network,
            )
            out_path = out_dir / out_name
            np.savez(out_path, **payload,
                     meta=np.asarray(json.dumps(provenance_base | {
                         "trial_type": trial, "scope": scope,
                         "network_restriction": args.network,
                     })))
            logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
