"""Within- vs between-network coherence of per-subject IN-OUT contrasts.

For a given feature on an atlas space (e.g. ``schaefer_400``), this computes:

  1. Per-subject contrast at each parcel: median(OUT) − median(IN), giving an
     array of shape (n_subjects, n_parcels).
  2. The parcel × parcel Pearson correlation across subjects.
  3. Per Yeo network: ⟨r⟩ within the network vs ⟨r⟩ between that network and
     all other networks.

Interpretation: features whose IN-OUT effect respects the Yeo decomposition
should show within > between for the networks they engage. Sanity check on
whether the parcellation is informative for this feature.

Output (one file per feature × trial_type, mirroring the stats layout):
    coherence_feature-<f>_yeo<N>_type-<trial>.npz
        feature              : str
        trial_type           : str
        network_names        : (n_networks,)
        within_r             : (n_networks,)        mean within-network r
        between_r            : (n_networks,)        mean between-network r
        corr_matrix          : (n_parcels, n_parcels)   per-subject r across parcels
        ch_names             : (n_parcels,) cortical parcels only
        n_subjects           : int
        meta                 : pickled provenance dict

Usage:
    python -m code.statistics.network_coherence \\
        --feature fooof_exponent --space schaefer_400 \\
        --trial-type correct --yeo 7
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.statistics.run_group_statistics import load_all_features_batched
from code.utils.config import load_config
from code.utils.paths import get_features_root, get_results_root
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


def _per_subject_contrast(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    aggregate: str = "median",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-subject OUT − IN contrast at each parcel.

    Mirrors the subject-aggregation in ``run_group_statistics._subject_aggregate``
    but exposes the per-subject contrast vector instead of the test result.

    Args:
        X: (n_features=1, n_trials, n_parcels) — trial-level features.
        y: (n_trials,) labels 0=IN, 1=OUT.
        groups: (n_trials,) subject indices.
        aggregate: 'median' (default) or 'mean'.

    Returns:
        (contrast, kept_subjects) where contrast is shape (n_subjects, n_parcels).
    """
    if X.shape[0] != 1:
        raise ValueError(
            f"Coherence expects a single feature per call; got {X.shape[0]}."
        )
    reducer = np.nanmedian if aggregate == "median" else np.nanmean

    subj_in: List[np.ndarray] = []
    subj_out: List[np.ndarray] = []
    kept: List = []
    for subj in np.unique(groups):
        sm = groups == subj
        in_mask = sm & (y == 0)
        out_mask = sm & (y == 1)
        if not (in_mask.any() and out_mask.any()):
            continue
        subj_in.append(reducer(X[0, in_mask, :], axis=0))
        subj_out.append(reducer(X[0, out_mask, :], axis=0))
        kept.append(subj)
    if not kept:
        raise ValueError("No subject contributed both IN and OUT trials.")
    return np.asarray(subj_out) - np.asarray(subj_in), np.asarray(kept)


def _corrcoef_skipnan(M: np.ndarray) -> np.ndarray:
    """Pearson correlation across rows ignoring NaNs.

    ``M`` has shape (n_subjects, n_parcels). Returns (n_parcels, n_parcels).
    Pairs with <2 finite overlapping rows give NaN.
    """
    # Center each column on its nanmean; replace NaNs with 0 only AFTER
    # computing per-pair denominators so they don't bias the correlation.
    n_subj, n_par = M.shape
    out = np.full((n_par, n_par), np.nan, dtype=float)
    mu = np.nanmean(M, axis=0)
    centered = M - mu  # NaNs propagate; we mask them below
    finite = np.isfinite(centered)

    for i in range(n_par):
        ci = centered[:, i]
        fi = finite[:, i]
        for j in range(i, n_par):
            cj = centered[:, j]
            fj = finite[:, j]
            mask = fi & fj
            n = int(mask.sum())
            if n < 2:
                continue
            xi, xj = ci[mask], cj[mask]
            num = float(np.sum(xi * xj))
            den = float(np.sqrt(np.sum(xi * xi) * np.sum(xj * xj)))
            if den == 0:
                continue
            r = num / den
            out[i, j] = r
            out[j, i] = r
    return out


def network_coherence(
    corr_matrix: np.ndarray,
    ch_names: List[str],
    n_networks: int = 7,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Compute mean within- and between-network r per network.

    Diagonal is excluded. Within = mean over distinct parcel pairs whose
    members are both in the network. Between = mean over parcel pairs where
    exactly one member is in the network.
    """
    nets = network_order(n_networks)
    pidx = network_parcel_indices(ch_names, n_networks=n_networks)

    within = np.full(len(nets), np.nan)
    between = np.full(len(nets), np.nan)

    for k, net in enumerate(nets):
        idx = pidx[net]
        if idx.size < 2:
            continue
        # Within: upper triangle of the network's submatrix, off-diagonal
        sub = corr_matrix[np.ix_(idx, idx)]
        iu = np.triu_indices(sub.shape[0], k=1)
        within[k] = float(np.nanmean(sub[iu]))

        # Between: rows in `idx`, cols outside
        other = np.setdiff1d(np.arange(corr_matrix.shape[0]), idx)
        if other.size == 0:
            continue
        cross = corr_matrix[np.ix_(idx, other)]
        between[k] = float(np.nanmean(cross))

    return within, between, nets


def run_one_feature(
    feature: str,
    space: str,
    trial_type: str,
    n_networks: int,
    config: Dict,
    inout_bounds: Tuple[int, int],
    aggregate: str = "median",
) -> Dict[str, np.ndarray]:
    """Full pipeline for one (feature, trial_type)."""
    logger.info(f"Loading feature '{feature}' [space={space}, trial={trial_type}]")
    bundle = load_all_features_batched(
        feature_types=[feature],
        space=space,
        inout_bounds=inout_bounds,
        config=config,
        trial_type=trial_type,
        drop_bad_trials=True,
    )
    X, y, groups, meta = bundle[feature]
    logger.info(f"  loaded {X.shape[1]} trials over {len(np.unique(groups))} subjects, "
                f"n_parcels={X.shape[2]}")

    ch_names: Optional[List[str]] = meta.get("ch_names")
    if ch_names is None:
        raise RuntimeError(
            f"Feature loader did not return ch_names for {feature}/{space}. "
            f"Backfill via code.utils.backfill_ch_names."
        )
    ch_names = [str(x) for x in ch_names]

    # Drop non-cortical (medial wall) parcels up front.
    assignments = get_network_assignments(ch_names, n_networks=n_networks)
    keep = assignments != UNKNOWN_NETWORK
    if not keep.all():
        logger.info(f"  dropping {(~keep).sum()} Unknown/medial-wall parcels")
    X = X[:, :, keep]
    ch_keep = [c for c, k in zip(ch_names, keep) if k]

    contrast, kept_subj = _per_subject_contrast(X, y, groups, aggregate=aggregate)
    logger.info(f"  per-subject contrasts: {contrast.shape} (subjects × parcels)")

    logger.info("  computing correlation matrix (this is the heavy step)...")
    R = _corrcoef_skipnan(contrast)

    within, between, nets = network_coherence(R, ch_keep, n_networks=n_networks)
    for net, w, b in zip(nets, within, between):
        delta = (w - b) if np.isfinite(w) and np.isfinite(b) else float("nan")
        logger.info(f"    {net}: within={w:+.3f}  between={b:+.3f}  Δ={delta:+.3f}")

    return {
        "feature": np.asarray(feature),
        "trial_type": np.asarray(trial_type),
        "network_names": np.asarray(nets),
        "within_r": within,
        "between_r": between,
        "corr_matrix": R,
        "ch_names": np.asarray(ch_keep),
        "n_subjects": np.asarray(kept_subj.size),
        "aggregate": np.asarray(aggregate),
    }


# Default feature set for the network-story panel: corrected PSD α/θ, FOOOF
# exponent + offset, LZc. Aligns with the panel's column ordering.
DEFAULT_FEATURES: Tuple[str, ...] = (
    "psd_corrected_alpha",
    "psd_corrected_theta",
    "fooof_exponent",
    "fooof_offset",
    "complexity_lzc_median",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Within/between Yeo-network coherence of IN-OUT contrasts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--feature", default=None,
                   help="Feature name (e.g. fooof_exponent). Omit to use the "
                        "default DEFAULT_FEATURES set.")
    p.add_argument("--space", required=True,
                   help="Atlas space (e.g. schaefer_400).")
    p.add_argument("--trial-type", default="all",
                   help="alltrials | correct | lapse | all (= all three)")
    p.add_argument("--yeo", type=int, default=7, choices=[7, 17])
    p.add_argument("--aggregate", default="median", choices=["median", "mean"],
                   help="Per-subject reducer (matches run_group_statistics default).")
    p.add_argument("--results-root", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    results_root = (Path(args.results_root) if args.results_root
                    else get_results_root(config))
    out_dir = results_root / f"statistics_{args.space}" / "group" / "networks"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    inout_bounds = tuple(config.get("analysis", {}).get("inout_bounds", [25, 75]))

    features = [args.feature] if args.feature else list(DEFAULT_FEATURES)
    trial_types = (
        ["alltrials", "correct", "lapse"]
        if args.trial_type == "all" else [args.trial_type]
    )

    provenance = {
        "script": "code.statistics.network_coherence",
        "timestamp": datetime.utcnow().isoformat(),
        "space": args.space,
        "yeo": args.yeo,
        "aggregate": args.aggregate,
        "inout_bounds": list(inout_bounds),
    }

    for feature in features:
        for trial in trial_types:
            logger.info(f"=== feature={feature}  trial={trial} ===")
            try:
                payload = run_one_feature(
                    feature=feature,
                    space=args.space,
                    trial_type=trial,
                    n_networks=args.yeo,
                    config=config,
                    inout_bounds=inout_bounds,
                    aggregate=args.aggregate,
                )
            except FileNotFoundError as exc:
                logger.warning(f"  skipping: {exc}")
                continue
            out_name = (f"coherence_feature-{feature}_"
                        f"yeo{args.yeo}_type-{trial}.npz")
            out_path = out_dir / out_name
            np.savez(out_path, **payload,
                     meta=np.asarray(json.dumps(provenance | {
                         "feature": feature, "trial_type": trial,
                     })))
            logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
