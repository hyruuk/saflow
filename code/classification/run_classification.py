"""Classification analysis for decoding IN vs OUT attentional states.

Loads per-trial features from the HPC-computed `.npz` outputs, splits trials
into IN / OUT zones using each subject's VTC distribution, and runs either:

- univariate (default): one classifier per channel/ROI with shared-permutation
  t-max correction across the spatial dimension.
- multivariate: pool spatial units into a single feature vector and run a
  single permutation-based classification (sklearn's permutation_test_score).

Supported feature families (folder layout under <features>/):
- fooof_{space}/                  -> fooof_exponent, fooof_offset, fooof_r_squared
- welch_psds_{space}/             -> psd_<band> (band averaged from configured bins)
- welch_psds_corrected_{space}/   -> psd_corrected_<band>
- complexity_{space}/             -> complexity_<metric> (lzc_median, entropy_*, fractal_*)

Usage:
    python -m code.classification.run_classification \
        --feature complexity_lzc_median \
        --space sensor --mode univariate --n-permutations 1000
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)
from tqdm import tqdm

from code.classification.classifiers import get_classifier
from code.statistics.corrections import (
    apply_bonferroni_correction,
    apply_fdr_correction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / provenance
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[Path] = None) -> Dict:
    if config_path is None:
        config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def inout_bounds_to_string(bounds: Tuple[int, int]) -> str:
    return f"{bounds[0]}{bounds[1]}"


# ---------------------------------------------------------------------------
# Feature loading from new npz format
# ---------------------------------------------------------------------------

def parse_feature(feature: str) -> Tuple[str, str, Optional[str]]:
    """Map a feature name to (folder_prefix, file_suffix, sub_key).

    Returns:
        (folder_prefix, file_pattern_suffix, sub_key)

    Examples:
        fooof_exponent          -> ("fooof", "desc-fooof.npz", "exponent")
        psd_alpha               -> ("welch_psds", "desc-welch_psds.npz", "alpha")
        psd_corrected_alpha     -> ("welch_psds_corrected",
                                    "desc-welch-corrected_psds.npz", "alpha")
        complexity_lzc_median   -> ("complexity", "desc-complexity.npz", "lzc_median")
    """
    if feature.startswith("fooof_"):
        return "fooof", "desc-fooof.npz", feature[len("fooof_"):]
    if feature.startswith("psd_corrected_"):
        return (
            "welch_psds_corrected",
            "desc-welch-corrected_psds.npz",
            feature[len("psd_corrected_"):],
        )
    if feature.startswith("psd_"):
        return "welch_psds", "desc-welch_psds.npz", feature[len("psd_"):]
    if feature.startswith("complexity_"):
        return "complexity", "desc-complexity.npz", feature[len("complexity_"):]
    raise ValueError(
        f"Unknown feature '{feature}'. Expected one of: fooof_*, psd_*, "
        f"psd_corrected_*, complexity_*"
    )


def load_classification_data(
    feature: str,
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict,
    subjects: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load per-trial feature data, split into IN/OUT using per-subject VTC.

    Args:
        feature: feature name (see parse_feature).
        space: "sensor", "source", or atlas name (e.g. "schaefer_400").
        inout_bounds: (low_pct, high_pct) for IN/OUT VTC zones.
        config: loaded config.yaml dictionary.
        subjects: list of subject IDs; defaults to config["bids"]["subjects"].

    Returns:
        X        : (n_trials, n_spatial)
        y        : (n_trials,) labels — 0 = IN, 1 = OUT
        groups   : (n_trials,) subject indices
        metadata : dict with shape, retention, and provenance info.
    """
    folder_prefix, file_suffix, sub_key = parse_feature(feature)

    data_root = Path(config["paths"]["data_root"])
    feature_root = data_root / config["paths"]["features"] / f"{folder_prefix}_{space}"
    if not feature_root.exists():
        raise FileNotFoundError(f"Feature folder not found: {feature_root}")

    if subjects is None:
        subjects = config["bids"]["subjects"]
    runs = config["bids"]["task_runs"]
    freq_bands = config.get("features", {}).get("frequency_bands", {})

    is_psd = folder_prefix in ("welch_psds", "welch_psds_corrected")
    if is_psd:
        if sub_key not in freq_bands:
            raise ValueError(
                f"Band '{sub_key}' not in config.features.frequency_bands. "
                f"Known bands: {list(freq_bands)}"
            )

    all_X: List[np.ndarray] = []
    all_y: List[int] = []
    all_groups: List[int] = []
    input_git_hashes: set = set()
    spatial_names: Optional[List[str]] = None
    n_in_total = 0
    n_out_total = 0

    for subj_idx, subject in enumerate(tqdm(subjects, desc="Loading", unit="subj")):
        subj_dir = feature_root / f"sub-{subject}"
        if not subj_dir.exists():
            continue

        subj_data = []
        subj_vtc = []
        subj_task = []

        for run in runs:
            files = list(subj_dir.glob(f"sub-{subject}_*_run-{run}_*_{file_suffix}"))
            if not files:
                continue
            file_path = files[0]

            try:
                npz = np.load(file_path, allow_pickle=True)
            except Exception as e:
                logger.warning(f"Could not load {file_path.name}: {e}")
                continue

            params_file = file_path.with_name(file_path.stem + "_params.json")
            if params_file.exists():
                try:
                    params = json.loads(params_file.read_text())
                    if "git_hash" in params:
                        input_git_hashes.add(params["git_hash"])
                except Exception:
                    pass

            meta = npz["trial_metadata"].item()

            if is_psd:
                if "psds" not in npz:
                    logger.warning(f"'psds' key missing in {file_path.name}")
                    continue
                psds = npz["psds"]  # (n_trials, n_spatial, n_freqs)
                freqs = npz["freqs"]
                fmin, fmax = freq_bands[sub_key]
                fmask = (freqs >= fmin) & (freqs <= fmax)
                feat = np.mean(psds[:, :, fmask], axis=2)  # (n_trials, n_spatial)
            else:
                if sub_key not in npz:
                    logger.warning(
                        f"'{sub_key}' missing in {file_path.name}; "
                        f"available: {list(npz.keys())}"
                    )
                    continue
                feat = npz[sub_key]  # (n_trials, n_spatial)

            if spatial_names is None and "ch_names" in npz:
                spatial_names = list(npz["ch_names"])

            subj_data.append(feat)
            subj_vtc.append(np.asarray(meta["VTC_filtered"], dtype=float))
            subj_task.append(np.asarray(meta["task"]))

        if not subj_data:
            continue

        subj_data = np.concatenate(subj_data, axis=0)
        subj_vtc = np.concatenate(subj_vtc)
        subj_task = np.concatenate(subj_task)

        inbound = np.nanpercentile(subj_vtc, inout_bounds[0])
        outbound = np.nanpercentile(subj_vtc, inout_bounds[1])

        task_mask = subj_task == "correct_commission"
        in_mask = task_mask & (subj_vtc <= inbound)
        out_mask = task_mask & (subj_vtc >= outbound)

        n_in = int(in_mask.sum())
        n_out = int(out_mask.sum())
        if n_in == 0 or n_out == 0:
            continue

        all_X.append(subj_data[in_mask])
        all_X.append(subj_data[out_mask])
        all_y.extend([0] * n_in + [1] * n_out)
        all_groups.extend([subj_idx] * (n_in + n_out))
        n_in_total += n_in
        n_out_total += n_out

    if not all_X:
        raise ValueError("No data loaded for any subject")

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y)
    groups = np.array(all_groups)

    metadata = {
        "feature": feature,
        "space": space,
        "inout_bounds": list(inout_bounds),
        "n_subjects": int(len(np.unique(groups))),
        "n_trials": int(len(y)),
        "n_spatial": int(X.shape[1]),
        "n_in": int(n_in_total),
        "n_out": int(n_out_total),
        "input_git_hashes": sorted(input_git_hashes),
    }
    if spatial_names is not None:
        metadata["spatial_names"] = spatial_names

    logger.info(
        f"Loaded {len(y)} trials from {metadata['n_subjects']} subjects "
        f"(IN: {n_in_total}, OUT: {n_out_total}, n_spatial: {X.shape[1]})"
    )
    return X, y, groups, metadata


# ---------------------------------------------------------------------------
# Class balancing (within-subject, within-class subsampling)
# ---------------------------------------------------------------------------

def balance_within_subject(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42069
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n_per_class = min((y[idx] == c).sum() for c in (0, 1))
        if n_per_class == 0:
            continue
        for c in (0, 1):
            class_idx = idx[y[idx] == c]
            keep.extend(rng.choice(class_idx, size=n_per_class, replace=False))
    keep = np.sort(np.array(keep))
    return X[keep], y[keep], groups[keep]


# ---------------------------------------------------------------------------
# CV strategy
# ---------------------------------------------------------------------------

def get_cv_strategy(name: str, n_splits: int = 5) -> object:
    if name == "logo":
        return LeaveOneGroupOut()
    if name == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    if name == "group":
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unknown cv: {name}")


def _is_group_cv(cv) -> bool:
    return isinstance(cv, (LeaveOneGroupOut, GroupKFold))


# ---------------------------------------------------------------------------
# Univariate classification with shared-permutation t-max
# ---------------------------------------------------------------------------

def _score_one_spatial(clf, X_col, y, cv, groups, scoring="roc_auc"):
    kw = {"groups": groups} if _is_group_cv(cv) else {}
    scores = cross_val_score(
        clf, X_col.reshape(-1, 1), y, cv=cv, scoring=scoring, n_jobs=1, **kw
    )
    return float(np.mean(scores))


def _permute_y_within_groups(y, groups, rng):
    y_perm = y.copy()
    for g in np.unique(groups):
        m = groups == g
        y_perm[m] = rng.permutation(y[m])
    return y_perm


def run_univariate_with_tmax(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_factory,
    cv,
    n_permutations: int,
    n_jobs: int = -1,
    seed: int = 42,
    scoring: str = "roc_auc",
) -> Dict:
    """Per-spatial classification with shared-permutation t-max correction.

    All channels share the same permuted labels at each iteration, so the per-
    iteration max across channels is well-defined and t-max p-values control
    FWER across the spatial dimension.

    Args:
        X: (n_trials, n_spatial)
        y: (n_trials,)
        groups: (n_trials,)
        clf_factory: callable returning a fresh sklearn classifier instance.
        cv: sklearn CV splitter.
        n_permutations: number of label permutations.
        n_jobs: parallel jobs over (channels, permutations).
        seed: RNG seed.
        scoring: sklearn scoring name.

    Returns:
        dict with: observed (n_spatial,), perm_scores (n_perms, n_spatial),
        pvals_uncorrected, pvals_tmax, pvals_fdr_bh, pvals_bonferroni.
    """
    n_trials, n_spatial = X.shape
    logger.info(
        f"Univariate t-max: n_trials={n_trials}, n_spatial={n_spatial}, "
        f"n_permutations={n_permutations}"
    )

    # Observed scores in parallel across spatial units
    logger.info("Computing observed scores per spatial unit…")
    observed = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_score_one_spatial)(clf_factory(), X[:, s], y, cv, groups, scoring)
            for s in range(n_spatial)
        )
    )

    # Permutation scores: outer loop over permutations (shared y_perm), inner
    # parallelism over channels.
    logger.info("Running permutations with shared label shuffles…")
    rng = np.random.default_rng(seed)
    perm_scores = np.zeros((n_permutations, n_spatial), dtype=float)

    for p in tqdm(range(n_permutations), desc="permutations", unit="perm"):
        y_perm = _permute_y_within_groups(y, groups, rng)
        scores_p = Parallel(n_jobs=n_jobs)(
            delayed(_score_one_spatial)(
                clf_factory(), X[:, s], y_perm, cv, groups, scoring
            )
            for s in range(n_spatial)
        )
        perm_scores[p, :] = scores_p

    # Uncorrected p-values: per-channel right-tail
    pvals_unc = (np.sum(perm_scores >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )

    # T-max corrected p-values: compare each observed to the max-across-channels
    # null distribution
    max_perm = perm_scores.max(axis=1)
    pvals_tmax = (np.sum(max_perm[:, None] >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )

    pvals_fdr = apply_fdr_correction(pvals_unc, alpha=0.05, method="bh")
    pvals_bonf = apply_bonferroni_correction(pvals_unc, alpha=0.05)

    n_sig_tmax = int(np.sum(pvals_tmax < 0.05))
    n_sig_fdr = int(np.sum(pvals_fdr < 0.05))
    logger.info(
        f"Significant @ alpha=0.05 — t-max: {n_sig_tmax}/{n_spatial}, "
        f"FDR-BH: {n_sig_fdr}/{n_spatial}, "
        f"max observed: {observed.max():.3f}"
    )

    return {
        "mode": "univariate",
        "observed": observed,
        "perm_scores": perm_scores,
        "pvals_uncorrected": pvals_unc,
        "pvals_tmax": pvals_tmax,
        "pvals_fdr_bh": pvals_fdr,
        "pvals_bonferroni": pvals_bonf,
    }


# ---------------------------------------------------------------------------
# Multivariate classification (single classifier on pooled features)
# ---------------------------------------------------------------------------

def run_multivariate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf,
    cv,
    n_permutations: int,
    scoring: str = "roc_auc",
) -> Dict:
    kw = {"groups": groups} if _is_group_cv(cv) else {}
    score, perm_scores, pvalue = permutation_test_score(
        clf,
        X,
        y,
        cv=cv,
        n_permutations=n_permutations,
        scoring=scoring,
        n_jobs=-1,
        **kw,
    )
    logger.info(
        f"Multivariate: score={score:.3f}, p-value={pvalue:.4f} "
        f"(perm mean={np.mean(perm_scores):.3f})"
    )
    return {
        "mode": "multivariate",
        "observed": float(score),
        "perm_scores": np.asarray(perm_scores),
        "pvalue": float(pvalue),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    feature: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    mode: str,
    results: Dict,
    metadata: Dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"feature-{feature}_space-{space}"
        f"_inout-{inout_bounds_to_string(inout_bounds)}"
        f"_clf-{clf_name}_cv-{cv_name}_mode-{mode}"
    )

    if mode == "univariate":
        np.savez_compressed(
            output_dir / f"{base}_scores.npz",
            observed=results["observed"],
            perm_scores=results["perm_scores"],
            pvals_uncorrected=results["pvals_uncorrected"],
            pvals_tmax=results["pvals_tmax"],
            pvals_fdr_bh=results["pvals_fdr_bh"],
            pvals_bonferroni=results["pvals_bonferroni"],
        )
        summary = {
            "max_score": float(results["observed"].max()),
            "mean_score": float(results["observed"].mean()),
            "n_significant_tmax_a05": int((results["pvals_tmax"] < 0.05).sum()),
            "n_significant_fdr_bh_a05": int((results["pvals_fdr_bh"] < 0.05).sum()),
            "n_permutations": int(results["perm_scores"].shape[0]),
        }
    else:
        np.savez_compressed(
            output_dir / f"{base}_scores.npz",
            observed=results["observed"],
            perm_scores=results["perm_scores"],
        )
        summary = {
            "score": float(results["observed"]),
            "pvalue": float(results["pvalue"]),
            "n_permutations": int(len(results["perm_scores"])),
        }

    meta_out = {
        "feature": feature,
        "space": space,
        "inout_bounds": list(inout_bounds),
        "classifier": clf_name,
        "cv_strategy": cv_name,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "data_metadata": metadata,
        "summary": summary,
    }
    meta_path = output_dir / f"{base}_metadata.json"
    meta_path.write_text(json.dumps(meta_out, indent=2))

    logger.info(f"Saved results -> {output_dir / (base + '_scores.npz')}")
    logger.info(f"Saved metadata -> {meta_path}")
    return meta_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify IN vs OUT attentional states (univariate t-max or multivariate)."
    )
    parser.add_argument(
        "--feature",
        required=True,
        help=(
            "Feature name. Examples: fooof_exponent, psd_alpha, psd_corrected_alpha, "
            "complexity_lzc_median, complexity_entropy_permutation, "
            "complexity_fractal_higuchi."
        ),
    )
    parser.add_argument(
        "--space",
        default="sensor",
        help="Analysis space: 'sensor', 'source', or atlas name (e.g. 'schaefer_400').",
    )
    parser.add_argument(
        "--mode",
        choices=["univariate", "multivariate"],
        default="univariate",
        help="univariate = per-channel/ROI + t-max; multivariate = pooled features.",
    )
    parser.add_argument(
        "--clf", default="lda", choices=["lda", "svm", "rf", "logistic"]
    )
    parser.add_argument(
        "--cv", default="logo", choices=["logo", "stratified", "group"]
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    inout_bounds = tuple(config["analysis"]["inout_bounds"])

    logger.info("=" * 78)
    logger.info("CLASSIFICATION (IN vs OUT)")
    logger.info("=" * 78)
    logger.info(f"feature={args.feature}  space={args.space}  mode={args.mode}")
    logger.info(f"clf={args.clf}  cv={args.cv}  inout={inout_bounds}")
    logger.info(f"n_permutations={args.n_permutations}  balance={not args.no_balance}")
    logger.info("=" * 78)

    X, y, groups, metadata = load_classification_data(
        feature=args.feature,
        space=args.space,
        inout_bounds=inout_bounds,
        config=config,
    )

    if not args.no_balance:
        X, y, groups = balance_within_subject(X, y, groups, seed=args.seed)
        logger.info(
            f"Balanced: {len(y)} trials (IN={int((y == 0).sum())}, "
            f"OUT={int((y == 1).sum())})"
        )

    cv = get_cv_strategy(args.cv, n_splits=args.n_splits)

    if args.mode == "univariate":
        results = run_univariate_with_tmax(
            X=X,
            y=y,
            groups=groups,
            clf_factory=lambda: get_classifier(args.clf),
            cv=cv,
            n_permutations=args.n_permutations,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
    else:
        results = run_multivariate(
            X=X,
            y=y,
            groups=groups,
            clf=get_classifier(args.clf),
            cv=cv,
            n_permutations=args.n_permutations,
        )

    data_root = Path(config["paths"]["data_root"])
    output_dir = (
        data_root / config["paths"]["features"] / f"classification_{args.space}" / "group"
    )
    save_results(
        output_dir=output_dir,
        feature=args.feature,
        space=args.space,
        inout_bounds=inout_bounds,
        clf_name=args.clf,
        cv_name=args.cv,
        mode=args.mode,
        results=results,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
