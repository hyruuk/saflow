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

FOOOF_FEATURES = ["fooof_exponent", "fooof_offset", "fooof_r_squared"]
COMPLEXITY_METRICS = [
    "lzc_median",
    "entropy_permutation",
    "entropy_spectral",
    "entropy_sample",
    "entropy_approximate",
    "entropy_svd",
    "fractal_higuchi",
    "fractal_petrosian",
    "fractal_katz",
    "fractal_dfa",
]


def expand_feature_set(name: str, config: Dict) -> List[str]:
    """Expand a feature-set name to a list of individual feature names.

    Sets:
        psds            -> psd_<band> for each band in config.features.frequency_bands
        psds_corrected  -> psd_corrected_<band> for each band
        fooof           -> fooof_exponent, fooof_offset, fooof_r_squared
        complexity      -> complexity_<metric> for every npz key in COMPLEXITY_METRICS
        all             -> union of fooof + psds + psds_corrected + complexity
    """
    bands = list(config.get("features", {}).get("frequency_bands", {}).keys())
    if name == "psds":
        return [f"psd_{b}" for b in bands]
    if name == "psds_corrected":
        return [f"psd_corrected_{b}" for b in bands]
    if name == "fooof":
        return list(FOOOF_FEATURES)
    if name == "complexity":
        return [f"complexity_{m}" for m in COMPLEXITY_METRICS]
    if name == "all":
        return (
            list(FOOOF_FEATURES)
            + [f"psd_{b}" for b in bands]
            + [f"psd_corrected_{b}" for b in bands]
            + [f"complexity_{m}" for m in COMPLEXITY_METRICS]
        )
    raise ValueError(
        f"Unknown feature-set '{name}'. Choose: psds, psds_corrected, fooof, "
        f"complexity, all."
    )


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


def load_combined_features(
    features: List[str],
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict,
    subjects: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load multiple features and stack along a new feature axis.

    Each feature must produce the same trial set (same y, same groups). When a
    feature has different per-subject trial coverage (rare), this raises.

    Returns:
        X: (n_trials, n_spatial, n_features)
        y, groups: as in load_classification_data
        metadata: includes per-feature metadata under ``per_feature``.
    """
    Xs = []
    y_ref = None
    groups_ref = None
    per_feature = {}
    spatial_names = None

    for feat in features:
        X_f, y_f, g_f, meta_f = load_classification_data(
            feature=feat,
            space=space,
            inout_bounds=inout_bounds,
            config=config,
            subjects=subjects,
        )
        if y_ref is None:
            y_ref = y_f
            groups_ref = g_f
        else:
            if not (
                len(y_f) == len(y_ref)
                and np.array_equal(y_f, y_ref)
                and np.array_equal(g_f, groups_ref)
            ):
                raise ValueError(
                    f"Feature '{feat}' produced a different trial set than "
                    f"'{features[0]}'. Cannot combine features that don't share "
                    f"identical trial alignment."
                )
        Xs.append(X_f)
        per_feature[feat] = {k: v for k, v in meta_f.items() if k != "spatial_names"}
        if spatial_names is None:
            spatial_names = meta_f.get("spatial_names")

    X = np.stack(Xs, axis=2)  # (n_trials, n_spatial, n_features)

    metadata = {
        "features": list(features),
        "space": space,
        "inout_bounds": list(inout_bounds),
        "n_subjects": int(len(np.unique(groups_ref))),
        "n_trials": int(len(y_ref)),
        "n_spatial": int(X.shape[1]),
        "n_features": int(X.shape[2]),
        "n_in": int((y_ref == 0).sum()),
        "n_out": int((y_ref == 1).sum()),
        "per_feature": per_feature,
    }
    if spatial_names is not None:
        metadata["spatial_names"] = spatial_names

    logger.info(
        f"Combined {len(features)} features into X of shape {X.shape}"
    )
    return X, y_ref, groups_ref, metadata


# ---------------------------------------------------------------------------
# Class balancing (within-subject, within-class subsampling)
# ---------------------------------------------------------------------------

def balance_within_subject(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42069
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample IN/OUT to equal counts per subject. Works for 2D or 3D X."""
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

def _score_one_spatial(clf, X_slice, y, cv, groups, scoring="roc_auc"):
    """Score one spatial unit. X_slice is (n_trials,) or (n_trials, n_features)."""
    if X_slice.ndim == 1:
        X_slice = X_slice.reshape(-1, 1)
    kw = {"groups": groups} if _is_group_cv(cv) else {}
    scores = cross_val_score(
        clf, X_slice, y, cv=cv, scoring=scoring, n_jobs=1, **kw
    )
    return float(np.mean(scores))


def _permute_y_within_groups(y, groups, rng):
    y_perm = y.copy()
    for g in np.unique(groups):
        m = groups == g
        y_perm[m] = rng.permutation(y[m])
    return y_perm


def _slice_spatial(X, s):
    """Slice the spatial dim. X is (n_trials, n_spatial) or (n_trials, n_spatial, n_features)."""
    return X[:, s] if X.ndim == 2 else X[:, s, :]


def chunk_range(n_spatial: int, n_chunks: int, chunk_idx: int) -> Tuple[int, int]:
    """Return (start, stop) indices for chunk_idx of n_chunks over n_spatial.

    Splits as evenly as possible. The first n_spatial % n_chunks chunks get one
    extra unit so every spatial index lands in exactly one chunk.
    """
    if not (0 <= chunk_idx < n_chunks):
        raise ValueError(f"chunk_idx={chunk_idx} out of range [0, {n_chunks})")
    base, rem = divmod(n_spatial, n_chunks)
    sizes = [base + 1 if i < rem else base for i in range(n_chunks)]
    start = sum(sizes[:chunk_idx])
    stop = start + sizes[chunk_idx]
    return start, stop


def _fit_full_and_get_importances(clf, X, y):
    """Fit on the full dataset and try to extract feature_importances_."""
    try:
        clf.fit(X, y)
    except Exception as exc:
        logger.debug(f"Importance fit failed: {exc}")
        return None
    return getattr(clf, "feature_importances_", None)


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
    fit_importances: bool = False,
) -> Dict:
    """Per-spatial classification with shared-permutation t-max correction.

    All channels share the same permuted labels at each iteration, so the per-
    iteration max across channels is well-defined and t-max p-values control
    FWER across the spatial dimension.

    Args:
        X: (n_trials, n_spatial) or (n_trials, n_spatial, n_features) for combined.
        y: (n_trials,)
        groups: (n_trials,)
        clf_factory: callable returning a fresh sklearn classifier instance.
        cv: sklearn CV splitter.
        n_permutations: number of label permutations.
        n_jobs: parallel jobs over (channels, permutations).
        seed: RNG seed.
        scoring: sklearn scoring name.
        fit_importances: if True, fit one classifier per spatial unit on the full
            dataset to extract feature_importances_ (only meaningful when X has a
            feature dim and the classifier exposes it, e.g. random forest).
    """
    n_trials, n_spatial = X.shape[0], X.shape[1]
    has_feature_dim = X.ndim == 3
    n_features = X.shape[2] if has_feature_dim else 1
    logger.info(
        f"Univariate t-max: n_trials={n_trials}, n_spatial={n_spatial}, "
        f"n_features={n_features}, n_permutations={n_permutations}"
    )

    logger.info("Computing observed scores per spatial unit…")
    observed = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_score_one_spatial)(
                clf_factory(), _slice_spatial(X, s), y, cv, groups, scoring
            )
            for s in range(n_spatial)
        )
    )

    logger.info("Running permutations with shared label shuffles…")
    rng = np.random.default_rng(seed)
    perm_scores = np.zeros((n_permutations, n_spatial), dtype=float)

    for p in tqdm(range(n_permutations), desc="permutations", unit="perm"):
        y_perm = _permute_y_within_groups(y, groups, rng)
        scores_p = Parallel(n_jobs=n_jobs)(
            delayed(_score_one_spatial)(
                clf_factory(), _slice_spatial(X, s), y_perm, cv, groups, scoring
            )
            for s in range(n_spatial)
        )
        perm_scores[p, :] = scores_p

    pvals_unc = (np.sum(perm_scores >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    max_perm = perm_scores.max(axis=1)
    pvals_tmax = (np.sum(max_perm[:, None] >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    pvals_fdr = apply_fdr_correction(pvals_unc, alpha=0.05, method="bh")
    pvals_bonf = apply_bonferroni_correction(pvals_unc, alpha=0.05)

    importances = None
    if fit_importances and has_feature_dim:
        logger.info("Extracting feature importances per spatial unit…")
        imp_list = Parallel(n_jobs=n_jobs)(
            delayed(_fit_full_and_get_importances)(
                clf_factory(), _slice_spatial(X, s), y
            )
            for s in range(n_spatial)
        )
        if all(imp is not None for imp in imp_list):
            importances = np.stack(imp_list, axis=0)  # (n_spatial, n_features)
        else:
            logger.warning(
                "Some classifiers did not expose feature_importances_; skipping."
            )

    n_sig_tmax = int(np.sum(pvals_tmax < 0.05))
    n_sig_fdr = int(np.sum(pvals_fdr < 0.05))
    logger.info(
        f"Significant @ alpha=0.05 — t-max: {n_sig_tmax}/{n_spatial}, "
        f"FDR-BH: {n_sig_fdr}/{n_spatial}, "
        f"max observed: {observed.max():.3f}"
    )

    out = {
        "mode": "univariate",
        "observed": observed,
        "perm_scores": perm_scores,
        "pvals_uncorrected": pvals_unc,
        "pvals_tmax": pvals_tmax,
        "pvals_fdr_bh": pvals_fdr,
        "pvals_bonferroni": pvals_bonf,
    }
    if importances is not None:
        out["feature_importances"] = importances  # (n_spatial, n_features)
    return out


# ---------------------------------------------------------------------------
# Multivariate classification (single classifier on pooled features)
# ---------------------------------------------------------------------------

def run_multivariate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_factory,
    cv,
    n_permutations: int,
    scoring: str = "roc_auc",
    fit_importances: bool = False,
) -> Dict:
    """Multivariate classification on a flat (n_trials, n_features_total) input."""
    if X.ndim > 2:
        # Flatten any extra dims (spatial, features) into one feature axis
        X = X.reshape(X.shape[0], -1)

    clf = clf_factory()
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
    out = {
        "mode": "multivariate",
        "observed": float(score),
        "perm_scores": np.asarray(perm_scores),
        "pvalue": float(pvalue),
    }
    if fit_importances:
        importances = _fit_full_and_get_importances(clf_factory(), X, y)
        if importances is not None:
            out["feature_importances"] = importances  # (n_features_total,)
        else:
            logger.warning(
                "Classifier did not expose feature_importances_; skipping."
            )
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def build_base_name(
    feature_label: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    mode: str,
    combined: bool,
) -> str:
    base = (
        f"feature-{feature_label}_space-{space}"
        f"_inout-{inout_bounds_to_string(inout_bounds)}"
        f"_clf-{clf_name}_cv-{cv_name}_mode-{mode}"
    )
    if combined:
        base += "_combined"
    return base


def save_results(
    output_dir: Path,
    feature_label: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    mode: str,
    combined: bool,
    feature_list: List[str],
    results: Dict,
    metadata: Dict,
    chunk_info: Optional[Dict] = None,
) -> Path:
    """Save NPZ scores + JSON metadata with provenance.

    If chunk_info is provided, this is a partial result for one spatial chunk:
    only `observed`, `perm_scores`, and `feature_importances` are stored; p-
    values are deferred to the aggregation step. The filename gets a
    ``_chunk-{idx}of{N}`` suffix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base = build_base_name(
        feature_label, space, inout_bounds, clf_name, cv_name, mode, combined
    )
    if chunk_info is not None:
        base += f"_chunk-{chunk_info['chunk_idx']}of{chunk_info['n_chunks']}"

    npz_payload: Dict[str, np.ndarray] = {}
    summary: Dict[str, object] = {}

    if chunk_info is not None:
        # Partial output: just observed + perm_scores (and importances if any)
        npz_payload.update(
            observed=results["observed"],
            perm_scores=results["perm_scores"],
            spatial_start=np.array(chunk_info["start"]),
            spatial_stop=np.array(chunk_info["stop"]),
            n_spatial_total=np.array(chunk_info["n_spatial_total"]),
        )
        summary = {
            "chunk_idx": chunk_info["chunk_idx"],
            "n_chunks": chunk_info["n_chunks"],
            "spatial_start": chunk_info["start"],
            "spatial_stop": chunk_info["stop"],
            "n_spatial_total": chunk_info["n_spatial_total"],
            "max_score_in_chunk": float(results["observed"].max()),
            "n_permutations": int(results["perm_scores"].shape[0]),
        }
    elif mode == "univariate":
        npz_payload.update(
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
        npz_payload.update(
            observed=np.asarray(results["observed"]),
            perm_scores=results["perm_scores"],
        )
        summary = {
            "score": float(results["observed"]),
            "pvalue": float(results["pvalue"]),
            "n_permutations": int(len(results["perm_scores"])),
        }

    if "feature_importances" in results:
        npz_payload["feature_importances"] = results["feature_importances"]

    np.savez_compressed(output_dir / f"{base}_scores.npz", **npz_payload)

    meta_out = {
        "feature": feature_label,
        "feature_list": feature_list,
        "combined": combined,
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
    if chunk_info is not None:
        meta_out["chunk"] = {
            "chunk_idx": chunk_info["chunk_idx"],
            "n_chunks": chunk_info["n_chunks"],
            "spatial_start": chunk_info["start"],
            "spatial_stop": chunk_info["stop"],
            "n_spatial_total": chunk_info["n_spatial_total"],
            "seed": chunk_info["seed"],
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
        nargs="+",
        default=None,
        help=(
            "One or more feature names. Examples: fooof_exponent, psd_alpha, "
            "psd_corrected_alpha, complexity_lzc_median. Pass any combination "
            "(space-separated). Use --feature-set for shortcuts."
        ),
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        choices=["psds", "psds_corrected", "fooof", "complexity", "all"],
        help=(
            "Shortcut to run a family of features in one call. Combined with "
            "--feature if both are given."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If a feature fails, log and continue with the next instead of exiting.",
    )
    parser.add_argument(
        "--combine-features",
        action="store_true",
        help=(
            "Combine all selected features into a single classification (stacked "
            "along a new feature axis). For mode=univariate this gives one "
            "multi-feature classifier per spatial unit; for mode=multivariate "
            "this flattens features × space into one big classifier."
        ),
    )
    parser.add_argument(
        "--importances",
        action="store_true",
        help=(
            "After scoring, fit each classifier on the full data and save "
            "feature_importances_ if available (e.g. random forest)."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help=(
            "Filename label when --combine-features is set "
            "(default: 'combined-{N}'). Ignored otherwise."
        ),
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=1,
        help=(
            "Split the spatial dimension into N chunks; this run only processes "
            "the chunk indexed by --chunk-idx. Used for parallelizing one "
            "classification across many SLURM tasks. Univariate mode only."
        ),
    )
    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=0,
        help="Zero-based index of the chunk this run will process (0..n_chunks-1).",
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

    # Resolve feature list from --feature and/or --feature-set
    features: List[str] = []
    if args.feature_set:
        features.extend(expand_feature_set(args.feature_set, config))
    if args.feature:
        features.extend(args.feature)
    # de-dupe while preserving order
    seen = set()
    features = [f for f in features if not (f in seen or seen.add(f))]
    if not features:
        raise SystemExit("Must pass --feature and/or --feature-set")

    logger.info("=" * 78)
    logger.info("CLASSIFICATION (IN vs OUT)")
    logger.info("=" * 78)
    logger.info(f"features ({len(features)}): {features}")
    logger.info(f"space={args.space}  mode={args.mode}  clf={args.clf}  cv={args.cv}")
    logger.info(f"inout={inout_bounds}  n_permutations={args.n_permutations}")
    logger.info("=" * 78)

    cv = get_cv_strategy(args.cv, n_splits=args.n_splits)
    data_root = Path(config["paths"]["data_root"])
    output_dir = (
        data_root / config["paths"]["features"] / f"classification_{args.space}" / "group"
    )

    chunked = args.n_chunks > 1
    if chunked and args.mode != "univariate":
        raise SystemExit("--n-chunks only applies to mode=univariate")

    def _run_one(label: str, feature_list: List[str], X, y, groups, metadata):
        if not args.no_balance:
            X_b, y_b, g_b = balance_within_subject(X, y, groups, seed=args.seed)
            logger.info(
                f"Balanced: {len(y_b)} trials (IN={int((y_b == 0).sum())}, "
                f"OUT={int((y_b == 1).sum())})"
            )
        else:
            X_b, y_b, g_b = X, y, groups

        chunk_info = None
        if chunked:
            n_spatial_total = X_b.shape[1]
            start, stop = chunk_range(n_spatial_total, args.n_chunks, args.chunk_idx)
            logger.info(
                f"Chunk {args.chunk_idx}/{args.n_chunks}: "
                f"spatial[{start}:{stop}] of {n_spatial_total}"
            )
            X_b = X_b[:, start:stop] if X_b.ndim == 2 else X_b[:, start:stop, :]
            chunk_info = {
                "chunk_idx": args.chunk_idx,
                "n_chunks": args.n_chunks,
                "start": int(start),
                "stop": int(stop),
                "n_spatial_total": int(n_spatial_total),
                "seed": int(args.seed),
            }

        if args.mode == "univariate":
            results = run_univariate_with_tmax(
                X=X_b,
                y=y_b,
                groups=g_b,
                clf_factory=lambda: get_classifier(args.clf),
                cv=cv,
                n_permutations=args.n_permutations,
                n_jobs=args.n_jobs,
                seed=args.seed,
                fit_importances=args.importances,
            )
        else:
            results = run_multivariate(
                X=X_b,
                y=y_b,
                groups=g_b,
                clf_factory=lambda: get_classifier(args.clf),
                cv=cv,
                n_permutations=args.n_permutations,
                fit_importances=args.importances,
            )

        save_results(
            output_dir=output_dir,
            feature_label=label,
            space=args.space,
            inout_bounds=inout_bounds,
            clf_name=args.clf,
            cv_name=args.cv,
            mode=args.mode,
            combined=args.combine_features,
            feature_list=feature_list,
            results=results,
            metadata=metadata,
            chunk_info=chunk_info,
        )

    failures: List[Tuple[str, str]] = []

    if args.combine_features:
        if len(features) < 2:
            logger.warning(
                "--combine-features requested but only one feature given; "
                "running as a normal single-feature classification."
            )
        label = args.label or f"combined-{len(features)}"
        logger.info(f"Combined run over {len(features)} feature(s) -> label '{label}'")
        try:
            X, y, groups, metadata = load_combined_features(
                features=features,
                space=args.space,
                inout_bounds=inout_bounds,
                config=config,
            )
            _run_one(label, features, X, y, groups, metadata)
        except Exception as exc:
            logger.error(f"Combined run failed: {exc}", exc_info=True)
            failures.append((label, str(exc)))
            if not args.continue_on_error:
                raise
    else:
        for i, feat in enumerate(features, start=1):
            logger.info("")
            logger.info(f"[{i}/{len(features)}] feature: {feat}")
            logger.info("-" * 78)
            try:
                X, y, groups, metadata = load_classification_data(
                    feature=feat,
                    space=args.space,
                    inout_bounds=inout_bounds,
                    config=config,
                )
                _run_one(feat, [feat], X, y, groups, metadata)
            except Exception as exc:
                logger.error(f"Feature '{feat}' failed: {exc}", exc_info=True)
                failures.append((feat, str(exc)))
                if not args.continue_on_error:
                    raise

    if failures:
        logger.warning(f"{len(failures)}/{len(features)} feature(s) failed:")
        for f, msg in failures:
            logger.warning(f"  - {f}: {msg}")


if __name__ == "__main__":
    main()
