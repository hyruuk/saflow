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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    permutation_test_score,
)
from tqdm import tqdm

# LOGO CV + label permutation routinely produces single-class test folds where
# ROC AUC is undefined; the NaN return is handled downstream. Silence the warning
# globally so it doesn't balloon slurm logs into hundreds of MB.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from code.classification.classifiers import get_classifier
from code.features.inout_selection import (
    DEFAULT_STRATEGY as DEFAULT_INOUT_STRATEGY,
    RunMeta,
    build_run_meta_from_welch,
    compute_inout_zones,
    concat_zones,
    inout_selection_token,
)
from code.features.utils import select_window_mask
from code.utils.bad_trials import compute_run_bad_mask
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


def parse_feature(
    feature: str, n_events_window: int = 1
) -> Tuple[str, str, Optional[str]]:
    """Map a feature name to (folder_prefix, file_suffix, sub_key).

    The file desc suffix is window-aware (``w{N}`` appended when
    ``n_events_window >= 2``) so the classifier loads the right artifacts.

    Returns:
        (folder_prefix, file_pattern_suffix, sub_key)

    Examples:
        fooof_exponent (w=1)    -> ("fooof", "desc-fooof.npz", "exponent")
        fooof_exponent (w=8)    -> ("fooof", "desc-fooofw8.npz", "exponent")
        psd_alpha (w=8)         -> ("welch_psds", "desc-welchw8_psds.npz", "alpha")
        psd_corrected_alpha (w=8) -> ("welch_psds_corrected",
                                      "desc-welch-corrw8_psds.npz", "alpha")
        complexity_lzc_median (w=8) -> ("complexity", "desc-complexityw8.npz",
                                        "lzc_median")
    """
    w = "" if n_events_window <= 1 else f"w{n_events_window}"
    if feature.startswith("fooof_"):
        return "fooof", f"desc-fooof{w}.npz", feature[len("fooof_"):]
    if feature.startswith("psd_corrected_"):
        corr_desc = (
            "welch-corrected" if n_events_window <= 1 else f"welch-corrw{n_events_window}"
        )
        return (
            "welch_psds_corrected",
            f"desc-{corr_desc}_psds.npz",
            feature[len("psd_corrected_"):],
        )
    if feature.startswith("psd_"):
        welch_desc = "welch" if n_events_window <= 1 else f"welchw{n_events_window}"
        return "welch_psds", f"desc-{welch_desc}_psds.npz", feature[len("psd_"):]
    if feature.startswith("complexity_"):
        return "complexity", f"desc-complexity{w}.npz", feature[len("complexity_"):]
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
    drop_bad_trials: bool = True,
    trial_type: str = "alltrials",
    zoning: str = "per-subject",
    n_events_window: int = 1,
    inout_selection: str = DEFAULT_INOUT_STRATEGY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load per-epoch feature data, split into IN/OUT using VTC quartiles.

    IN/OUT thresholds are computed from VTC over **all** epochs (including
    those flagged ``bad_ar2``). Epochs flagged ``bad_ar2`` by the autoreject
    second pass are dropped after masking when ``drop_bad_trials=True``.

    Args:
        feature: feature name (see parse_feature).
        space: "sensor", "source", or atlas name (e.g. "schaefer_400").
        inout_bounds: (low_pct, high_pct) for IN/OUT VTC zones.
        config: loaded config.yaml dictionary.
        subjects: list of subject IDs; defaults to config["bids"]["subjects"].
        drop_bad_trials: drop epochs flagged ``bad_ar2`` (default True).
        trial_type: filter mode applied per epoch (mirrors cc_saflow's
            ``select_epoch``). One of ``alltrials`` (default, matches
            cc_saflow's main analysis), ``correct``, ``rare``, ``lapse``,
            or ``correct_commission`` (saflow's previous single-trial default).
        zoning: how IN/OUT VTC percentiles are computed. ``per-subject``
            pools all of a subject's epochs (saflow's previous behaviour),
            ``per-run`` computes bounds within each run (matches cc_saflow).

    Returns:
        X        : (n_trials, n_spatial)
        y        : (n_trials,) labels — 0 = IN, 1 = OUT
        groups   : (n_trials,) subject indices
        metadata : dict with shape, retention, and provenance info.
    """
    folder_prefix, file_suffix, sub_key = parse_feature(
        feature, n_events_window=n_events_window
    )

    analysis_cfg = config.get("analysis", {})
    bad_trial_rule = str(analysis_cfg.get("bad_trial_rule", "ar2"))
    interp_reject_threshold = int(analysis_cfg.get("interp_reject_threshold", 0) or 0)

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
    n_bad_excluded = 0
    per_subject: Dict[str, Dict[str, int]] = {}
    bad_metadata_present = False

    for subj_idx, subject in enumerate(tqdm(subjects, desc="Loading", unit="subj")):
        subj_dir = feature_root / f"sub-{subject}"
        if not subj_dir.exists():
            continue

        subj_data = []
        subj_task = []
        subj_inc_task: List[List[np.ndarray]] = []  # length-N per-epoch task arrays
        subj_bad: List[np.ndarray] = []
        subj_run_idx: List[np.ndarray] = []  # per-epoch run index (for per-run zoning)
        run_metas: List[RunMeta] = []

        for run_pos, run in enumerate(runs):
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
            rm = build_run_meta_from_welch(meta)
            run_metas.append(rm)
            subj_task.append(np.asarray(meta["task"]))

            # included_task: length-N arrays per epoch (windowed mode) or
            # singletons (single-trial fallback)
            if "included_task" in meta:
                subj_inc_task.append([np.asarray(t) for t in meta["included_task"]])
            else:
                subj_inc_task.append([np.array([t]) for t in meta["task"]])

            run_n = rm.n_windows
            subj_bad.append(
                compute_run_bad_mask(meta, run_n, bad_trial_rule, interp_reject_threshold)
            )
            subj_run_idx.append(np.full(run_n, run_pos, dtype=int))

        if not subj_data:
            continue

        subj_data = np.concatenate(subj_data, axis=0)
        subj_task = np.concatenate(subj_task)
        subj_inc_task_flat = [arr for run_list in subj_inc_task for arr in run_list]
        subj_bad_arr = np.concatenate(subj_bad) if subj_bad else np.array([], dtype=bool)
        subj_run_idx_arr = np.concatenate(subj_run_idx) if subj_run_idx else np.array([], dtype=int)
        if subj_bad_arr.any():
            bad_metadata_present = True

        # Window-level IN/OUT zones from the configured selection strategy.
        # ``per-run`` matches cc_saflow (bounds within each run); ``per-subject``
        # pools all runs for this subject (saflow default).
        per_run_zones = compute_inout_zones(
            run_metas,
            strategy=inout_selection,
            inout_bounds=inout_bounds,
            zoning=zoning,
        )
        in_zone, out_zone = concat_zones(per_run_zones)

        # Trial-type filter (mirrors cc_saflow's select_epoch)
        task_mask = select_window_mask(
            included_task_per_epoch=subj_inc_task_flat,
            task_per_epoch=subj_task,
            type_how=trial_type,
        )
        in_mask_full = task_mask & in_zone
        out_mask_full = task_mask & out_zone
        mid_mask_full = task_mask & ~in_mask_full & ~out_mask_full

        n_bad_in = int((in_mask_full & subj_bad_arr).sum()) if drop_bad_trials else 0
        n_bad_out = int((out_mask_full & subj_bad_arr).sum()) if drop_bad_trials else 0

        if drop_bad_trials:
            in_mask = in_mask_full & ~subj_bad_arr
            out_mask = out_mask_full & ~subj_bad_arr
        else:
            in_mask = in_mask_full
            out_mask = out_mask_full

        n_in = int(in_mask.sum())
        n_out = int(out_mask.sum())
        per_subject[subject] = {
            "n_total": int(in_zone.size),
            "n_in": n_in,
            "n_out": n_out,
            "n_mid": int(mid_mask_full.sum()),
            "n_bad_in": n_bad_in,
            "n_bad_out": n_bad_out,
            "n_in_before_bad_filter": int(in_mask_full.sum()),
            "n_out_before_bad_filter": int(out_mask_full.sum()),
        }
        if n_in == 0 or n_out == 0:
            logger.warning(
                f"sub-{subject}: dropped — n_in={n_in}, n_out={n_out} "
                f"(bad_in={n_bad_in}, bad_out={n_bad_out})"
            )
            continue

        all_X.append(subj_data[in_mask])
        all_X.append(subj_data[out_mask])
        all_y.extend([0] * n_in + [1] * n_out)
        all_groups.extend([subj_idx] * (n_in + n_out))
        n_in_total += n_in
        n_out_total += n_out
        n_bad_excluded += n_bad_in + n_bad_out

    if not all_X:
        raise ValueError("No data loaded for any subject")

    if drop_bad_trials and not bad_metadata_present:
        logger.warning(
            "drop_bad_trials=True but no bad_ar2 column was found in any "
            "trial_metadata — feature files predate the bad-trial flag. "
            "Run `python -m code.utils.backfill_bad_trials` to backfill, "
            "or recompute features."
        )

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y)
    groups = np.array(all_groups)

    metadata = {
        "feature": feature,
        "space": space,
        "inout_bounds": list(inout_bounds),
        "trial_type": trial_type,
        "zoning": zoning,
        "n_subjects": int(len(np.unique(groups))),
        "n_trials": int(len(y)),
        "n_spatial": int(X.shape[1]),
        "n_in": int(n_in_total),
        "n_out": int(n_out_total),
        "n_bad_excluded": int(n_bad_excluded),
        "drop_bad_trials": bool(drop_bad_trials),
        "bad_trial_rule": bad_trial_rule,
        "interp_reject_threshold": interp_reject_threshold,
        "bad_ar2_metadata_present": bool(bad_metadata_present),
        "per_subject": per_subject,
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
    drop_bad_trials: bool = True,
    trial_type: str = "alltrials",
    zoning: str = "per-subject",
    n_events_window: int = 1,
    inout_selection: str = DEFAULT_INOUT_STRATEGY,
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
            drop_bad_trials=drop_bad_trials,
            trial_type=trial_type,
            zoning=zoning,
            n_events_window=n_events_window,
            inout_selection=inout_selection,
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
        "trial_type": trial_type,
        "zoning": zoning,
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
# Trial averaging (collapse each subject to one mean vector per class)
# ---------------------------------------------------------------------------

def standardize_within_subject(
    X: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Z-score features within each subject across the trial axis.

    Removes between-subject absolute-scale differences (e.g. raw log-power
    that varies by orders of magnitude between sensor calibrations / head
    sizes but only fractions of a log within a subject). Without this, a
    leave-one-subject-out classifier on an absolute-scale feature predicts
    a single class for every held-out trial — balanced accuracy then pins
    at exactly 0.5 regardless of any within-subject IN-vs-OUT effect.

    Works for 2D (n_trials, n_spatial) and 3D (n_trials, n_spatial, n_features)
    X. Spatial units with zero within-subject std are left unchanged (sd→1).
    NaN-aware via np.nanmean / np.nanstd.
    """
    X = X.astype(float, copy=True)
    for g in np.unique(groups):
        idx = groups == g
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(X[idx], axis=0, keepdims=True)
            sd = np.nanstd(X[idx], axis=0, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        X[idx] = (X[idx] - mu) / sd
    return X


def average_within_subject(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse each subject's trials to one mean feature vector per class.

    Each subject contributes exactly two rows (mean of its IN trials, mean of
    its OUT trials). ``groups`` stays the subject index so GroupKFold/LOSO keep
    a subject's IN and OUT samples together in the same fold. Works for 2D or
    3D X. NaN-aware: a NaN in a spatial unit for some trials does not poison
    that subject's average (handled by np.nanmean; downstream code drops any
    spatial unit that ends up all-NaN).
    """
    Xa: List[np.ndarray] = []
    ya: List[int] = []
    ga: List[int] = []
    for g in np.unique(groups):
        for c in np.unique(y):
            sel = (groups == g) & (y == c)
            if not sel.any():
                continue
            with warnings.catch_warnings():
                # All-NaN slices for a dropped spatial unit are expected.
                warnings.simplefilter("ignore", RuntimeWarning)
                Xa.append(np.nanmean(X[sel], axis=0))
            ya.append(int(c))
            ga.append(int(g))
    return np.stack(Xa), np.array(ya), np.array(ga)


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
    """Score one spatial unit. X_slice is (n_trials,) or (n_trials, n_features).

    Trials with a NaN in this spatial unit (e.g. failed FOOOF fits in fooof /
    psd_corrected features) are dropped before CV — sklearn estimators reject
    NaN. The NaN positions depend only on X, not on labels, so the same rows are
    dropped for observed and permuted runs, keeping the permutation test valid.
    """
    if X_slice.ndim == 1:
        X_slice = X_slice.reshape(-1, 1)
    finite = ~np.isnan(X_slice).any(axis=1)
    if not finite.all():
        X_slice, y, groups = X_slice[finite], y[finite], groups[finite]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    if _is_group_cv(cv) and len(np.unique(groups)) < 2:
        return float("nan")
    kw = {"groups": groups} if _is_group_cv(cv) else {}
    scores = cross_val_score(
        clf, X_slice, y, cv=cv, scoring=scoring, n_jobs=1, **kw
    )
    return float(np.mean(scores))


# Auxiliary metrics computed alongside the primary `scoring` metric. The
# primary metric is what permutation tests / p-values are based on; the
# auxiliaries are observed-only — no permutation null — and exist so the
# panel can show e.g. balanced_accuracy + AUC side by side post-hoc.
_AUX_METRICS = ("roc_auc", "balanced_accuracy", "accuracy")


def _compute_metrics_one_spatial(clf_factory, X_slice, y, cv, groups
                                 ) -> Dict[str, object]:
    """Return per-CV-averaged metrics + pooled confusion matrix for one slice.

    Output keys: ``roc_auc``, ``balanced_accuracy``, ``accuracy``,
    ``confusion_matrix`` (shape (2, 2), pooled across CV folds via
    ``cross_val_predict``).

    Degenerate slices (NaN features, single-class subset, single-group CV
    folds) return NaN metrics + a zero CM, mirroring ``_score_one_spatial``.
    """
    nan_out: Dict[str, object] = {m: float("nan") for m in _AUX_METRICS}
    nan_out["confusion_matrix"] = np.zeros((2, 2), dtype=int)

    if X_slice.ndim == 1:
        X_slice = X_slice.reshape(-1, 1)
    finite = ~np.isnan(X_slice).any(axis=1)
    if not finite.all():
        X_slice, y, groups = X_slice[finite], y[finite], groups[finite]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return nan_out
    if _is_group_cv(cv) and len(np.unique(groups)) < 2:
        return nan_out

    kw = {"groups": groups} if _is_group_cv(cv) else {}
    try:
        cv_res = cross_validate(
            clf_factory(), X_slice, y, cv=cv,
            scoring=list(_AUX_METRICS), n_jobs=1, **kw,
        )
        out: Dict[str, object] = {
            m: float(np.mean(cv_res[f"test_{m}"])) for m in _AUX_METRICS
        }
        # Pool predictions across folds for a single representative CM.
        # (Per-fold CMs aren't useful — most folds are <50 trials.)
        y_pred = cross_val_predict(
            clf_factory(), X_slice, y, cv=cv, n_jobs=1, **kw,
        )
        labels = np.unique(y)
        cm = _sk_confusion_matrix(y, y_pred, labels=labels)
        if cm.shape != (2, 2):
            full = np.zeros((2, 2), dtype=int)
            full[: cm.shape[0], : cm.shape[1]] = cm
            cm = full
        out["confusion_matrix"] = cm.astype(int)
        return out
    except Exception as exc:
        logger.debug(f"  multi-metric fail on slice: {exc}")
        return nan_out


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

    # Auxiliary metrics + confusion matrices — observed only (no permutation
    # null). Cheap relative to the n_permutations × n_spatial permutation
    # phase below (factor of ~2 vs the observed pass).
    logger.info("Computing auxiliary metrics + confusion matrices…")
    aux = Parallel(n_jobs=n_jobs)(
        delayed(_compute_metrics_one_spatial)(
            clf_factory, _slice_spatial(X, s), y, cv, groups
        )
        for s in range(n_spatial)
    )
    metrics_per_spatial: Dict[str, np.ndarray] = {
        f"metrics_{m}": np.asarray([d[m] for d in aux], dtype=float)
        for m in _AUX_METRICS
    }
    confusion_matrices = np.stack(
        [d["confusion_matrix"] for d in aux], axis=0,
    ).astype(int)  # (n_spatial, 2, 2)

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
    max_perm = np.nanmax(perm_scores, axis=1)
    pvals_tmax = (np.sum(max_perm[:, None] >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    # Channels with no finite observed score (all trials NaN) carry no signal —
    # NaN >= NaN is False, which would otherwise yield a spuriously small pval.
    nan_obs = np.isnan(observed)
    if nan_obs.any():
        pvals_unc[nan_obs] = 1.0
        pvals_tmax[nan_obs] = 1.0
        logger.warning(
            f"{int(nan_obs.sum())}/{n_spatial} spatial units had no finite "
            f"score (all-NaN feature); their p-values set to 1.0"
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
        f"max observed: {np.nanmax(observed):.3f}"
    )

    out = {
        "mode": "univariate",
        "observed": observed,
        "perm_scores": perm_scores,
        "pvals_uncorrected": pvals_unc,
        "pvals_tmax": pvals_tmax,
        "pvals_fdr_bh": pvals_fdr,
        "pvals_bonferroni": pvals_bonf,
        "confusion_matrices": confusion_matrices,
        "scoring": scoring,
    }
    out.update(metrics_per_spatial)  # metrics_roc_auc / balanced_accuracy / accuracy
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

    # Auxiliary metrics + confusion matrix on the same data.
    aux = _compute_metrics_one_spatial(clf_factory, X, y, cv, groups)
    out = {
        "mode": "multivariate",
        "observed": float(score),
        "perm_scores": np.asarray(perm_scores),
        "pvalue": float(pvalue),
        "scoring": scoring,
        "confusion_matrices": aux["confusion_matrix"][None, ...],  # (1, 2, 2) for uniformity
    }
    for m in _AUX_METRICS:
        out[f"metrics_{m}"] = float(aux[m])
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
    trial_type: str = "alltrials",
    analysis_level: Optional[str] = None,
    inout_selection: str = DEFAULT_INOUT_STRATEGY,
) -> str:
    base = (
        f"feature-{feature_label}_space-{space}"
        f"_inout-{inout_bounds_to_string(inout_bounds)}{inout_selection_token(inout_selection)}"
        f"_clf-{clf_name}_cv-{cv_name}_mode-{mode}"
    )
    if analysis_level is not None:
        base += f"_level-{analysis_level}"
    base += f"_type-{trial_type}"
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
    trial_type: str = "alltrials",
    analysis_level: Optional[str] = None,
    inout_selection: str = DEFAULT_INOUT_STRATEGY,
) -> Path:
    """Save NPZ scores + JSON metadata with provenance.

    If chunk_info is provided, this is a partial result for one spatial chunk:
    only `observed`, `perm_scores`, and `feature_importances` are stored; p-
    values are deferred to the aggregation step. The filename gets a
    ``_chunk-{idx}of{N}`` suffix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base = build_base_name(
        feature_label, space, inout_bounds, clf_name, cv_name, mode, combined,
        trial_type=trial_type, analysis_level=analysis_level,
        inout_selection=inout_selection,
    )
    if chunk_info is not None:
        base += f"_chunk-{chunk_info['chunk_idx']}of{chunk_info['n_chunks']}"

    npz_payload: Dict[str, np.ndarray] = {}
    summary: Dict[str, object] = {}

    # Auxiliary metrics + confusion matrices live alongside the primary
    # `observed` array in every output (univariate, multivariate, and
    # chunked). Pulled out once here so each branch can include them.
    aux_payload: Dict[str, np.ndarray] = {
        k: np.asarray(v) for k, v in results.items()
        if k.startswith("metrics_") or k == "confusion_matrices"
    }

    if chunk_info is not None:
        # Partial output: observed + perm_scores + the aux metrics/CMs so
        # the aggregator can concatenate without re-running. P-values are
        # still deferred to the merge step.
        npz_payload.update(
            observed=results["observed"],
            perm_scores=results["perm_scores"],
            spatial_start=np.array(chunk_info["start"]),
            spatial_stop=np.array(chunk_info["stop"]),
            n_spatial_total=np.array(chunk_info["n_spatial_total"]),
        )
        npz_payload.update(aux_payload)
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
        npz_payload.update(aux_payload)
        summary = {
            "max_score": float(results["observed"].max()),
            "mean_score": float(results["observed"].mean()),
            "n_significant_tmax_a05": int((results["pvals_tmax"] < 0.05).sum()),
            "n_significant_fdr_bh_a05": int((results["pvals_fdr_bh"] < 0.05).sum()),
            "n_permutations": int(results["perm_scores"].shape[0]),
        }
        for m in _AUX_METRICS:
            key = f"metrics_{m}"
            if key in results:
                arr = np.asarray(results[key], dtype=float)
                summary[f"mean_{m}"] = float(np.nanmean(arr))
                summary[f"max_{m}"] = float(np.nanmax(arr))
    else:
        npz_payload.update(
            observed=np.asarray(results["observed"]),
            perm_scores=results["perm_scores"],
        )
        npz_payload.update(aux_payload)
        summary = {
            "score": float(results["observed"]),
            "pvalue": float(results["pvalue"]),
            "n_permutations": int(len(results["perm_scores"])),
        }
        for m in _AUX_METRICS:
            key = f"metrics_{m}"
            if key in results:
                summary[m] = float(results[key])

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
        "scoring": results.get("scoring"),
        "metrics_computed": list(_AUX_METRICS),
        "has_confusion_matrices": "confusion_matrices" in results,
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
# Analysis-mode dispatch
# ---------------------------------------------------------------------------

ANALYSIS_LEVELS = ("epoch", "average")


def _resolve_levels(analysis_level: str) -> List[str]:
    """Expand --analysis-level into a list of levels to run.

    'both' (default) → ['epoch', 'average']. Single values pass through.
    """
    if analysis_level == "both":
        return list(ANALYSIS_LEVELS)
    if analysis_level in ANALYSIS_LEVELS:
        return [analysis_level]
    raise SystemExit(
        f"--analysis-level={analysis_level!r} invalid. "
        f"Choose 'both' or one of {ANALYSIS_LEVELS}."
    )


def _loader_for_level(level: str, feature: str) -> str:
    """Map (analysis level, feature) to the per-feature loader path.

    - epoch: per-epoch features as upstream produced them (sliding-window
      epochs in our pipeline). Trial loader for all features.
    - average:
        psd_/fooof_   → subject-spectrum (pool each (subj, cond)'s good-
                        epoch PSDs into one mean and fit FOOOF once → 1 IN
                        + 1 OUT row per subject).
        complexity_   → subject-mean (per-epoch features averaged per
                        (subj, class) → 1 IN + 1 OUT row per subject —
                        complexity has no aperiodic fit to do at subject
                        level).
    """
    if level == "epoch":
        return "trial"
    is_complexity = feature.startswith("complexity")
    return "subject-mean" if is_complexity else "subject-spectrum"


def _cv_for_level(level: str) -> str:
    """Default CV strategy for an analysis level.

    'average' produces 1 IN + 1 OUT per subject → GroupKFold(k=6) over
    subjects. 'epoch' keeps held-out subjects' trials out of training with
    leave-one-subject-out.
    """
    return "logo" if level == "epoch" else "group"


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
        "--clf", default="logistic", choices=["lda", "svm", "rf", "logistic"],
        help="Classifier. Default 'logistic' matches cc_saflow's run_classifs.py."
    )
    parser.add_argument(
        "--cv", default="auto", choices=["auto", "logo", "stratified", "group"],
        help=(
            "Cross-validation strategy. 'auto' (default) is resolved from "
            "--analysis-level: GroupKFold(k=6) over subjects for "
            "level=average, leave-one-subject-out for level=epoch. 'logo', "
            "'stratified', 'group' force a specific splitter."
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5,
                        help="n_splits for an explicit --cv group/stratified. "
                             "Ignored when --cv=auto (auto uses k=6).")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--scoring", default=None,
        help=(
            "sklearn scoring metric for permutation_test_score / cross_val. "
            "Default: config classification.scoring (roc_auc), matching the "
            "manuscript Fig.3 classifications."
        ),
    )
    parser.add_argument(
        "--analysis-level", default="both",
        choices=["both", *ANALYSIS_LEVELS],
        help=(
            "Aggregation level. 'epoch': per-epoch features (each 'epoch' "
            "is the sliding window produced by the feature pipeline, e.g. "
            "8 trials wide for n_events_window=8), many rows per subject; "
            "default cv = leave-one-subject-out. 'average': 1 IN + 1 OUT "
            "row per subject — for psd_*/fooof_* via subject-spectrum (pool "
            "each (subj, cond)'s good-epoch PSDs and fit FOOOF once), for "
            "complexity_* via the mean of per-epoch features; default cv = "
            "GroupKFold(k=6). 'both' (default) runs both levels in turn, "
            "saving two output files per feature (one per level)."
        ),
    )
    parser.add_argument(
        "--keep-bad-trials",
        action="store_true",
        default=False,
        help="Skip the bad_ar2 filter (keeps trials inside autoreject-rejected "
             "BAD_AR2 windows). Default is to drop them.",
    )
    parser.add_argument(
        "--trial-type",
        default="alltrials",
        choices=["alltrials", "correct", "rare", "lapse", "correct_commission"],
        help=(
            "Trial-type filter applied per epoch (mirrors cc_saflow's "
            "select_epoch). 'alltrials' (default) = no filter, matches "
            "cc_saflow's main analysis. 'correct' = window contains only "
            "CC+CO. 'rare' = window contains a rare-stim outcome. 'lapse' = "
            "window contains a commission error. 'correct_commission' = "
            "saflow's previous default (all trials must be CC)."
        ),
    )
    parser.add_argument(
        "--zoning",
        default="per-run",
        choices=["per-run", "per-subject"],
        help=(
            "How IN/OUT VTC percentile bounds are computed. 'per-run' "
            "(default, matches cc_saflow) computes within each run; "
            "'per-subject' pools all runs (saflow's previous behaviour)."
        ),
    )
    parser.add_argument(
        "--standardize",
        default="auto",
        choices=["auto", "none", "per-subject"],
        help=(
            "Per-subject feature standardization applied before averaging / "
            "balancing / classification. 'per-subject' z-scores each "
            "subject's epochs independently, so a held-out subject's "
            "feature distribution overlaps the training set's. Without "
            "this, absolute-scale features (raw log-power) drive LOSO "
            "balanced-accuracy to exactly 0.5 because every held-out epoch "
            "lands on the same side of the learned decision boundary. "
            "'auto' (default) = per-subject for level=epoch, none for "
            "level=average (already normalized by the per-subject FOOOF "
            "aperiodic fit / per-subject mean)."
        ),
    )
    parser.add_argument(
        "--n-events-window",
        type=int,
        default=8,
        help=(
            "Window size used by upstream feature extraction. Determines "
            "which desc-suffixed feature files to load. Default 8 matches "
            "cc_saflow's sliding window."
        ),
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    inout_bounds = tuple(config["analysis"]["inout_bounds"])
    inout_selection = str(
        config.get("analysis", {}).get("inout_selection", DEFAULT_INOUT_STRATEGY)
    )
    scoring = args.scoring or config.get("classification", {}).get(
        "scoring", "roc_auc"
    )

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

    levels = _resolve_levels(args.analysis_level)

    logger.info("=" * 78)
    logger.info("CLASSIFICATION (IN vs OUT)")
    logger.info("=" * 78)
    logger.info(f"features ({len(features)}): {features}")
    logger.info(f"space={args.space}  mode={args.mode}  clf={args.clf}")
    logger.info(
        f"scoring={scoring}  analysis_level={args.analysis_level} "
        f"({'+'.join(levels)})  cv={args.cv}"
    )
    logger.info(f"inout={inout_bounds}  n_permutations={args.n_permutations}")
    logger.info("=" * 78)

    data_root = Path(config["paths"]["data_root"])
    output_dir = (
        data_root / config["paths"]["results"] / f"classification_{args.space}" / "group"
    )

    chunked = args.n_chunks > 1
    if chunked and args.mode != "univariate":
        raise SystemExit("--n-chunks only applies to mode=univariate")

    def _resolve_cv(level: str) -> Tuple[str, int]:
        if args.cv == "auto":
            cv_name = _cv_for_level(level)
            n_splits = 6 if cv_name == "group" else args.n_splits
        else:
            cv_name = args.cv
            n_splits = args.n_splits
        return cv_name, n_splits

    def _run_one(label, feature_list, X, y, groups, metadata, level, loader):
        # Resolve standardization. 'auto' = per-subject for level=epoch
        # (between-subject absolute scale dominates), none for level=average
        # (already normalized by per-subject FOOOF / per-subject mean).
        standardize = (
            args.standardize if args.standardize != "auto"
            else ("per-subject" if level == "epoch" else "none")
        )
        cv_name, n_splits = _resolve_cv(level)
        cv = get_cv_strategy(cv_name, n_splits=n_splits)
        logger.info(
            f"level={level} ({loader})  cv={cv_name}"
            + (f"(k={n_splits})" if cv_name in ("group", "stratified") else "")
            + f"  standardize={standardize}"
        )

        metadata["analysis_level"] = level
        metadata["loader"] = loader
        metadata["cv_strategy"] = cv_name

        if loader == "subject-spectrum":
            # Loader emits X as (1, 2N, n_spatial) — exactly 1 IN + 1 OUT
            # row per subject, 1:1 balanced. No averaging/balancing needed;
            # drop the leading feature axis.
            if X.ndim == 3 and X.shape[0] == 1:
                X = X[0]
            metadata["n_trials"] = int(len(y))
            metadata["n_in"] = int((y == 0).sum())
            metadata["n_out"] = int((y == 1).sum())
            if standardize == "per-subject":
                X = standardize_within_subject(X, groups)
                logger.info("Standardized features per subject (z-score)")
            X_b, y_b, g_b = X, y, groups
            logger.info(
                f"subject-spectrum: {len(y)} rows "
                f"({metadata.get('n_subjects', '?')} subjects × IN/OUT)"
            )
        else:
            # Trial loader: X is per-epoch (n_trials, n_spatial[, n_features]).
            if standardize == "per-subject":
                X = standardize_within_subject(X, groups)
                logger.info(
                    f"Standardized features per subject (z-score) — "
                    f"{X.shape[0]} epochs × {X.shape[1]} spatial units"
                )
            if loader == "subject-mean":
                X, y, groups = average_within_subject(X, y, groups)
                metadata["n_trials"] = int(len(y))
                metadata["n_in"] = int((y == 0).sum())
                metadata["n_out"] = int((y == 1).sum())
                logger.info(
                    f"Averaged within subject×class: {len(y)} samples "
                    f"({metadata['n_subjects']} subjects × IN/OUT)"
                )

            if not args.no_balance:
                X_b, y_b, g_b = balance_within_subject(
                    X, y, groups, seed=args.seed
                )
                logger.info(
                    f"Balanced: {len(y_b)} epochs (IN={int((y_b == 0).sum())}, "
                    f"OUT={int((y_b == 1).sum())})"
                )
            else:
                X_b, y_b, g_b = X, y, groups

        metadata["standardize"] = standardize

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
                scoring=scoring,
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
                scoring=scoring,
                fit_importances=args.importances,
            )

        save_results(
            output_dir=output_dir,
            feature_label=label,
            space=args.space,
            inout_bounds=inout_bounds,
            clf_name=args.clf,
            cv_name=cv_name,
            mode=args.mode,
            combined=args.combine_features,
            feature_list=feature_list,
            results=results,
            metadata=metadata,
            chunk_info=chunk_info,
            trial_type=args.trial_type,
            analysis_level=level,
            inout_selection=inout_selection,
        )

    failures: List[Tuple[str, str]] = []

    if args.combine_features:
        if "average" in levels and any(
            not f.startswith("complexity") for f in features
        ):
            logger.warning(
                "--combine-features uses the trial-level loader; PSD/FOOOF "
                "features in level=average will fall back to the "
                "'subject-mean' loader (per-epoch features averaged within "
                "subject) instead of the subject-spectrum (Fig.3) path."
            )
        if len(features) < 2:
            logger.warning(
                "--combine-features requested but only one feature given; "
                "running as a normal single-feature classification."
            )
        label = args.label or f"combined-{len(features)}"
        logger.info(f"Combined run over {len(features)} feature(s) -> label '{label}'")
        try:
            X_combined, y_combined, g_combined, metadata_combined = (
                load_combined_features(
                    features=features,
                    space=args.space,
                    inout_bounds=inout_bounds,
                    config=config,
                    drop_bad_trials=not args.keep_bad_trials,
                    trial_type=args.trial_type,
                    zoning=args.zoning,
                    n_events_window=args.n_events_window,
                    inout_selection=inout_selection,
                )
            )
            for level in levels:
                # Combined runs share the trial loader; in 'average' mode it
                # acts as the subject-mean path (per-epoch features → mean
                # per subj×class). Standardize/CV resolve per-level inside.
                loader = "trial" if level == "epoch" else "subject-mean"
                logger.info("")
                logger.info(f"Combined run, level={level}")
                logger.info("-" * 78)
                _run_one(
                    label, features,
                    np.asarray(X_combined), y_combined.copy(), g_combined.copy(),
                    dict(metadata_combined),
                    level, loader,
                )
        except Exception as exc:
            logger.error(f"Combined run failed: {exc}", exc_info=True)
            failures.append((label, str(exc)))
            if not args.continue_on_error:
                raise
    else:
        # Outer loop = level so the subject-spectrum loader is shared across
        # features within a level. Inner loop = feature.
        for level in levels:
            logger.info("")
            logger.info("=" * 78)
            logger.info(f"Analysis level: {level}")
            logger.info("=" * 78)
            for i, feat in enumerate(features, start=1):
                loader = _loader_for_level(level, feat)
                logger.info("")
                logger.info(
                    f"[{i}/{len(features)}] feature: {feat}  "
                    f"(level={level}, loader={loader})"
                )
                logger.info("-" * 78)
                try:
                    if loader == "subject-spectrum":
                        # Manuscript Fig.3 path: pool each (subj, cond)'s
                        # good-epoch PSDs into one mean PSD and fit FOOOF
                        # once per condition (2 fits/subject). Classifier
                        # sees one IN + one OUT row per subject (same
                        # data the paired t-test uses).
                        from code.statistics.subject_spectrum import (
                            load_subject_spectrum_features,
                        )
                        ss = load_subject_spectrum_features(
                            feature_types=[feat],
                            space=args.space,
                            inout_bounds=inout_bounds,
                            config=config,
                            drop_bad_trials=not args.keep_bad_trials,
                            n_jobs=args.n_jobs,
                            trial_type=args.trial_type,
                            zoning=args.zoning,
                            n_events_window=args.n_events_window,
                            inout_selection=inout_selection,
                        )
                        X, y, groups, metadata = ss[feat]
                    else:
                        X, y, groups, metadata = load_classification_data(
                            feature=feat,
                            space=args.space,
                            inout_bounds=inout_bounds,
                            config=config,
                            drop_bad_trials=not args.keep_bad_trials,
                            trial_type=args.trial_type,
                            zoning=args.zoning,
                            n_events_window=args.n_events_window,
                            inout_selection=inout_selection,
                        )
                    _run_one(feat, [feat], X, y, groups, metadata, level, loader)
                except Exception as exc:
                    logger.error(
                        f"Feature '{feat}' (level={level}) failed: {exc}",
                        exc_info=True,
                    )
                    failures.append((f"{feat}@{level}", str(exc)))
                    if not args.continue_on_error:
                        raise

    if failures:
        logger.warning(f"{len(failures)} run(s) failed:")
        for f, msg in failures:
            logger.warning(f"  - {f}: {msg}")


if __name__ == "__main__":
    main()
