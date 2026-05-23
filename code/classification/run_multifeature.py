"""Multi-feature classification with axis-aware decoding and feature importance.

Given a stack of features ``X: (n_trials, n_spatial, n_features)``, this script
runs IN-vs-OUT classification along one of four axes:

- ``per-spatial``  : loop spatial units; classifier sees (n_trials, n_features).
                    Output: scores(n_spatial,), importance(n_spatial, n_features).
- ``per-feature``  : loop features; classifier sees (n_trials, n_spatial).
                    Output: scores(n_features,), importance(n_features, n_spatial).
- ``per-cell``     : loop (s, f) pairs; classifier sees (n_trials, 1).
                    Output: scores(n_spatial, n_features).
- ``joint``        : flatten to (n_trials, n_spatial * n_features); single fit.
                    Output: scalar score, importance(n_spatial, n_features).

The looped-axis classifiers share permuted-label sequences, so per-permutation
max-across-axis gives t-max-corrected p-values for the looped axis.

Three feature-importance backends are supported via ``--importance``:
- ``permutation`` (default): model-agnostic, computed on each CV test fold and
  averaged across folds.
- ``coef``: signed coefficients from linear models (LDA, logistic, linear SVM).
- ``tree``: ``feature_importances_`` from tree ensembles (RF).

Scaling pipeline (in order):
1. Per-subject z-score across trials (outside CV; within-subject only, no leak).
2. (Optional) Per-feature global ``StandardScaler`` inside an sklearn
   ``Pipeline`` so the scaler is fit on the training fold only. Default ON;
   matters most for ``joint`` and ``per-spatial`` where features with wildly
   different magnitudes are stacked.

Usage:
    python -m code.classification.run_multifeature \
        --feature-set all --space sensor --axis all \
        --clf logistic --cv logo --importance permutation \
        --n-permutations 1000
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from code.classification.classifiers import get_classifier
from code.classification.run_classification import (
    _is_group_cv,
    _permute_y_within_groups,
    balance_within_subject,
    chunk_range,
    expand_feature_set,
    get_cv_strategy,
    get_git_hash,
    inout_bounds_to_string,
    load_classification_data,
    load_combined_features,
    load_config,
    standardize_within_subject,
)
from code.statistics.corrections import (
    apply_bonferroni_correction,
    apply_fdr_correction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


AXES = ("per-spatial", "per-feature", "per-cell", "joint")
IMPORTANCE_METHODS = ("permutation", "coef", "tree", "none")
CLF_CHOICES = ("lda", "svm", "rf", "logistic")


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline_factory(
    clf_name: str,
    per_feature_scale: bool = True,
    seed: int = 42,
) -> Callable[[], object]:
    """Return a callable that builds a fresh sklearn estimator.

    When ``per_feature_scale`` is True the classifier is wrapped in a
    ``Pipeline([StandardScaler, clf])`` so the scaler is fit on each CV
    training fold only — fixes the cross-feature-scale dominance issue when
    stacking heterogeneous features (PSD log-power vs LZc).
    """
    def _factory() -> object:
        clf = get_classifier(clf_name, random_state=seed)
        if not per_feature_scale:
            return clf
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    return _factory


def _extract_inner_estimator(estimator) -> object:
    """Return the final classifier inside a Pipeline (or the estimator itself)."""
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


# ---------------------------------------------------------------------------
# CV-fold feature importance
# ---------------------------------------------------------------------------

def _drop_nan_rows(X: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    finite = ~np.isnan(X).any(axis=1)
    if finite.all():
        return (X,) + arrays
    return (X[finite],) + tuple(a[finite] for a in arrays)


def cv_permutation_importance(
    clf_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv,
    scoring: str,
    n_repeats: int = 5,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Permutation importance averaged across CV test folds.

    Fits a fresh estimator on each training fold, then runs sklearn's
    ``permutation_importance`` on that fold's test set. Importances are
    averaged across folds to give one ``(n_features,)`` vector.

    Returns None if no fold could be scored (e.g. all-NaN feature).
    """
    X, y, groups = _drop_nan_rows(X.astype(float), y, groups)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None

    split_kw = {"groups": groups} if _is_group_cv(cv) else {}
    fold_imps: List[np.ndarray] = []
    for train_idx, test_idx in cv.split(X, y, **split_kw):
        if len(np.unique(y[test_idx])) < 2:
            continue
        est = clf_factory()
        try:
            est.fit(X[train_idx], y[train_idx])
        except Exception as exc:
            logger.debug(f"Importance fold fit failed: {exc}")
            continue
        try:
            r = permutation_importance(
                est, X[test_idx], y[test_idx],
                n_repeats=n_repeats, scoring=scoring,
                random_state=seed, n_jobs=1,
            )
        except Exception as exc:
            logger.debug(f"permutation_importance failed: {exc}")
            continue
        fold_imps.append(r.importances_mean)

    if not fold_imps:
        return None
    return np.mean(np.stack(fold_imps, axis=0), axis=0)


def _coef_importance(estimator) -> Optional[np.ndarray]:
    inner = _extract_inner_estimator(estimator)
    coef = getattr(inner, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef)
    # Binary classification: coef_ shape is (1, n_features). Squeeze the class axis.
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]
    elif coef.ndim == 2:
        coef = np.linalg.norm(coef, axis=0)  # multi-class fallback
    return coef


def _tree_importance(estimator) -> Optional[np.ndarray]:
    inner = _extract_inner_estimator(estimator)
    return getattr(inner, "feature_importances_", None)


def fit_full_importance(
    clf_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    method: str,
) -> Optional[np.ndarray]:
    """Fit a classifier on the full dataset and extract importance via coef/tree."""
    X, y = _drop_nan_rows(X.astype(float), y)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    est = clf_factory()
    try:
        est.fit(X, y)
    except Exception as exc:
        logger.debug(f"Full-data fit for importance failed: {exc}")
        return None
    if method == "coef":
        return _coef_importance(est)
    if method == "tree":
        return _tree_importance(est)
    return None


def compute_importance_one_slice(
    clf_factory: Callable[[], object],
    X_slice: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv,
    scoring: str,
    method: str,
    n_repeats: int = 5,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Compute one ``(n_features_in_slice,)`` importance vector."""
    if method == "none":
        return None
    if method == "permutation":
        return cv_permutation_importance(
            clf_factory, X_slice, y, groups, cv,
            scoring=scoring, n_repeats=n_repeats, seed=seed,
        )
    # coef / tree backends fit once on the full data
    return fit_full_importance(clf_factory, X_slice, y, method)


# ---------------------------------------------------------------------------
# Scoring helpers (mirror run_classification but Pipeline-aware)
# ---------------------------------------------------------------------------

def _score_slice(
    clf_factory: Callable[[], object],
    X_slice: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv,
    scoring: str,
) -> float:
    """Cross-validated score on one slice. Drops rows with any NaN in X_slice."""
    if X_slice.ndim == 1:
        X_slice = X_slice.reshape(-1, 1)
    X_slice, y, groups = _drop_nan_rows(X_slice.astype(float), y, groups)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    if _is_group_cv(cv) and len(np.unique(groups)) < 2:
        return float("nan")
    kw = {"groups": groups} if _is_group_cv(cv) else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            clf_factory(), X_slice, y, cv=cv, scoring=scoring, n_jobs=1, **kw
        )
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Generic per-axis runner (per-spatial, per-feature)
# ---------------------------------------------------------------------------

def _slicer_for_axis(X: np.ndarray, axis: str) -> Tuple[Callable[[int], np.ndarray], int, int]:
    """Return (slicer, n_iters, n_features_in_slice) for a given axis.

    X: (n_trials, n_spatial, n_features).
    """
    n_trials, n_spatial, n_features = X.shape
    if axis == "per-spatial":
        return (lambda s: X[:, s, :]), n_spatial, n_features
    if axis == "per-feature":
        return (lambda f: X[:, :, f]), n_features, n_spatial
    raise ValueError(f"axis '{axis}' is not a looped-axis (use per-spatial or per-feature)")


def run_per_axis(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    axis: str,
    clf_factory: Callable[[], object],
    cv,
    n_permutations: int,
    scoring: str,
    importance_method: str,
    n_jobs: int = -1,
    seed: int = 42,
    importance_n_repeats: int = 5,
) -> Dict:
    """Loop over one axis of X with shared-permutation t-max correction.

    Returns dict with: observed, perm_scores, pvals_uncorrected, pvals_tmax,
    pvals_fdr_bh, pvals_bonferroni, (optionally) importances of shape
    ``(n_iters, n_features_in_slice)``.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X of shape (n_trials, n_spatial, n_features); got {X.shape}")

    slicer, n_iters, n_inner = _slicer_for_axis(X, axis)
    logger.info(
        f"{axis}: looping {n_iters} units, each classifier sees {n_inner} features "
        f"({n_permutations} permutations, importance={importance_method})"
    )

    logger.info("Computing observed scores…")
    observed = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_score_slice)(
                clf_factory, slicer(i), y, groups, cv, scoring
            )
            for i in range(n_iters)
        )
    )

    logger.info("Running permutations with shared label shuffles…")
    rng = np.random.default_rng(seed)
    perm_scores = np.zeros((n_permutations, n_iters), dtype=float)
    for p in tqdm(range(n_permutations), desc="permutations", unit="perm"):
        y_perm = _permute_y_within_groups(y, groups, rng)
        scores_p = Parallel(n_jobs=n_jobs)(
            delayed(_score_slice)(
                clf_factory, slicer(i), y_perm, groups, cv, scoring
            )
            for i in range(n_iters)
        )
        perm_scores[p, :] = scores_p

    pvals_unc = (np.sum(perm_scores >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    max_perm = np.nanmax(perm_scores, axis=1)
    pvals_tmax = (np.sum(max_perm[:, None] >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    nan_obs = np.isnan(observed)
    if nan_obs.any():
        pvals_unc[nan_obs] = 1.0
        pvals_tmax[nan_obs] = 1.0
        logger.warning(
            f"{int(nan_obs.sum())}/{n_iters} units had no finite score "
            f"(all-NaN slice); their p-values set to 1.0"
        )
    pvals_fdr = apply_fdr_correction(pvals_unc, alpha=0.05, method="bh")
    pvals_bonf = apply_bonferroni_correction(pvals_unc, alpha=0.05)

    out: Dict = {
        "mode": axis,
        "observed": observed,
        "perm_scores": perm_scores,
        "pvals_uncorrected": pvals_unc,
        "pvals_tmax": pvals_tmax,
        "pvals_fdr_bh": pvals_fdr,
        "pvals_bonferroni": pvals_bonf,
    }

    if importance_method != "none":
        logger.info(
            f"Computing {importance_method} importance across {n_iters} units…"
        )
        imps = Parallel(n_jobs=n_jobs)(
            delayed(compute_importance_one_slice)(
                clf_factory, slicer(i), y, groups, cv,
                scoring, importance_method,
                importance_n_repeats, seed,
            )
            for i in range(n_iters)
        )
        if all(imp is not None for imp in imps):
            importances = np.stack(imps, axis=0)  # (n_iters, n_inner)
            out["importances"] = importances
            out["importance_method"] = importance_method
        else:
            n_missing = sum(1 for imp in imps if imp is None)
            logger.warning(
                f"{n_missing}/{n_iters} units returned no importance; skipping."
            )

    n_sig_tmax = int(np.sum(pvals_tmax < 0.05))
    n_sig_fdr = int(np.sum(pvals_fdr < 0.05))
    logger.info(
        f"{axis} significant @ alpha=0.05 — tmax: {n_sig_tmax}/{n_iters}, "
        f"FDR-BH: {n_sig_fdr}/{n_iters}, max observed: {np.nanmax(observed):.3f}"
    )
    return out


# ---------------------------------------------------------------------------
# Per-cell runner (one classifier per (spatial, feature) pair)
# ---------------------------------------------------------------------------

def run_per_cell(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_factory: Callable[[], object],
    cv,
    n_permutations: int,
    scoring: str,
    n_jobs: int = -1,
    seed: int = 42,
    spatial_range: Optional[Tuple[int, int]] = None,
) -> Dict:
    """One univariate classifier per (spatial, feature) cell.

    Returns observed/perm_scores of shape (n_perms, n_spatial, n_features).
    No importance is computed (each classifier has only one input feature).
    Supports chunking over the spatial dimension via ``spatial_range``.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X of shape (n_trials, n_spatial, n_features); got {X.shape}")

    n_trials, n_spatial, n_features = X.shape
    s_start, s_stop = (0, n_spatial) if spatial_range is None else spatial_range
    spatial_indices = list(range(s_start, s_stop))

    pairs = [(s, f) for s in spatial_indices for f in range(n_features)]
    logger.info(
        f"per-cell: {len(pairs)} (spatial, feature) pairs "
        f"[spatial[{s_start}:{s_stop}] × {n_features} features], "
        f"{n_permutations} permutations"
    )

    def _cell_slice(s: int, f: int) -> np.ndarray:
        return X[:, s, f]

    logger.info("Computing observed scores per cell…")
    observed_flat = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_score_slice)(
                clf_factory, _cell_slice(s, f), y, groups, cv, scoring
            )
            for (s, f) in pairs
        )
    )
    observed = observed_flat.reshape(len(spatial_indices), n_features)

    logger.info("Running permutations with shared label shuffles…")
    rng = np.random.default_rng(seed)
    perm_scores = np.zeros(
        (n_permutations, len(spatial_indices), n_features), dtype=float
    )
    for p in tqdm(range(n_permutations), desc="permutations", unit="perm"):
        y_perm = _permute_y_within_groups(y, groups, rng)
        scores_p = Parallel(n_jobs=n_jobs)(
            delayed(_score_slice)(
                clf_factory, _cell_slice(s, f), y_perm, groups, cv, scoring
            )
            for (s, f) in pairs
        )
        perm_scores[p, :, :] = np.asarray(scores_p).reshape(
            len(spatial_indices), n_features
        )

    obs_flat = observed.ravel()
    perm_flat = perm_scores.reshape(n_permutations, -1)
    pvals_unc_flat = (np.sum(perm_flat >= obs_flat[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    max_perm = np.nanmax(perm_flat, axis=1)
    pvals_tmax_flat = (np.sum(max_perm[:, None] >= obs_flat[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    nan_obs = np.isnan(obs_flat)
    if nan_obs.any():
        pvals_unc_flat[nan_obs] = 1.0
        pvals_tmax_flat[nan_obs] = 1.0
    pvals_fdr_flat = apply_fdr_correction(pvals_unc_flat, alpha=0.05, method="bh")
    pvals_bonf_flat = apply_bonferroni_correction(pvals_unc_flat, alpha=0.05)

    cell_shape = observed.shape
    out: Dict = {
        "mode": "per-cell",
        "observed": observed,
        "perm_scores": perm_scores,
        "pvals_uncorrected": pvals_unc_flat.reshape(cell_shape),
        "pvals_tmax": pvals_tmax_flat.reshape(cell_shape),
        "pvals_fdr_bh": pvals_fdr_flat.reshape(cell_shape),
        "pvals_bonferroni": pvals_bonf_flat.reshape(cell_shape),
    }
    n_sig = int((out["pvals_tmax"] < 0.05).sum())
    logger.info(
        f"per-cell significant @ alpha=0.05 (tmax over cells): {n_sig}/{obs_flat.size}, "
        f"max observed: {np.nanmax(observed):.3f}"
    )
    return out


# ---------------------------------------------------------------------------
# Joint runner (single classifier over flattened spatial × features)
# ---------------------------------------------------------------------------

def run_joint(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_factory: Callable[[], object],
    cv,
    n_permutations: int,
    scoring: str,
    importance_method: str,
    n_jobs: int = -1,
    seed: int = 42,
    importance_n_repeats: int = 5,
) -> Dict:
    """Single classifier on (n_trials, n_spatial * n_features)."""
    if X.ndim != 3:
        raise ValueError(f"Expected X of shape (n_trials, n_spatial, n_features); got {X.shape}")

    n_trials, n_spatial, n_features = X.shape
    X_flat = X.reshape(n_trials, n_spatial * n_features)

    # NaN-safe: drop trial rows containing NaN in any feature (sklearn rejects NaN).
    X_flat, y_safe, groups_safe = _drop_nan_rows(X_flat.astype(float), y, groups)
    if len(y_safe) < 2 or len(np.unique(y_safe)) < 2:
        raise ValueError("Joint run: insufficient finite trials with both classes")

    kw = {"groups": groups_safe} if _is_group_cv(cv) else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, perm_scores, pvalue = permutation_test_score(
            clf_factory(), X_flat, y_safe,
            cv=cv, n_permutations=n_permutations, scoring=scoring,
            n_jobs=n_jobs, random_state=seed, **kw,
        )
    logger.info(
        f"joint: score={score:.3f}, p-value={pvalue:.4f} "
        f"(perm mean={np.mean(perm_scores):.3f})"
    )

    out: Dict = {
        "mode": "joint",
        "observed": float(score),
        "perm_scores": np.asarray(perm_scores),
        "pvalue": float(pvalue),
    }

    if importance_method != "none":
        logger.info(f"Computing {importance_method} importance for joint…")
        if importance_method == "permutation":
            imp = cv_permutation_importance(
                clf_factory, X_flat, y_safe, groups_safe, cv,
                scoring=scoring, n_repeats=importance_n_repeats, seed=seed,
            )
        else:
            imp = fit_full_importance(clf_factory, X_flat, y_safe, importance_method)
        if imp is not None:
            # Unflatten (n_spatial * n_features,) → (n_spatial, n_features)
            out["importances"] = np.asarray(imp).reshape(n_spatial, n_features)
            out["importance_method"] = importance_method
        else:
            logger.warning("Joint importance not available; skipping.")

    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def build_mf_base_name(
    feature_label: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    axis: str,
    importance: str,
    trial_type: str,
    analysis_level: Optional[str] = None,
) -> str:
    base = (
        f"feature-{feature_label}_space-{space}"
        f"_inout-{inout_bounds_to_string(inout_bounds)}"
        f"_clf-{clf_name}_cv-{cv_name}"
        f"_axis-{axis}_imp-{importance}"
    )
    if analysis_level is not None:
        base += f"_level-{analysis_level}"
    base += f"_type-{trial_type}_mf"
    return base


def save_mf_results(
    output_dir: Path,
    feature_label: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    axis: str,
    importance: str,
    feature_list: List[str],
    results: Dict,
    metadata: Dict,
    args_dict: Dict,
    trial_type: str,
    analysis_level: Optional[str] = None,
    chunk_info: Optional[Dict] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = build_mf_base_name(
        feature_label, space, inout_bounds, clf_name, cv_name, axis,
        importance, trial_type, analysis_level=analysis_level,
    )
    if chunk_info is not None:
        base += f"_chunk-{chunk_info['chunk_idx']}of{chunk_info['n_chunks']}"

    npz_payload: Dict[str, np.ndarray] = {}
    summary: Dict[str, object] = {}

    if axis == "joint":
        npz_payload["observed"] = np.asarray(results["observed"])
        npz_payload["perm_scores"] = results["perm_scores"]
        summary = {
            "score": float(results["observed"]),
            "pvalue": float(results["pvalue"]),
            "n_permutations": int(len(results["perm_scores"])),
        }
    else:
        npz_payload["observed"] = results["observed"]
        npz_payload["perm_scores"] = results["perm_scores"]
        npz_payload["pvals_uncorrected"] = results["pvals_uncorrected"]
        npz_payload["pvals_tmax"] = results["pvals_tmax"]
        npz_payload["pvals_fdr_bh"] = results["pvals_fdr_bh"]
        npz_payload["pvals_bonferroni"] = results["pvals_bonferroni"]
        summary = {
            "max_score": float(np.nanmax(results["observed"])),
            "mean_score": float(np.nanmean(results["observed"])),
            "n_significant_tmax_a05": int(
                np.nansum(results["pvals_tmax"] < 0.05)
            ),
            "n_significant_fdr_bh_a05": int(
                np.nansum(results["pvals_fdr_bh"] < 0.05)
            ),
            "n_permutations": int(results["perm_scores"].shape[0]),
        }

    if "importances" in results:
        npz_payload["importances"] = results["importances"]
    # Pass through auxiliary metrics + confusion matrices when present.
    # ``run_*`` populates them when the cell scoring path computes them; if
    # not, this is a no-op and downstream code falls back to ``observed``.
    for k, v in results.items():
        if k.startswith("metrics_") or k == "confusion_matrices":
            npz_payload[k] = np.asarray(v)

    # Save spatial/feature index info so the aggregator can align files.
    # Fall back to numeric labels when the feature npz didn't expose ch_names
    # (fooof/welch features predate the ch_names key).
    spatial_names = metadata.get("spatial_names")
    if spatial_names is None:
        n_spatial = int(metadata.get("n_spatial", 0))
        if n_spatial > 0:
            spatial_names = [f"s-{i}" for i in range(n_spatial)]
    if spatial_names is not None:
        npz_payload["spatial_names"] = np.asarray(spatial_names)
    npz_payload["feature_names"] = np.asarray(feature_list)

    np.savez_compressed(output_dir / f"{base}_scores.npz", **npz_payload)

    meta_out = {
        "feature": feature_label,
        "feature_list": feature_list,
        "space": space,
        "inout_bounds": list(inout_bounds),
        "classifier": clf_name,
        "cv_strategy": cv_name,
        "axis": axis,
        "scoring": args_dict.get("scoring") or results.get("scoring"),
        "importance_method": (
            results.get("importance_method", importance)
            if importance != "none" else "none"
        ),
        "trial_type": trial_type,
        "analysis_level": analysis_level,
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "args": args_dict,
        "data_metadata": metadata,
        "summary": summary,
    }
    if chunk_info is not None:
        meta_out["chunk"] = chunk_info
    meta_path = output_dir / f"{base}_metadata.json"
    meta_path.write_text(json.dumps(meta_out, indent=2, default=str))

    logger.info(f"Saved scores  -> {output_dir / (base + '_scores.npz')}")
    logger.info(f"Saved metadata -> {meta_path}")
    return meta_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_axes(axis_arg: str) -> List[str]:
    if axis_arg == "all":
        return list(AXES)
    if axis_arg in AXES:
        return [axis_arg]
    raise SystemExit(f"--axis={axis_arg!r} invalid. Choose 'all' or one of {AXES}.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-feature IN/OUT classification with axis-aware decoding and "
            "feature importance."
        )
    )
    parser.add_argument(
        "--feature", nargs="+", default=None,
        help="One or more feature names (e.g. fooof_exponent psd_alpha).",
    )
    parser.add_argument(
        "--feature-set", default=None,
        choices=["psds", "psds_corrected", "fooof", "complexity", "all"],
        help="Shortcut to expand a family of features.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Filename label for the feature stack (default: 'combined-{N}').",
    )
    parser.add_argument(
        "--space", default="sensor",
        help="'sensor', 'source', or an atlas name (e.g. 'schaefer_400').",
    )
    parser.add_argument(
        "--axis", default="per-spatial",
        choices=("all",) + AXES,
        help=(
            "Looping axis. 'per-spatial' loops sensors/ROIs using all features. "
            "'per-feature' loops features using all spatial units. 'per-cell' is "
            "the (spatial, feature) grid. 'joint' pools everything. 'all' runs "
            "all four."
        ),
    )
    parser.add_argument(
        "--clf", default="logistic", choices=CLF_CHOICES,
        help="Classifier. Default 'logistic' (L2).",
    )
    parser.add_argument(
        "--cv", default="logo", choices=["logo", "group", "stratified"],
        help="Cross-validation strategy. Default 'logo' (leave-one-subject-out).",
    )
    parser.add_argument("--n-splits", type=int, default=5,
                        help="n_splits for --cv group/stratified.")
    parser.add_argument(
        "--importance", default="permutation",
        choices=IMPORTANCE_METHODS,
        help=(
            "Feature-importance backend. 'permutation' (default): model-"
            "agnostic, averaged over CV test folds. 'coef': signed coefficients "
            "(linear models only). 'tree': feature_importances_ (RF only). "
            "'none': skip importance."
        ),
    )
    parser.add_argument(
        "--importance-n-repeats", type=int, default=5,
        help="Permutations per fold for --importance=permutation.",
    )
    parser.add_argument(
        "--no-per-feature-scale", action="store_true",
        help="Disable per-feature StandardScaler inside the CV pipeline.",
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Skip within-subject IN/OUT class balancing.",
    )
    parser.add_argument(
        "--keep-bad-trials", action="store_true",
        help="Skip the bad_ar2 filter (default drops them).",
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trial-type", default="alltrials",
        choices=["alltrials", "correct", "rare", "lapse", "correct_commission"],
    )
    parser.add_argument("--zoning", default="per-run",
                        choices=["per-run", "per-subject"])
    parser.add_argument("--n-events-window", type=int, default=8)
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Restrict to a subset of subject IDs (default: all from config).",
    )
    parser.add_argument("--scoring", default=None,
                        help="sklearn scoring metric (default: roc_auc from config).")
    parser.add_argument(
        "--standardize", default="per-subject",
        choices=["per-subject", "none"],
        help=(
            "Per-subject z-score across trials (outside CV). 'per-subject' "
            "(default) avoids LOSO collapse on absolute-scale features."
        ),
    )
    parser.add_argument(
        "--analysis-level", default="epoch",
        choices=["epoch"],
        help=(
            "Currently only 'epoch' is supported (per-epoch features). "
            "Subject-spectrum is single-feature by construction."
        ),
    )
    parser.add_argument("--n-chunks", type=int, default=1,
                        help="Split spatial dim into N chunks (per-cell only).")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Zero-based chunk index (per-cell only).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: <results>/classification_<space>/group/).",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    inout_bounds = tuple(config["analysis"]["inout_bounds"])
    scoring = args.scoring or config.get("classification", {}).get(
        "scoring", "roc_auc"
    )

    # Resolve features
    features: List[str] = []
    if args.feature_set:
        features.extend(expand_feature_set(args.feature_set, config))
    if args.feature:
        features.extend(args.feature)
    seen = set()
    features = [f for f in features if not (f in seen or seen.add(f))]
    if not features:
        raise SystemExit("Must pass --feature and/or --feature-set")

    axes = _resolve_axes(args.axis)
    label = args.label or f"combined-{len(features)}"

    if args.n_chunks > 1 and "per-cell" not in axes:
        logger.warning(
            "--n-chunks > 1 only takes effect for axis=per-cell; ignored."
        )
    if args.n_chunks > 1 and len(axes) > 1:
        raise SystemExit(
            "--n-chunks > 1 requires a single axis (per-cell). "
            "Re-run with --axis per-cell."
        )

    logger.info("=" * 78)
    logger.info("MULTI-FEATURE CLASSIFICATION (IN vs OUT)")
    logger.info("=" * 78)
    logger.info(f"features ({len(features)}): {features}")
    logger.info(f"label={label}  space={args.space}  axes={axes}")
    logger.info(f"clf={args.clf}  cv={args.cv}  importance={args.importance}")
    logger.info(f"per_feature_scale={not args.no_per_feature_scale}  "
                f"standardize={args.standardize}")
    logger.info(f"inout={inout_bounds}  n_permutations={args.n_permutations}")
    logger.info("=" * 78)

    # Load stacked features → (n_trials, n_spatial, n_features)
    if len(features) == 1:
        X, y, groups, metadata = load_classification_data(
            feature=features[0],
            space=args.space,
            inout_bounds=inout_bounds,
            config=config,
            subjects=args.subjects,
            drop_bad_trials=not args.keep_bad_trials,
            trial_type=args.trial_type,
            zoning=args.zoning,
            n_events_window=args.n_events_window,
        )
        X = X[:, :, None]
        metadata = dict(metadata)
        metadata["features"] = features
        metadata["n_features"] = 1
    else:
        X, y, groups, metadata = load_combined_features(
            features=features,
            space=args.space,
            inout_bounds=inout_bounds,
            config=config,
            subjects=args.subjects,
            drop_bad_trials=not args.keep_bad_trials,
            trial_type=args.trial_type,
            zoning=args.zoning,
            n_events_window=args.n_events_window,
        )
    X = np.asarray(X)

    # Per-subject z-score (outside CV; within-subject only)
    if args.standardize == "per-subject":
        X = standardize_within_subject(X, groups)
        logger.info(
            f"Standardized per subject: shape={X.shape}"
        )

    # Within-subject IN/OUT balance
    if not args.no_balance:
        X, y, groups = balance_within_subject(X, y, groups, seed=args.seed)
        logger.info(
            f"Balanced: {len(y)} epochs "
            f"(IN={int((y == 0).sum())}, OUT={int((y == 1).sum())})"
        )

    # CV + classifier factory (Pipeline with StandardScaler when scale enabled)
    cv = get_cv_strategy(args.cv, n_splits=args.n_splits)
    clf_factory = make_pipeline_factory(
        args.clf,
        per_feature_scale=not args.no_per_feature_scale,
        seed=args.seed,
    )

    # Importance backend compatibility check
    if args.importance == "coef" and args.clf not in ("lda", "logistic", "svm"):
        logger.warning(
            f"--importance=coef requested but clf={args.clf} has no coef_; "
            f"importance will be skipped per slice."
        )
    if args.importance == "tree" and args.clf != "rf":
        logger.warning(
            f"--importance=tree requested but clf={args.clf} has no "
            f"feature_importances_; importance will be skipped per slice."
        )

    # Output directory
    data_root = Path(config["paths"]["data_root"])
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            data_root / config["paths"]["results"]
            / f"classification_{args.space}" / "group_mf"
        )

    args_dict = {k: v for k, v in vars(args).items()}

    chunk_info = None
    if args.n_chunks > 1 and "per-cell" in axes:
        n_spatial_total = X.shape[1]
        start, stop = chunk_range(n_spatial_total, args.n_chunks, args.chunk_idx)
        chunk_info = {
            "chunk_idx": args.chunk_idx,
            "n_chunks": args.n_chunks,
            "start": int(start),
            "stop": int(stop),
            "n_spatial_total": int(n_spatial_total),
            "seed": int(args.seed),
        }
        logger.info(
            f"per-cell chunk {args.chunk_idx}/{args.n_chunks}: "
            f"spatial[{start}:{stop}] of {n_spatial_total}"
        )

    for axis in axes:
        logger.info("")
        logger.info("=" * 78)
        logger.info(f"AXIS: {axis}")
        logger.info("=" * 78)

        if axis in ("per-spatial", "per-feature"):
            results = run_per_axis(
                X=X, y=y, groups=groups, axis=axis,
                clf_factory=clf_factory, cv=cv,
                n_permutations=args.n_permutations,
                scoring=scoring,
                importance_method=args.importance,
                n_jobs=args.n_jobs, seed=args.seed,
                importance_n_repeats=args.importance_n_repeats,
            )
        elif axis == "per-cell":
            spatial_range = None
            if chunk_info is not None:
                spatial_range = (chunk_info["start"], chunk_info["stop"])
            results = run_per_cell(
                X=X, y=y, groups=groups,
                clf_factory=clf_factory, cv=cv,
                n_permutations=args.n_permutations,
                scoring=scoring,
                n_jobs=args.n_jobs, seed=args.seed,
                spatial_range=spatial_range,
            )
        elif axis == "joint":
            results = run_joint(
                X=X, y=y, groups=groups,
                clf_factory=clf_factory, cv=cv,
                n_permutations=args.n_permutations,
                scoring=scoring,
                importance_method=args.importance,
                n_jobs=args.n_jobs, seed=args.seed,
                importance_n_repeats=args.importance_n_repeats,
            )
        else:
            raise ValueError(f"Unknown axis: {axis}")

        save_mf_results(
            output_dir=output_dir,
            feature_label=label,
            space=args.space,
            inout_bounds=inout_bounds,
            clf_name=args.clf,
            cv_name=args.cv,
            axis=axis,
            importance=args.importance,
            feature_list=features,
            results=results,
            metadata=metadata,
            args_dict=args_dict,
            trial_type=args.trial_type,
            analysis_level=args.analysis_level,
            chunk_info=chunk_info if axis == "per-cell" else None,
        )


if __name__ == "__main__":
    main()
