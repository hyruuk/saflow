"""Cross-subject generalization sweep for IN/OUT decoding.

Motivation
----------
The ``joint`` axis of :mod:`code.classification.run_multifeature` fits one
classifier on the flattened ``(n_spatial * n_features)`` vector with a fixed
``logistic(C=1)``. On Schaefer-400 that is 9600 dimensions per window, and
leave-one-subject-out (LOSO) AUC sits barely above chance. Piloting showed the
binding constraint is dimensionality, not the classifier family: collapsing the
400 parcels to the 7 Yeo networks gains ~5 AUC points and runs ~40x faster,
while tuning C / swapping in SVM / RF / boosting on the flat vector gains
essentially nothing.

This module therefore sweeps the axes that actually matter, in three stages:

``sweep``
    Grid over {spatial reduction} x {feature set} x {estimator} x
    {per-subject normalization} x {IN/OUT bounds}, each scored with LOSO.
    Saves *per-fold* (= per-subject) AUC, which ``run_multifeature`` does not
    keep — it is what lets us say "N of 32 held-out subjects above chance".

``confirm``
    Nested LOSO on the winning cell (inner GroupKFold over training subjects
    tunes the estimator's hyperparameter) so the headline number is not
    contaminated by the sweep, plus a within-subject label-permutation null.

``importance``
    Haufe-transformed activation patterns and block permutation importance for
    the winning cell — the "alpha in this network / exponent in that one"
    readout. Blocks are whole features (across spatial units) and whole spatial
    units (across features), which is both cheaper and more interpretable than
    permuting 9600 individual columns.

Deliberately *not* included: temporal smoothing of the decision function.
``load_classification_data`` returns each subject's windows as an IN block
followed by an OUT block, so smoothing in array order averages within a class
and inflates AUC. Doing it honestly needs the ``alignment_keys`` onsets to
restore true temporal order, and even then it trades on the autocorrelation of
the VTC-derived labels; out of scope here.

Usage:
    python -m code.classification.run_multifeature_sweep --stage sweep --space schaefer_400
    python -m code.classification.run_multifeature_sweep --stage confirm --space schaefer_400
    python -m code.classification.run_multifeature_sweep --stage importance --space schaefer_400
    python -m code.classification.run_multifeature_sweep --stage all --space schaefer_400
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    LeaveOneGroupOut,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC

from code.classification.run_classification import (
    _permute_y_within_groups,
    expand_feature_set,
    get_git_hash,
    inout_bounds_to_string,
    load_combined_features,
    standardize_within_subject,
)
from code.features.inout_selection import (
    DEFAULT_STRATEGY as DEFAULT_INOUT_STRATEGY,
    inout_selection_token,
)
from code.statistics.corrections import apply_fdr_correction
from code.utils.config import load_config
from code.utils.yeo_networks import network_parcel_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


STAGES = ("sweep", "merge", "confirm", "importance", "nested-select", "all")
NORMALIZATIONS = ("zscore", "rank", "none")

# Curated defaults. Ordered cheapest-first on purpose: the network-level cells
# cost well under a second each while `flat` (up to 400 x 24 = 9600 dims) costs
# minutes, so a job that runs out of wall time still leaves the informative
# cells on disk — the CSV is flushed after every cell and a re-submission
# resumes where it stopped.
DEFAULT_REDUCTIONS = ("yeo7-mean", "yeo7-meansd", "hemi-yeo7", "global-mean",
                      "pca-20", "pca-50", "flat")

# Tree ensembles on the flat vector cost ~10x a linear model and, in piloting,
# scored below it — the sweep skips them above this many input dimensions
# rather than spending hours confirming that.
NONLINEAR_FAMILIES = ("hgb", "rf")
DEFAULT_FEATURE_SETS = ("all", "fooof", "psds", "psds_corrected", "complexity")
DEFAULT_ESTIMATORS = (
    "logistic:C=0.0001", "logistic:C=0.001", "logistic:C=0.01",
    "logistic:C=0.1", "logistic:C=1", "logistic:C=10",
    "lda:shrinkage=auto", "linearsvc:C=0.01", "linearsvc:C=1", "hgb",
)
DEFAULT_NORMALIZATIONS = ("zscore", "rank")
DEFAULT_BOUNDS = ((25, 75),)

# Candidate set for the `nested-select` stage. Deliberately restricted to the
# cheap, label-independent reductions: that stage re-runs selection inside all
# 32 outer folds, so a candidate costs 32x what it does in the sweep.
DEFAULT_SELECT_REDUCTIONS = ("yeo7-mean", "yeo7-meansd", "hemi-yeo7", "global-mean")
DEFAULT_SELECT_ESTIMATORS = ("logistic:C=0.01", "logistic:C=1", "lda:shrinkage=auto")


# ---------------------------------------------------------------------------
# Per-subject normalization
# ---------------------------------------------------------------------------

def rank_within_subject(X: np.ndarray, groups: np.ndarray, seed: int = 42) -> np.ndarray:
    """Per-subject rank-Gaussianization of every feature column.

    Same purpose as :func:`standardize_within_subject` — remove the
    between-subject offset that otherwise dominates a cross-subject fit — but
    robust to the heavy tails of log-power and complexity features. Applied
    outside CV; it only ever touches one subject's own X, never y, so no label
    information crosses the LOSO boundary.
    """
    X = X.astype(float, copy=True)
    orig_shape = X.shape
    for g in np.unique(groups):
        idx = groups == g
        block = X[idx].reshape(int(idx.sum()), -1)
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, block.shape[0]),
            subsample=None,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[idx] = qt.fit_transform(block).reshape((-1,) + orig_shape[1:])
    return X


def apply_normalization(X: np.ndarray, groups: np.ndarray, mode: str, seed: int = 42
                        ) -> np.ndarray:
    if mode == "zscore":
        return standardize_within_subject(X, groups)
    if mode == "rank":
        return rank_within_subject(X, groups, seed=seed)
    if mode == "none":
        return X.astype(float, copy=False)
    raise ValueError(f"Unknown normalization {mode!r}; choose from {NORMALIZATIONS}")


# ---------------------------------------------------------------------------
# Spatial reductions
# ---------------------------------------------------------------------------

def _hemisphere_of(name: str) -> Optional[str]:
    """Return 'LH'/'RH' for a Schaefer parcel label, else None."""
    for h in ("LH", "RH"):
        if name.startswith(h + "_") or f"_{h}_" in name:
            return h
    return None


def build_reduction(
    name: str, spatial_names: Sequence[str]
) -> Tuple[Callable[[np.ndarray], np.ndarray], List[str], Optional[int]]:
    """Return ``(reduce_fn, reduced_spatial_names, pca_k)``.

    ``reduce_fn`` maps ``(n_trials, n_spatial, n_features)`` to a reduced
    ``(n_trials, n_reduced, n_features)``. Every reduction here is a fixed,
    label-independent regrouping of the spatial axis, so it is safe to apply
    once outside the CV loop. The one data-dependent reduction (PCA) is
    returned as ``pca_k`` instead and is fit inside the pipeline, on the
    training fold only.
    """
    names = list(spatial_names)

    if name == "flat":
        return (lambda X: X), names, None

    if name.startswith("pca-"):
        return (lambda X: X), names, int(name.split("-", 1)[1])

    if name == "global-mean":
        return (lambda X: np.nanmean(X, axis=1, keepdims=True)), ["Global"], None

    nets = {k: v for k, v in network_parcel_indices(names, 7).items() if len(v)}
    if not nets:
        raise ValueError(
            f"Reduction {name!r} needs Yeo-parseable parcel labels; got e.g. "
            f"{names[:2]}. Only 'flat', 'pca-K' and 'global-mean' work in this space."
        )

    if name == "yeo7-mean":
        idx = list(nets.values())
        labels = list(nets)
        return (lambda X: np.stack([np.nanmean(X[:, i, :], axis=1) for i in idx], axis=1)
                ), labels, None

    if name == "yeo7-meansd":
        idx = list(nets.values())
        labels = [f"{k}_mean" for k in nets] + [f"{k}_sd" for k in nets]

        def _reduce(X, idx=idx):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mu = np.stack([np.nanmean(X[:, i, :], axis=1) for i in idx], axis=1)
                sd = np.stack([np.nanstd(X[:, i, :], axis=1) for i in idx], axis=1)
            return np.concatenate([mu, sd], axis=1)

        return _reduce, labels, None

    if name == "hemi-yeo7":
        groups: Dict[str, np.ndarray] = {}
        for net, idx in nets.items():
            for h in ("LH", "RH"):
                sel = np.array([i for i in idx if _hemisphere_of(names[i]) == h])
                if len(sel):
                    groups[f"{h}_{net}"] = sel
        if not groups:
            raise ValueError("hemi-yeo7: no LH/RH prefix found in parcel labels")
        idx = list(groups.values())
        return (lambda X: np.stack([np.nanmean(X[:, i, :], axis=1) for i in idx], axis=1)
                ), list(groups), None

    if name.startswith("net-"):
        net = name.split("-", 1)[1]
        if net not in nets:
            raise ValueError(f"Unknown network {net!r}; known: {list(nets)}")
        sel = nets[net]
        return (lambda X, s=sel: X[:, s, :]), [names[i] for i in sel], None

    raise ValueError(f"Unknown reduction {name!r}")


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def parse_estimator(spec: str, seed: int = 42) -> Tuple[object, Dict[str, List]]:
    """Parse ``"family:key=value,key=value"`` into ``(estimator, nested_grid)``.

    ``nested_grid`` is the search space the ``confirm`` stage tunes over with
    an inner GroupKFold; keys are prefixed for use inside the pipeline.
    """
    family, _, rest = spec.partition(":")
    params: Dict[str, object] = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        try:
            params[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except ValueError:
            params[k] = None if v == "none" else v

    if family == "logistic":
        est = LogisticRegression(
            C=params.get("C", 1.0), max_iter=int(params.get("max_iter", 2000)),
            random_state=seed,
        )
        grid = {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]}
    elif family == "linearsvc":
        est = LinearSVC(C=params.get("C", 1.0), dual="auto", random_state=seed,
                        max_iter=int(params.get("max_iter", 5000)))
        grid = {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
    elif family == "lda":
        shrink = params.get("shrinkage", "auto")
        est = LinearDiscriminantAnalysis(
            solver="svd" if shrink is None else "lsqr", shrinkage=shrink
        )
        grid = {"clf__shrinkage": ["auto", 0.1, 0.5, 0.9]}
    elif family == "hgb":
        est = HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 200)),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=seed,
        )
        grid = {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [15, 31]}
    elif family == "rf":
        est = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            random_state=seed, n_jobs=1,
        )
        grid = {"clf__max_depth": [None, 10, 20]}
    else:
        raise ValueError(f"Unknown estimator family {family!r}")
    return est, grid


def make_pipeline(est: object, pca_k: Optional[int] = None) -> Pipeline:
    """Impute -> scale -> (PCA) -> classify, all fit on the training fold only.

    The imputer matters: on the flat 400x24 vector a single failed FOOOF fit
    anywhere in a window would otherwise drop the whole window, and with 9600
    columns that can silently discard most of the data.
    """
    steps: List[Tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ]
    if pca_k:
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=pca_k, random_state=42)))
    steps.append(("clf", est))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def oof_scores(pipe: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               cv, n_jobs: int = -1) -> np.ndarray:
    """Out-of-fold continuous decision scores (one per sample)."""
    method = "decision_function" if hasattr(pipe[-1], "decision_function") else "predict_proba"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = cross_val_predict(pipe, X, y, cv=cv, groups=groups,
                                method=method, n_jobs=n_jobs)
    return out if out.ndim == 1 else out[:, 1]


def per_subject_metrics(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        threshold: float) -> Dict[str, np.ndarray]:
    """Per-held-out-subject AUC and balanced accuracy from OOF scores.

    With LOSO each subject is exactly one test fold, so these are per-fold
    scores. ``threshold`` is 0.0 for ``decision_function`` and 0.5 for
    ``predict_proba``, which reproduces what ``predict`` would return.
    """
    subs = np.unique(groups)
    auc, bacc = [], []
    for s in subs:
        m = groups == s
        if len(np.unique(y[m])) < 2:
            auc.append(np.nan)
            bacc.append(np.nan)
            continue
        auc.append(roc_auc_score(y[m], scores[m]))
        bacc.append(balanced_accuracy_score(y[m], (scores[m] > threshold).astype(int)))
    return {"subjects": subs, "auc": np.asarray(auc), "bacc": np.asarray(bacc)}


def summarize(auc: np.ndarray) -> Dict[str, float]:
    """Mean / median / 95% CI over subjects, plus a Wilcoxon test vs 0.5."""
    a = auc[np.isfinite(auc)]
    n = len(a)
    out = {
        "mean_auc": float(np.mean(a)),
        "median_auc": float(np.median(a)),
        "std_auc": float(np.std(a, ddof=1)) if n > 1 else float("nan"),
        "n_subjects": int(n),
        "n_above_chance": int((a > 0.5).sum()),
    }
    if n > 1:
        sem = stats.sem(a)
        lo, hi = stats.t.interval(0.95, n - 1, loc=np.mean(a), scale=sem)
        out["ci95_low"], out["ci95_high"] = float(lo), float(hi)
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a - 0.5).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Data cache
# ---------------------------------------------------------------------------

class DataCache:
    """Loads the union of every requested feature once per IN/OUT bound pair.

    Feature sets are then column subsets of that one tensor, and each
    per-subject normalization is materialized once and reused across every
    (feature set x reduction x estimator) cell.
    """

    def __init__(self, features: List[str], space: str, config: Dict,
                 trial_type: str, n_events_window: int, inout_selection: str,
                 keep_bad_trials: bool, seed: int):
        self.features = features
        self.space = space
        self.config = config
        self.trial_type = trial_type
        self.n_events_window = n_events_window
        self.inout_selection = inout_selection
        self.keep_bad_trials = keep_bad_trials
        self.seed = seed
        self._bounds: Optional[Tuple[int, int]] = None
        self._norm_cache: Dict[str, np.ndarray] = {}
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.groups: Optional[np.ndarray] = None
        self.metadata: Dict = {}
        self.spatial_names: List[str] = []

    def load(self, bounds: Tuple[int, int]) -> None:
        if self._bounds == bounds:
            return
        t0 = time.time()
        X, y, groups, meta = load_combined_features(
            features=self.features, space=self.space, inout_bounds=bounds,
            config=self.config, drop_bad_trials=not self.keep_bad_trials,
            trial_type=self.trial_type, n_events_window=self.n_events_window,
            inout_selection=self.inout_selection,
        )
        self.X = np.asarray(X, dtype=float)
        self.y, self.groups, self.metadata = y, groups, meta
        names = meta.get("spatial_names")
        self.spatial_names = list(names) if names is not None else [
            f"s-{i}" for i in range(self.X.shape[1])
        ]
        self._bounds = bounds
        self._norm_cache = {}
        n_nan = int(np.isnan(self.X).sum())
        logger.info(
            f"inout={bounds}: X={self.X.shape}  n_subjects={len(np.unique(groups))}  "
            f"NaN cells={n_nan} ({100 * n_nan / self.X.size:.4f}%)  "
            f"[{time.time() - t0:.0f}s]"
        )

    def normalized(self, mode: str) -> np.ndarray:
        if mode not in self._norm_cache:
            t0 = time.time()
            self._norm_cache[mode] = apply_normalization(
                self.X, self.groups, mode, seed=self.seed
            )
            logger.info(f"  normalization={mode} materialized [{time.time() - t0:.0f}s]")
        return self._norm_cache[mode]


def build_cell(cache: DataCache, feature_set: str, reduction: str,
               feature_index: Dict[str, List[int]], normalization: str
               ) -> Tuple[np.ndarray, List[str], List[str], Optional[int]]:
    """Return ``(X2d, reduced_spatial_names, feature_names, pca_k)`` for one cell."""
    cols = feature_index[feature_set]
    Xn = cache.normalized(normalization)[:, :, cols]
    reduce_fn, red_names, pca_k = build_reduction(reduction, cache.spatial_names)
    Xr = reduce_fn(Xn)
    feat_names = [cache.features[c] for c in cols]
    return Xr.reshape(Xr.shape[0], -1), red_names, feat_names, pca_k


# ---------------------------------------------------------------------------
# Stage: sweep
# ---------------------------------------------------------------------------

SWEEP_FIELDS = [
    "config_id", "inout", "normalization", "feature_set", "reduction", "estimator",
    "n_input_dims", "n_reduced_spatial", "n_features", "mean_auc", "median_auc",
    "std_auc", "ci95_low", "ci95_high", "n_above_chance", "n_subjects",
    "wilcoxon_p", "wilcoxon_p_fdr", "mean_bacc", "seconds",
]


def _config_id(bounds, norm, fset, red, est) -> str:
    return f"{inout_bounds_to_string(bounds)}|{norm}|{fset}|{red}|{est}"


def enumerate_cells(args) -> List[Tuple]:
    """Deterministic full-grid ordering: (bounds, normalization, feature_set,
    reduction, estimator). Every shard derives its work from this one list, so
    shards never overlap and never miss a cell."""
    return [
        (bounds, norm, fset, red, est)
        for bounds in args.inout_bounds
        for norm in args.normalizations
        for fset in args.feature_sets
        for red in args.reductions
        for est in args.estimators
    ]


def shard_cells(cells: List[Tuple], n_shards: int, shard_idx: int) -> List[Tuple]:
    """Round-robin slice of the grid, re-sorted for cache locality.

    Round-robin (rather than contiguous blocks) balances cost: the expensive
    ``flat`` cells are spread evenly instead of piling into the last shards.
    The re-sort then groups a shard's own cells by (bounds, normalization,
    feature_set, reduction) so each shard loads each bounds once and
    materializes each normalization once.
    """
    mine = cells[shard_idx::n_shards] if n_shards > 1 else list(cells)
    order = {c: i for i, c in enumerate(cells)}
    return sorted(mine, key=lambda c: (c[0], c[1], c[2], c[3], order[c]))


def _shard_suffix(n_shards: int, shard_idx: int) -> str:
    return f"_shard-{shard_idx}of{n_shards}" if n_shards > 1 else ""


def stage_sweep(cache: DataCache, feature_index: Dict[str, List[int]],
                args, out_base: Path) -> Path:
    """Grid over the sweep axes; one LOSO evaluation per cell.

    With ``--n-shards N`` this evaluates only shard ``--shard-idx`` of the grid
    and writes its own ``*_shard-KofN`` files; ``--stage merge`` then folds the
    shards into the single ranked CSV that confirm/importance read.
    """
    suffix = _shard_suffix(args.n_shards, args.shard_idx)
    csv_path = out_base.with_name(out_base.name + f"_sweep{suffix}.csv")
    folds_path = out_base.with_name(out_base.name + f"_sweep-folds{suffix}.npz")

    done: Dict[str, Dict] = {}
    fold_store: Dict[str, np.ndarray] = {}
    if csv_path.exists() and not args.force:
        with csv_path.open() as fh:
            done = {r["config_id"]: r for r in csv.DictReader(fh)}
        if folds_path.exists():
            with np.load(folds_path, allow_pickle=True) as npz:
                fold_store = {k: npz[k] for k in npz.files if k.startswith("auc/")}
        logger.info(f"Resuming: {len(done)} cell(s) already in {csv_path.name}")

    cv = LeaveOneGroupOut()
    rows: List[Dict] = list(done.values())
    my_cells = shard_cells(enumerate_cells(args), args.n_shards, args.shard_idx)
    n_total = len(my_cells)
    logger.info(
        f"Sweep shard {args.shard_idx}/{args.n_shards}: {n_total} cell(s) "
        f"of {len(enumerate_cells(args))} in the full grid"
    )

    built: Optional[Tuple] = None   # (bounds, norm, fset, red) currently in X2d
    X2d = red_names = feat_names = pca_k = None

    for i, (bounds, norm, fset, red, est_spec) in enumerate(my_cells, start=1):
        cid = _config_id(bounds, norm, fset, red, est_spec)
        if cid in done:
            continue
        if built != (bounds, norm, fset, red):
            cache.load(bounds)
            try:
                X2d, red_names, feat_names, pca_k = build_cell(
                    cache, fset, red, feature_index, norm
                )
            except ValueError as exc:
                logger.warning(f"skip reduction={red} fset={fset}: {exc}")
                built = None
                continue
            built = (bounds, norm, fset, red)
        if (est_spec.partition(":")[0] in NONLINEAR_FAMILIES
                and X2d.shape[1] > args.nonlinear_max_dims):
            logger.info(
                f"[{i}/{n_total}] skip {cid}: {X2d.shape[1]} dims "
                f"> --nonlinear-max-dims={args.nonlinear_max_dims}"
            )
            continue
        est, _ = parse_estimator(est_spec, seed=args.seed)
        pipe = make_pipeline(est, pca_k)
        thr = 0.0 if hasattr(pipe[-1], "decision_function") else 0.5
        t0 = time.time()
        try:
            scores = oof_scores(pipe, X2d, cache.y, cache.groups, cv,
                                n_jobs=args.n_jobs)
        except Exception as exc:  # keep the shard alive
            logger.warning(f"[{i}/{n_total}] FAILED {cid}: {exc}")
            continue
        m = per_subject_metrics(scores, cache.y, cache.groups, thr)
        summ = summarize(m["auc"])
        row = {
            "config_id": cid,
            "inout": inout_bounds_to_string(bounds),
            "normalization": norm, "feature_set": fset,
            "reduction": red, "estimator": est_spec,
            "n_input_dims": X2d.shape[1],
            "n_reduced_spatial": len(red_names),
            "n_features": len(feat_names),
            "mean_bacc": float(np.nanmean(m["bacc"])),
            "seconds": round(time.time() - t0, 1),
            **{k: summ.get(k) for k in
               ("mean_auc", "median_auc", "std_auc", "ci95_low", "ci95_high",
                "n_above_chance", "n_subjects", "wilcoxon_p")},
        }
        rows.append(row)
        fold_store[f"auc/{cid}"] = m["auc"]
        logger.info(
            f"[{i}/{n_total}] {cid}  dims={X2d.shape[1]}  "
            f"AUC={row['mean_auc']:.4f}  "
            f"n>0.5={row['n_above_chance']}/{row['n_subjects']}  "
            f"({row['seconds']}s)"
        )
        _write_sweep(csv_path, folds_path, rows, fold_store, cache)

    _write_sweep(csv_path, folds_path, rows, fold_store, cache)
    logger.info(f"Sweep shard complete: {len(rows)} cell(s) -> {csv_path}")
    return csv_path


def stage_merge(out_base: Path) -> Path:
    """Fold per-shard sweep outputs into one ranked CSV + one folds npz."""
    csv_path = out_base.with_name(out_base.name + "_sweep.csv")
    folds_path = out_base.with_name(out_base.name + "_sweep-folds.npz")
    shard_csvs = sorted(out_base.parent.glob(out_base.name + "_sweep_shard-*.csv"))
    if not shard_csvs:
        # An unsharded sweep wrote the merged path directly. Merge is still
        # worth running (it is what computes the grid-wide FDR), and the SLURM
        # chain always calls it, so re-rank that file in place rather than
        # failing and stranding confirm/importance behind an afterok.
        if csv_path.exists():
            logger.info(f"No shard CSVs; re-ranking unsharded {csv_path.name}")
            shard_csvs = [csv_path]
        else:
            raise SystemExit(f"No sweep CSVs to merge next to {out_base}")

    rows: Dict[str, Dict] = {}
    for path in shard_csvs:
        with path.open() as fh:
            for r in csv.DictReader(fh):
                rows[r["config_id"]] = r
    fold_store: Dict[str, np.ndarray] = {}
    subjects = None
    fold_paths = sorted(out_base.parent.glob(out_base.name + "_sweep-folds_shard-*.npz"))
    if not fold_paths and folds_path.exists():
        fold_paths = [folds_path]
    for path in fold_paths:
        with np.load(path, allow_pickle=True) as npz:
            for k in npz.files:
                if k.startswith("auc/"):
                    fold_store[k] = npz[k]
                elif k == "subjects":
                    subjects = npz[k]

    # The per-cell Wilcoxon is one test per cell; across a grid this size that
    # needs correcting. BH across every cell that produced a p-value.
    ranked = sorted(rows.values(), key=lambda r: -float(r.get("mean_auc") or 0))
    have_p = [r for r in ranked if r.get("wilcoxon_p") not in (None, "", "nan")]
    if have_p:
        q = apply_fdr_correction(
            np.array([float(r["wilcoxon_p"]) for r in have_p]), method="bh"
        )
        for r, qi in zip(have_p, q):
            r["wilcoxon_p_fdr"] = float(qi)
        logger.info(
            f"FDR-BH across {len(have_p)} cell(s): "
            f"{int((q < 0.05).sum())} significant at q<0.05"
        )

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SWEEP_FIELDS)
        w.writeheader()
        for r in ranked:
            w.writerow({k: r.get(k) for k in SWEEP_FIELDS})
    np.savez_compressed(
        folds_path,
        subjects=subjects if subjects is not None else np.array([]),
        **fold_store,
    )
    logger.info(
        f"Merged {len(shard_csvs)} shard(s), {len(rows)} cell(s) -> {csv_path}"
    )
    return csv_path


def _write_sweep(csv_path: Path, folds_path: Path, rows: List[Dict],
                 fold_store: Dict[str, np.ndarray], cache: DataCache) -> None:
    """Flush after every cell so a wall-time kill still leaves usable results."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SWEEP_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: -float(r.get("mean_auc") or 0)):
            w.writerow({k: r.get(k) for k in SWEEP_FIELDS})
    np.savez_compressed(folds_path, subjects=np.unique(cache.groups), **fold_store)


def read_winner(csv_path: Path, select_by: str = "mean_auc") -> Dict[str, str]:
    with csv_path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r.get(select_by)]
    if not rows:
        raise SystemExit(f"No usable rows in {csv_path}")
    best = max(rows, key=lambda r: float(r[select_by]))
    logger.info(
        f"Winner by {select_by}: {best['config_id']}  "
        f"AUC={float(best['mean_auc']):.4f}  "
        f"n>0.5={best['n_above_chance']}/{best['n_subjects']}"
    )
    return best


# ---------------------------------------------------------------------------
# Stage: confirm (nested LOSO + permutation null)
# ---------------------------------------------------------------------------

def _parse_inout(token: str, candidates: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    for b in candidates:
        if inout_bounds_to_string(b) == token:
            return b
    return int(token[:2]), int(token[2:])


def _fold_auc(pipe: Pipeline, Xtr, ytr, Xte, yte) -> Tuple[float, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(Xtr, ytr)
    if hasattr(pipe[-1], "decision_function"):
        s = pipe.decision_function(Xte)
    else:
        s = pipe.predict_proba(Xte)[:, 1]
    if len(np.unique(yte)) < 2:
        return float("nan"), s
    return float(roc_auc_score(yte, s)), s


def _nested_fold(X, y, groups, train, test, pipe, grid, inner_splits, seed):
    """One outer LOSO fold with an inner GroupKFold hyperparameter search."""
    Xtr, ytr, gtr = X[train], y[train], groups[train]
    n_inner = min(inner_splits, len(np.unique(gtr)))
    inner = list(GroupKFold(n_splits=n_inner).split(Xtr, ytr, gtr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="roc_auc",
                          n_jobs=1, refit=True)
        gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    if hasattr(best[-1], "decision_function"):
        s = best.decision_function(X[test])
    else:
        s = best.predict_proba(X[test])[:, 1]
    yte = y[test]
    auc = (float(roc_auc_score(yte, s)) if len(np.unique(yte)) > 1 else float("nan"))
    return auc, gs.best_params_


def _mean_auc(pipe, X, y, groups, cv, thr, n_jobs=1) -> float:
    s = oof_scores(clone(pipe), X, y, groups, cv, n_jobs=n_jobs)
    return float(np.nanmean(per_subject_metrics(s, y, groups, thr)["auc"]))


def stage_confirm(cache: DataCache, feature_index, args, out_base: Path,
                  cell: Dict[str, str]) -> Path:
    """Nested LOSO on one cell + a within-subject label-permutation null.

    The nested estimate is the honest headline number: the hyperparameter is
    re-chosen inside each training set, so the sweep that picked this cell
    cannot leak into it. The permutation null is run at *fixed*
    hyperparameters — nesting it too would multiply cost by the size of the
    grid for no gain in the null's calibration.
    """
    from joblib import Parallel, delayed

    bounds = _parse_inout(cell["inout"], args.inout_bounds)
    cache.load(bounds)
    X2d, red_names, feat_names, pca_k = build_cell(
        cache, cell["feature_set"], cell["reduction"], feature_index,
        cell["normalization"],
    )
    est, grid = parse_estimator(cell["estimator"], seed=args.seed)
    pipe = make_pipeline(est, pca_k)
    thr = 0.0 if hasattr(pipe[-1], "decision_function") else 0.5
    y, groups = cache.y, cache.groups
    cv = LeaveOneGroupOut()
    splits = list(cv.split(X2d, y, groups))

    logger.info(f"confirm: cell={cell['config_id']}  dims={X2d.shape[1]}")
    logger.info(f"confirm: nested LOSO ({len(splits)} outer x {args.inner_splits} inner), "
                f"grid={grid}")
    t0 = time.time()
    nested = Parallel(n_jobs=args.n_jobs)(
        delayed(_nested_fold)(X2d, y, groups, tr, te, pipe, grid,
                              args.inner_splits, args.seed)
        for tr, te in splits
    )
    nested_auc = np.array([a for a, _ in nested])
    best_params = [p for _, p in nested]
    logger.info(f"confirm: nested done [{time.time() - t0:.0f}s]")

    # Fixed-hyperparameter reference + permutation null.
    fixed_auc_per_subject = per_subject_metrics(
        oof_scores(pipe, X2d, y, groups, cv, n_jobs=args.n_jobs), y, groups, thr
    )["auc"]
    observed_fixed = float(np.nanmean(fixed_auc_per_subject))

    perm = np.array([])
    if args.n_permutations > 0:
        logger.info(f"confirm: {args.n_permutations} within-subject label permutations")
        t0 = time.time()
        rngs = [np.random.default_rng(args.seed + i) for i in range(args.n_permutations)]
        perm = np.array(Parallel(n_jobs=args.n_jobs)(
            delayed(_mean_auc)(pipe, X2d, _permute_y_within_groups(y, groups, r),
                               groups, cv, thr, 1)
            for r in rngs
        ))
        logger.info(f"confirm: permutations done [{time.time() - t0:.0f}s]")

    summ_nested = summarize(nested_auc)
    pvalue = (float((np.sum(perm >= observed_fixed) + 1) / (len(perm) + 1))
              if perm.size else float("nan"))

    out_path = out_base.with_name(out_base.name + "_confirm.npz")
    np.savez_compressed(
        out_path,
        nested_auc_per_subject=nested_auc,
        fixed_auc_per_subject=fixed_auc_per_subject,
        perm_mean_auc=perm,
        subjects=np.unique(groups),
        best_params=np.asarray([json.dumps(p) for p in best_params]),
        spatial_names=np.asarray(red_names),
        feature_names=np.asarray(feat_names),
    )
    meta = {
        "cell": cell,
        "nested": summ_nested,
        "observed_fixed_hyperparams": observed_fixed,
        "permutation_pvalue": pvalue,
        "n_permutations": int(perm.size),
        "n_input_dims": int(X2d.shape[1]),
        "inner_splits": args.inner_splits,
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
    }
    out_base.with_name(out_base.name + "_confirm.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )
    logger.info(
        f"confirm: nested AUC={summ_nested['mean_auc']:.4f} "
        f"[{summ_nested.get('ci95_low', float('nan')):.4f}, "
        f"{summ_nested.get('ci95_high', float('nan')):.4f}]  "
        f"n>0.5={summ_nested['n_above_chance']}/{summ_nested['n_subjects']}  "
        f"wilcoxon p={summ_nested.get('wilcoxon_p', float('nan')):.2e}  "
        f"perm p={pvalue:.4f}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Stage: importance (Haufe patterns + block permutation)
# ---------------------------------------------------------------------------

def _haufe_pattern(pipe: Pipeline, Xtr: np.ndarray) -> Optional[np.ndarray]:
    """Activation pattern A ~ Cov(Z) w / var(Zw), in the scaled feature space.

    Weights of a multivariate model are not interpretable on their own: a large
    weight can mark a noise-suppressing channel rather than a signal-carrying
    one (Haufe et al., 2014). The pattern is, which is what the panel should
    plot. Units are per-SD of each column, so heterogeneous features (log-power
    vs LZc) stay comparable.
    """
    clf = pipe[-1]
    if not hasattr(clf, "coef_"):
        return None
    w = np.ravel(clf.coef_)
    Z = Xtr
    for name, step in pipe.steps[:-1]:
        if name == "pca":
            break
        Z = step.transform(Z)
    if "pca" in pipe.named_steps:
        w = pipe.named_steps["pca"].components_.T @ w
    s = Z @ w
    var = float(np.var(s))
    Zc = Z - Z.mean(axis=0, keepdims=True)
    A = Zc.T @ (s - s.mean()) / (len(s) - 1)
    return A / var if var > 0 else A


def _block_columns(n_spatial: int, n_features: int) -> Tuple[Dict[str, np.ndarray],
                                                             Dict[str, np.ndarray]]:
    """Column indices of each feature block and each spatial block.

    ``X2d`` is ``(n_trials, n_spatial, n_features)`` reshaped in C order, so
    column ``s * n_features + f`` holds spatial unit ``s``, feature ``f``.
    """
    feat_blocks = {
        str(f): np.array([s * n_features + f for s in range(n_spatial)])
        for f in range(n_features)
    }
    spat_blocks = {
        str(s): np.arange(s * n_features, (s + 1) * n_features)
        for s in range(n_spatial)
    }
    return feat_blocks, spat_blocks


def _importance_fold(X, y, groups, train, test, pipe, n_repeats, seed,
                     n_spatial, n_features):
    """Pattern + block-permutation AUC drops for one LOSO fold."""
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    base, _ = _fold_auc(clone(pipe), Xtr, ytr, Xte, yte)
    fitted = clone(pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(Xtr, ytr)
    pattern = _haufe_pattern(fitted, Xtr)

    def _score(Xm):
        if hasattr(fitted[-1], "decision_function"):
            s = fitted.decision_function(Xm)
        else:
            s = fitted.predict_proba(Xm)[:, 1]
        return float(roc_auc_score(yte, s))

    feat_blocks, spat_blocks = _block_columns(n_spatial, n_features)
    rng = np.random.default_rng(seed)
    drops = {}
    if np.isfinite(base):
        for label, blocks in (("feature", feat_blocks), ("spatial", spat_blocks)):
            d = np.zeros(len(blocks))
            for bi, cols in enumerate(blocks.values()):
                vals = []
                for _ in range(n_repeats):
                    Xp = Xte.copy()
                    Xp[:, cols] = Xte[rng.permutation(len(Xte))][:, cols]
                    vals.append(base - _score(Xp))
                d[bi] = float(np.mean(vals))
            drops[label] = d
    return base, pattern, drops


def _fold_significance(folds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Wilcoxon vs 0 across folds for each column, plus BH-corrected q-values.

    ``folds`` is ``(n_folds, n_blocks)``; under LOSO each row is one held-out
    subject, so this asks "is this block's contribution consistently non-zero
    across held-out subjects". The training sets overlap heavily between folds,
    so treat these as consistency measures rather than strict independent
    tests — the subject-level AUC in the confirm stage is the inferential
    headline.
    """
    n_blocks = folds.shape[1]
    pvals = np.full(n_blocks, np.nan)
    for j in range(n_blocks):
        col = folds[:, j]
        col = col[np.isfinite(col)]
        if len(col) < 2 or np.allclose(col, 0):
            continue
        try:
            pvals[j] = float(stats.wilcoxon(col).pvalue)
        except ValueError:
            continue
    finite = np.isfinite(pvals)
    qvals = np.full(n_blocks, np.nan)
    if finite.any():
        qvals[finite] = apply_fdr_correction(pvals[finite], method="bh")
    return pvals, qvals


def stage_importance(cache: DataCache, feature_index, args, out_base: Path,
                     cell: Dict[str, str]) -> Path:
    """Haufe patterns + block permutation importance for one cell."""
    from joblib import Parallel, delayed

    bounds = _parse_inout(cell["inout"], args.inout_bounds)
    cache.load(bounds)
    X2d, red_names, feat_names, pca_k = build_cell(
        cache, cell["feature_set"], cell["reduction"], feature_index,
        cell["normalization"],
    )
    n_spatial, n_features = len(red_names), len(feat_names)
    est, _ = parse_estimator(cell["estimator"], seed=args.seed)
    pipe = make_pipeline(est, pca_k)
    y, groups = cache.y, cache.groups
    splits = list(LeaveOneGroupOut().split(X2d, y, groups))

    n_blocks = n_spatial + n_features
    logger.info(
        f"importance: cell={cell['config_id']}  dims={X2d.shape[1]}  "
        f"blocks={n_features} feature + {n_spatial} spatial = {n_blocks}, "
        f"{args.importance_n_repeats} repeat(s) x {len(splits)} folds"
    )
    t0 = time.time()
    out = Parallel(n_jobs=args.n_jobs)(
        delayed(_importance_fold)(X2d, y, groups, tr, te, pipe,
                                  args.importance_n_repeats, args.seed + i,
                                  n_spatial, n_features)
        for i, (tr, te) in enumerate(splits)
    )
    logger.info(f"importance: done [{time.time() - t0:.0f}s]")

    patterns = np.array([p for _, p, _ in out if p is not None])
    feat_drop = np.array([d["feature"] for _, _, d in out if "feature" in d])
    spat_drop = np.array([d["spatial"] for _, _, d in out if "spatial" in d])

    payload: Dict[str, np.ndarray] = {
        "spatial_names": np.asarray(red_names),
        "feature_names": np.asarray(feat_names),
        "fold_auc": np.asarray([b for b, _, _ in out]),
        "subjects": np.unique(groups),
    }
    if patterns.size:
        mean_pat = patterns.mean(axis=0)
        sd_pat = patterns.std(axis=0, ddof=1) if len(patterns) > 1 else np.zeros_like(mean_pat)
        payload["haufe_pattern"] = mean_pat.reshape(n_spatial, n_features)
        payload["haufe_pattern_sd"] = sd_pat.reshape(n_spatial, n_features)
        # Stability: |mean| / sd across folds — how reproducibly signed a cell is.
        with np.errstate(divide="ignore", invalid="ignore"):
            payload["haufe_stability"] = np.abs(
                np.divide(mean_pat, sd_pat, out=np.zeros_like(mean_pat),
                          where=sd_pat > 0)
            ).reshape(n_spatial, n_features)
        payload["haufe_pattern_folds"] = patterns.reshape(len(patterns), n_spatial,
                                                          n_features)
    if patterns.size and len(patterns) > 1:
        # Is each (spatial, feature) cell of the pattern consistently signed
        # across held-out subjects? BH across all cells of the pattern.
        pp, pq = _fold_significance(patterns)
        payload["haufe_pvalue"] = pp.reshape(n_spatial, n_features)
        payload["haufe_pvalue_fdr"] = pq.reshape(n_spatial, n_features)
    if feat_drop.size:
        payload["importance_by_feature"] = feat_drop.mean(axis=0)
        payload["importance_by_feature_sem"] = stats.sem(feat_drop, axis=0)
        payload["importance_by_feature_folds"] = feat_drop
        fp, fq = _fold_significance(feat_drop)
        payload["importance_by_feature_pvalue"] = fp
        payload["importance_by_feature_pvalue_fdr"] = fq
    if spat_drop.size:
        payload["importance_by_spatial"] = spat_drop.mean(axis=0)
        payload["importance_by_spatial_sem"] = stats.sem(spat_drop, axis=0)
        payload["importance_by_spatial_folds"] = spat_drop
        sp_, sq = _fold_significance(spat_drop)
        payload["importance_by_spatial_pvalue"] = sp_
        payload["importance_by_spatial_pvalue_fdr"] = sq

    out_path = out_base.with_name(out_base.name + "_importance.npz")
    np.savez_compressed(out_path, **payload)
    out_base.with_name(out_base.name + "_importance.json").write_text(json.dumps({
        "cell": cell,
        "n_spatial": n_spatial,
        "n_features": n_features,
        "importance_n_repeats": args.importance_n_repeats,
        "mean_fold_auc": float(np.nanmean(payload["fold_auc"])),
        "pattern_available": bool(patterns.size),
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
    }, indent=2, default=str))

    for label, names, key in (("features", feat_names, "importance_by_feature"),
                              ("spatial units", red_names, "importance_by_spatial")):
        if key not in payload:
            continue
        vals = payload[key]
        q = payload.get(f"{key}_pvalue_fdr")
        order = np.argsort(-vals)[:5]
        logger.info(
            f"importance: top {label} by AUC drop — " + ", ".join(
                f"{names[i]}={vals[i]:+.4f}"
                + (f" (q={q[i]:.3f})" if q is not None and np.isfinite(q[i]) else "")
                for i in order
            )
        )
        if q is not None:
            logger.info(
                f"importance: {int(np.nansum(q < 0.05))}/{len(q)} {label} "
                f"significant at q<0.05 (BH over blocks)"
            )
    logger.info(f"Saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Stage: nested-select (selection inside the outer CV loop)
# ---------------------------------------------------------------------------

def _inner_select_score(X, y, groups, train, est_spec, pca_k, inner_splits, seed
                        ) -> float:
    """Mean inner-CV AUC of one candidate, fit only on the outer training set."""
    Xtr, ytr, gtr = X[train], y[train], groups[train]
    n_inner = min(inner_splits, len(np.unique(gtr)))
    if n_inner < 2:
        return float("nan")
    est, _ = parse_estimator(est_spec, seed=seed)
    pipe = make_pipeline(est, pca_k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            pipe, Xtr, ytr, cv=GroupKFold(n_splits=n_inner), groups=gtr,
            scoring="roc_auc", n_jobs=1,
        )
    return float(np.mean(scores))


def _outer_refit_score(X, pca_k, est_spec, y, train, test, seed) -> float:
    """Refit the fold's chosen candidate on its training subjects, score the held-out one."""
    est, _ = parse_estimator(est_spec, seed=seed)
    auc, _ = _fold_auc(make_pipeline(est, pca_k), X[train], y[train],
                       X[test], y[test])
    return auc


def stage_nested_select(cache: DataCache, feature_index, args, out_base: Path) -> Path:
    """Leave-one-subject-out with the *whole cell choice* made inside each fold.

    The sweep picks its winning cell by looking at all 32 subjects, so the
    winner's own score is optimistic no matter how carefully the estimator
    hyperparameter is nested. Here every outer fold re-runs the candidate
    search on its 31 training subjects only (inner GroupKFold), picks a cell,
    refits, and is scored once on the held-out subject. The resulting AUC
    carries no selection leak at all, and the per-fold choices double as a
    stability check on the winner.

    Restricted to a single IN/OUT bound: different bounds retain different
    windows, so their held-out sets are not comparable within one outer fold.
    """
    from joblib import Parallel, delayed

    if len(args.inout_bounds) > 1:
        logger.warning(
            f"nested-select uses one IN/OUT bound; using {args.inout_bounds[0]} "
            f"and ignoring {args.inout_bounds[1:]} (different bounds retain "
            f"different windows, so their held-out sets are not comparable)."
        )
    bounds = args.inout_bounds[0]
    cache.load(bounds)
    y, groups = cache.y, cache.groups

    candidates = [
        (norm, fset, red, est)
        for norm in args.normalizations
        for fset in args.feature_sets
        for red in args.reductions
        for est in args.estimators
    ]

    # Build each (norm, feature_set, reduction) matrix once. All three are
    # label-independent, so materializing them outside the outer loop is not
    # leakage — only the estimator ever sees y.
    mats: Dict[Tuple[str, str, str], Tuple[np.ndarray, Optional[int]]] = {}
    for norm, fset, red in {(c[0], c[1], c[2]) for c in candidates}:
        try:
            X2d, _, _, pca_k = build_cell(cache, fset, red, feature_index, norm)
        except ValueError as exc:
            logger.warning(f"nested-select: dropping {norm}/{fset}/{red}: {exc}")
            continue
        mats[(norm, fset, red)] = (X2d, pca_k)
    candidates = [c for c in candidates if (c[0], c[1], c[2]) in mats]
    if not candidates:
        raise SystemExit("nested-select: no usable candidates")

    cand_ids = [f"{inout_bounds_to_string(bounds)}|{n}|{f}|{r}|{e}"
                for n, f, r, e in candidates]
    splits = list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    logger.info(
        f"nested-select: {len(candidates)} candidate(s) x {len(splits)} outer "
        f"fold(s), inner GroupKFold({args.inner_splits}) on the training subjects"
    )

    t0 = time.time()
    flat_scores = Parallel(n_jobs=args.n_jobs)(
        delayed(_inner_select_score)(
            mats[(c[0], c[1], c[2])][0], y, groups, train, c[3],
            mats[(c[0], c[1], c[2])][1], args.inner_splits, args.seed,
        )
        for train, _ in splits
        for c in candidates
    )
    inner = np.asarray(flat_scores).reshape(len(splits), len(candidates))
    logger.info(f"nested-select: inner search done [{time.time() - t0:.0f}s]")

    chosen = np.nanargmax(np.where(np.isfinite(inner), inner, -np.inf), axis=1)
    t0 = time.time()
    outer_auc = np.asarray(Parallel(n_jobs=args.n_jobs)(
        delayed(_outer_refit_score)(
            *mats[candidates[k][:3]], candidates[k][3], y, train, test, args.seed
        )
        for k, (train, test) in zip(chosen, splits)
    ))
    logger.info(f"nested-select: outer scoring done [{time.time() - t0:.0f}s]")

    summ = summarize(outer_auc)
    chosen_ids = [cand_ids[k] for k in chosen]
    counts: Dict[str, int] = {}
    for cid in chosen_ids:
        counts[cid] = counts.get(cid, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])

    out_path = out_base.with_name(out_base.name + "_nested-select.npz")
    np.savez_compressed(
        out_path,
        auc_per_subject=outer_auc,
        subjects=np.unique(groups),
        inner_scores=inner,
        candidate_ids=np.asarray(cand_ids),
        chosen_index=chosen,
        chosen_ids=np.asarray(chosen_ids),
    )
    out_base.with_name(out_base.name + "_nested-select.json").write_text(json.dumps({
        "inout_bounds": list(bounds),
        "n_candidates": len(candidates),
        "inner_splits": args.inner_splits,
        "summary": summ,
        "selection_counts": dict(top),
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
    }, indent=2, default=str))

    logger.info(
        f"nested-select: AUC={summ['mean_auc']:.4f} "
        f"[{summ.get('ci95_low', float('nan')):.4f}, "
        f"{summ.get('ci95_high', float('nan')):.4f}]  "
        f"n>0.5={summ['n_above_chance']}/{summ['n_subjects']}  "
        f"wilcoxon p={summ.get('wilcoxon_p', float('nan')):.2e}"
    )
    for cid, n in top[:3]:
        logger.info(f"nested-select: chosen in {n}/{len(splits)} folds — {cid}")
    logger.info(f"Saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_cell(args, csv_path: Path) -> Dict[str, str]:
    """The cell confirm/importance operate on: --cell if given, else the sweep winner."""
    if args.cell:
        parts = args.cell.split("|")
        if len(parts) != 5:
            raise SystemExit(
                "--cell must be 'inout|normalization|feature_set|reduction|estimator'"
            )
        cell = dict(zip(
            ["inout", "normalization", "feature_set", "reduction", "estimator"], parts
        ))
        cell["config_id"] = args.cell
        return cell
    if not csv_path.exists():
        raise SystemExit(
            f"No sweep results at {csv_path}. Run --stage sweep (then merge) "
            f"first, or pass --cell explicitly."
        )
    return read_winner(csv_path, args.select_by)


def _parse_bounds_arg(values: Sequence[str]) -> List[Tuple[int, int]]:
    out = []
    for v in values:
        lo, _, hi = v.partition(",")
        out.append((int(lo), int(hi)))
    return out


def build_output_base(out_dir: Path, space: str, trial_type: str,
                      n_events_window: int, inout_selection: str) -> Path:
    return out_dir / (
        f"space-{space}_type-{trial_type}_w{n_events_window}"
        f"{inout_selection_token(inout_selection)}_sweep"
    )


def main():
    p = argparse.ArgumentParser(
        description="Cross-subject generalization sweep for IN/OUT decoding."
    )
    p.add_argument("--stage", default="all", choices=STAGES)
    p.add_argument("--space", default="schaefer_400")
    p.add_argument("--trial-type", default="alltrials")
    p.add_argument("--n-events-window", type=int, default=8)
    p.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_FEATURE_SETS),
                   help="Feature-set shortcuts (psds, psds_corrected, fooof, "
                        "complexity, all). Their union is loaded once.")
    p.add_argument("--reductions", nargs="+", default=None,
                   help="flat | yeo7-mean | yeo7-meansd | hemi-yeo7 | global-mean "
                        "| pca-K | net-<Network>")
    p.add_argument("--estimators", nargs="+", default=None,
                   help="family[:k=v,...] — logistic, linearsvc, lda, hgb, rf")
    p.add_argument("--normalizations", nargs="+", default=list(DEFAULT_NORMALIZATIONS),
                   choices=list(NORMALIZATIONS))
    p.add_argument("--inout-bounds", nargs="+", default=None,
                   help="One or more 'low,high' pairs (default: config value).")
    p.add_argument("--select-by", default="mean_auc",
                   help="Sweep column the confirm/importance stages maximise.")
    p.add_argument("--cell", default=None,
                   help="Explicit config_id 'inout|norm|featureset|reduction|estimator' "
                        "for confirm/importance instead of the sweep winner.")
    p.add_argument("--inner-splits", type=int, default=5,
                   help="Inner GroupKFold splits for nested CV (confirm stage).")
    p.add_argument("--n-permutations", type=int, default=200,
                   help="Within-subject label permutations in the confirm stage.")
    p.add_argument("--importance-n-repeats", type=int, default=5)
    p.add_argument("--n-shards", type=int, default=1,
                   help="Split the grid into N shards (one SLURM array task each). "
                        "Each shard writes its own *_shard-KofN files; run "
                        "--stage merge afterwards to combine them.")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="Which shard this process evaluates (0-based).")
    p.add_argument("--nonlinear-max-dims", type=int, default=2000,
                   help=f"Skip {'/'.join(NONLINEAR_FAMILIES)} cells whose input "
                        f"exceeds this many dimensions (default 2000).")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-bad-trials", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--force", action="store_true",
                   help="Recompute sweep cells already present in the CSV.")
    args = p.parse_args()

    # nested-select re-runs the candidate search inside all 32 outer folds, so
    # a candidate costs 32x what it does in the sweep — it gets its own,
    # deliberately cheaper default candidate set.
    if args.stage == "nested-select":
        args.reductions = args.reductions or list(DEFAULT_SELECT_REDUCTIONS)
        args.estimators = args.estimators or list(DEFAULT_SELECT_ESTIMATORS)
    else:
        args.reductions = args.reductions or list(DEFAULT_REDUCTIONS)
        args.estimators = args.estimators or list(DEFAULT_ESTIMATORS)

    config = load_config(Path(args.config))
    inout_selection = str(
        config.get("analysis", {}).get("inout_selection", DEFAULT_INOUT_STRATEGY)
    )
    if args.inout_bounds:
        args.inout_bounds = _parse_bounds_arg(args.inout_bounds)
    else:
        args.inout_bounds = [tuple(config["analysis"]["inout_bounds"])]

    data_root = Path(config["paths"]["data_root"])
    out_dir = (Path(args.output_dir) if args.output_dir else
               data_root / config["paths"]["results"]
               / f"classification_{args.space}" / "group_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = build_output_base(out_dir, args.space, args.trial_type,
                                 args.n_events_window, inout_selection)

    # Resolve the target cell before loading anything when this process is not
    # sweeping: confirm/importance need one cell's features, not the union of
    # every swept feature set, which saves both the load time and the memory.
    cell: Optional[Dict[str, str]] = None
    csv_path = out_base.with_name(out_base.name + "_sweep.csv")
    if args.stage in ("confirm", "importance"):
        cell = _resolve_cell(args, csv_path)
        args.feature_sets = [cell["feature_set"]]
        args.inout_bounds = [_parse_inout(cell["inout"], args.inout_bounds)]

    # Union of every requested feature set, loaded once per bounds.
    feature_index: Dict[str, List[int]] = {}
    features: List[str] = []
    for fset in args.feature_sets:
        for f in expand_feature_set(fset, config):
            if f not in features:
                features.append(f)
    for fset in args.feature_sets:
        wanted = set(expand_feature_set(fset, config))
        feature_index[fset] = [i for i, f in enumerate(features) if f in wanted]

    logger.info("=" * 78)
    logger.info("CROSS-SUBJECT GENERALIZATION SWEEP (IN vs OUT)")
    logger.info("=" * 78)
    logger.info(f"stage={args.stage}  space={args.space}  trial_type={args.trial_type}")
    logger.info(f"features ({len(features)}): {features}")
    logger.info(f"feature_sets={args.feature_sets}")
    logger.info(f"reductions={args.reductions}")
    logger.info(f"estimators={args.estimators}")
    logger.info(f"normalizations={args.normalizations}  inout={args.inout_bounds}")
    logger.info(f"output -> {out_base}_*")
    logger.info("=" * 78)

    cache = DataCache(
        features=features, space=args.space, config=config,
        trial_type=args.trial_type, n_events_window=args.n_events_window,
        inout_selection=inout_selection, keep_bad_trials=args.keep_bad_trials,
        seed=args.seed,
    )

    if args.shard_idx >= args.n_shards or args.shard_idx < 0:
        raise SystemExit(
            f"--shard-idx={args.shard_idx} out of range for --n-shards={args.n_shards}"
        )

    if args.stage in ("sweep", "all"):
        shard_csv = stage_sweep(cache, feature_index, args, out_base)
        if args.n_shards == 1:
            csv_path = shard_csv
        elif args.stage == "all":
            raise SystemExit(
                "--stage all with --n-shards > 1 would pick a winner from one "
                "shard only. Run --stage sweep per shard, then --stage merge, "
                "then --stage confirm / importance."
            )

    if args.stage == "merge":
        csv_path = stage_merge(out_base)

    if args.stage == "nested-select":
        stage_nested_select(cache, feature_index, args, out_base)
        return

    if args.stage in ("confirm", "importance", "all"):
        if cell is None:
            cell = _resolve_cell(args, csv_path)
        if cell["feature_set"] not in feature_index:
            raise SystemExit(
                f"Winning cell uses feature_set={cell['feature_set']!r}, which is "
                f"not among --feature-sets {args.feature_sets}."
            )

        if args.stage in ("confirm", "all"):
            stage_confirm(cache, feature_index, args, out_base, cell)
        if args.stage in ("importance", "all"):
            stage_importance(cache, feature_index, args, out_base, cell)


if __name__ == "__main__":
    main()
