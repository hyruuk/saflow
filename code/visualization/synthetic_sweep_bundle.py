"""Generate a synthetic multifeature-sweep output set for panel prototyping.

Writes exactly the files ``code.classification.run_multifeature_sweep`` writes
— ``*_sweep.csv``, ``*_sweep-folds.npz``, ``*_confirm.{npz,json}``,
``*_importance.{npz,json}``, ``*_nested-select.{npz,json}`` — so the panel can
be built and reviewed before the real cluster run lands.

Field names and the output basename are imported from the sweep module rather
than restated, so a schema change there breaks this loudly instead of silently
producing a bundle the panel mis-reads.

The numbers are shaped to look like the fooof pilot: network-level reductions
around AUC 0.57-0.60, the flat 400-parcel vector near 0.52, tree ensembles
below the linear models, and a handful of networks carrying the signal.

Usage:
    python -m code.visualization.synthetic_sweep_bundle --out /tmp/synth_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from code.classification.run_multifeature_sweep import (
    DEFAULT_ESTIMATORS,
    DEFAULT_FEATURE_SETS,
    DEFAULT_NORMALIZATIONS,
    DEFAULT_REDUCTIONS,
    DEFAULT_SELECT_ESTIMATORS,
    DEFAULT_SELECT_REDUCTIONS,
    SWEEP_FIELDS,
    build_output_base,
    summarize,
)
from code.classification.run_classification import expand_feature_set
from code.statistics.corrections import apply_fdr_correction
from code.utils.config import load_config
from code.utils.yeo_networks import network_order

N_SUBJECTS = 32

# Plausible mean AUC per reduction, and the dimensionality each implies for a
# 23-feature stack on Schaefer-400. Network-level wins; flat drowns.
REDUCTION_AUC = {
    "yeo7-mean": 0.598,
    "yeo7-meansd": 0.596,
    "hemi-yeo7": 0.591,
    "pca-50": 0.567,
    "pca-20": 0.559,
    "global-mean": 0.548,
    "flat": 0.521,
}
REDUCTION_SPATIAL = {
    "yeo7-mean": 7, "yeo7-meansd": 14, "hemi-yeo7": 14,
    "global-mean": 1, "pca-20": 20, "pca-50": 50, "flat": 400,
}
# Additive offsets: linear models on top, boosting behind, weak C penalised.
ESTIMATOR_OFFSET = {
    "logistic:C=0.0001": -0.012, "logistic:C=0.001": -0.002,
    "logistic:C=0.01": 0.0, "logistic:C=0.1": -0.003,
    "logistic:C=1": -0.008, "logistic:C=10": -0.014,
    "lda:shrinkage=auto": 0.002, "linearsvc:C=0.01": -0.001,
    "linearsvc:C=1": -0.010, "hgb": -0.028, "rf": -0.032,
}
FEATURE_SET_OFFSET = {
    "all": 0.0, "psds_corrected": -0.004, "fooof": -0.012,
    "psds": -0.016, "complexity": -0.030,
}
NORMALIZATION_OFFSET = {"zscore": 0.0, "rank": 0.002, "none": -0.020}


def _feature_names(config: Dict, feature_set: str) -> List[str]:
    return expand_feature_set(feature_set, config)


def _cell_auc(bounds, norm, fset, red, est, rng) -> float:
    """Mean AUC for one grid cell, plus a little irreducible noise."""
    base = REDUCTION_AUC.get(red, 0.55)
    base += ESTIMATOR_OFFSET.get(est, -0.01)
    base += FEATURE_SET_OFFSET.get(fset, -0.01)
    base += NORMALIZATION_OFFSET.get(norm, 0.0)
    # The 10/90 split is a cleaner contrast: fewer windows, larger effect.
    if bounds == (10, 90):
        base += 0.022
    return float(base + rng.normal(0, 0.0035))


def _subject_aucs(mean_auc: float, rng, spread: float = 0.07) -> np.ndarray:
    return np.clip(rng.normal(mean_auc, spread, N_SUBJECTS), 0.15, 0.95)


def write_synthetic(out_dir: Path, space: str, trial_type: str, seed: int,
                    config: Dict) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = build_output_base(out_dir, space, trial_type, 8, "strict")

    bounds_list = [(25, 75), (10, 90)]
    subjects = np.arange(N_SUBJECTS)

    # ---- sweep grid -------------------------------------------------------
    rows: List[Dict] = []
    fold_store: Dict[str, np.ndarray] = {}
    for bounds in bounds_list:
        for norm in DEFAULT_NORMALIZATIONS:
            for fset in DEFAULT_FEATURE_SETS:
                n_feat = len(_feature_names(config, fset))
                for red in DEFAULT_REDUCTIONS:
                    n_spatial = REDUCTION_SPATIAL[red]
                    dims = (n_spatial if red.startswith("pca-")
                            else n_spatial * n_feat)
                    for est in DEFAULT_ESTIMATORS:
                        # Mirror the real skip rule for tree ensembles.
                        if est.split(":")[0] in ("hgb", "rf") and dims > 2000:
                            continue
                        mean_auc = _cell_auc(bounds, norm, fset, red, est, rng)
                        per_subj = _subject_aucs(mean_auc, rng)
                        summ = summarize(per_subj)
                        cid = f"{bounds[0]}{bounds[1]}|{norm}|{fset}|{red}|{est}"
                        rows.append({
                            "config_id": cid,
                            "inout": f"{bounds[0]}{bounds[1]}",
                            "normalization": norm, "feature_set": fset,
                            "reduction": red, "estimator": est,
                            "n_input_dims": dims,
                            "n_reduced_spatial": n_spatial,
                            "n_features": n_feat,
                            "mean_bacc": float(0.5 + (summ["mean_auc"] - 0.5) * 0.75),
                            "seconds": round(float(rng.uniform(0.2, 240)), 1),
                            **{k: summ.get(k) for k in
                               ("mean_auc", "median_auc", "std_auc", "ci95_low",
                                "ci95_high", "n_above_chance", "n_subjects",
                                "wilcoxon_p")},
                        })
                        fold_store[f"auc/{cid}"] = per_subj

    ranked = sorted(rows, key=lambda r: -r["mean_auc"])
    q = apply_fdr_correction(np.array([r["wilcoxon_p"] for r in ranked]), method="bh")
    for r, qi in zip(ranked, q):
        r["wilcoxon_p_fdr"] = float(qi)

    csv_path = out_base.with_name(out_base.name + "_sweep.csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SWEEP_FIELDS)
        w.writeheader()
        for r in ranked:
            w.writerow({k: r.get(k) for k in SWEEP_FIELDS})
    np.savez_compressed(
        out_base.with_name(out_base.name + "_sweep-folds.npz"),
        subjects=subjects, **fold_store,
    )

    # ---- confirm ----------------------------------------------------------
    winner = ranked[0]
    win_features = _feature_names(config, winner["feature_set"])
    networks = list(network_order(7))
    nested_auc = fold_store[f"auc/{winner['config_id']}"] - rng.normal(0.002, 0.004,
                                                                      N_SUBJECTS)
    fixed_auc = fold_store[f"auc/{winner['config_id']}"]
    perm = rng.normal(0.5, 0.012, 200)
    observed_fixed = float(np.mean(fixed_auc))
    np.savez_compressed(
        out_base.with_name(out_base.name + "_confirm.npz"),
        nested_auc_per_subject=nested_auc,
        fixed_auc_per_subject=fixed_auc,
        perm_mean_auc=perm,
        subjects=subjects,
        best_params=np.asarray([json.dumps({"clf__C": 0.01})] * N_SUBJECTS),
        spatial_names=np.asarray(networks),
        feature_names=np.asarray(win_features),
    )
    out_base.with_name(out_base.name + "_confirm.json").write_text(json.dumps({
        "cell": {k: winner[k] for k in
                 ("config_id", "inout", "normalization", "feature_set",
                  "reduction", "estimator")},
        "nested": summarize(nested_auc),
        "observed_fixed_hyperparams": observed_fixed,
        "permutation_pvalue": float((np.sum(perm >= observed_fixed) + 1) / (len(perm) + 1)),
        "n_permutations": int(len(perm)),
        "n_input_dims": int(winner["n_input_dims"]),
        "inner_splits": 5,
        "synthetic": True,
        "timestamp": datetime.now().isoformat(),
    }, indent=2, default=str))

    # ---- importance -------------------------------------------------------
    n_sp, n_ft = len(networks), len(win_features)
    # Give a few (network, feature) cells a real signed effect so the panel has
    # something to show: alpha/low-beta in DorsAttn + SomMot, exponent in Default.
    pattern_folds = rng.normal(0, 0.010, (N_SUBJECTS, n_sp, n_ft))

    def _bump(net: str, feat_match: str, size: float) -> None:
        if net not in networks:
            return
        i = networks.index(net)
        for j, f in enumerate(win_features):
            if feat_match in f:
                pattern_folds[:, i, j] += size

    _bump("DorsAttn", "alpha", -0.055)
    _bump("SomMot", "alpha", -0.048)
    _bump("SomMot", "lobeta", -0.040)
    _bump("Default", "exponent", 0.043)
    _bump("Cont", "offset", 0.031)
    _bump("Limbic", "gamma", 0.026)

    feat_folds = np.abs(pattern_folds).mean(axis=1) * 1.4 + rng.normal(0, 0.004,
                                                                      (N_SUBJECTS, n_ft))
    spat_folds = np.abs(pattern_folds).mean(axis=2) * 1.4 + rng.normal(0, 0.004,
                                                                      (N_SUBJECTS, n_sp))

    def _sig(folds: np.ndarray):
        from scipy import stats as _st
        p = np.array([_st.wilcoxon(folds[:, j]).pvalue for j in range(folds.shape[1])])
        return p, apply_fdr_correction(p, method="bh")

    flat_pat = pattern_folds.reshape(N_SUBJECTS, -1)
    pat_p, pat_q = _sig(flat_pat)
    ft_p, ft_q = _sig(feat_folds)
    sp_p, sp_q = _sig(spat_folds)
    mean_pat = pattern_folds.mean(axis=0)
    sd_pat = pattern_folds.std(axis=0, ddof=1)

    np.savez_compressed(
        out_base.with_name(out_base.name + "_importance.npz"),
        spatial_names=np.asarray(networks),
        feature_names=np.asarray(win_features),
        fold_auc=fixed_auc,
        subjects=subjects,
        haufe_pattern=mean_pat,
        haufe_pattern_sd=sd_pat,
        haufe_stability=np.abs(mean_pat / np.where(sd_pat > 0, sd_pat, 1.0)),
        haufe_pattern_folds=pattern_folds,
        haufe_pvalue=pat_p.reshape(n_sp, n_ft),
        haufe_pvalue_fdr=pat_q.reshape(n_sp, n_ft),
        importance_by_feature=feat_folds.mean(axis=0),
        importance_by_feature_sem=feat_folds.std(axis=0, ddof=1) / np.sqrt(N_SUBJECTS),
        importance_by_feature_folds=feat_folds,
        importance_by_feature_pvalue=ft_p,
        importance_by_feature_pvalue_fdr=ft_q,
        importance_by_spatial=spat_folds.mean(axis=0),
        importance_by_spatial_sem=spat_folds.std(axis=0, ddof=1) / np.sqrt(N_SUBJECTS),
        importance_by_spatial_folds=spat_folds,
        importance_by_spatial_pvalue=sp_p,
        importance_by_spatial_pvalue_fdr=sp_q,
    )
    out_base.with_name(out_base.name + "_importance.json").write_text(json.dumps({
        "cell": {k: winner[k] for k in ("config_id", "reduction", "estimator")},
        "n_spatial": n_sp, "n_features": n_ft,
        "importance_n_repeats": 5,
        "mean_fold_auc": float(np.mean(fixed_auc)),
        "pattern_available": True,
        "synthetic": True,
    }, indent=2, default=str))

    # ---- nested-select ----------------------------------------------------
    cand_ids, cand_mean = [], []
    for norm in DEFAULT_NORMALIZATIONS:
        for fset in DEFAULT_FEATURE_SETS:
            for red in DEFAULT_SELECT_REDUCTIONS:
                for est in DEFAULT_SELECT_ESTIMATORS:
                    cand_ids.append(f"2575|{norm}|{fset}|{red}|{est}")
                    cand_mean.append(_cell_auc((25, 75), norm, fset, red, est, rng))
    cand_mean = np.asarray(cand_mean)
    inner = cand_mean[None, :] + rng.normal(0, 0.006, (N_SUBJECTS, len(cand_ids)))
    chosen = inner.argmax(axis=1)
    ns_auc = _subject_aucs(float(cand_mean.max()) - 0.003, rng)
    chosen_ids = [cand_ids[k] for k in chosen]
    counts: Dict[str, int] = {}
    for cid in chosen_ids:
        counts[cid] = counts.get(cid, 0) + 1

    np.savez_compressed(
        out_base.with_name(out_base.name + "_nested-select.npz"),
        auc_per_subject=ns_auc, subjects=subjects, inner_scores=inner,
        candidate_ids=np.asarray(cand_ids), chosen_index=chosen,
        chosen_ids=np.asarray(chosen_ids),
    )
    out_base.with_name(out_base.name + "_nested-select.json").write_text(json.dumps({
        "inout_bounds": [25, 75],
        "n_candidates": len(cand_ids),
        "inner_splits": 5,
        "summary": summarize(ns_auc),
        "selection_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "synthetic": True,
    }, indent=2, default=str))

    print(f"Synthetic sweep bundle -> {out_base}_*")
    print(f"  {len(ranked)} grid cells; winner: {winner['config_id']} "
          f"(AUC={winner['mean_auc']:.4f})")
    return out_base


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Directory to write the bundle into.")
    p.add_argument("--space", default="schaefer_400")
    p.add_argument("--trial-type", default="alltrials")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    write_synthetic(Path(args.out), args.space, args.trial_type, args.seed,
                    load_config(Path(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
