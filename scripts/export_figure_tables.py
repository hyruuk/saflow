"""Export per-figure CSV tables from result files for the manuscript.

Tables are exhaustive (more inclusive than the cherry-picked figure panels):
they contain every feature, trial-type and analysis-level for which a result
file exists. The figure panels select a subset of these for the headline plot;
the manuscript text and supplementary tables can read the full picture from
these CSVs.

Outputs (under reports/tables/, override with --out-dir):
  - fig2_behavior_per_subject.csv, fig2_behavior_summary.csv
  - fig3_stats_per_roi.csv      (long: per feature × trial_type × ROI)
  - fig3_stats_summary.csv      (one row per feature × trial_type)
  - fig3_classif_per_roi.csv    (long: per feature × trial_type × level × ROI)
  - fig3_classif_summary.csv    (one row per feature × trial_type × level)
  - fig4_*.csv, fig5_*.csv      (stub warnings until results files exist)

Stats and classification values use the same keys as
code/visualization/stats_classif_panel.py — `pvals_cluster_perm` and
`pvals_tmax` respectively — so n_significant counts in this CSV will match
the `n=X sig` overlays printed on the panel figure.

Usage:
    python scripts/export_figure_tables.py --fig all
    python scripts/export_figure_tables.py --fig 3 --space schaefer_400
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region-name parsing (Schaefer 400, 7-network ordering)
# ---------------------------------------------------------------------------

_SCHAEFER_RE = re.compile(
    r"7Networks_(?P<hemi>LH|RH)_(?P<network>[A-Za-z]+)_(?P<region>.+)-(?P<hemi_suffix>lh|rh)$"
)


def _parse_schaefer_label(name: str) -> Tuple[str, str, str]:
    """Return (network, hemisphere, sub-region) for a Schaefer ROI label.

    Background/medial-wall labels return ("Background", hemi, "MedialWall").
    """
    if name.startswith("Background"):
        hemi = "lh" if name.endswith("-lh") else "rh"
        return "Background", hemi, "MedialWall"
    m = _SCHAEFER_RE.match(name)
    if not m:
        return "Unknown", "unknown", name
    return m.group("network"), m.group("hemi").lower(), m.group("region")


def _load_schaefer_400_roi_names() -> List[str]:
    import mne
    sd = mne.get_config("SUBJECTS_DIR")
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc="Schaefer2018_400Parcels_7Networks_order",
        subjects_dir=sd,
        verbose=False,
    )
    return sorted(label.name for label in labels)


# ---------------------------------------------------------------------------
# Fig 2 — behavior
# ---------------------------------------------------------------------------

def export_fig2(out_dir: Path, inout_bounds: Tuple[int, int] = (25, 75)) -> None:
    from scipy.stats import ttest_rel

    from code.utils.config import load_config
    from code.visualization.plot_behavior import (
        get_behavior_dict,
        replace_missing_values,
    )

    config = load_config(None)
    data_root = Path(config["paths"]["data_root"])
    logs_dir = data_root / "sourcedata" / "behav"
    subjects = config["bids"]["subjects"]
    runs = config["bids"]["task_runs"]
    filt_config = config["behavioral"]["vtc"]["filter"]
    files_list = os.listdir(logs_dir)

    df = get_behavior_dict(
        files_list, subjects, runs, logs_dir, tuple(inout_bounds), filt_config
    )

    numeric_cols = [
        "lapse_rate", "omission_error_rate",
        "commission_error", "correct_omission",
        "omission_error", "correct_commission", "n_rare",
        "RT_preCE", "RT_preOC", "RT_preCC",
    ]
    subj_avg = (
        df.groupby(["subject", "cond"])[numeric_cols].mean().reset_index()
    )

    per_subject = subj_avg.rename(columns={
        "cond": "condition",
        "RT_preCE": "RT_pre_lapse",
        "RT_preOC": "RT_pre_CO",
        "RT_preCC": "RT_pre_baseline",
    })
    per_subject_path = out_dir / "fig2_behavior_per_subject.csv"
    per_subject.to_csv(per_subject_path, index=False)
    logger.info("wrote %s (%d rows)", per_subject_path, len(per_subject))

    df_corr = replace_missing_values(df.copy(), "RT_preCE")
    rt_cols = ["RT_preCE", "RT_preOC", "RT_preCC"]
    subj_avg_rt = (
        df_corr.groupby(["subject", "cond"])[rt_cols].mean().reset_index()
    )

    rows = []
    metric_specs = [
        ("lapse_rate",          "lapse_rate",          subj_avg),
        ("omission_error_rate", "omission_error_rate", subj_avg),
        ("RT_pre_lapse",        "RT_preCE",            subj_avg_rt),
        ("RT_pre_CO",           "RT_preOC",            subj_avg_rt),
        ("RT_pre_baseline",     "RT_preCC",            subj_avg_rt),
    ]
    for out_name, col, frame in metric_specs:
        wide = frame.pivot(index="subject", columns="cond", values=col).dropna()
        n = len(wide)
        t, p = ttest_rel(wide["IN"], wide["OUT"])
        diff = wide["IN"] - wide["OUT"]
        cohens_d = float(diff.mean() / diff.std(ddof=1))
        rows.append({
            "metric":   out_name,
            "n":        int(n),
            "mean_IN":  float(wide["IN"].mean()),
            "sd_IN":    float(wide["IN"].std(ddof=1)),
            "mean_OUT": float(wide["OUT"].mean()),
            "sd_OUT":   float(wide["OUT"].std(ddof=1)),
            "t":        float(t),
            "df":       int(n - 1),
            "p":        float(p),
            "cohens_d": cohens_d,
        })
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "fig2_behavior_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("wrote %s (%d rows)", summary_path, len(summary))


# ---------------------------------------------------------------------------
# Fig 3 — feature catalogue
# ---------------------------------------------------------------------------

PSD_BANDS = ("delta", "theta", "alpha", "lobeta", "hibeta", "gamma1", "gamma2", "gamma3")
FOOOF_PARAMS = ("exponent", "offset", "r_squared")
COMPLEXITY_FEATURES = (
    "complexity_lzc_median",
    "complexity_entropy_permutation",
    "complexity_entropy_spectral",
    "complexity_entropy_svd",
    "complexity_fractal_petrosian",
)


def _enumerate_features() -> List[Tuple[str, str, str]]:
    """Return list of (feature_id, feature_group, display_name)."""
    out: List[Tuple[str, str, str]] = []
    band_display = {
        "delta": "Delta (2-4 Hz)",
        "theta": "Theta (4-8 Hz)",
        "alpha": "Alpha (8-12 Hz)",
        "lobeta": "Low Beta (12-20 Hz)",
        "hibeta": "High Beta (20-30 Hz)",
        "gamma1": "Gamma 1 (30-60 Hz)",
        "gamma2": "Gamma 2 (60-90 Hz)",
        "gamma3": "Gamma 3 (90-120 Hz)",
    }
    for b in PSD_BANDS:
        out.append((f"psd_{b}", "raw_psd", band_display[b]))
    for p in FOOOF_PARAMS:
        out.append((f"fooof_{p}",
                    "fooof",
                    {"exponent": "FOOOF exponent",
                     "offset": "FOOOF offset",
                     "r_squared": "FOOOF R^2"}[p]))
    for b in PSD_BANDS:
        out.append((f"psd_corrected_{b}",
                    "corrected_psd",
                    f"Corrected {band_display[b]}"))
    for c in COMPLEXITY_FEATURES:
        out.append((c, "complexity", c.replace("complexity_", "").replace("_", " ").title()))
    return out


# ---------------------------------------------------------------------------
# Fig 3 — stats (per-ROI long + per-feature summary)
# ---------------------------------------------------------------------------

STATS_KEYS = {
    "tvals": "tvals",
    "contrast": "contrast",
    "pvals_uncorrected": "p_uncorrected",
    "pvals_cluster_perm": "p_cluster_perm",
    "effectsize_cohens_d_paired": "cohens_d",
    "effectsize_hedges_g_paired": "hedges_g",
    "effectsize_eta_squared": "eta_squared",
}


def _stats_path(stats_dir: Path, feature: str, trial_type: str) -> Path:
    return stats_dir / (
        f"feature-{feature}_inout-2575_test-paired_ttest_"
        f"level-average_type-{trial_type}_results.npz"
    )


def _load_stats_arrays(path: Path) -> dict:
    out = {}
    with np.load(path, allow_pickle=True) as npz:
        for src_key, dst_key in STATS_KEYS.items():
            if src_key in npz.files:
                out[dst_key] = np.asarray(npz[src_key]).flatten()
            else:
                out[dst_key] = None
    return out


def _classif_path(clf_dir: Path, feature: str, space: str,
                  trial_type: str, level: str) -> Path:
    cv = "logo" if level == "epoch" else "group"
    return clf_dir / (
        f"feature-{feature}_space-{space}_inout-2575_clf-logistic_"
        f"cv-{cv}_mode-univariate_level-{level}_type-{trial_type}_scores.npz"
    )


CLASSIF_KEYS = {
    "observed": "observed",
    "metrics_roc_auc": "AUC",
    "metrics_balanced_accuracy": "balanced_accuracy",
    "metrics_accuracy": "accuracy",
    "pvals_uncorrected": "p_uncorrected",
    "pvals_tmax": "p_tmax",
    "pvals_fdr_bh": "p_fdr_bh",
    "pvals_bonferroni": "p_bonferroni",
}


def _load_classif_arrays(path: Path) -> dict:
    out = {}
    with np.load(path, allow_pickle=True) as npz:
        for src_key, dst_key in CLASSIF_KEYS.items():
            if src_key in npz.files:
                out[dst_key] = np.asarray(npz[src_key]).flatten()
            else:
                out[dst_key] = None
    return out


def export_fig3(
    results_root: Path,
    out_dir: Path,
    space: str = "schaefer_400",
    trial_types: Iterable[str] = ("alltrials", "correct", "lapse"),
    levels: Iterable[str] = ("average", "epoch"),
    alpha: float = 0.05,
) -> None:
    stats_dir = results_root / f"statistics_{space}"
    clf_dir = results_root / f"classification_{space}" / "group"
    roi_names = _load_schaefer_400_roi_names()
    n_expected = len(roi_names)

    parsed = [_parse_schaefer_label(n) for n in roi_names]
    networks = [p[0] for p in parsed]
    hemis = [p[1] for p in parsed]
    subregions = [p[2] for p in parsed]

    features = _enumerate_features()

    stats_rows: List[dict] = []
    stats_summary: List[dict] = []
    classif_rows: List[dict] = []
    classif_summary: List[dict] = []

    for feature, group, display in features:
        for trial_type in trial_types:
            stats_fp = _stats_path(stats_dir, feature, trial_type)
            if stats_fp.exists():
                arrs = _load_stats_arrays(stats_fp)
                if arrs["tvals"] is not None:
                    n = arrs["tvals"].shape[0]
                    if n != n_expected:
                        logger.warning(
                            "stats %s/%s: %d ROIs (expected %d) — ROI alignment may be off",
                            feature, trial_type, n, n_expected,
                        )
                    rn = roi_names[:n]
                    rn_net = networks[:n]
                    rn_hemi = hemis[:n]
                    rn_sub = subregions[:n]
                    sig_clust = (arrs["p_cluster_perm"] < alpha).astype(bool) \
                        if arrs["p_cluster_perm"] is not None else np.zeros(n, bool)
                    sig_unc = (arrs["p_uncorrected"] < alpha).astype(bool) \
                        if arrs["p_uncorrected"] is not None else np.zeros(n, bool)
                    for i in range(n):
                        row = {
                            "feature": feature,
                            "feature_group": group,
                            "display": display,
                            "trial_type": trial_type,
                            "roi_name": rn[i],
                            "network": rn_net[i],
                            "hemisphere": rn_hemi[i],
                            "subregion": rn_sub[i],
                        }
                        for k in ("tvals", "contrast", "p_uncorrected",
                                  "p_cluster_perm", "cohens_d", "hedges_g",
                                  "eta_squared"):
                            arr = arrs[k]
                            out_name = "t" if k == "tvals" else k
                            row[out_name] = float(arr[i]) if arr is not None else np.nan
                        row["sig_cluster_perm"] = bool(sig_clust[i])
                        row["sig_uncorrected"] = bool(sig_unc[i])
                        stats_rows.append(row)

                    valid = np.where(np.isfinite(arrs["tvals"]))[0]
                    if valid.size:
                        order = valid[np.argsort(-np.abs(arrs["tvals"][valid]))]
                    else:
                        order = np.array([], int)
                    pos = [i for i in order if arrs["tvals"][i] > 0][:3]
                    neg = [i for i in order if arrs["tvals"][i] < 0][:3]
                    stats_summary.append({
                        "feature": feature,
                        "feature_group": group,
                        "display": display,
                        "trial_type": trial_type,
                        "n_significant_cluster_perm": int(sig_clust.sum()),
                        "n_significant_uncorrected": int(sig_unc.sum()),
                        "n_positive_t_sig": int((sig_clust & (arrs["tvals"] > 0)).sum()),
                        "n_negative_t_sig": int((sig_clust & (arrs["tvals"] < 0)).sum()),
                        "max_abs_t": float(np.nanmax(np.abs(arrs["tvals"]))),
                        "max_pos_t": float(np.nanmax(arrs["tvals"])),
                        "max_neg_t": float(np.nanmin(arrs["tvals"])),
                        "min_p_cluster_perm": float(np.nanmin(arrs["p_cluster_perm"]))
                            if arrs["p_cluster_perm"] is not None else np.nan,
                        "min_p_uncorrected": float(np.nanmin(arrs["p_uncorrected"]))
                            if arrs["p_uncorrected"] is not None else np.nan,
                        "mean_cohens_d": float(np.nanmean(arrs["cohens_d"]))
                            if arrs["cohens_d"] is not None else np.nan,
                        "top3_positive_t":
                            "; ".join(f"{rn[i]} (t={arrs['tvals'][i]:+.2f}, "
                                      f"p_cp={arrs['p_cluster_perm'][i]:.3g})"
                                      for i in pos),
                        "top3_negative_t":
                            "; ".join(f"{rn[i]} (t={arrs['tvals'][i]:+.2f}, "
                                      f"p_cp={arrs['p_cluster_perm'][i]:.3g})"
                                      for i in neg),
                    })

            for level in levels:
                clf_fp = _classif_path(clf_dir, feature, space, trial_type, level)
                if not clf_fp.exists():
                    continue
                carrs = _load_classif_arrays(clf_fp)
                if carrs["AUC"] is None:
                    continue
                n = carrs["AUC"].shape[0]
                if n != n_expected:
                    logger.warning(
                        "classif %s/%s/%s: %d ROIs (expected %d)",
                        feature, trial_type, level, n, n_expected,
                    )
                rn = roi_names[:n]
                rn_net = networks[:n]
                rn_hemi = hemis[:n]
                rn_sub = subregions[:n]
                sig_tmax = (carrs["p_tmax"] < alpha).astype(bool) \
                    if carrs["p_tmax"] is not None else np.zeros(n, bool)
                sig_fdr = (carrs["p_fdr_bh"] < alpha).astype(bool) \
                    if carrs["p_fdr_bh"] is not None else np.zeros(n, bool)
                for i in range(n):
                    row = {
                        "feature": feature,
                        "feature_group": group,
                        "display": display,
                        "trial_type": trial_type,
                        "analysis_level": level,
                        "cv": "logo" if level == "epoch" else "group",
                        "roi_name": rn[i],
                        "network": rn_net[i],
                        "hemisphere": rn_hemi[i],
                        "subregion": rn_sub[i],
                    }
                    for k in ("AUC", "balanced_accuracy", "accuracy",
                              "p_uncorrected", "p_tmax", "p_fdr_bh", "p_bonferroni"):
                        arr = carrs[k]
                        row[k] = float(arr[i]) if arr is not None else np.nan
                    row["sig_tmax"] = bool(sig_tmax[i])
                    row["sig_fdr_bh"] = bool(sig_fdr[i])
                    classif_rows.append(row)

                valid = np.where(np.isfinite(carrs["AUC"]))[0]
                if valid.size:
                    order = valid[np.argsort(-carrs["AUC"][valid])]
                else:
                    order = np.array([], int)
                top = list(order[:3])
                classif_summary.append({
                    "feature": feature,
                    "feature_group": group,
                    "display": display,
                    "trial_type": trial_type,
                    "analysis_level": level,
                    "cv": "logo" if level == "epoch" else "group",
                    "n_significant_tmax": int(sig_tmax.sum()),
                    "n_significant_fdr_bh": int(sig_fdr.sum()),
                    "mean_AUC": float(np.nanmean(carrs["AUC"])),
                    "max_AUC": float(np.nanmax(carrs["AUC"])),
                    "min_p_tmax": float(np.nanmin(carrs["p_tmax"]))
                        if carrs["p_tmax"] is not None else np.nan,
                    "top3_parcels_by_AUC":
                        "; ".join(f"{rn[i]} (AUC={carrs['AUC'][i]:.3f}, "
                                  f"p_tmax={carrs['p_tmax'][i]:.3g})"
                                  for i in top),
                })

    stats_long_df = pd.DataFrame(stats_rows)
    stats_long_path = out_dir / "fig3_stats_per_roi.csv"
    stats_long_df.to_csv(stats_long_path, index=False)
    logger.info("wrote %s (%d rows)", stats_long_path, len(stats_long_df))

    stats_summary_df = pd.DataFrame(stats_summary)
    stats_summary_path = out_dir / "fig3_stats_summary.csv"
    stats_summary_df.to_csv(stats_summary_path, index=False)
    logger.info("wrote %s (%d rows)", stats_summary_path, len(stats_summary_df))

    classif_long_df = pd.DataFrame(classif_rows)
    classif_long_path = out_dir / "fig3_classif_per_roi.csv"
    classif_long_df.to_csv(classif_long_path, index=False)
    logger.info("wrote %s (%d rows)", classif_long_path, len(classif_long_df))

    classif_summary_df = pd.DataFrame(classif_summary)
    classif_summary_path = out_dir / "fig3_classif_summary.csv"
    classif_summary_df.to_csv(classif_summary_path, index=False)
    logger.info("wrote %s (%d rows)", classif_summary_path, len(classif_summary_df))


# ---------------------------------------------------------------------------
# Fig 4 / Fig 5 — stubs
# ---------------------------------------------------------------------------

def export_fig4(out_dir: Path) -> None:
    logger.warning(
        "Fig 4 (multifeature classification) — results files not yet produced; "
        "no CSV written. Re-run after running the multifeature pipeline."
    )


def export_fig5(out_dir: Path) -> None:
    logger.warning(
        "Fig 5 (network analysis with lapses vs correct) — results files not yet "
        "produced; no CSV written. Re-run after running the network panel pipeline."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig", choices=["2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--space", default="schaefer_400")
    parser.add_argument("--trial-types", nargs="+",
                        default=["alltrials", "correct", "lapse"])
    parser.add_argument("--levels", nargs="+",
                        default=["average", "epoch"])
    parser.add_argument("--results-root", default=None,
                        help="Override results path; defaults to paths.results in config.yaml")
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory (default: reports/tables)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--inout-bounds", type=int, nargs=2, default=[25, 75])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from code.utils.config import load_config
    config = load_config(None)

    results_root = (
        Path(args.results_root) if args.results_root
        else Path(config["paths"]["results"])
    )
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else PROJECT_ROOT / "reports" / "tables"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", out_dir)

    figs = ["2", "3", "4", "5"] if args.fig == "all" else [args.fig]
    for f in figs:
        if f == "2":
            export_fig2(out_dir, inout_bounds=tuple(args.inout_bounds))
        elif f == "3":
            export_fig3(
                results_root, out_dir,
                space=args.space,
                trial_types=tuple(args.trial_types),
                levels=tuple(args.levels),
                alpha=args.alpha,
            )
        elif f == "4":
            export_fig4(out_dir)
        elif f == "5":
            export_fig5(out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
