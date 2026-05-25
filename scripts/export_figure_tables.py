"""Export per-figure CSV tables from result files for the manuscript.

For each figure in the manuscript, produces a long CSV (one row per
ROI/sensor x feature) and a short summary CSV (one row per feature).
Tables are written to reports/tables/.

Tables for Fig 4 (multifeature) and Fig 5 (networks) are stubbed and
emit a warning until their result files are produced.

Usage:
    python scripts/export_figure_tables.py --fig all
    python scripts/export_figure_tables.py --fig 3 --space schaefer_400
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


SCHAEFER_400_FEATURES: Tuple[Tuple[str, str, str], ...] = (
    ("psd_theta",            "Theta (4-8 Hz)",           "raw"),
    ("psd_alpha",            "Alpha (8-12 Hz)",          "raw"),
    ("psd_lobeta",           "Low Beta (12-20 Hz)",      "raw"),
    ("psd_hibeta",           "High Beta (20-30 Hz)",     "raw"),
    ("psd_gamma1",           "Gamma 1 (30-60 Hz)",       "raw"),
    ("psd_gamma2",           "Gamma 2 (60-90 Hz)",       "raw"),
    ("psd_gamma3",           "Gamma 3 (90-120 Hz)",      "raw"),
    ("fooof_exponent",       "FOOOF exponent",           "fooof"),
    ("fooof_offset",         "FOOOF offset",             "fooof"),
    ("fooof_r_squared",      "FOOOF R^2",                "fooof"),
    ("psd_corrected_theta",  "Corrected Theta",          "corrected"),
    ("psd_corrected_alpha",  "Corrected Alpha",          "corrected"),
    ("psd_corrected_lobeta", "Corrected Low Beta",       "corrected"),
    ("psd_corrected_hibeta", "Corrected High Beta",      "corrected"),
    ("psd_corrected_gamma1", "Corrected Gamma 1",        "corrected"),
    ("psd_corrected_gamma2", "Corrected Gamma 2",        "corrected"),
    ("psd_corrected_gamma3", "Corrected Gamma 3",        "corrected"),
)


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
# Fig 3 — stats + classification (Schaefer 400)
# ---------------------------------------------------------------------------

def _load_schaefer_400_roi_names() -> list:
    import mne
    sd = mne.get_config("SUBJECTS_DIR")
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc="Schaefer2018_400Parcels_7Networks_order",
        subjects_dir=sd,
        verbose=False,
    )
    return sorted(label.name for label in labels)


def _stats_path(stats_dir: Path, feature: str, trial_type: str) -> Path:
    return stats_dir / (
        f"feature-{feature}_inout-2575_test-paired_ttest_"
        f"level-average_type-{trial_type}_results.npz"
    )


def _classif_path(clf_dir: Path, feature: str, space: str, trial_type: str) -> Path:
    return clf_dir / (
        f"feature-{feature}_space-{space}_inout-2575_clf-logistic_"
        f"cv-logo_mode-univariate_level-epoch_type-{trial_type}_scores.npz"
    )


def export_fig3(
    results_root: Path,
    out_dir: Path,
    space: str = "schaefer_400",
    trial_type: str = "alltrials",
    alpha: float = 0.05,
) -> None:
    stats_dir = results_root / f"statistics_{space}"
    clf_dir = results_root / f"classification_{space}" / "group"
    roi_names = _load_schaefer_400_roi_names()
    n_rois_expected = len(roi_names)

    long_rows = []
    summary_rows = []

    for feature, display, group in SCHAEFER_400_FEATURES:
        stats_fp = _stats_path(stats_dir, feature, trial_type)
        clf_fp = _classif_path(clf_dir, feature, space, trial_type)
        if not stats_fp.exists():
            logger.warning("missing stats: %s", stats_fp.name)
            continue
        if not clf_fp.exists():
            logger.warning("missing classif: %s", clf_fp.name)
            continue

        sd = np.load(stats_fp, allow_pickle=True)
        cd = np.load(clf_fp, allow_pickle=True)

        tvals = np.asarray(sd["tvals"]).squeeze()
        contrast = np.asarray(sd["contrast"]).squeeze()
        pvals_unc = np.asarray(sd["pvals_uncorrected"]).squeeze()
        pvals_cp = np.asarray(sd["pvals_cluster_perm"]).squeeze()
        cohens = np.asarray(sd["effectsize_cohens_d_paired"]).squeeze()
        auc = np.asarray(cd["metrics_roc_auc"])
        p_tmax = np.asarray(cd["pvals_tmax"])

        n = tvals.shape[0]
        if n != n_rois_expected:
            logger.warning(
                "feature %s: %d ROIs (expected %d) — names may be misaligned",
                feature, n, n_rois_expected,
            )
        rn = roi_names[:n]

        sig_stats = pvals_cp < alpha
        sig_clf = p_tmax < alpha

        for i in range(n):
            long_rows.append({
                "feature":       feature,
                "feature_group": group,
                "display":       display,
                "roi_name":      rn[i],
                "contrast":      float(contrast[i]),
                "t":             float(tvals[i]),
                "p_uncorrected": float(pvals_unc[i]),
                "p_cluster_perm": float(pvals_cp[i]),
                "cohens_d":      float(cohens[i]),
                "AUC":           float(auc[i]),
                "p_tmax":        float(p_tmax[i]),
                "sig_stats":     bool(sig_stats[i]),
                "sig_classif":   bool(sig_clf[i]),
            })

        valid_t = np.where(np.isfinite(tvals))[0]
        idx_t = valid_t[np.argsort(-np.abs(tvals[valid_t]))[:3]] if valid_t.size else np.array([], dtype=int)
        valid_auc = np.where(np.isfinite(auc))[0]
        idx_auc = valid_auc[np.argsort(-auc[valid_auc])[:3]] if valid_auc.size else np.array([], dtype=int)

        summary_rows.append({
            "feature":               feature,
            "feature_group":         group,
            "display":               display,
            "n_significant_stats":   int(sig_stats.sum()),
            "max_abs_t":             float(np.nanmax(np.abs(tvals))),
            "min_p_cluster_perm":    float(np.nanmin(pvals_cp)),
            "mean_cohens_d":         float(np.nanmean(cohens)),
            "top3_rois_by_abs_t":    "; ".join(
                f"{rn[i]} (t={tvals[i]:+.2f})" for i in idx_t
            ),
            "n_significant_classif": int(sig_clf.sum()),
            "mean_AUC":              float(np.nanmean(auc)),
            "max_AUC":               float(np.nanmax(auc)),
            "top3_rois_by_AUC":      "; ".join(
                f"{rn[i]} (AUC={auc[i]:.3f})" for i in idx_auc
            ),
        })

    long_df = pd.DataFrame(long_rows)
    long_path = out_dir / "fig3_stats_classif_per_roi.csv"
    long_df.to_csv(long_path, index=False)
    logger.info("wrote %s (%d rows)", long_path, len(long_df))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "fig3_stats_classif_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("wrote %s (%d rows)", summary_path, len(summary_df))


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
    parser.add_argument("--trial-type", default="alltrials")
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
                trial_type=args.trial_type,
                alpha=args.alpha,
            )
        elif f == "4":
            export_fig4(out_dir)
        elif f == "5":
            export_fig5(out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
