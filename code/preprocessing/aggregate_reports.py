"""Aggregate preprocessing reports for subject-level and dataset-level summaries.

This module reads per-run *_params.json metadata files (no raw MEG data loading)
and generates:
1. Subject-level reports: quality summary across all runs for one subject
2. Dataset-level reports: quality summary across all subjects

Usage:
    python -m code.preprocessing.aggregate_reports -s 04           # subject report
    python -m code.preprocessing.aggregate_reports --dataset        # dataset report
    python -m code.preprocessing.aggregate_reports -s 04 --dataset  # both
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from code.utils.config import load_config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def collect_run_params(subject: str, derivatives_root: Path, task_runs: list) -> dict:
    """Load *_params.json for each run.

    Args:
        subject: Subject ID (e.g. "04").
        derivatives_root: Path to derivatives directory.
        task_runs: List of run IDs (e.g. ["02", "03", ...]).

    Returns:
        Dict mapping run ID to params dict (or None if missing/invalid).
    """
    run_params = {}
    for run in task_runs:
        params_path = (
            derivatives_root / "preprocessed" / f"sub-{subject}" / "meg"
            / f"sub-{subject}_task-gradCPT_run-{run}_proc-clean_params.json"
        )
        if not params_path.exists():
            logger.warning(f"Missing params file: {params_path}")
            run_params[run] = None
            continue
        try:
            with open(params_path) as f:
                run_params[run] = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Invalid params file {params_path}: {e}")
            run_params[run] = None
    return run_params


def collect_all_subject_params(
    derivatives_root: Path, subjects: list, task_runs: list
) -> dict:
    """Collect params for all subjects.

    Args:
        derivatives_root: Path to derivatives directory.
        subjects: List of subject IDs.
        task_runs: List of run IDs.

    Returns:
        Dict mapping subject ID to {run: params_dict_or_None}.
    """
    all_params = {}
    for subject in subjects:
        all_params[subject] = collect_run_params(subject, derivatives_root, task_runs)
    return all_params


# =============================================================================
# Metric Aggregation
# =============================================================================

def _safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current


def aggregate_subject_metrics(run_params: dict) -> dict:
    """Aggregate preprocessing metrics across runs for one subject.

    Args:
        run_params: Dict mapping run ID to params dict (or None).

    Returns:
        Dict with per_run metrics, summary stats, outlier info.
    """
    per_run = []
    for run, params in run_params.items():
        if params is None:
            per_run.append({
                "run": run,
                "complete": False,
            })
            continue

        p = params.get("parameters", params)

        ar1 = p.get("autoreject_first_pass", {})
        ar2 = p.get("autoreject_second_pass", {})
        threshold = p.get("threshold_detection", {})
        ica = p.get("ica", {})
        three_way = p.get("three_way_comparison", {})
        event_counts = p.get("event_counts", {})
        isi_stats = p.get("isi_statistics", {})
        retention = p.get("retention", {})
        pre_ar2 = p.get("pre_ar2_filter", {})

        ar1_pct = ar1.get("pct_bad")
        ar2_pct = ar2.get("pct_bad") if ar2 else None
        threshold_pct = threshold.get("pct_bad")
        n_total = ar1.get("n_total_epochs")

        ecg_components = ica.get("ecg_components", [])
        eog_components = ica.get("eog_components", [])
        n_ecg = len(ecg_components) if isinstance(ecg_components, list) else 0
        n_eog = len(eog_components) if isinstance(eog_components, list) else 0

        # ICA scores (available from future runs, None for old data)
        ecg_scores = _safe_get(ica, "ecg_scores")
        eog_scores = _safe_get(ica, "eog_scores")
        ecg_threshold = _safe_get(ica, "ecg_threshold")
        eog_threshold = _safe_get(ica, "eog_threshold")
        ecg_forced = _safe_get(ica, "ecg_forced")
        eog_forced = _safe_get(ica, "eog_forced")

        # Compute max scores if available
        ecg_max_score = None
        eog_max_score = None
        if ecg_scores is not None and isinstance(ecg_scores, list) and len(ecg_scores) > 0:
            ecg_max_score = float(max(abs(s) for s in ecg_scores))
        if eog_scores is not None and isinstance(eog_scores, list) and len(eog_scores) > 0:
            eog_max_score = float(max(abs(s) for s in eog_scores))

        rescue_rate = _safe_get(
            three_way, "ica_effectiveness", "rescue_rate_pct"
        )
        pairwise_kappa = _safe_get(three_way, "pairwise_kappa")

        # New fields: event counts, ISI, proper retention
        n_freq = event_counts.get("Freq")
        n_rare = event_counts.get("Rare")
        n_resp = event_counts.get("Resp")
        n_stimulus_epochs = _safe_get(retention, "n_total_stimulus_epochs", default=n_total)
        proper_retention = _safe_get(retention, "retention_rate_pct")
        isi_mean = _safe_get(isi_stats, "mean")
        isi_std = _safe_get(isi_stats, "std")

        per_run.append({
            "run": run,
            "complete": True,
            "ar1_pct_bad": ar1_pct,
            "ar1_n_bad": ar1.get("n_bad_epochs"),
            "ar2_pct_bad": ar2_pct,
            "ar2_n_bad": ar2.get("n_bad_epochs") if ar2 else None,
            "threshold_pct_bad": threshold_pct,
            "threshold_n_bad": threshold.get("n_bad_epochs"),
            "n_total_epochs": n_total,
            "n_stimulus_epochs": n_stimulus_epochs,
            "n_freq": n_freq,
            "n_rare": n_rare,
            "n_resp": n_resp,
            "n_ecg_ics": n_ecg,
            "n_eog_ics": n_eog,
            "ecg_components": ecg_components,
            "eog_components": eog_components,
            "ecg_scores": ecg_scores,
            "eog_scores": eog_scores,
            "ecg_max_score": ecg_max_score,
            "eog_max_score": eog_max_score,
            "ecg_threshold": ecg_threshold,
            "eog_threshold": eog_threshold,
            "ecg_forced": ecg_forced,
            "eog_forced": eog_forced,
            "rescue_rate": rescue_rate,
            "pairwise_kappa": pairwise_kappa,
            "retention_rate": proper_retention,
            "isi_mean": isi_mean,
            "isi_std": isi_std,
            "pre_ar2_outliers": _safe_get(pre_ar2, "n_outliers"),
        })

    # Compute summary statistics from complete runs
    complete_runs = [r for r in per_run if r.get("complete")]

    def _stats(values):
        arr = np.array([v for v in values if v is not None], dtype=float)
        if len(arr) == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def _stats_cv(values):
        """Stats with coefficient of variation."""
        arr = np.array([v for v in values if v is not None], dtype=float)
        if len(arr) == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "cv": None}
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return {
            "mean": mean,
            "std": std,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "cv": float(std / mean) if mean > 0 else None,
        }

    ar1_pcts = [r["ar1_pct_bad"] for r in complete_runs]
    ar2_pcts = [r["ar2_pct_bad"] for r in complete_runs]
    threshold_pcts = [r["threshold_pct_bad"] for r in complete_runs]
    rescue_rates = [r["rescue_rate"] for r in complete_runs]
    n_totals = [r["n_total_epochs"] for r in complete_runs]
    isi_means = [r["isi_mean"] for r in complete_runs]
    n_stimulus = [r["n_stimulus_epochs"] for r in complete_runs]

    # Retention rate: use proper retention from params if available, else fall back
    proper_retentions = [r["retention_rate"] for r in complete_runs]
    has_proper_retention = any(v is not None for v in proper_retentions)
    if has_proper_retention:
        retention_rates = proper_retentions
    else:
        # Fallback: 100 - ar1_pct_bad
        retention_rates = [
            100.0 - v for v in ar1_pcts if v is not None
        ]

    summary = {
        "ar1_pct_bad": _stats(ar1_pcts),
        "ar2_pct_bad": _stats(ar2_pcts),
        "threshold_pct_bad": _stats(threshold_pcts),
        "rescue_rate": _stats(rescue_rates),
        "retention_rate": _stats(retention_rates),
        "isi_mean": _stats(isi_means),
        "n_stimulus_epochs": _stats_cv(n_stimulus),
    }

    total_epochs = sum(v for v in n_totals if v is not None)

    # Outlier detection: runs with >30% bad epochs or 0 ICA components
    outlier_runs = []
    for r in complete_runs:
        reasons = []
        if r["ar1_pct_bad"] is not None and r["ar1_pct_bad"] > 30:
            reasons.append(f"AR1 bad rate {r['ar1_pct_bad']:.1f}% > 30%")
        if r["n_ecg_ics"] == 0:
            reasons.append("0 ECG components removed")
        if r["n_eog_ics"] == 0:
            reasons.append("0 EOG components removed")
        if reasons:
            outlier_runs.append({"run": r["run"], "reasons": reasons})

    return {
        "per_run": per_run,
        "summary": summary,
        "total_epochs": total_epochs,
        "outlier_runs": outlier_runs,
        "n_runs_complete": len(complete_runs),
        "n_runs_total": len(per_run),
    }


def aggregate_dataset_metrics(all_subject_params: dict) -> dict:
    """Aggregate preprocessing metrics across all subjects.

    Args:
        all_subject_params: Dict mapping subject ID to {run: params_dict_or_None}.

    Returns:
        Dict with per-subject aggregates, dataset-wide stats, and outlier subjects.
    """
    per_subject = {}
    for subject, run_params in all_subject_params.items():
        metrics = aggregate_subject_metrics(run_params)
        per_subject[subject] = metrics

    # Collect per-subject summary values for dataset-wide stats
    mean_ar1_pcts = []
    mean_retention = []
    mean_rescue_rates = []
    total_epochs_list = []

    for subject, metrics in per_subject.items():
        s = metrics["summary"]
        if s["ar1_pct_bad"]["mean"] is not None:
            mean_ar1_pcts.append(s["ar1_pct_bad"]["mean"])
        if s["retention_rate"]["mean"] is not None:
            mean_retention.append(s["retention_rate"]["mean"])
        if s["rescue_rate"]["mean"] is not None:
            mean_rescue_rates.append(s["rescue_rate"]["mean"])
        total_epochs_list.append(metrics["total_epochs"])

    def _dataset_stats(values):
        arr = np.array(values, dtype=float)
        if len(arr) == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "median": None}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    dataset_stats = {
        "mean_ar1_pct_bad": _dataset_stats(mean_ar1_pcts),
        "mean_retention_rate": _dataset_stats(mean_retention),
        "mean_rescue_rate": _dataset_stats(mean_rescue_rates),
        "total_epochs": _dataset_stats(total_epochs_list),
    }

    # MAD-based outlier detection across subjects
    outlier_subjects = []
    subjects_ordered = list(per_subject.keys())

    def _mad_outliers(values, subject_ids, metric_name, threshold=3.0):
        """Flag subjects > threshold MAD from median."""
        arr = np.array(values, dtype=float)
        if len(arr) < 3:
            return []
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            return []
        outliers = []
        for val, subj in zip(arr, subject_ids):
            deviation = abs(val - median) / mad
            if deviation > threshold:
                outliers.append({
                    "subject": subj,
                    "metric": metric_name,
                    "value": float(val),
                    "median": float(median),
                    "mad": float(mad),
                    "deviation_mad": float(deviation),
                })
        return outliers

    # MAD outliers on key metrics
    if mean_ar1_pcts:
        subj_ids_ar1 = [
            s for s in subjects_ordered
            if per_subject[s]["summary"]["ar1_pct_bad"]["mean"] is not None
        ]
        outlier_subjects.extend(
            _mad_outliers(mean_ar1_pcts, subj_ids_ar1, "mean_ar1_pct_bad")
        )

    if mean_retention:
        subj_ids_ret = [
            s for s in subjects_ordered
            if per_subject[s]["summary"]["retention_rate"]["mean"] is not None
        ]
        outlier_subjects.extend(
            _mad_outliers(mean_retention, subj_ids_ret, "mean_retention_rate")
        )

    # Absolute outliers
    for subject, metrics in per_subject.items():
        s = metrics["summary"]
        reasons = []
        mean_ar1 = s["ar1_pct_bad"]["mean"]
        mean_ret = s["retention_rate"]["mean"]
        if mean_ar1 is not None and mean_ar1 > 30:
            reasons.append(f"mean bad epoch rate {mean_ar1:.1f}% > 30%")
        if mean_ret is not None and mean_ret < 60:
            reasons.append(f"mean retention {mean_ret:.1f}% < 60%")
        if reasons:
            outlier_subjects.append({
                "subject": subject,
                "metric": "absolute_threshold",
                "reasons": reasons,
            })

    # Epoch count consistency check: flag subjects with varying per-run counts
    epoch_count_flags = []
    for subject, metrics in per_subject.items():
        complete_runs = [r for r in metrics["per_run"] if r.get("complete")]
        stim_counts = [r.get("n_stimulus_epochs") for r in complete_runs
                       if r.get("n_stimulus_epochs") is not None]
        if len(stim_counts) >= 2:
            unique_counts = set(stim_counts)
            if len(unique_counts) > 1:
                epoch_count_flags.append({
                    "subject": subject,
                    "metric": "epoch_count_inconsistency",
                    "counts": stim_counts,
                    "unique": sorted(unique_counts),
                })
                outlier_subjects.append({
                    "subject": subject,
                    "metric": "epoch_count_inconsistency",
                    "reasons": [f"per-run epoch counts vary: {sorted(unique_counts)}"],
                })

    return {
        "per_subject": per_subject,
        "dataset_stats": dataset_stats,
        "outlier_subjects": outlier_subjects,
        "epoch_count_flags": epoch_count_flags,
        "n_subjects": len(per_subject),
        "n_subjects_complete": sum(
            1 for m in per_subject.values() if m["n_runs_complete"] > 0
        ),
    }


# =============================================================================
# Figure Generation
# =============================================================================

def _create_subject_epoch_bar_chart(run_metrics: list, subject: str) -> plt.Figure:
    """Grouped bar chart: bad epoch counts per run, grouped by method.

    Args:
        run_metrics: List of per-run metric dicts from aggregate_subject_metrics.
        subject: Subject ID for title.

    Returns:
        Matplotlib figure.
    """
    complete_runs = [r for r in run_metrics if r.get("complete")]
    if not complete_runs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No complete runs", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    runs = [r["run"] for r in complete_runs]
    ar1_bad = [r.get("ar1_n_bad", 0) or 0 for r in complete_runs]
    ar2_bad = [r.get("ar2_n_bad", 0) or 0 for r in complete_runs]
    threshold_bad = [r.get("threshold_n_bad", 0) or 0 for r in complete_runs]

    x = np.arange(len(runs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.5), 5))
    bars1 = ax.bar(x - width, ar1_bad, width, label="AR1 (pre-ICA)", color="#1f77b4")
    bars2 = ax.bar(x, ar2_bad, width, label="AR2 (post-ICA)", color="#ff7f0e")
    bars3 = ax.bar(x + width, threshold_bad, width, label="Threshold", color="#2ca02c")

    ax.set_xlabel("Run")
    ax.set_ylabel("Bad Epoch Count")
    ax.set_title(f"Bad Epochs per Run - sub-{subject}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"run-{r}" for r in runs])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def _create_dataset_distribution_figure(dataset_metrics: dict) -> plt.Figure:
    """4-panel figure with histograms across subjects.

    Panels:
    - Histogram: AR1 bad epoch rates across subjects
    - Histogram: retention rates across subjects
    - Histogram: ICA rescue rates (if available)
    - Scatter: AR1 bad % vs AR2 bad % per subject

    Args:
        dataset_metrics: Output of aggregate_dataset_metrics.

    Returns:
        Matplotlib figure.
    """
    per_subject = dataset_metrics["per_subject"]

    # Collect per-subject mean values
    subjects = []
    ar1_means = []
    ar2_means = []
    retention_means = []
    rescue_means = []

    for subj, metrics in per_subject.items():
        s = metrics["summary"]
        subjects.append(subj)
        ar1_means.append(s["ar1_pct_bad"]["mean"])
        ar2_means.append(s["ar2_pct_bad"]["mean"])
        retention_means.append(s["retention_rate"]["mean"])
        rescue_means.append(s["rescue_rate"]["mean"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: AR1 bad epoch rate distribution
    ax = axes[0, 0]
    valid_ar1 = [v for v in ar1_means if v is not None]
    if valid_ar1:
        ax.hist(valid_ar1, bins=min(15, max(5, len(valid_ar1) // 2)),
                color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.axvline(30, color="red", linestyle="--", linewidth=1.5, label="30% threshold")
        ax.legend()
    ax.set_xlabel("Mean AR1 Bad Epoch Rate (%)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("AR1 Bad Epoch Rates Across Subjects")

    # Panel 2: Retention rate distribution
    ax = axes[0, 1]
    valid_ret = [v for v in retention_means if v is not None]
    if valid_ret:
        ax.hist(valid_ret, bins=min(15, max(5, len(valid_ret) // 2)),
                color="#2ca02c", edgecolor="black", alpha=0.7)
        ax.axvline(60, color="red", linestyle="--", linewidth=1.5, label="60% threshold")
        ax.legend()
    ax.set_xlabel("Mean Retention Rate (%)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Retention Rates Across Subjects")

    # Panel 3: ICA rescue rate distribution
    ax = axes[1, 0]
    valid_rescue = [v for v in rescue_means if v is not None]
    if valid_rescue:
        ax.hist(valid_rescue, bins=min(15, max(5, len(valid_rescue) // 2)),
                color="#9467bd", edgecolor="black", alpha=0.7)
    else:
        ax.text(0.5, 0.5, "No rescue rate data\n(requires second AR pass)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Mean ICA Rescue Rate (%)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("ICA Rescue Rates Across Subjects")

    # Panel 4: AR1 vs AR2 scatter
    ax = axes[1, 1]
    scatter_ar1 = []
    scatter_ar2 = []
    scatter_labels = []
    for subj, a1, a2 in zip(subjects, ar1_means, ar2_means):
        if a1 is not None and a2 is not None:
            scatter_ar1.append(a1)
            scatter_ar2.append(a2)
            scatter_labels.append(subj)

    if scatter_ar1:
        ax.scatter(scatter_ar1, scatter_ar2, c="#ff7f0e", edgecolors="black",
                   alpha=0.7, s=50)
        # Add diagonal reference line
        lim_max = max(max(scatter_ar1), max(scatter_ar2)) * 1.1
        ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, label="x=y")
        # Label outlier points
        for x, y, label in zip(scatter_ar1, scatter_ar2, scatter_labels):
            if x > 30 or y > 30:
                ax.annotate(f"sub-{label}", (x, y), fontsize=8,
                            xytext=(5, 5), textcoords="offset points")
        ax.legend()
    ax.set_xlabel("Mean AR1 Bad %")
    ax.set_ylabel("Mean AR2 Bad %")
    ax.set_title("AR1 vs AR2 Bad Epoch Rate Per Subject")

    fig.suptitle("Dataset-Level Preprocessing Quality", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# =============================================================================
# Report Generation
# =============================================================================

def _shared_css_js() -> str:
    """Return shared CSS and JS for interactive reports."""
    return """
<style>
    .sortable th { cursor: pointer; user-select: none; position: relative; }
    .sortable th:hover { background-color: #e0e0e0; }
    .sortable th::after { content: ' \\2195'; font-size: 10px; color: #999; }
    .sortable th.sort-asc::after { content: ' \\2191'; color: #333; }
    .sortable th.sort-desc::after { content: ' \\2193'; color: #333; }
    .collapsible-header { cursor: pointer; user-select: none; }
    .collapsible-header::before { content: '\\25BC '; font-size: 10px; }
    .collapsible-header.collapsed::before { content: '\\25B6 '; }
    .collapsible-content { overflow: hidden; transition: max-height 0.3s ease; }
    .collapsible-content.collapsed { max-height: 0 !important; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
             font-size: 11px; font-weight: bold; }
    .badge-green { background-color: #d4edda; color: #155724; }
    .badge-yellow { background-color: #fff3cd; color: #856404; }
    .badge-red { background-color: #f8d7da; color: #721c24; }
    .breadcrumb { margin-bottom: 15px; font-size: 14px; }
    .breadcrumb a { color: #0066cc; text-decoration: none; }
    .breadcrumb a:hover { text-decoration: underline; }
    .forced-text { color: #856404; font-weight: bold; }
    .above-thresh-text { color: #155724; }
    table { border-collapse: collapse; }
    td, th { padding: 4px 8px; }
</style>
<script>
function makeSortable(table) {
    var headers = table.querySelectorAll('th');
    headers.forEach(function(header, index) {
        header.addEventListener('click', function() {
            var rows = Array.from(table.querySelectorAll('tbody tr'));
            if (rows.length === 0) {
                rows = Array.from(table.querySelectorAll('tr'));
                rows.shift(); // remove header row
            }
            var ascending = !header.classList.contains('sort-asc');
            headers.forEach(function(h) { h.classList.remove('sort-asc', 'sort-desc'); });
            header.classList.add(ascending ? 'sort-asc' : 'sort-desc');
            rows.sort(function(a, b) {
                var cellA = a.cells[index] ? a.cells[index].textContent.trim() : '';
                var cellB = b.cells[index] ? b.cells[index].textContent.trim() : '';
                var numA = parseFloat(cellA.replace('%', '').replace('s', ''));
                var numB = parseFloat(cellB.replace('%', '').replace('s', ''));
                if (!isNaN(numA) && !isNaN(numB)) {
                    return ascending ? numA - numB : numB - numA;
                }
                return ascending ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
            });
            var parent = rows[0].parentNode;
            rows.forEach(function(row) { parent.appendChild(row); });
        });
    });
}
function makeCollapsible(header) {
    header.addEventListener('click', function() {
        this.classList.toggle('collapsed');
        var content = this.nextElementSibling;
        if (content) content.classList.toggle('collapsed');
    });
}
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('table.sortable').forEach(makeSortable);
    document.querySelectorAll('.collapsible-header').forEach(makeCollapsible);
});
</script>
"""


def _retention_badge(rate) -> str:
    """Return a colored badge for retention rate."""
    if rate is None:
        return "-"
    if rate >= 85:
        return f'<span class="badge badge-green">{rate:.1f}%</span>'
    elif rate >= 60:
        return f'<span class="badge badge-yellow">{rate:.1f}%</span>'
    else:
        return f'<span class="badge badge-red">{rate:.1f}%</span>'


def _ica_forced_badge(forced) -> str:
    """Return a colored indicator for forced ICA selection."""
    if forced is None:
        return "N/A"
    if forced:
        return '<span class="forced-text">Yes</span>'
    return '<span class="above-thresh-text">No</span>'


def _build_subject_html(subject: str, metrics: dict, config: dict) -> str:
    """Build HTML content for subject-level report.

    Args:
        subject: Subject ID.
        metrics: Output of aggregate_subject_metrics.
        config: Configuration dict.

    Returns:
        HTML string.
    """
    per_run = metrics["per_run"]
    summary = metrics["summary"]
    outlier_runs = metrics["outlier_runs"]
    n_complete = metrics["n_runs_complete"]
    n_total = metrics["n_runs_total"]

    # Shared CSS/JS
    html = _shared_css_js()

    # Breadcrumb navigation
    html += """
    <div class="breadcrumb">
        <a href="../group_preprocessing-summary.html">&larr; Dataset Report</a>
    </div>
    """

    # Overview section
    html += f"""
    <h2>Subject Report: sub-{subject}</h2>
    <p><b>Timestamp:</b> {datetime.now().isoformat()}</p>
    <p><b>Runs:</b> {n_complete}/{n_total} complete</p>
    <p><b>Total epochs:</b> {metrics['total_epochs']}</p>
    """

    # Per-run metrics table
    html += """
    <h3 class="collapsible-header">Per-Run Metrics</h3>
    <div class="collapsible-content">
    <table border="1" class="sortable" style="border-collapse: collapse; width: 100%; font-size: 12px;">
    <tr style="background-color: #f0f0f0;">
        <th>Run</th><th>Stim Epochs</th>
        <th>Freq</th><th>Rare</th><th>Resp</th>
        <th>ISI Mean</th>
        <th>AR1 Bad</th><th>AR1 %</th>
        <th>AR2 Bad</th><th>AR2 %</th>
        <th>Threshold %</th>
        <th>Retention %</th>
        <th>ECG ICs</th><th>EOG ICs</th>
        <th>Rescue %</th>
        <th>Report</th>
    </tr>
    """
    # Build per-run report link lookup
    derivatives_root = Path(config["paths"]["data_root"]) / config["paths"]["derivatives"]

    for r in per_run:
        if not r.get("complete"):
            html += f'<tr style="background-color: #fff3cd;"><td>{r["run"]}</td>'
            html += '<td colspan="15"><i>Missing or incomplete</i></td></tr>'
            continue

        # Highlight runs with >30% bad
        row_style = ""
        if r.get("ar1_pct_bad") is not None and r["ar1_pct_bad"] > 30:
            row_style = ' style="background-color: #f8d7da;"'

        def _fmt(val, fmt=".1f"):
            return f"{val:{fmt}}" if val is not None else "-"

        def _fmt_int(val):
            return str(int(val)) if val is not None else "-"

        rescue = _fmt(r.get("rescue_rate"), ".1f")
        if rescue != "-":
            rescue += "%"

        retention_badge = _retention_badge(r.get("retention_rate"))

        isi_mean = _fmt(r.get("isi_mean"), ".3f")
        if isi_mean != "-":
            isi_mean += "s"

        # Per-run report link (relative path from subject report dir)
        run_id = r["run"]
        report_name = f"sub-{subject}_task-gradCPT_run-{run_id}_meg_desc-report.html"
        report_abs_path = derivatives_root / "preprocessed" / f"sub-{subject}" / "meg" / report_name
        report_rel = f"meg/{report_name}"
        if report_abs_path.exists():
            report_link = f'<a href="{report_rel}">View</a>'
        else:
            report_link = '<i>N/A</i>'

        html += f"""<tr{row_style}>
            <td>{r['run']}</td>
            <td>{_fmt_int(r.get('n_stimulus_epochs'))}</td>
            <td>{_fmt_int(r.get('n_freq'))}</td>
            <td>{_fmt_int(r.get('n_rare'))}</td>
            <td>{_fmt_int(r.get('n_resp'))}</td>
            <td>{isi_mean}</td>
            <td>{_fmt_int(r.get('ar1_n_bad'))}</td>
            <td>{_fmt(r.get('ar1_pct_bad'))}%</td>
            <td>{_fmt_int(r.get('ar2_n_bad'))}</td>
            <td>{_fmt(r.get('ar2_pct_bad'))}%</td>
            <td>{_fmt(r.get('threshold_pct_bad'))}%</td>
            <td>{retention_badge}</td>
            <td>{r.get('n_ecg_ics', '-')}</td>
            <td>{r.get('n_eog_ics', '-')}</td>
            <td>{rescue}</td>
            <td>{report_link}</td>
        </tr>"""

    html += "</table></div>"

    # ISI statistics section
    isi_summary = summary.get("isi_mean", {})
    if isi_summary.get("mean") is not None:
        html += f"""
        <h3>ISI Statistics (across runs)</h3>
        <p>Mean ISI: {isi_summary['mean']:.3f}s (std={isi_summary.get('std', 0):.3f}s,
        range: {isi_summary['min']:.3f}s - {isi_summary['max']:.3f}s)</p>
        """

    # Enhanced ICA summary table
    html += """
    <h3 class="collapsible-header">ICA Component Summary</h3>
    <div class="collapsible-content">
    <table border="1" class="sortable" style="border-collapse: collapse; width: 90%; font-size: 13px;">
    <tr style="background-color: #f0f0f0;">
        <th>Run</th><th>ECG ICs</th><th>ECG Max Score</th><th>ECG Forced?</th>
        <th>EOG ICs</th><th>EOG Max Score</th><th>EOG Forced?</th>
    </tr>
    """
    for r in per_run:
        if not r.get("complete"):
            continue
        ecg = r.get("ecg_components", [])
        eog = r.get("eog_components", [])
        ecg_max = r.get("ecg_max_score")
        eog_max = r.get("eog_max_score")
        ecg_max_str = f"{ecg_max:.3f}" if ecg_max is not None else "N/A"
        eog_max_str = f"{eog_max:.3f}" if eog_max is not None else "N/A"

        # Color the max score based on forced status
        ecg_forced = r.get("ecg_forced")
        eog_forced = r.get("eog_forced")
        if ecg_forced is not None:
            ecg_score_class = "forced-text" if ecg_forced else "above-thresh-text"
            ecg_max_str = f'<span class="{ecg_score_class}">{ecg_max_str}</span>'
        if eog_forced is not None:
            eog_score_class = "forced-text" if eog_forced else "above-thresh-text"
            eog_max_str = f'<span class="{eog_score_class}">{eog_max_str}</span>'

        html += f"""<tr>
            <td>{r['run']}</td>
            <td>{ecg if ecg else 'None'}</td>
            <td>{ecg_max_str}</td>
            <td>{_ica_forced_badge(ecg_forced)}</td>
            <td>{eog if eog else 'None'}</td>
            <td>{eog_max_str}</td>
            <td>{_ica_forced_badge(eog_forced)}</td>
        </tr>"""
    html += "</table></div>"

    # Outlier flags
    if outlier_runs:
        html += '<h3 class="collapsible-header">Outlier Flags</h3><div class="collapsible-content"><ul>'
        for o in outlier_runs:
            reasons_str = "; ".join(o["reasons"])
            html += f'<li style="color: #c62828;"><b>Run {o["run"]}:</b> {reasons_str}</li>'
        html += "</ul></div>"
    else:
        html += '<h3 class="collapsible-header">Outlier Flags</h3><div class="collapsible-content"><p style="color: green;">No outlier runs detected.</p></div>'

    # Summary statistics
    html += '<h3 class="collapsible-header">Summary Statistics</h3>'
    html += '<div class="collapsible-content">'
    html += '<table border="1" class="sortable" style="border-collapse: collapse; width: 80%; font-size: 13px;">'
    html += '<tr style="background-color: #f0f0f0;"><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>'
    for metric_name, metric_key in [
        ("AR1 Bad %", "ar1_pct_bad"),
        ("AR2 Bad %", "ar2_pct_bad"),
        ("Threshold Bad %", "threshold_pct_bad"),
        ("Retention Rate %", "retention_rate"),
        ("Rescue Rate %", "rescue_rate"),
        ("ISI Mean (s)", "isi_mean"),
    ]:
        s = summary.get(metric_key, {})
        def _f(v):
            return f"{v:.1f}" if v is not None else "-"
        html += f'<tr><td>{metric_name}</td><td>{_f(s.get("mean"))}</td><td>{_f(s.get("std"))}</td><td>{_f(s.get("min"))}</td><td>{_f(s.get("max"))}</td></tr>'

    # Stimulus epoch count with CV
    stim_stats = summary.get("n_stimulus_epochs", {})
    if stim_stats.get("mean") is not None:
        cv = stim_stats.get("cv")
        cv_str = f"{cv:.3f}" if cv is not None else "-"
        html += f'<tr><td>Stimulus Epochs</td><td>{stim_stats["mean"]:.0f}</td><td>{stim_stats["std"]:.1f}</td><td>{stim_stats["min"]:.0f}</td><td>{stim_stats["max"]:.0f}</td></tr>'
        html += f'<tr><td>Epoch Count CV</td><td colspan="4">{cv_str}</td></tr>'

    html += "</table></div>"

    return html


def generate_subject_report(
    subject: str,
    derivatives_root: Path,
    task_runs: list,
    config: dict,
) -> Path:
    """Generate subject-level HTML report + JSON summary.

    Args:
        subject: Subject ID.
        derivatives_root: Path to derivatives directory.
        task_runs: List of run IDs.
        config: Configuration dict.

    Returns:
        Path to the generated HTML report.
    """
    import mne

    run_params = collect_run_params(subject, derivatives_root, task_runs)
    metrics = aggregate_subject_metrics(run_params)

    # Generate HTML report using mne.Report
    report = mne.Report(title=f"Preprocessing Summary - sub-{subject}", verbose=False)

    # Add HTML overview
    html_content = _build_subject_html(subject, metrics, config)
    report.add_html(html_content, title="Overview")

    # Add bar chart figure
    fig = _create_subject_epoch_bar_chart(metrics["per_run"], subject)
    report.add_figure(fig, title="Bad Epochs per Run")
    plt.close(fig)

    # Output paths
    out_dir = derivatives_root / "preprocessed" / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"sub-{subject}_preprocessing-summary.html"
    json_path = out_dir / f"sub-{subject}_preprocessing-summary.json"

    report.save(html_path, open_browser=False, overwrite=True)
    logger.info(f"Subject report saved: {html_path}")

    # Save JSON summary
    json_data = {
        "subject": subject,
        "timestamp": datetime.now().isoformat(),
        "n_runs_complete": metrics["n_runs_complete"],
        "n_runs_total": metrics["n_runs_total"],
        "total_epochs": metrics["total_epochs"],
        "summary": metrics["summary"],
        "outlier_runs": metrics["outlier_runs"],
        "per_run": metrics["per_run"],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Subject JSON saved: {json_path}")

    return html_path


def _build_dataset_html(dataset_metrics: dict) -> str:
    """Build HTML content for dataset-level report.

    Args:
        dataset_metrics: Output of aggregate_dataset_metrics.

    Returns:
        HTML string.
    """
    per_subject = dataset_metrics["per_subject"]
    dataset_stats = dataset_metrics["dataset_stats"]
    outlier_subjects = dataset_metrics["outlier_subjects"]
    n_subjects = dataset_metrics["n_subjects"]
    n_complete = dataset_metrics["n_subjects_complete"]

    # Shared CSS/JS
    html = _shared_css_js()

    # Overview
    html += f"""
    <h2>Dataset Preprocessing Report</h2>
    <p><b>Timestamp:</b> {datetime.now().isoformat()}</p>
    <p><b>Subjects:</b> {n_complete}/{n_subjects} with data</p>
    """

    # Per-subject summary table
    html += """
    <h3 class="collapsible-header">Per-Subject Summary</h3>
    <div class="collapsible-content">
    <table border="1" class="sortable" style="border-collapse: collapse; width: 100%; font-size: 11px;">
    <tr style="background-color: #f0f0f0;">
        <th>Subject</th><th>Runs</th><th>Total Epochs</th>
        <th>Stim Epochs</th><th>ISI Mean</th>
        <th>Mean AR1 %</th><th>Mean AR2 %</th>
        <th>Mean Threshold %</th><th>Mean Retention %</th>
        <th>Mean Rescue %</th>
        <th>Mean ECG ICs</th><th>Mean EOG ICs</th><th>Forced ICA Runs</th>
        <th>Outlier Runs</th>
    </tr>
    """

    # Collect outlier subject IDs for highlighting
    outlier_subj_ids = set()
    for o in outlier_subjects:
        outlier_subj_ids.add(o["subject"])

    for subj in sorted(per_subject.keys()):
        metrics = per_subject[subj]
        s = metrics["summary"]

        row_style = ""
        if subj in outlier_subj_ids:
            row_style = ' style="background-color: #f8d7da;"'

        def _f(v):
            return f"{v:.1f}" if v is not None else "-"

        def _f3(v):
            return f"{v:.3f}" if v is not None else "-"

        n_outliers = len(metrics["outlier_runs"])

        stim_mean = _safe_get(s, "n_stimulus_epochs", "mean")
        isi_mean = _safe_get(s, "isi_mean", "mean")

        # ICA aggregates across runs
        complete_runs = [r for r in metrics["per_run"] if r.get("complete")]
        ecg_ics = [r.get("n_ecg_ics", 0) for r in complete_runs if r.get("n_ecg_ics") is not None]
        eog_ics = [r.get("n_eog_ics", 0) for r in complete_runs if r.get("n_eog_ics") is not None]
        mean_ecg = f"{np.mean(ecg_ics):.1f}" if ecg_ics else "-"
        mean_eog = f"{np.mean(eog_ics):.1f}" if eog_ics else "-"

        # Count runs with forced ICA selection
        forced_runs = [r for r in complete_runs if r.get("ecg_forced") or r.get("eog_forced")]
        has_forced_data = any(r.get("ecg_forced") is not None or r.get("eog_forced") is not None for r in complete_runs)
        forced_str = str(len(forced_runs)) if has_forced_data else "N/A"
        if has_forced_data and len(forced_runs) > 0:
            forced_str = f'<span class="badge badge-yellow">{len(forced_runs)}</span>'

        # Retention badge
        retention_val = s['retention_rate']['mean']
        retention_cell = _retention_badge(retention_val) if retention_val is not None else "-"

        # Subject link (relative to dataset report location)
        subject_link = f'<a href="sub-{subj}/sub-{subj}_preprocessing-summary.html">sub-{subj}</a>'

        html += f"""<tr{row_style}>
            <td>{subject_link}</td>
            <td>{metrics['n_runs_complete']}/{metrics['n_runs_total']}</td>
            <td>{metrics['total_epochs']}</td>
            <td>{f'{stim_mean:.0f}' if stim_mean is not None else '-'}</td>
            <td>{_f3(isi_mean)}s</td>
            <td>{_f(s['ar1_pct_bad']['mean'])}</td>
            <td>{_f(s['ar2_pct_bad']['mean'])}</td>
            <td>{_f(s['threshold_pct_bad']['mean'])}</td>
            <td>{retention_cell}</td>
            <td>{_f(s['rescue_rate']['mean'])}</td>
            <td>{mean_ecg}</td>
            <td>{mean_eog}</td>
            <td>{forced_str}</td>
            <td>{'<b style="color:red;">' + str(n_outliers) + '</b>' if n_outliers > 0 else '0'}</td>
        </tr>"""

    html += "</table></div>"

    # Epoch count consistency flags
    epoch_flags = dataset_metrics.get("epoch_count_flags", [])
    if epoch_flags:
        html += '<h3 class="collapsible-header">Epoch Count Consistency Warnings</h3><div class="collapsible-content"><ul>'
        for flag in epoch_flags:
            html += f'<li style="color: #c62828;">sub-{flag["subject"]}: per-run counts vary: {flag["unique"]}</li>'
        html += "</ul></div>"

    # Overall statistics
    html += '<h3 class="collapsible-header">Overall Statistics</h3>'
    html += '<div class="collapsible-content">'
    html += '<table border="1" class="sortable" style="border-collapse: collapse; width: 80%; font-size: 13px;">'
    html += '<tr style="background-color: #f0f0f0;"><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Median</th></tr>'
    for metric_name, metric_key in [
        ("Mean AR1 Bad %", "mean_ar1_pct_bad"),
        ("Mean Retention %", "mean_retention_rate"),
        ("Mean Rescue Rate %", "mean_rescue_rate"),
        ("Total Epochs", "total_epochs"),
    ]:
        s = dataset_stats[metric_key]
        def _f(v):
            return f"{v:.1f}" if v is not None else "-"
        html += f'<tr><td>{metric_name}</td><td>{_f(s["mean"])}</td><td>{_f(s["std"])}</td><td>{_f(s["min"])}</td><td>{_f(s["max"])}</td><td>{_f(s["median"])}</td></tr>'
    html += "</table></div>"

    # Outlier summary
    if outlier_subjects:
        html += '<h3 class="collapsible-header">Outlier Subjects</h3>'
        html += '<div class="collapsible-content">'
        html += '<table border="1" class="sortable" style="border-collapse: collapse; width: 100%; font-size: 12px;">'
        html += '<tr style="background-color: #f0f0f0;"><th>Subject</th><th>Metric</th><th>Details</th></tr>'
        for o in outlier_subjects:
            if o.get("metric") == "absolute_threshold":
                details = "; ".join(o.get("reasons", []))
            else:
                def _fmt(v, fmt=".1f"):
                    try:
                        return f"{float(v):{fmt}}"
                    except (TypeError, ValueError):
                        return "-"
                details = (
                    f"value={_fmt(o.get('value'))}, "
                    f"median={_fmt(o.get('median'))}, "
                    f"MAD={_fmt(o.get('mad'), '.2f')}, "
                    f"deviation={_fmt(o.get('deviation_mad'))} MAD"
                )
            html += f'<tr style="background-color: #f8d7da;"><td>sub-{o["subject"]}</td><td>{o.get("metric", "")}</td><td>{details}</td></tr>'
        html += "</table></div>"
    else:
        html += '<h3 class="collapsible-header">Outlier Subjects</h3><div class="collapsible-content"><p style="color: green;">No outlier subjects detected.</p></div>'

    return html


def generate_dataset_report(
    derivatives_root: Path,
    subjects: list,
    task_runs: list,
    config: dict,
) -> Path:
    """Generate dataset-level HTML report + JSON summary.

    Args:
        derivatives_root: Path to derivatives directory.
        subjects: List of subject IDs.
        task_runs: List of run IDs.
        config: Configuration dict.

    Returns:
        Path to the generated HTML report.
    """
    import mne

    all_params = collect_all_subject_params(derivatives_root, subjects, task_runs)
    dataset_metrics = aggregate_dataset_metrics(all_params)

    # Generate HTML report
    report = mne.Report(title="Dataset Preprocessing Summary", verbose=False)

    # Add HTML overview
    html_content = _build_dataset_html(dataset_metrics)
    report.add_html(html_content, title="Overview")

    # Add distribution figure
    fig = _create_dataset_distribution_figure(dataset_metrics)
    report.add_figure(fig, title="Distribution Plots")
    plt.close(fig)

    # Output paths
    out_dir = derivatives_root / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "group_preprocessing-summary.html"
    json_path = out_dir / "group_preprocessing-summary.json"

    report.save(html_path, open_browser=False, overwrite=True)
    logger.info(f"Dataset report saved: {html_path}")

    # Save JSON summary
    # Simplify per_subject for JSON (remove nested per_run details)
    per_subject_json = {}
    for subj, metrics in dataset_metrics["per_subject"].items():
        per_subject_json[subj] = {
            "n_runs_complete": metrics["n_runs_complete"],
            "n_runs_total": metrics["n_runs_total"],
            "total_epochs": metrics["total_epochs"],
            "summary": metrics["summary"],
            "outlier_runs": metrics["outlier_runs"],
        }

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "n_subjects": dataset_metrics["n_subjects"],
        "n_subjects_complete": dataset_metrics["n_subjects_complete"],
        "dataset_stats": dataset_metrics["dataset_stats"],
        "outlier_subjects": dataset_metrics["outlier_subjects"],
        "per_subject": per_subject_json,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Dataset JSON saved: {json_path}")

    return html_path


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for aggregate reports."""
    parser = argparse.ArgumentParser(
        description="Generate aggregate preprocessing reports"
    )
    parser.add_argument(
        "-s", "--subject",
        type=str,
        default=None,
        help="Subject ID (generates subject-level report)",
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        default=False,
        help="Generate dataset-level report",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    if not args.subject and not args.dataset:
        parser.error("At least one of -s/--subject or --dataset is required")

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = load_config()
    derivatives_root = Path(config["paths"]["data_root"]) / config["paths"]["derivatives"]
    task_runs = config["bids"]["task_runs"]
    subjects = config["bids"]["subjects"]

    if args.subject:
        print(f"Generating subject-level report for sub-{args.subject}...")
        report_path = generate_subject_report(
            args.subject, derivatives_root, task_runs, config
        )
        print(f"Subject report: {report_path}")

    if args.dataset:
        print(f"Generating dataset-level report for {len(subjects)} subjects...")
        report_path = generate_dataset_report(
            derivatives_root, subjects, task_runs, config
        )
        print(f"Dataset report: {report_path}")


if __name__ == "__main__":
    main()
