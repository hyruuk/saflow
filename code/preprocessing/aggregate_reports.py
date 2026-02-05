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

        ar1_pct = ar1.get("pct_bad")
        ar2_pct = ar2.get("pct_bad") if ar2 else None
        threshold_pct = threshold.get("pct_bad")
        n_total = ar1.get("n_total_epochs")

        ecg_components = ica.get("ecg_components", [])
        eog_components = ica.get("eog_components", [])
        n_ecg = len(ecg_components) if isinstance(ecg_components, list) else 0
        n_eog = len(eog_components) if isinstance(eog_components, list) else 0

        rescue_rate = _safe_get(
            three_way, "ica_effectiveness", "rescue_rate_pct"
        )
        pairwise_kappa = _safe_get(three_way, "pairwise_kappa")

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
            "n_ecg_ics": n_ecg,
            "n_eog_ics": n_eog,
            "ecg_components": ecg_components,
            "eog_components": eog_components,
            "rescue_rate": rescue_rate,
            "pairwise_kappa": pairwise_kappa,
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

    ar1_pcts = [r["ar1_pct_bad"] for r in complete_runs]
    ar2_pcts = [r["ar2_pct_bad"] for r in complete_runs]
    threshold_pcts = [r["threshold_pct_bad"] for r in complete_runs]
    rescue_rates = [r["rescue_rate"] for r in complete_runs]
    n_totals = [r["n_total_epochs"] for r in complete_runs]

    # Retention rate: 100 - ar1_pct_bad (epochs surviving first pass)
    retention_rates = [
        100.0 - v for v in ar1_pcts if v is not None
    ]

    summary = {
        "ar1_pct_bad": _stats(ar1_pcts),
        "ar2_pct_bad": _stats(ar2_pcts),
        "threshold_pct_bad": _stats(threshold_pcts),
        "rescue_rate": _stats(rescue_rates),
        "retention_rate": _stats(retention_rates),
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

    return {
        "per_subject": per_subject,
        "dataset_stats": dataset_stats,
        "outlier_subjects": outlier_subjects,
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

    # Overview section
    html = f"""
    <h2>Subject Report: sub-{subject}</h2>
    <p><b>Timestamp:</b> {datetime.now().isoformat()}</p>
    <p><b>Runs:</b> {n_complete}/{n_total} complete</p>
    <p><b>Total epochs:</b> {metrics['total_epochs']}</p>
    """

    # Per-run metrics table
    html += """
    <h3>Per-Run Metrics</h3>
    <table border="1" style="border-collapse: collapse; width: 100%; font-size: 13px;">
    <tr style="background-color: #f0f0f0;">
        <th>Run</th><th>Total Epochs</th>
        <th>AR1 Bad</th><th>AR1 %</th>
        <th>AR2 Bad</th><th>AR2 %</th>
        <th>Threshold Bad</th><th>Threshold %</th>
        <th>ECG ICs</th><th>EOG ICs</th>
        <th>Rescue Rate</th>
    </tr>
    """
    for r in per_run:
        if not r.get("complete"):
            html += f'<tr style="background-color: #fff3cd;"><td>{r["run"]}</td>'
            html += '<td colspan="10"><i>Missing or incomplete</i></td></tr>'
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

        html += f"""<tr{row_style}>
            <td>{r['run']}</td>
            <td>{_fmt_int(r.get('n_total_epochs'))}</td>
            <td>{_fmt_int(r.get('ar1_n_bad'))}</td>
            <td>{_fmt(r.get('ar1_pct_bad'))}%</td>
            <td>{_fmt_int(r.get('ar2_n_bad'))}</td>
            <td>{_fmt(r.get('ar2_pct_bad'))}%</td>
            <td>{_fmt_int(r.get('threshold_n_bad'))}</td>
            <td>{_fmt(r.get('threshold_pct_bad'))}%</td>
            <td>{r.get('n_ecg_ics', '-')}</td>
            <td>{r.get('n_eog_ics', '-')}</td>
            <td>{rescue}</td>
        </tr>"""

    html += "</table>"

    # ICA summary table
    html += """
    <h3>ICA Component Summary</h3>
    <table border="1" style="border-collapse: collapse; width: 60%; font-size: 13px;">
    <tr style="background-color: #f0f0f0;">
        <th>Run</th><th>ECG Indices</th><th>EOG Indices</th>
    </tr>
    """
    for r in per_run:
        if not r.get("complete"):
            continue
        ecg = r.get("ecg_components", [])
        eog = r.get("eog_components", [])
        html += f"""<tr>
            <td>{r['run']}</td>
            <td>{ecg if ecg else 'None'}</td>
            <td>{eog if eog else 'None'}</td>
        </tr>"""
    html += "</table>"

    # Outlier flags
    if outlier_runs:
        html += "<h3>Outlier Flags</h3><ul>"
        for o in outlier_runs:
            reasons_str = "; ".join(o["reasons"])
            html += f'<li style="color: #c62828;"><b>Run {o["run"]}:</b> {reasons_str}</li>'
        html += "</ul>"
    else:
        html += '<h3>Outlier Flags</h3><p style="color: green;">No outlier runs detected.</p>'

    # Summary statistics
    html += "<h3>Summary Statistics</h3>"
    html += '<table border="1" style="border-collapse: collapse; width: 80%; font-size: 13px;">'
    html += '<tr style="background-color: #f0f0f0;"><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>'
    for metric_name, metric_key in [
        ("AR1 Bad %", "ar1_pct_bad"),
        ("AR2 Bad %", "ar2_pct_bad"),
        ("Threshold Bad %", "threshold_pct_bad"),
        ("Retention Rate %", "retention_rate"),
        ("Rescue Rate %", "rescue_rate"),
    ]:
        s = summary[metric_key]
        def _f(v):
            return f"{v:.1f}" if v is not None else "-"
        html += f'<tr><td>{metric_name}</td><td>{_f(s["mean"])}</td><td>{_f(s["std"])}</td><td>{_f(s["min"])}</td><td>{_f(s["max"])}</td></tr>'
    html += "</table>"

    # Links to per-run reports
    derivatives_root = Path(config["paths"]["data_root"]) / config["paths"]["derivatives"]
    html += "<h3>Per-Run Reports</h3><ul>"
    for r in per_run:
        run_id = r["run"]
        report_name = f"sub-{subject}_task-gradCPT_run-{run_id}_proc-clean_report_meg.html"
        report_path = derivatives_root / "preprocessed" / f"sub-{subject}" / "meg" / report_name
        if report_path.exists():
            html += f'<li><a href="{report_path}">Run {run_id} report</a></li>'
        else:
            html += f'<li>Run {run_id}: <i>report not found</i></li>'
    html += "</ul>"

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

    # Overview
    html = f"""
    <h2>Dataset Preprocessing Report</h2>
    <p><b>Timestamp:</b> {datetime.now().isoformat()}</p>
    <p><b>Subjects:</b> {n_complete}/{n_subjects} with data</p>
    """

    # Per-subject summary table
    html += """
    <h3>Per-Subject Summary</h3>
    <table border="1" style="border-collapse: collapse; width: 100%; font-size: 12px;">
    <tr style="background-color: #f0f0f0;">
        <th>Subject</th><th>Runs</th><th>Total Epochs</th>
        <th>Mean AR1 %</th><th>Mean AR2 %</th>
        <th>Mean Threshold %</th><th>Mean Retention %</th>
        <th>Mean Rescue %</th><th>Outlier Runs</th>
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

        n_outliers = len(metrics["outlier_runs"])

        html += f"""<tr{row_style}>
            <td>sub-{subj}</td>
            <td>{metrics['n_runs_complete']}/{metrics['n_runs_total']}</td>
            <td>{metrics['total_epochs']}</td>
            <td>{_f(s['ar1_pct_bad']['mean'])}</td>
            <td>{_f(s['ar2_pct_bad']['mean'])}</td>
            <td>{_f(s['threshold_pct_bad']['mean'])}</td>
            <td>{_f(s['retention_rate']['mean'])}</td>
            <td>{_f(s['rescue_rate']['mean'])}</td>
            <td>{'<b style="color:red;">' + str(n_outliers) + '</b>' if n_outliers > 0 else '0'}</td>
        </tr>"""

    html += "</table>"

    # Overall statistics
    html += "<h3>Overall Statistics</h3>"
    html += '<table border="1" style="border-collapse: collapse; width: 80%; font-size: 13px;">'
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
    html += "</table>"

    # Outlier summary
    if outlier_subjects:
        html += "<h3>Outlier Subjects</h3>"
        html += '<table border="1" style="border-collapse: collapse; width: 100%; font-size: 12px;">'
        html += '<tr style="background-color: #f0f0f0;"><th>Subject</th><th>Metric</th><th>Details</th></tr>'
        for o in outlier_subjects:
            if o.get("metric") == "absolute_threshold":
                details = "; ".join(o.get("reasons", []))
            else:
                details = (
                    f"value={o.get('value', ''):.1f}, "
                    f"median={o.get('median', ''):.1f}, "
                    f"MAD={o.get('mad', ''):.2f}, "
                    f"deviation={o.get('deviation_mad', ''):.1f} MAD"
                )
            html += f'<tr style="background-color: #f8d7da;"><td>sub-{o["subject"]}</td><td>{o.get("metric", "")}</td><td>{details}</td></tr>'
        html += "</table>"
    else:
        html += '<h3>Outlier Subjects</h3><p style="color: green;">No outlier subjects detected.</p>'

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
