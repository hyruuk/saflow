"""Aggregate preprocessing reports for subject-level and dataset-level summaries.

This module reads per-run *_params.json metadata files (no raw MEG data loading)
and generates:
1. Subject-level reports: quality summary across all runs for one subject
2. Dataset-level reports: quality summary across all subjects

Usage:
    python -m code.preprocessing.aggregate_reports -s 04   # single subject report
    python -m code.preprocessing.aggregate_reports --dataset  # all subject reports + dataset report
"""

import argparse
import base64
import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

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
        bad_chans = p.get("bad_channels", {}) or {}
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
        n_ica_components = _safe_get(ica, "n_components_actual")

        # ICA scores (available from future runs, None for old data)
        ecg_scores = _safe_get(ica, "ecg_scores")
        eog_scores = _safe_get(ica, "eog_scores")
        ecg_threshold = _safe_get(ica, "ecg_threshold")
        eog_threshold = _safe_get(ica, "eog_threshold")
        ecg_forced = _safe_get(ica, "ecg_forced")
        eog_forced = _safe_get(ica, "eog_forced")
        pca_explained_variance = _safe_get(ica, "pca_explained_variance")

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
            "n_ica_components": n_ica_components,
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
            "pca_explained_variance": pca_explained_variance,
            "rescue_rate": rescue_rate,
            "pairwise_kappa": pairwise_kappa,
            "retention_rate": proper_retention,
            "isi_mean": isi_mean,
            "isi_std": isi_std,
            "pre_ar2_outliers": _safe_get(pre_ar2, "n_outliers"),
            "bad_channels_n": int(bad_chans.get("n_bad", 0)) if isinstance(bad_chans, dict) else 0,
            "bad_channels_names": list(bad_chans.get("names", [])) if isinstance(bad_chans, dict) else [],
            "bad_channels_threshold": bad_chans.get("threshold") if isinstance(bad_chans, dict) else None,
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

    n_ica_components = [r["n_ica_components"] for r in complete_runs]

    summary = {
        "ar1_pct_bad": _stats(ar1_pcts),
        "ar2_pct_bad": _stats(ar2_pcts),
        "threshold_pct_bad": _stats(threshold_pcts),
        "rescue_rate": _stats(rescue_rates),
        "retention_rate": _stats(retention_rates),
        "isi_mean": _stats(isi_means),
        "n_stimulus_epochs": _stats_cv(n_stimulus),
        "n_ica_components": _stats(n_ica_components),
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

def _create_epoch_bar_html(metrics: dict, subject: str, include_plotlyjs: bool = True) -> str:
    """Interactive Plotly grouped bar chart: bad epoch counts per run.

    Falls back to a base64-embedded matplotlib PNG if Plotly is unavailable.

    Args:
        metrics: Output of aggregate_subject_metrics.
        subject: Subject ID.
        include_plotlyjs: Whether to embed the Plotly JS library inline.

    Returns:
        HTML string.
    """
    complete_runs = [r for r in metrics["per_run"] if r.get("complete")]
    if not complete_runs:
        return "<p><i>No complete runs.</i></p>"

    runs = [f"run-{r['run']}" for r in complete_runs]
    ar1_bad = [r.get("ar1_n_bad") or 0 for r in complete_runs]
    ar2_bad = [r.get("ar2_n_bad") or 0 for r in complete_runs]
    thr_bad  = [r.get("threshold_n_bad") or 0 for r in complete_runs]

    if _PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=runs, y=ar1_bad, name="AR1 (pre-ICA)", marker_color="#1f77b4",
            text=ar1_bad, textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=runs, y=ar2_bad, name="AR2 (post-ICA)", marker_color="#ff7f0e",
            text=ar2_bad, textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=runs, y=thr_bad, name="Threshold", marker_color="#2ca02c",
            text=thr_bad, textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            title=f"Bad Epochs per Run — sub-{subject}",
            xaxis_title="Run", yaxis_title="Bad Epoch Count",
            template="plotly_white", height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False)

    # Matplotlib fallback
    x = np.arange(len(runs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.5), 5))
    for offset, vals, label, color in [
        (-width, ar1_bad, "AR1 (pre-ICA)", "#1f77b4"),
        (0,      ar2_bad, "AR2 (post-ICA)", "#ff7f0e"),
        (width,  thr_bad,  "Threshold",      "#2ca02c"),
    ]:
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(str(int(h)),
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Run"); ax.set_ylabel("Bad Epoch Count")
    ax.set_title(f"Bad Epochs per Run — sub-{subject}")
    ax.set_xticks(x); ax.set_xticklabels(runs); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0); b64 = base64.b64encode(buf.read()).decode(); plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'


def _create_dataset_distribution_html(dataset_metrics: dict) -> str:
    """4-panel interactive Plotly figure with distributions across subjects.

    Panels:
    - Histogram: AR1 bad epoch rates across subjects (hover shows subject IDs)
    - Histogram: retention rates across subjects
    - Histogram: ICA rescue rates (if available)
    - Scatter: AR1 bad % vs AR2 bad % per subject (hover shows subject ID)

    Args:
        dataset_metrics: Output of aggregate_dataset_metrics.

    Returns:
        HTML string with embedded Plotly figure, or empty string if Plotly unavailable.
    """
    if not _PLOTLY_AVAILABLE:
        return ""

    per_subject = dataset_metrics["per_subject"]

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

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "AR1 Bad Epoch Rates Across Subjects",
            "Retention Rates Across Subjects",
            "ICA Rescue Rates Across Subjects",
            "AR1 vs AR2 Bad Epoch Rate Per Subject",
        ),
    )

    # Helper: build custom histogram bins with per-bin subject lists for hover
    def _hist_with_hover(values, labels, nbins, color, row, col, name):
        valid = [(v, l) for v, l in zip(values, labels) if v is not None]
        if not valid:
            return
        vals, labs = zip(*valid)
        vals = list(vals)
        labs = list(labs)
        lo, hi = min(vals), max(vals)
        if lo == hi:
            edges = [lo - 0.5, hi + 0.5]
        else:
            step = (hi - lo) / nbins
            edges = [lo + i * step for i in range(nbins + 1)]
        bin_counts = []
        bin_subjects = []
        bin_centers = []
        for i in range(len(edges) - 1):
            lo_e, hi_e = edges[i], edges[i + 1]
            in_bin = [l for v, l in zip(vals, labs) if lo_e <= v < hi_e or (i == len(edges) - 2 and v == hi_e)]
            bin_counts.append(len(in_bin))
            bin_subjects.append("<br>".join(f"sub-{l}" for l in in_bin) if in_bin else "")
            bin_centers.append((lo_e + hi_e) / 2)
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=bin_counts,
                name=name,
                marker_color=color,
                marker_line_color="black",
                marker_line_width=0.8,
                opacity=0.75,
                width=edges[1] - edges[0],
                customdata=bin_subjects,
                hovertemplate="<b>%{y} subjects</b><br>%{customdata}<extra></extra>",
            ),
            row=row, col=col,
        )

    nbins_ar1 = min(15, max(5, len([v for v in ar1_means if v is not None]) // 2))
    nbins_ret = min(15, max(5, len([v for v in retention_means if v is not None]) // 2))
    nbins_rsc = min(15, max(5, len([v for v in rescue_means if v is not None]) // 2))

    _hist_with_hover(ar1_means, subjects, nbins_ar1, "#1f77b4", 1, 1, "AR1 bad rate")
    _hist_with_hover(retention_means, subjects, nbins_ret, "#2ca02c", 1, 2, "Retention rate")
    _hist_with_hover(rescue_means, subjects, nbins_rsc, "#9467bd", 2, 1, "ICA rescue rate")

    # Threshold lines
    fig.add_vline(x=30, line_dash="dash", line_color="red", row=1, col=1,
                  annotation_text="30%", annotation_position="top right")
    fig.add_vline(x=60, line_dash="dash", line_color="red", row=1, col=2,
                  annotation_text="60%", annotation_position="top left")

    # Scatter: AR1 vs AR2
    scatter_data = [(a1, a2, s) for a1, a2, s in zip(ar1_means, ar2_means, subjects)
                    if a1 is not None and a2 is not None]
    if scatter_data:
        sx, sy, sl = zip(*scatter_data)
        lim_max = max(max(sx), max(sy)) * 1.1
        fig.add_trace(
            go.Scatter(
                x=list(sx), y=list(sy),
                mode="markers",
                marker=dict(color="#ff7f0e", size=9, line=dict(color="black", width=1)),
                text=[f"sub-{s}" for s in sl],
                hovertemplate="<b>%{text}</b><br>AR1: %{x:.1f}%<br>AR2: %{y:.1f}%<extra></extra>",
                name="subjects",
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[0, lim_max], y=[0, lim_max],
                mode="lines",
                line=dict(color="black", dash="dash", width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Mean AR1 Bad Epoch Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Mean Retention Rate (%)", row=1, col=2)
    fig.update_xaxes(title_text="Mean ICA Rescue Rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Mean AR1 Bad %", row=2, col=2)
    fig.update_yaxes(title_text="Number of Subjects", row=1, col=1)
    fig.update_yaxes(title_text="Number of Subjects", row=1, col=2)
    fig.update_yaxes(title_text="Number of Subjects", row=2, col=1)
    fig.update_yaxes(title_text="Mean AR2 Bad %", row=2, col=2)

    fig.update_layout(
        title_text="Dataset-Level Preprocessing Quality",
        title_font_size=16,
        height=800,
        showlegend=False,
        template="plotly_white",
    )

    return fig.to_html(include_plotlyjs=True, full_html=False)


def _create_dataset_distribution_figure(dataset_metrics: dict) -> plt.Figure:
    """4-panel matplotlib figure (fallback when Plotly unavailable)."""
    per_subject = dataset_metrics["per_subject"]

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
        lim_max = max(max(scatter_ar1), max(scatter_ar2)) * 1.1
        ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, label="x=y")
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


def _inject_nav_bar(html_path: Path, links: list) -> None:
    """Inject a sticky navigation bar into an MNE Report HTML file.

    The bar is inserted immediately after the <body> opening tag so it is
    always visible regardless of which tab/section the user has selected.

    Args:
        html_path: Path to the saved HTML file to patch in-place.
        links: List of (label, href) tuples rendered left-to-right.
    """
    import re

    try:
        content = html_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Cannot read {html_path} for nav-bar injection: {exc}")
        return

    link_parts = " &nbsp;|&nbsp; ".join(
        f'<a href="{href}" style="color:#5dade2;text-decoration:none;">{label}</a>'
        for label, href in links
    )
    nav_html = (
        '\n<div style="position:sticky;top:0;z-index:10000;background:#1a252f;'
        'color:#ecf0f1;padding:6px 16px;font-size:13px;'
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        'display:flex;align-items:center;gap:12px;border-bottom:2px solid #2980b9;">'
        f'{link_parts}'
        '</div>\n'
    )

    modified = re.sub(
        r'(<body[^>]*>)',
        lambda m: m.group(1) + nav_html,
        content,
        count=1,
    )
    if modified == content:
        logger.warning(f"Nav-bar injection: no <body> tag found in {html_path}")
        return

    try:
        html_path.write_text(modified, encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Cannot write nav bar to {html_path}: {exc}")


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
        <th>Bad Chans</th>
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
        report_name = f"sub-{subject}_task-gradCPT_run-{run_id}_desc-report_meg.html"
        report_abs_path = derivatives_root / "preprocessed" / f"sub-{subject}" / "meg" / report_name
        report_rel = f"meg/{report_name}"
        if report_abs_path.exists():
            report_link = f'<a href="{report_rel}">View</a>'
        else:
            report_link = '<i>N/A</i>'

        bad_chan_n = r.get("bad_channels_n") or 0
        bad_chan_names = r.get("bad_channels_names") or []
        bad_chan_cell = (
            f'<span title="{", ".join(bad_chan_names)}">{int(bad_chan_n)}</span>'
            if bad_chan_n
            else "0"
        )

        html += f"""<tr{row_style}>
            <td>{r['run']}</td>
            <td>{_fmt_int(r.get('n_stimulus_epochs'))}</td>
            <td>{_fmt_int(r.get('n_freq'))}</td>
            <td>{_fmt_int(r.get('n_rare'))}</td>
            <td>{_fmt_int(r.get('n_resp'))}</td>
            <td>{isi_mean}</td>
            <td>{bad_chan_cell}</td>
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
        <th>Run</th><th>N Components</th>
        <th>ECG ICs</th><th>ECG Max Score</th><th>ECG Forced?</th>
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
        n_comp = r.get("n_ica_components")
        n_comp_str = str(int(n_comp)) if n_comp is not None else "N/A"

        # Color the max score based on forced status
        ecg_forced = r.get("ecg_forced")
        eog_forced = r.get("eog_forced")
        if ecg_forced is not None:
            ecg_score_class = "forced-text" if ecg_forced else "above-thresh-text"
            ecg_max_str = f'<span class="{ecg_score_class}">{ecg_max_str}</span>'
        if eog_forced is not None:
            eog_score_class = "forced-text" if eog_forced else "above-thresh-text"
            eog_max_str = f'<span class="{eog_score_class}">{eog_max_str}</span>'

        # Link run label to per-run report
        run_id = r["run"]
        run_report_name = f"sub-{subject}_task-gradCPT_run-{run_id}_desc-report_meg.html"
        run_report_abs = derivatives_root / "preprocessed" / f"sub-{subject}" / "meg" / run_report_name
        if run_report_abs.exists():
            run_label = f'<a href="meg/{run_report_name}">{run_id}</a>'
        else:
            run_label = run_id

        html += f"""<tr>
            <td>{run_label}</td>
            <td>{n_comp_str}</td>
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
        ("ICA Components", "n_ica_components"),
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

    # ---- Interactive charts (always visible — embedded inline) ----
    html += "<h3>Bad Epochs per Run</h3>"
    html += _create_epoch_bar_html(metrics, subject, include_plotlyjs=True)

    html += "<h3>ICA Scores (y-axis 0–1)</h3>"
    html += _create_ica_score_html(metrics, subject, include_plotlyjs=False)

    html += "<h3>PCA Cumulative Explained Variance</h3>"
    html += _create_cumulative_variance_html(metrics, subject, include_plotlyjs=False)

    return html


def _create_ica_score_html(metrics: dict, subject: str, include_plotlyjs: bool = False) -> str:
    """Interactive Plotly ICA score bar charts (all runs, y-axis fixed 0–1).

    One row of subplots per run, two columns (ECG | EOG). Excluded components
    shown in red so they stand out immediately. Falls back to base64 PNG if
    Plotly is unavailable.

    Args:
        metrics: Output of aggregate_subject_metrics.
        subject: Subject ID.
        include_plotlyjs: Whether to embed the Plotly JS library inline.

    Returns:
        HTML string.
    """
    run_data = [
        r for r in metrics["per_run"]
        if r.get("complete") and (r.get("ecg_scores") or r.get("eog_scores"))
    ]
    if not run_data:
        return "<p><i>No ICA score data available.</i></p>"

    n_runs = len(run_data)

    if _PLOTLY_AVAILABLE:
        titles = []
        for r in run_data:
            titles += [f"Run {r['run']} — ECG", f"Run {r['run']} — EOG"]

        fig = make_subplots(rows=n_runs, cols=2, subplot_titles=titles,
                            vertical_spacing=0.08 / max(n_runs, 1))

        for row_i, r in enumerate(run_data, start=1):
            ecg_scores = r.get("ecg_scores") or []
            eog_scores = r.get("eog_scores") or []
            ecg_inds   = set(r.get("ecg_components", []))
            eog_inds   = set(r.get("eog_components", []))
            ecg_thresh = r.get("ecg_threshold")
            eog_thresh = r.get("eog_threshold")

            # ECG column
            if ecg_scores:
                n = len(ecg_scores)
                abs_s = [abs(s) for s in ecg_scores]
                colors = ["#d9534f" if i in ecg_inds else "#5bc0de" for i in range(n)]
                fig.add_trace(go.Bar(
                    x=list(range(n)), y=abs_s,
                    marker_color=colors, marker_line_color="black",
                    marker_line_width=0.5, showlegend=False,
                    hovertemplate="IC%{x}<br>|score|=%{y:.3f}<extra>ECG</extra>",
                ), row=row_i, col=1)
                if ecg_thresh is not None:
                    fig.add_hline(y=ecg_thresh, line_dash="dash", line_color="red",
                                  line_width=1, row=row_i, col=1)

            # EOG column
            if eog_scores:
                n = len(eog_scores)
                abs_s = [abs(s) for s in eog_scores]
                colors = ["#d9534f" if i in eog_inds else "#5bc0de" for i in range(n)]
                fig.add_trace(go.Bar(
                    x=list(range(n)), y=abs_s,
                    marker_color=colors, marker_line_color="black",
                    marker_line_width=0.5, showlegend=False,
                    hovertemplate="IC%{x}<br>|score|=%{y:.3f}<extra>EOG</extra>",
                ), row=row_i, col=2)
                if eog_thresh is not None:
                    fig.add_hline(y=eog_thresh, line_dash="dash", line_color="red",
                                  line_width=1, row=row_i, col=2)

        # Fix y-axis to [0, 1] for all subplots so runs are comparable
        for i in range(1, n_runs + 1):
            fig.update_yaxes(range=[0, 1], row=i, col=1)
            fig.update_yaxes(range=[0, 1], row=i, col=2)

        fig.update_layout(
            title=f"ICA Scores — sub-{subject}  (red = excluded, dashed = threshold)",
            height=320 * n_runs,
            template="plotly_white",
        )
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False)

    # Matplotlib fallback — one figure per run, embedded as base64
    html = ""
    for r in run_data:
        ecg_scores = r.get("ecg_scores") or []
        eog_scores = r.get("eog_scores") or []
        ecg_inds   = r.get("ecg_components", [])
        eog_inds   = r.get("eog_components", [])
        n_max = max(len(ecg_scores), len(eog_scores), 1)
        fig, axes = plt.subplots(1, 2, figsize=(max(14, n_max * 0.4), 4))
        for ax, scores, inds, label in [
            (axes[0], ecg_scores, ecg_inds, "ECG"),
            (axes[1], eog_scores, eog_inds, "EOG"),
        ]:
            if scores:
                n = len(scores)
                colors = ["#d9534f" if i in inds else "#5bc0de" for i in range(n)]
                ax.bar(range(n), [abs(s) for s in scores], color=colors,
                       edgecolor="black", linewidth=0.5)
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, f"No {label} scores", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(f"{label} — excluded: {inds}")
            ax.set_xlabel("ICA Component"); ax.set_ylabel("|Score|")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Run {r['run']}", fontweight="bold"); fig.tight_layout()
        buf = BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        buf.seek(0); b64 = base64.b64encode(buf.read()).decode(); plt.close(fig)
        html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%;display:block;margin:8px 0;"/>'
    return html


def _create_cumulative_variance_html(metrics: dict, subject: str, include_plotlyjs: bool = True) -> str:
    """Interactive Plotly (or static matplotlib) cumulative PCA variance plot.

    Shows one line per run, x = component index, y = cumulative explained
    variance (%).  Falls back to a matplotlib PNG embedded as base64 when
    Plotly is not available.

    Args:
        metrics: Output of aggregate_subject_metrics.
        subject: Subject ID.

    Returns:
        HTML string ready for report.add_html().
    """
    runs_data = []
    for r in metrics["per_run"]:
        if not r.get("complete"):
            continue
        pev = r.get("pca_explained_variance")
        if pev and isinstance(pev, list) and len(pev) > 0:
            runs_data.append((r["run"], pev))

    if not runs_data:
        return "<p><i>No PCA explained-variance data available (requires reprocessed runs).</i></p>"

    def _to_cumvar_pct(pev):
        """Cumulative % of total retained PCA variance, regardless of units.

        pca_explained_variance_ stores raw eigenvalues (not ratios), so we
        always normalise by the total before cumsum-ing.
        """
        arr = np.array(pev, dtype=float)
        total = arr.sum()
        if total == 0:
            return [0.0] * len(arr)
        return list(np.cumsum(arr) / total * 100)

    if _PLOTLY_AVAILABLE:
        fig = go.Figure()
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        ]
        for i, (run, pev) in enumerate(runs_data):
            cumvar = _to_cumvar_pct(pev)
            n_ica_actual = next(
                (r.get("n_ica_components") for r in metrics["per_run"] if r.get("run") == run),
                None,
            )
            hover = [
                f"Run {run}<br>Component {j}<br>Cumulative: {v:.1f}%"
                for j, v in enumerate(cumvar)
            ]
            # Marker to show where ICA cuts off
            marker_x = None
            marker_y = None
            if n_ica_actual is not None and n_ica_actual <= len(cumvar):
                marker_x = n_ica_actual - 1
                marker_y = cumvar[n_ica_actual - 1]

            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=list(range(len(cumvar))),
                y=cumvar,
                mode="lines",
                name=f"run-{run}",
                line=dict(color=color, width=2),
                text=hover,
                hoverinfo="text",
            ))
            if marker_x is not None:
                fig.add_trace(go.Scatter(
                    x=[marker_x],
                    y=[marker_y],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="diamond",
                                line=dict(color="black", width=1)),
                    name=f"run-{run} ICA cutoff",
                    hovertemplate=f"Run {run} ICA cutoff<br>Component {marker_x}<br>{marker_y:.1f}% variance<extra></extra>",
                    showlegend=True,
                ))

        fig.update_layout(
            title=f"Cumulative PCA Explained Variance — sub-{subject}",
            xaxis_title="PCA Component Index",
            yaxis_title="Cumulative Explained Variance (%)",
            yaxis_range=[0, 101],
            template="plotly_white",
            height=450,
            hovermode="closest",
        )
        fig.add_hline(y=90, line_dash="dash", line_color="gray",
                      annotation_text="90%", annotation_position="top left")
        fig.add_hline(y=99, line_dash="dot", line_color="gray",
                      annotation_text="99%", annotation_position="top left")
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False)

    # --- Matplotlib fallback ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for run, pev in runs_data:
        cumvar = np.array(_to_cumvar_pct(pev))
        ax.plot(range(len(cumvar)), cumvar, label=f"run-{run}", linewidth=1.8)
        n_ica_actual = next(
            (r.get("n_ica_components") for r in metrics["per_run"] if r.get("run") == run),
            None,
        )
        if n_ica_actual is not None and n_ica_actual <= len(cumvar):
            ax.plot(n_ica_actual - 1, cumvar[n_ica_actual - 1], "D", markersize=7)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(99, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylim(0, 101)
    ax.set_xlabel("PCA Component Index")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title(f"Cumulative PCA Explained Variance — sub-{subject}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'


def _create_ica_topography_slider_html(
    ica, run: str, ecg_inds: list, eog_inds: list
) -> str:
    """Generate an HTML snippet with a JS slider to browse ICA topographies.

    Each ICA component is rendered individually as a PNG and embedded as
    base64.  Navigation is handled by a range slider + prev/next buttons.
    ECG and EOG components are highlighted with coloured badges.

    Args:
        ica: Fitted MNE ICA object.
        run: Run ID string (used to namespace JS variables).
        ecg_inds: List of ECG component indices.
        eog_inds: List of EOG component indices.

    Returns:
        HTML string.
    """
    import matplotlib
    matplotlib.use("Agg")

    n_components = ica.n_components_
    safe_run = run.replace("-", "_")

    images = []
    for comp_idx in range(n_components):
        try:
            figs = ica.plot_components(picks=[comp_idx], show=False)
            if not isinstance(figs, list):
                figs = [figs]
            fig = figs[0]
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
        except Exception:
            b64 = ""

        if comp_idx in ecg_inds and comp_idx in eog_inds:
            comp_type = "ECG+EOG"
            badge_color = "#d62728"
        elif comp_idx in ecg_inds:
            comp_type = "ECG"
            badge_color = "#e377c2"
        elif comp_idx in eog_inds:
            comp_type = "EOG"
            badge_color = "#ff7f0e"
        else:
            comp_type = "kept"
            badge_color = "#2ca02c"

        images.append({
            "b64": b64,
            "type": comp_type,
            "color": badge_color,
            "idx": comp_idx,
        })

    if not images:
        return "<p><i>No ICA topographies to display.</i></p>"

    # Build JS data array (only src + metadata, no big inline object)
    js_images = json.dumps([
        {"src": f"data:image/png;base64,{img['b64']}",
         "type": img["type"],
         "color": img["color"],
         "idx": img["idx"]}
        for img in images
    ])

    html = f"""
<div id="ica-slider-{safe_run}" style="text-align:center; font-family: sans-serif; padding: 10px;">
  <div style="margin-bottom:8px;">
    <span id="topo-label-{safe_run}" style="font-weight:bold; font-size:15px;"></span>
    &nbsp;
    <span id="topo-badge-{safe_run}"
          style="display:inline-block; padding:2px 10px; border-radius:10px;
                 font-size:12px; font-weight:bold; color:white; background:#2ca02c;"></span>
  </div>
  <div style="margin-bottom:6px;">
    <button onclick="prevICA_{safe_run}()"
            style="padding:4px 12px; margin-right:8px; cursor:pointer;">&#8592; Prev</button>
    <input type="range" id="slider-{safe_run}" min="0" max="{len(images)-1}" value="0"
           style="width:300px; vertical-align:middle;"
           oninput="gotoICA_{safe_run}(parseInt(this.value))">
    <button onclick="nextICA_{safe_run}()"
            style="padding:4px 12px; margin-left:8px; cursor:pointer;">Next &#8594;</button>
  </div>
  <div>
    <img id="topo-img-{safe_run}" src="" style="max-width:480px; border:1px solid #ccc; border-radius:4px;"/>
  </div>
</div>
<script>
(function() {{
  var _data_{safe_run} = {js_images};
  var _cur_{safe_run} = 0;

  function _render_{safe_run}(idx) {{
    _cur_{safe_run} = idx;
    var d = _data_{safe_run}[idx];
    document.getElementById('topo-img-{safe_run}').src = d.src;
    document.getElementById('topo-label-{safe_run}').textContent =
      'IC' + d.idx + '  (' + (idx+1) + ' / ' + _data_{safe_run}.length + ')';
    var badge = document.getElementById('topo-badge-{safe_run}');
    badge.textContent = d.type;
    badge.style.background = d.color;
    document.getElementById('slider-{safe_run}').value = idx;
  }}

  window.gotoICA_{safe_run} = function(idx) {{ _render_{safe_run}(idx); }};
  window.prevICA_{safe_run} = function() {{
    if (_cur_{safe_run} > 0) _render_{safe_run}(_cur_{safe_run} - 1);
  }};
  window.nextICA_{safe_run} = function() {{
    if (_cur_{safe_run} < _data_{safe_run}.length - 1) _render_{safe_run}(_cur_{safe_run} + 1);
  }};

  // Keyboard navigation when hovering the container
  document.getElementById('ica-slider-{safe_run}').addEventListener('keydown', function(e) {{
    if (e.key === 'ArrowLeft') {{ window.prevICA_{safe_run}(); e.preventDefault(); }}
    if (e.key === 'ArrowRight') {{ window.nextICA_{safe_run}(); e.preventDefault(); }}
  }});
  document.getElementById('ica-slider-{safe_run}').setAttribute('tabindex', '0');

  // Init
  _render_{safe_run}(0);
}})();
</script>
"""
    return html


def _load_ica_and_plot_components(
    subject: str, run: str, derivatives_root: Path, ecg_inds: list, eog_inds: list
) -> list:
    """Try to load saved ICA object and generate component property figures.

    Args:
        subject: Subject ID.
        run: Run ID.
        derivatives_root: Path to derivatives directory.
        ecg_inds: ECG component indices.
        eog_inds: EOG component indices.

    Returns:
        List of (figure, title) tuples, or empty list if ICA not available.
    """
    import mne
    from mne.preprocessing import read_ica

    # Try to find ICA file
    ica_path = (
        derivatives_root / "preprocessed" / f"sub-{subject}" / "meg"
        / f"sub-{subject}_task-gradCPT_run-{run}_desc-ica_meg-ica.fif"
    )
    if not ica_path.exists():
        return []

    try:
        ica = read_ica(ica_path)
    except Exception as e:
        logger.warning(f"Could not load ICA for sub-{subject} run-{run}: {e}")
        return []

    # Returns list of (html_str, title) tuples (html=None means use figure instead)
    results = []

    # Interactive topography slider (one per run)
    try:
        slider_html = _create_ica_topography_slider_html(ica, run, ecg_inds, eog_inds)
        results.append((slider_html, f"Run {run} - ICA topographies (slider)"))
    except Exception as e:
        logger.warning(f"Could not create ICA topography slider for run {run}: {e}")

    # Note: plot_properties is intentionally NOT called here.
    # The per-run report (run_preprocessing.py) already does this correctly using
    # create_ecg/eog_epochs(raw) — the original uncleaned raw — so the artifact
    # component activation is intact. Any epoch file available here would be
    # post-ICA (cleaned), making the excluded component's activation near-zero
    # and the ERP curve invisible. Refer to the per-run report for IC properties.

    return results


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
    report = mne.Report(title=f"Preprocessing Summary - sub-{subject}", image_format="png", verbose=False)

    # Add HTML overview — epoch bar, ICA scores, and PCA variance are now
    # embedded inline here so they are always visible (no hidden-tab issue).
    html_content = _build_subject_html(subject, metrics, config)
    report.add_html(html_content, title="Overview")

    # Add ICA topography slider + component property figures (from saved ICA objects)
    for r in metrics["per_run"]:
        if not r.get("complete"):
            continue
        run = r["run"]
        ecg_inds = r.get("ecg_components", [])
        eog_inds = r.get("eog_components", [])
        ica_results = _load_ica_and_plot_components(
            subject, run, derivatives_root, ecg_inds, eog_inds
        )
        for item, title in ica_results:
            if isinstance(item, str):
                # HTML string (e.g. interactive topography slider)
                report.add_html(item, title=title)
            else:
                # Matplotlib figure
                report.add_figure(item, title=title)
                plt.close(item)

    # Output paths
    out_dir = derivatives_root / "preprocessed" / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"sub-{subject}_preprocessing-summary.html"
    json_path = out_dir / f"sub-{subject}_preprocessing-summary.json"

    report.save(html_path, open_browser=False, overwrite=True)
    logger.info(f"Subject report saved: {html_path}")

    # Inject sticky nav bar so the link to the dataset report is always visible
    _inject_nav_bar(html_path, [
        ("← Dataset Report", "../group_preprocessing-summary.html"),
    ])

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
        <th>Mean ICA N</th><th>Mean ECG ICs</th><th>Mean EOG ICs</th><th>Forced ICA Runs</th>
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
        n_comps = [r.get("n_ica_components") for r in complete_runs if r.get("n_ica_components") is not None]
        mean_n_comp = f"{np.mean(n_comps):.0f}" if n_comps else "N/A"
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
            <td>{mean_n_comp}</td>
            <td>{mean_ecg}</td>
            <td>{mean_eog}</td>
            <td>{forced_str}</td>
            <td>{'<b style="color:red;">' + str(n_outliers) + '</b>' if n_outliers > 0 else '0'}</td>
        </tr>"""

    html += "</table></div>"

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

    Written as a standalone HTML file (not mne.Report) so that Plotly figures
    are visible on load and not buried in hidden Bootstrap tabs.

    Args:
        derivatives_root: Path to derivatives directory.
        subjects: List of subject IDs.
        task_runs: List of run IDs.
        config: Configuration dict.

    Returns:
        Path to the generated HTML report.
    """
    all_params = collect_all_subject_params(derivatives_root, subjects, task_runs)
    dataset_metrics = aggregate_dataset_metrics(all_params)

    # Build distribution figure HTML
    if _PLOTLY_AVAILABLE:
        dist_html = _create_dataset_distribution_html(dataset_metrics)
    else:
        fig = _create_dataset_distribution_figure(dataset_metrics)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        dist_html = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'

    overview_html = _build_dataset_html(dataset_metrics)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Dataset Preprocessing Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 20px; background: #f8f9fa; color: #212529; }}
    .container {{ max-width: 1400px; margin: 0 auto; background: white;
                 padding: 24px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
    h2 {{ color: #1a252f; }}
    h3 {{ color: #2c3e50; margin-top: 24px; }}
    hr {{ border: none; border-top: 1px solid #dee2e6; margin: 24px 0; }}
  </style>
  {_shared_css_js()}
</head>
<body>
<div class="container">
{overview_html}
<hr/>
<h3>Distribution Plots</h3>
{dist_html}
</div>
</body>
</html>"""

    # Output paths
    out_dir = derivatives_root / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "group_preprocessing-summary.html"
    json_path = out_dir / "group_preprocessing-summary.json"

    html_path.write_text(full_html, encoding="utf-8")
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
        print(f"Generating subject-level reports for {len(subjects)} subjects...")
        for subj in subjects:
            try:
                subj_report_path = generate_subject_report(
                    subj, derivatives_root, task_runs, config
                )
                print(f"  sub-{subj}: {subj_report_path}")
            except Exception as exc:
                print(f"  sub-{subj}: SKIPPED ({exc})")

        print(f"Generating dataset-level report for {len(subjects)} subjects...")
        report_path = generate_dataset_report(
            derivatives_root, subjects, task_runs, config
        )
        print(f"Dataset report: {report_path}")


if __name__ == "__main__":
    main()
