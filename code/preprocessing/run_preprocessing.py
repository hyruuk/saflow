"""MEG preprocessing pipeline for saflow.

Stage 1 of the saflow pipeline: Raw BIDS → Preprocessed data.

This script:
1. Loads raw BIDS data
2. Applies gradient compensation
3. Filters data (bandpass + notch)
4. Identifies bad epochs with AutoReject
5. Removes artifacts with ICA (ECG + EOG components)
6. Saves preprocessed continuous data, epochs, logs, and HTML report

Usage:
    # Use paths from config
    python code/preprocessing/run_preprocessing.py -s 04

    # Process specific runs
    python code/preprocessing/run_preprocessing.py -s 04 -r 02 03

    # Override paths
    python code/preprocessing/run_preprocessing.py -s 04 --bids-root /path/to/bids

Author: Claude (Anthropic)
Date: 2026-01-30
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import numpy as np
import pandas as pd
from mne_bids import read_raw_bids, write_raw_bids
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from code.preprocessing.autoreject_pipeline import (
    get_good_epochs_mask,
    run_autoreject,
    run_autoreject_both,
)
from code.preprocessing.ica_pipeline import run_ica_pipeline
from code.preprocessing.utils import (
    add_bad_epoch_annotations,
    apply_filtering,
    compute_event_counts,
    compute_isi_statistics,
    compute_or_load_noise_cov,
    create_epochs,
    create_preprocessing_paths,
    detect_bad_epochs_data_driven,
    detect_bad_epochs_threshold,
    filter_events_by_type,
    pre_ar2_outlier_filter,
)
from code.utils.config import load_config
from code.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
console = Console()


def _cohen_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute Cohen's kappa between two boolean arrays."""
    n = len(y1)
    if n == 0:
        return 0.0
    observed_agreement = np.mean(y1 == y2)
    p1 = np.mean(y1)
    p2 = np.mean(y2)
    expected_agreement = p1 * p2 + (1 - p1) * (1 - p2)
    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def _create_epoch_comparison_figure(
    ar1_flags: np.ndarray,
    ar2_flags: np.ndarray,
    threshold_flags: np.ndarray,
) -> plt.Figure:
    """Create a 3-column heatmap comparing bad epoch flags across methods.

    Args:
        ar1_flags: Boolean array (M,) — AR1 flags mapped to surviving epochs.
        ar2_flags: Boolean array (M,) — AR2 flags on post-ICA epochs.
        threshold_flags: Boolean array (M,) — Threshold flags on post-ICA epochs.

    Returns:
        Matplotlib figure.
    """
    n_epochs = len(ar1_flags)
    data = np.column_stack([ar1_flags.astype(float),
                            ar2_flags.astype(float),
                            threshold_flags.astype(float)])

    fig, ax = plt.subplots(figsize=(5, max(4, n_epochs * 0.04)))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "#d32f2f"])
    ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["AR1\n(pre-ICA)", "AR2\n(post-ICA)", "Threshold\n(post-ICA)"])
    ax.set_ylabel("Epoch index")
    ax.set_title("3-Way Bad Epoch Comparison (red = bad)")

    # Annotate counts at bottom
    counts = [int(ar1_flags.sum()), int(ar2_flags.sum()), int(threshold_flags.sum())]
    for i, c in enumerate(counts):
        ax.text(i, n_epochs + 0.5, f"{c} bad",
                ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_ylim(n_epochs + 1.5, -0.5)
    fig.tight_layout()
    return fig


def _create_ptp_distribution_figure(
    epochs_pre_ica: mne.Epochs,
    epochs_post_ica: mne.Epochs,
    ar1_flags: np.ndarray,
    ar2_flags: np.ndarray,
    threshold_flags: np.ndarray,
    threshold_stats: dict,
) -> plt.Figure:
    """Create PTP amplitude distribution figure with two side-by-side subplots.

    Args:
        epochs_pre_ica: Pre-ICA epochs (1Hz filtered, N epochs).
        epochs_post_ica: Post-ICA epochs (same count as ar2_flags).
        ar1_flags: Boolean array (N,) — AR1 flags on pre-ICA epochs.
        ar2_flags: Boolean array — AR2 flags on post-ICA epochs.
        threshold_flags: Boolean array — Threshold flags on post-ICA epochs.
        threshold_stats: Statistics from threshold detection (includes reject_threshold).

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # Pre-ICA PTP
    picks_mag = mne.pick_types(epochs_pre_ica.info, meg="mag")
    data_pre = epochs_pre_ica.get_data(picks=picks_mag)
    ptp_pre = np.max(np.ptp(data_pre, axis=2), axis=1)  # max PTP across channels

    good_pre = ~ar1_flags
    ax1.hist(ptp_pre[good_pre] * 1e15, bins=30, color="#4caf50", alpha=0.7, label="Good (AR1)")
    if ar1_flags.any():
        ax1.hist(ptp_pre[ar1_flags] * 1e15, bins=30, color="#d32f2f", alpha=0.7, label="Bad (AR1)")
    ax1.set_xlabel("Max PTP amplitude (fT)")
    ax1.set_ylabel("Count")
    ax1.set_title("Pre-ICA Epochs")
    ax1.legend()

    # Post-ICA PTP
    picks_mag_post = mne.pick_types(epochs_post_ica.info, meg="mag")
    data_post = epochs_post_ica.get_data(picks=picks_mag_post)
    ptp_post = np.max(np.ptp(data_post, axis=2), axis=1)

    # Categorize: good, AR2-bad, threshold-bad, both-bad
    both_bad = ar2_flags & threshold_flags
    only_ar2 = ar2_flags & ~threshold_flags
    only_thresh = threshold_flags & ~ar2_flags
    good_post = ~ar2_flags & ~threshold_flags

    ax2.hist(ptp_post[good_post] * 1e15, bins=30, color="#4caf50", alpha=0.7, label="Good")
    if only_ar2.any():
        ax2.hist(ptp_post[only_ar2] * 1e15, bins=30, color="#d32f2f", alpha=0.7, label="Bad (AR2 only)")
    if only_thresh.any():
        ax2.hist(ptp_post[only_thresh] * 1e15, bins=30, color="#ff9800", alpha=0.7, label="Bad (Threshold only)")
    if both_bad.any():
        ax2.hist(ptp_post[both_bad] * 1e15, bins=30, color="#9c27b0", alpha=0.7, label="Bad (AR2 + Threshold)")

    # Add threshold line(s)
    reject_threshold = threshold_stats.get("reject_threshold", {})
    if "mag" in reject_threshold:
        thresh_val = reject_threshold["mag"]
        mode = threshold_stats.get("mode", "fixed")
        label = f"Threshold ({thresh_val*1e15:.0f} fT, {mode})"
        ax2.axvline(thresh_val * 1e15, color="red", linestyle="--",
                     linewidth=2, label=label)

    ax2.set_xlabel("Max PTP amplitude (fT)")
    ax2.set_title("Post-ICA Epochs")
    ax2.legend()

    fig.suptitle("Peak-to-Peak Amplitude Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _create_isi_distribution_figure(isi_stats: dict, events_stim: np.ndarray, sfreq: float) -> plt.Figure:
    """Create ISI distribution histogram.

    Args:
        isi_stats: ISI statistics dict from compute_isi_statistics.
        events_stim: Stimulus events array.
        sfreq: Sampling frequency.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(events_stim) < 2:
        ax.text(0.5, 0.5, "Not enough events for ISI", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    isi_sec = np.diff(events_stim[:, 0]) / sfreq
    ax.hist(isi_sec, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
    if isi_stats.get("mean") is not None:
        ax.axvline(isi_stats["mean"], color="red", linestyle="--", linewidth=2,
                   label=f"Mean={isi_stats['mean']:.3f}s")
        ax.axvline(isi_stats["median"], color="orange", linestyle="--", linewidth=2,
                   label=f"Median={isi_stats['median']:.3f}s")
    ax.set_xlabel("Inter-Stimulus Interval (s)")
    ax.set_ylabel("Count")
    ax.set_title("ISI Distribution (Stimulus Events)")
    ax.legend()
    fig.tight_layout()
    return fig


def generate_preprocessing_report(
    raw: mne.io.Raw,
    cleaned_raw: mne.io.Raw,
    raw_filt: mne.io.Raw,
    epochs_filt: mne.Epochs,
    epochs_preproc: mne.Epochs,
    epochs_interpolated: mne.Epochs,
    reject_log_first: object,
    reject_log_second: object,
    threshold_bad_mask: np.ndarray,
    threshold_stats: dict,
    ica,
    ecg_inds: list,
    eog_inds: list,
    noise_cov: mne.Covariance,
    picks: list,
    event_counts: dict = None,
    isi_stats: dict = None,
    events_stim: np.ndarray = None,
    sfreq: float = None,
    pre_ar2_stats: dict = None,
) -> mne.Report:
    """Generate HTML report for preprocessing with three-way comparison.

    Args:
        raw: Original raw data.
        cleaned_raw: ICA-cleaned continuous data.
        raw_filt: Filtered data for ICA (1 Hz highpass).
        epochs_filt: Pre-ICA epochs (1 Hz filtered, N epochs).
        epochs_preproc: All ICA-cleaned epochs (Freq+Rare, no reject_dict).
        epochs_interpolated: AR2-interpolated epochs (or None).
        reject_log_first: AutoReject log from first pass (for ICA).
        reject_log_second: AutoReject log from second pass (or None).
        threshold_bad_mask: Boolean mask from threshold-based detection.
        threshold_stats: Statistics from threshold-based detection.
        ica: Fitted ICA object.
        ecg_inds: ECG component indices.
        eog_inds: EOG component indices.
        noise_cov: Noise covariance matrix.
        picks: Channel picks.
        event_counts: Dict of event counts per type.
        isi_stats: ISI statistics dict.
        events_stim: Stimulus events array (for ISI figure).
        sfreq: Sampling frequency.
        pre_ar2_stats: Pre-AR2 filter statistics (or None).

    Returns:
        MNE Report object.
    """
    logger.info("Generating HTML report")

    report = mne.Report(verbose=False)

    # Safe start time for plotting (handle cropped data)
    plot_start = min(10, raw.times[-1] - 20) if raw.times[-1] > 20 else 0

    # Raw data
    report.add_raw(raw, title="Raw data")
    fig = raw.plot(duration=20, start=plot_start, show=False)
    report.add_figure(fig, title="Time series (raw)")
    fig = raw.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (raw)")

    # Filtered data
    report.add_raw(cleaned_raw, title="Filtered data (ICA-cleaned)")
    fig = cleaned_raw.plot(duration=20, start=plot_start, show=False)
    report.add_figure(fig, title="Time series (ICA-cleaned)")
    fig = cleaned_raw.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (ICA-cleaned)")

    # Event counts and ISI section
    event_html = ""
    if event_counts:
        event_html += "<h3>Event Counts</h3><table border='1' style='border-collapse:collapse;'>"
        event_html += "<tr style='background-color:#f0f0f0;'><th>Type</th><th>Count</th></tr>"
        for k, v in sorted(event_counts.items()):
            event_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        event_html += "</table>"

    if isi_stats and isi_stats.get("mean") is not None:
        event_html += f"""<h3>ISI Statistics</h3>
        <table border='1' style='border-collapse:collapse;'>
        <tr><td>Mean</td><td>{isi_stats['mean']:.3f}s</td></tr>
        <tr><td>Std</td><td>{isi_stats['std']:.3f}s</td></tr>
        <tr><td>Min</td><td>{isi_stats['min']:.3f}s</td></tr>
        <tr><td>Max</td><td>{isi_stats['max']:.3f}s</td></tr>
        <tr><td>Median</td><td>{isi_stats['median']:.3f}s</td></tr>
        </table>"""

    if event_html:
        report.add_html(event_html, title="Event Summary")

    # ISI distribution figure
    if events_stim is not None and sfreq is not None and isi_stats is not None:
        fig_isi = _create_isi_distribution_figure(isi_stats, events_stim, sfreq)
        report.add_figure(fig_isi, title="ISI Distribution")
        plt.close(fig_isi)

    # Evoked responses before cleaning (from pre-ICA 1Hz epochs)
    for cond in ["Freq", "Rare"]:
        if cond in epochs_filt.event_id:
            evoked = epochs_filt[cond].average()
            fig = evoked.plot(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - before ICA")
            fig = evoked.plot_joint(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - Joint - before ICA")

    # First AutoReject pass (for ICA fitting) — uses epochs_filt (N epochs)
    if np.sum(reject_log_first.bad_epochs) > 0:
        try:
            fig = epochs_filt[reject_log_first.bad_epochs].plot(show=False)
            report.add_figure(fig, title="Bad epochs (1st AR pass)")
        except Exception as e:
            logger.warning(f"Could not plot bad epochs: {e}")

    fig = reject_log_first.plot("horizontal", show=False)
    report.add_figure(fig, title="AutoReject decisions (1st pass - for ICA)")

    # Second AutoReject pass (after ICA)
    if reject_log_second is not None:
        if np.sum(reject_log_second.bad_epochs) > 0:
            try:
                fig = epochs_preproc[reject_log_second.bad_epochs].plot(show=False)
                report.add_figure(fig, title="Bad epochs (2nd AR pass - post-ICA)")
            except Exception as e:
                logger.warning(f"Could not plot bad epochs: {e}")

        fig = reject_log_second.plot("horizontal", show=False)
        report.add_figure(fig, title="AutoReject decisions (2nd pass - post-ICA)")

    # Comprehensive summary with detailed metrics
    n_bad_first = np.sum(reject_log_first.bad_epochs)
    n_total_filt = len(epochs_filt)  # N: total pre-ICA stimulus epochs
    n_total_preproc = len(epochs_preproc)  # Same count (1:1 mapping)
    n_good_first = n_total_filt - n_bad_first

    # AR1 flags map 1:1 to epochs_preproc (same events, same count)
    ar1_flags = reject_log_first.bad_epochs

    # Second AR pass stats
    if reject_log_second is not None:
        ar2_flags = reject_log_second.bad_epochs
    else:
        ar2_flags = np.zeros(n_total_preproc, dtype=bool)

    # --- 3-Way Comparison Figures ---
    # Now direct 1:1 comparison — no mapping needed
    has_3way = (reject_log_second is not None
                and len(ar1_flags) == len(ar2_flags)
                and len(ar2_flags) == len(threshold_bad_mask))

    if has_3way:
        ar1_m = ar1_flags
        ar2_m = ar2_flags
        thr_m = threshold_bad_mask
        M = len(ar1_m)

        # 3-way heatmap
        fig_heatmap = _create_epoch_comparison_figure(ar1_m, ar2_m, thr_m)
        report.add_figure(fig_heatmap, title="3-Way Bad Epoch Comparison")
        plt.close(fig_heatmap)

        # PTP distribution
        fig_ptp = _create_ptp_distribution_figure(
            epochs_filt, epochs_preproc,
            ar1_m, ar2_m, thr_m,
            threshold_stats,
        )
        report.add_figure(fig_ptp, title="PTP Amplitude Distribution")
        plt.close(fig_ptp)

        # Compute 3-way agreement categories
        all_bad = ar1_m & ar2_m & thr_m
        all_good = ~ar1_m & ~ar2_m & ~thr_m
        ar1_ar2_only = ar1_m & ar2_m & ~thr_m
        ar1_thr_only = ar1_m & ~ar2_m & thr_m
        ar2_thr_only = ~ar1_m & ar2_m & thr_m
        only_ar1 = ar1_m & ~ar2_m & ~thr_m
        only_ar2 = ~ar1_m & ar2_m & ~thr_m
        only_thr = ~ar1_m & ~ar2_m & thr_m

        # ICA effectiveness
        n_ar1_bad = int(ar1_m.sum())
        n_still_bad = int((ar1_m & ar2_m).sum())
        n_rescued = n_ar1_bad - n_still_bad
        rescue_rate = 100 * n_rescued / n_ar1_bad if n_ar1_bad > 0 else 0.0

        # Pairwise kappas
        kappa_ar1_ar2 = _cohen_kappa(ar1_m, ar2_m) if M > 0 else 0.0
        kappa_ar1_thr = _cohen_kappa(ar1_m, thr_m) if M > 0 else 0.0
        kappa_ar2_thr = _cohen_kappa(ar2_m, thr_m) if M > 0 else 0.0
        agree_ar1_ar2 = 100 * np.mean(ar1_m == ar2_m)
        agree_ar1_thr = 100 * np.mean(ar1_m == thr_m)
        agree_ar2_thr = 100 * np.mean(ar2_m == thr_m)

        # AR2 retention rate
        n_ar2_bad = int(ar2_m.sum())
        ar2_retention = 100 * (M - n_ar2_bad) / M if M > 0 else 100.0

        three_way_html = f"""
    <h3>Table 1: 3-Way Bad Epoch Comparison</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Method</th><th>Stage</th><th>Input Epochs</th><th>Bad Detected</th><th>% Bad</th>
    </tr>
    <tr>
        <td>AR Pass 1</td><td>Pre-ICA</td><td>{n_total_filt}</td>
        <td>{int(n_bad_first)}</td><td>{100*n_bad_first/n_total_filt:.1f}%</td>
    </tr>
    <tr>
        <td>AR Pass 2</td><td>Post-ICA</td><td>{M}</td>
        <td>{n_ar2_bad}</td><td>{100*n_ar2_bad/M:.1f}%</td>
    </tr>
    <tr>
        <td>Threshold ({threshold_stats.get('mode', 'fixed')})</td><td>Post-ICA</td><td>{M}</td>
        <td>{int(thr_m.sum())}</td><td>{100*thr_m.sum()/M:.1f}%</td>
    </tr>
    </table>
    <p><b>AR2 Retention Rate:</b> {ar2_retention:.1f}% ({M - n_ar2_bad}/{M} epochs)</p>

    <h3>Table 2: ICA Effectiveness on Bad Epochs</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Metric</th><th>Count</th>
    </tr>
    <tr><td>Bad before ICA (AR1)</td><td>{n_ar1_bad}</td></tr>
    <tr><td>Still bad after ICA (AR2)</td><td>{n_still_bad}</td></tr>
    <tr style="background-color: #d4edda;"><td>Rescued by ICA</td><td>{n_rescued}</td></tr>
    <tr style="background-color: #d4edda;"><td>Rescue rate</td><td>{rescue_rate:.1f}%</td></tr>
    </table>

    <h3>Table 3: 3-Way Agreement Analysis (on {M} post-ICA epochs)</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Category</th><th>Count</th><th>%</th><th>Description</th>
    </tr>
    <tr style="background-color: #f8d7da;">
        <td>All 3 agree: bad</td><td>{int(all_bad.sum())}</td><td>{100*all_bad.sum()/M:.1f}%</td>
        <td>Clearly bad epochs</td>
    </tr>
    <tr style="background-color: #d4edda;">
        <td>All 3 agree: good</td><td>{int(all_good.sum())}</td><td>{100*all_good.sum()/M:.1f}%</td>
        <td>Clean epochs</td>
    </tr>
    <tr>
        <td>AR1 + AR2 only</td><td>{int(ar1_ar2_only.sum())}</td><td>{100*ar1_ar2_only.sum()/M:.1f}%</td>
        <td>Bad before &amp; after ICA, within amplitude threshold</td>
    </tr>
    <tr>
        <td>AR1 + Threshold only</td><td>{int(ar1_thr_only.sum())}</td><td>{100*ar1_thr_only.sum()/M:.1f}%</td>
        <td>Bad before ICA + high amplitude, but AR2 says ok</td>
    </tr>
    <tr>
        <td>AR2 + Threshold only</td><td>{int(ar2_thr_only.sum())}</td><td>{100*ar2_thr_only.sum()/M:.1f}%</td>
        <td>Post-ICA artifacts (ICA may have introduced)</td>
    </tr>
    <tr style="background-color: #fff3cd;">
        <td>Only AR1</td><td>{int(only_ar1.sum())}</td><td>{100*only_ar1.sum()/M:.1f}%</td>
        <td>Fixed by ICA (most common scenario)</td>
    </tr>
    <tr>
        <td>Only AR2</td><td>{int(only_ar2.sum())}</td><td>{100*only_ar2.sum()/M:.1f}%</td>
        <td>Subtle post-ICA artifacts</td>
    </tr>
    <tr>
        <td>Only Threshold</td><td>{int(only_thr.sum())}</td><td>{100*only_thr.sum()/M:.1f}%</td>
        <td>High amplitude but AR says ok</td>
    </tr>
    </table>

    <h3>Table 4: Pairwise Agreement</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Pair</th><th>Agreement %</th><th>Cohen's Kappa</th>
    </tr>
    <tr><td>AR1 vs AR2</td><td>{agree_ar1_ar2:.1f}%</td><td>{kappa_ar1_ar2:.3f}</td></tr>
    <tr><td>AR1 vs Threshold</td><td>{agree_ar1_thr:.1f}%</td><td>{kappa_ar1_thr:.3f}</td></tr>
    <tr><td>AR2 vs Threshold</td><td>{agree_ar2_thr:.1f}%</td><td>{kappa_ar2_thr:.3f}</td></tr>
    </table>
    """
    else:
        three_way_html = """
    <h3>3-Way Comparison</h3>
    <p><i>3-way comparison not available.</i></p>
    """

    # Pre-AR2 filter section
    pre_ar2_html = ""
    if pre_ar2_stats and pre_ar2_stats.get("n_outliers", 0) > 0:
        pre_ar2_html = f"""
    <h3>Pre-AR2 Outlier Filter</h3>
    <p>Removed {pre_ar2_stats['n_outliers']} extreme outlier epochs
    (&gt;{pre_ar2_stats['ptp_multiplier']}x median PTP = {pre_ar2_stats['threshold']*1e15:.0f} fT)
    before AR2 fitting.</p>
    """

    summary_text = f"""
    <h2>Preprocessing Summary</h2>

    <h3>Data Processing Flow</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Stage</th><th>Epochs</th><th>Change</th><th>Description</th>
    </tr>
    <tr>
        <td><b>Stimulus epochs (Freq+Rare)</b></td>
        <td>{n_total_filt}</td>
        <td>-</td>
        <td>Only Freq+Rare events epoched (no Resp, no reject_dict)</td>
    </tr>
    <tr style="background-color: #fff9e6;">
        <td><b>After AR Pass 1</b></td>
        <td>{int(n_good_first)}</td>
        <td>-{int(n_bad_first)} ({100*n_bad_first/n_total_filt:.1f}%)</td>
        <td>Bad epochs flagged for ICA fitting</td>
    </tr>
    <tr style="background-color: #e6f3ff;">
        <td><b>ICA-cleaned epochs</b></td>
        <td>{n_total_preproc}</td>
        <td>-</td>
        <td>ALL stimulus epochs re-epoched from ICA-cleaned Raw (1:1 with AR1)</td>
    </tr>
    <tr style="background-color: #d4edda;">
        <td><b>Output</b></td>
        <td colspan="3"><b>{n_total_preproc} epochs saved</b> (use AR2 reject_log to filter in analysis)</td>
    </tr>
    </table>

    {pre_ar2_html}
    {three_way_html}

    <h3>ICA Components Removed</h3>
    <ul>
    <li><b>ECG components:</b> {len(ecg_inds)} (ICs: {ecg_inds})</li>
    <li><b>EOG components:</b> {len(eog_inds)} (ICs: {eog_inds})</li>
    <li><b>Total components removed:</b> {len(ecg_inds) + len(eog_inds)}</li>
    </ul>

    <h3>Output Files</h3>
    <ul>
    <li><b>Continuous (ICA-cleaned + BAD_AR2 annotations):</b> proc-clean_meg.fif</li>
    <li><b>Epochs (ICA, all):</b> proc-ica (n={n_total_preproc}, with AR2 reject_log)</li>
    <li><b>Epochs (AR2-interpolated):</b> proc-ar2interp</li>
    </ul>
    """
    report.add_html(summary_text, title="Preprocessing Summary")

    # Noise covariance
    report.add_covariance(
        noise_cov, info=cleaned_raw.info, title="Noise covariance matrix"
    )

    # ICA sources (timeseries)
    fig = ica.plot_sources(cleaned_raw, show=False)
    report.add_figure(fig, title="ICA sources")

    # ICA component topographies
    try:
        figs = ica.plot_components(show=False)
        if not isinstance(figs, list):
            figs = [figs]
        for i, fig_topo in enumerate(figs):
            report.add_figure(fig_topo, title=f"ICA component topographies (page {i+1})")
            plt.close(fig_topo)
    except Exception as e:
        logger.warning(f"Could not plot ICA component topographies: {e}")

    # ICA properties for excluded components (ECG, EOG)
    excluded_inds = list(set(ecg_inds + eog_inds))
    if excluded_inds:
        try:
            figs_props = ica.plot_properties(epochs_preproc, picks=excluded_inds, show=False)
            if not isinstance(figs_props, list):
                figs_props = [figs_props]
            for idx, fig_prop in zip(excluded_inds, figs_props):
                label = "ECG" if idx in ecg_inds else "EOG"
                report.add_figure(fig_prop, title=f"ICA component {idx} properties ({label})")
                plt.close(fig_prop)
        except Exception as e:
            logger.warning(f"Could not plot ICA properties for excluded components: {e}")

    # PSD before vs after ICA overlay
    try:
        fig_psd_overlay, ax_psd = plt.subplots(figsize=(10, 5))

        psd_pre = epochs_filt.compute_psd(picks="mag")
        psd_data_pre = psd_pre.get_data().mean(axis=(0, 1))
        freqs_pre = psd_pre.freqs

        psd_post = epochs_preproc.compute_psd(picks="mag")
        psd_data_post = psd_post.get_data().mean(axis=(0, 1))
        freqs_post = psd_post.freqs

        ax_psd.semilogy(freqs_pre, psd_data_pre, color="blue", alpha=0.8, label="Pre-ICA")
        ax_psd.semilogy(freqs_post, psd_data_post, color="red", alpha=0.8, label="Post-ICA")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD (T\u00b2/Hz)")
        ax_psd.set_title("Average PSD: Pre-ICA vs Post-ICA")
        ax_psd.legend()
        ax_psd.grid(True, alpha=0.3)
        fig_psd_overlay.tight_layout()
        report.add_figure(fig_psd_overlay, title="PSD Before vs After ICA")
        plt.close(fig_psd_overlay)
    except Exception as e:
        logger.warning(f"Could not create PSD overlay figure: {e}")

    # Cleaned data - ICA only (all epochs)
    fig = cleaned_raw.plot(duration=20, start=plot_start, show=False)
    report.add_figure(fig, title="Time series (ICA cleaned)")
    fig = cleaned_raw.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (ICA cleaned)")
    fig = epochs_preproc.plot(show=False)
    report.add_figure(fig, title="Epochs (ICA, all)")
    fig = epochs_preproc.plot_psd(average=False, picks="mag", show=False)
    report.add_figure(fig, title="PSD epochs (ICA, all)")

    # AR2-interpolated epochs
    if epochs_interpolated is not None:
        fig = epochs_interpolated.plot(show=False)
        report.add_figure(fig, title="Epochs (AR2 interpolated)")
        fig = epochs_interpolated.plot_psd(average=False, picks="mag", show=False)
        report.add_figure(fig, title="PSD epochs (AR2 interpolated)")

    # Evoked responses - ICA version (Freq+Rare only)
    evokeds_ica = []
    titles_ica = []
    for cond in ["Freq", "Rare"]:
        if cond in epochs_preproc.event_id:
            evoked = epochs_preproc[cond].average()
            fig = evoked.plot(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - ICA")
            fig = evoked.plot_joint(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - Joint - ICA")
            evokeds_ica.append(evoked)
            titles_ica.append(f"Evoked ({cond}) - ICA")

    # Evoked responses - AR2-interpolated version
    evokeds_ar2 = []
    titles_ar2 = []
    if epochs_interpolated is not None:
        for cond in ["Freq", "Rare"]:
            if cond in epochs_interpolated.event_id:
                evoked = epochs_interpolated[cond].average()
                fig = evoked.plot(show=False)
                report.add_figure(fig, title=f"Evoked ({cond}) - AR2 interp")
                fig = evoked.plot_joint(show=False)
                report.add_figure(fig, title=f"Evoked ({cond}) - Joint - AR2 interp")
                evokeds_ar2.append(evoked)
                titles_ar2.append(f"Evoked ({cond}) - AR2")

    # Difference waves: Rare - Freq
    if "Rare" in epochs_preproc.event_id and "Freq" in epochs_preproc.event_id:
        evoked_diff_ica = mne.combine_evoked(
            [epochs_preproc["Rare"].average(), -epochs_preproc["Freq"].average()],
            weights="equal",
        )
        fig = evoked_diff_ica.plot(show=False)
        report.add_figure(fig, title="Evoked (Rare - Freq) - ICA")
        fig = evoked_diff_ica.plot_joint(show=False)
        report.add_figure(fig, title="Evoked (Rare - Freq) - Joint - ICA")
        evokeds_ica.append(evoked_diff_ica)
        titles_ica.append("Evoked (Rare - Freq) - ICA")

        if epochs_interpolated is not None and "Rare" in epochs_interpolated.event_id:
            evoked_diff_ar2 = mne.combine_evoked(
                [epochs_interpolated["Rare"].average(), -epochs_interpolated["Freq"].average()],
                weights="equal",
            )
            fig = evoked_diff_ar2.plot(show=False)
            report.add_figure(fig, title="Evoked (Rare - Freq) - AR2 interp")
            fig = evoked_diff_ar2.plot_joint(show=False)
            report.add_figure(fig, title="Evoked (Rare - Freq) - Joint - AR2 interp")
            evokeds_ar2.append(evoked_diff_ar2)
            titles_ar2.append("Evoked (Rare - Freq) - AR2")

    if evokeds_ica:
        report.add_evokeds(evokeds_ica, titles=titles_ica)
    if evokeds_ar2:
        report.add_evokeds(evokeds_ar2, titles=titles_ar2)

    logger.info("Report generation complete")
    return report


def save_preprocessing_metadata(
    output_path: Path,
    config: dict,
    subject: str,
    run: str,
    params: dict,
):
    """Save preprocessing parameters and provenance to JSON.

    Args:
        output_path: Path to save metadata JSON.
        config: Configuration dictionary.
        subject: Subject ID.
        run: Run number.
        params: Preprocessing parameters used.
    """
    metadata = {
        "script": "code/preprocessing/run_preprocessing.py",
        "timestamp": datetime.now().isoformat(),
        "subject": subject,
        "run": run,
        "parameters": params,
        "config": {
            "filter": config["preprocessing"]["filter"],
            "ica": config["preprocessing"]["ica"],
            "autoreject": config["preprocessing"]["autoreject"],
        },
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=_NumpyEncoder)

    logger.debug(f"Saved preprocessing metadata to {output_path}")


def save_preprocessing_summary(
    output_path: Path,
    subject: str,
    run: str,
    n_total_epochs: int,
    reject_log_first,
    reject_log_second,
    threshold_bad_mask: np.ndarray,
    threshold_stats: dict,
    ecg_inds: list,
    eog_inds: list,
    event_counts: dict = None,
    isi_stats: dict = None,
    pre_ar2_stats: dict = None,
):
    """Save text-based preprocessing summary for easy parsing.

    Args:
        output_path: Path to save summary text file.
        subject: Subject ID.
        run: Run number.
        n_total_epochs: Total number of stimulus epochs.
        reject_log_first: First AR reject log.
        reject_log_second: Second AR reject log (can be None).
        threshold_bad_mask: Boolean mask from threshold detection.
        threshold_stats: Statistics from threshold detection.
        ecg_inds: ECG component indices removed.
        eog_inds: EOG component indices removed.
        event_counts: Dict of event counts per type.
        isi_stats: ISI statistics dict.
        pre_ar2_stats: Pre-AR2 filter statistics.
    """
    n_bad_first = int(np.sum(reject_log_first.bad_epochs))
    ar1_flags = reject_log_first.bad_epochs

    # Event counts section
    event_section = ""
    if event_counts:
        event_section = "\nEvent Counts\n-----------\n"
        for k, v in sorted(event_counts.items()):
            event_section += f"  {k}: {v}\n"

    # ISI section
    isi_section = ""
    if isi_stats and isi_stats.get("mean") is not None:
        isi_section = f"""
ISI Statistics
--------------
  Mean:   {isi_stats['mean']:.3f}s
  Std:    {isi_stats['std']:.3f}s
  Min:    {isi_stats['min']:.3f}s
  Max:    {isi_stats['max']:.3f}s
  Median: {isi_stats['median']:.3f}s
"""

    # Second AR section
    if reject_log_second is not None:
        n_bad_second = int(np.sum(reject_log_second.bad_epochs))
        ar2_flags = reject_log_second.bad_epochs
        both_bad = int(np.sum(ar2_flags & threshold_bad_mask))
        only_ar = int(np.sum(ar2_flags & ~threshold_bad_mask))
        only_thresh = int(np.sum(threshold_bad_mask & ~ar2_flags))
        agreement = float(np.mean(ar2_flags == threshold_bad_mask) * 100)
        n_ar2_good = n_total_epochs - n_bad_second
        retention_rate = 100 * n_ar2_good / n_total_epochs if n_total_epochs > 0 else 100.0

        ar2_section = f"""
Second AutoReject Pass (post-ICA):
  - Bad epochs detected: {n_bad_second} ({100*n_bad_second/n_total_epochs:.1f}%)
  - Retention rate: {retention_rate:.1f}% ({n_ar2_good}/{n_total_epochs})

Threshold vs AutoReject Comparison
----------------------------------
  - Threshold mode: {threshold_stats.get('mode', 'fixed')}
  - Agreement: {agreement:.1f}%
  - Both methods reject: {both_bad} epochs
  - Only AutoReject rejects: {only_ar} epochs
  - Only Threshold rejects: {only_thresh} epochs
"""
    else:
        ar2_section = "\nSecond AutoReject Pass: Not performed\n"
        ar2_flags = np.zeros(n_total_epochs, dtype=bool)
        retention_rate = 100.0

    # Pre-AR2 filter section
    pre_ar2_section = ""
    if pre_ar2_stats and pre_ar2_stats.get("n_outliers", 0) > 0:
        pre_ar2_section = f"""
Pre-AR2 Outlier Filter
---------------------
  Removed: {pre_ar2_stats['n_outliers']} extreme epochs
  Threshold: >{pre_ar2_stats['ptp_multiplier']}x median PTP ({pre_ar2_stats['threshold']*1e15:.0f} fT)
"""

    # 3-way comparison (1:1 now, no mapping needed)
    three_way_section = ""
    if (reject_log_second is not None
            and len(ar1_flags) == len(ar2_flags)
            and len(ar2_flags) == len(threshold_bad_mask)):
        ar1_m = ar1_flags
        ar2_m = ar2_flags
        thr_m = threshold_bad_mask
        M = len(ar1_m)

        n_ar1_bad = int(ar1_m.sum())
        n_still_bad = int((ar1_m & ar2_m).sum())
        n_rescued = n_ar1_bad - n_still_bad
        rescue_rate = 100 * n_rescued / n_ar1_bad if n_ar1_bad > 0 else 0.0

        all_bad = int((ar1_m & ar2_m & thr_m).sum())
        all_good = int((~ar1_m & ~ar2_m & ~thr_m).sum())
        only_ar1_ct = int((ar1_m & ~ar2_m & ~thr_m).sum())
        only_ar2_ct = int((~ar1_m & ar2_m & ~thr_m).sum())
        only_thr_ct = int((~ar1_m & ~ar2_m & thr_m).sum())

        agree_ar1_ar2 = 100 * np.mean(ar1_m == ar2_m)
        agree_ar1_thr = 100 * np.mean(ar1_m == thr_m)
        agree_ar2_thr = 100 * np.mean(ar2_m == thr_m)

        kappa_ar1_ar2 = _cohen_kappa(ar1_m, ar2_m) if M > 0 else 0.0
        kappa_ar1_thr = _cohen_kappa(ar1_m, thr_m) if M > 0 else 0.0
        kappa_ar2_thr = _cohen_kappa(ar2_m, thr_m) if M > 0 else 0.0

        three_way_section = f"""
3-Way Bad Epoch Comparison (on {M} epochs, 1:1 mapping)
-------------------------------------------------------
  AR1 (pre-ICA):    {n_ar1_bad} bad ({100*n_ar1_bad/M:.1f}%)
  AR2 (post-ICA):   {int(ar2_m.sum())} bad ({100*ar2_m.sum()/M:.1f}%)
  Threshold:        {int(thr_m.sum())} bad ({100*thr_m.sum()/M:.1f}%)

ICA Effectiveness
-----------------
  Bad before ICA (AR1):     {n_ar1_bad}
  Still bad after ICA (AR2): {n_still_bad}
  Rescued by ICA:           {n_rescued}
  Rescue rate:              {rescue_rate:.1f}%

Agreement Categories
--------------------
  All 3 agree bad:  {all_bad}
  All 3 agree good: {all_good}
  Only AR1 bad:     {only_ar1_ct} (fixed by ICA)
  Only AR2 bad:     {only_ar2_ct}
  Only Threshold:   {only_thr_ct}

Pairwise Agreement
------------------
  AR1 vs AR2:       {agree_ar1_ar2:.1f}% (kappa={kappa_ar1_ar2:.3f})
  AR1 vs Threshold: {agree_ar1_thr:.1f}% (kappa={kappa_ar1_thr:.3f})
  AR2 vs Threshold: {agree_ar2_thr:.1f}% (kappa={kappa_ar2_thr:.3f})
"""

    summary = f"""Preprocessing Summary
====================
Subject: sub-{subject}
Run: {run}
Timestamp: {datetime.now().isoformat()}
{event_section}{isi_section}
Data Quality Metrics
-------------------
Total stimulus epochs (Freq+Rare): {n_total_epochs}

First AutoReject Pass (for ICA fitting):
  - Bad epochs detected: {n_bad_first} ({100*n_bad_first/n_total_epochs:.1f}%)
  - Good epochs for ICA: {n_total_epochs - n_bad_first} ({100*(n_total_epochs-n_bad_first)/n_total_epochs:.1f}%)

ICA Artifact Removal:
  - ECG components removed: {len(ecg_inds)} (ICs: {ecg_inds})
  - EOG components removed: {len(eog_inds)} (ICs: {eog_inds})
  - Total ICA components removed: {len(ecg_inds) + len(eog_inds)}

Threshold-Based Detection (post-ICA, {threshold_stats.get('mode', 'fixed')}):
  - Bad epochs detected: {threshold_stats['n_bad']} ({threshold_stats['pct_bad']:.1f}%)
  - Reject threshold: {threshold_stats['reject_threshold']}
  - Flat threshold: {threshold_stats['flat_threshold']}
{pre_ar2_section}{ar2_section}{three_way_section}
Output Files
-----------
Continuous (ICA + BAD_AR2 annotations): derivatives/preprocessed/sub-{subject}/meg/*_proc-clean_meg.fif
Epochs (ICA, all): derivatives/epochs/sub-{subject}/meg/*_proc-ica (n={n_total_epochs})
Epochs (AR2 interpolated): derivatives/epochs/sub-{subject}/meg/*_proc-ar2interp

Data Retention
--------------
AR2 retention rate: {retention_rate:.1f}%
"""

    with open(output_path, "w") as f:
        f.write(summary)

    logger.info(f"Saved preprocessing summary to {output_path}")


def preprocess_run(
    subject: str,
    run: str,
    bids_root: Path,
    derivatives_root: Path,
    config: dict,
    skip_existing: bool = True,
    crop: float = None,
):
    """Preprocess a single MEG run.

    New pipeline flow:
    Raw -> Filter (0.1Hz + 1Hz) -> Epoch Freq+Rare only (no reject_dict)
      -> AR1 fit-only -> flag bad epochs
      -> Fit ICA on good epochs, apply to continuous raw
      -> Epoch from ICA-cleaned continuous (ALL Freq+Rare, no reject_dict)
      -> Pre-AR2 safety filter (drop >10x median PTP)
      -> AR2 fit -> get reject_log (flags) + transform (interpolated epochs)
      -> Data-driven threshold detection (for comparison)
      -> Save: continuous Raw + BAD annotations, epochs-dropped, epochs-interpolated

    Args:
        subject: Subject ID.
        run: Run number.
        bids_root: BIDS dataset root directory.
        derivatives_root: Derivatives directory.
        config: Configuration dictionary.
        skip_existing: Skip if preprocessed file already exists.
        crop: If set, crop raw data to first N seconds (for quick testing).
    """
    logger.info("=" * 80)
    logger.info(f"Preprocessing subject {subject}, run {run}")
    logger.info("=" * 80)

    # Create file paths
    paths = create_preprocessing_paths(subject, run, bids_root, derivatives_root)

    # Check if already processed
    if skip_existing and paths["preproc"].fpath.exists():
        logger.info(f"Preprocessed file already exists, skipping: {paths['preproc']}")
        return

    # Load configuration parameters
    filter_cfg = config["preprocessing"]["filter"]
    ica_cfg = config["preprocessing"]["ica"]
    ar_cfg = config["preprocessing"]["autoreject"]
    epoch_events_cfg = config["preprocessing"].get("epoch_events", {})
    threshold_cfg = config["preprocessing"].get("threshold_rejection", {})
    pre_ar2_cfg = config["preprocessing"].get("pre_ar2_filter", {})

    # Epoch timing from config
    epochs_cfg = config["analysis"]["epochs"]
    tmin = epochs_cfg["tmin"]
    tmax = epochs_cfg["tmax"]
    logger.info(f"Epoch timing: tmin={tmin}s, tmax={tmax}s (duration={tmax-tmin:.3f}s)")

    # Load raw BIDS data (FIF format with renamed channels)
    logger.info(f"Loading raw data: {paths['raw']}")
    console.print(f"[yellow]\u23f3 Loading raw BIDS data...[/yellow]")
    raw = mne.io.read_raw_fif(str(paths["raw"].fpath), preload=True, verbose=False)
    raw.info["line_freq"] = 60  # Set line frequency for notch filtering
    console.print(f"[green]\u2713 Loaded {raw.n_times} samples ({len(raw.ch_names)} channels)[/green]")

    # Rename and set channel types for ECG/EOG (recorded on EEG channels in CTF)
    channels_cfg = config["preprocessing"].get("channels", {})
    ecg_ch = channels_cfg.get("ecg_channel", "EEG059")
    veog_ch = channels_cfg.get("veog_channel", "EEG057")
    heog_ch = channels_cfg.get("heog_channel", "EEG058")

    rename_map = {}
    if ecg_ch in raw.ch_names:
        rename_map[ecg_ch] = "ECG"
    if veog_ch in raw.ch_names:
        rename_map[veog_ch] = "vEOG"
    if heog_ch in raw.ch_names:
        rename_map[heog_ch] = "hEOG"

    if rename_map:
        mne.rename_channels(raw.info, rename_map)
        logger.info(f"Renamed channels: {rename_map}")

    channel_types = {}
    if "ECG" in raw.ch_names:
        channel_types["ECG"] = "ecg"
    if "vEOG" in raw.ch_names:
        channel_types["vEOG"] = "eog"
    if "hEOG" in raw.ch_names:
        channel_types["hEOG"] = "eog"

    if channel_types:
        raw.set_channel_types(channel_types)
        console.print(f"[green]\u2713 Renamed and set channel types: ECG, vEOG, hEOG[/green]")

    # Crop if requested (for quick testing)
    if crop is not None:
        original_duration = raw.times[-1]
        logger.info(f"Cropping raw data to first {crop} seconds (original: {original_duration:.1f}s)")
        console.print(f"[yellow]\u23f3 Cropping to first {crop} seconds (testing mode)...[/yellow]")
        raw.crop(tmin=0, tmax=crop)
        console.print(f"[green]\u2713 Cropped to {raw.times[-1]:.1f}s ({raw.n_times} samples)[/green]")

    # Resample if configured (do this early, before computing event samples)
    resample_sfreq = config["preprocessing"].get("resample_sfreq")
    if resample_sfreq is not None and raw.info["sfreq"] != resample_sfreq:
        original_sfreq = raw.info["sfreq"]
        logger.info(f"Resampling from {original_sfreq} Hz to {resample_sfreq} Hz")
        console.print(f"[yellow]\u23f3 Resampling from {original_sfreq} Hz to {resample_sfreq} Hz...[/yellow]")
        raw.resample(resample_sfreq, verbose=False)
        console.print(f"[green]\u2713 Resampled to {raw.info['sfreq']} Hz ({raw.n_times} samples)[/green]")

    # ================================================================
    # EVENT LOADING AND FILTERING
    # ================================================================
    events_tsv_path = str(paths["raw"].fpath).replace("_meg.fif", "_events.tsv")
    events_df = pd.read_csv(events_tsv_path, sep="\t")

    # Build full events array (all event types)
    sfreq = raw.info["sfreq"]
    events_list = []
    event_id_all = {}
    for _, row in events_df.iterrows():
        sample = int(row["onset"] * sfreq)
        trial_type = str(row.get("trial_type", row.get("value", "unknown")))
        if trial_type not in event_id_all:
            event_id_all[trial_type] = len(event_id_all) + 1
        events_list.append([sample, 0, event_id_all[trial_type]])
    events_all = np.array(events_list)

    # Compute event counts and ISI stats on ALL events
    event_counts = compute_event_counts(events_df)
    console.print(f"[green]\u2713 Detected {len(events_all)} events: {event_counts}[/green]")

    # Filter to stimulus events only (Freq + Rare, excluding Resp)
    include_types = epoch_events_cfg.get("include", ["Freq", "Rare"])
    events_stim, event_id_stim = filter_events_by_type(events_all, event_id_all, include_types)
    console.print(f"[green]\u2713 Stimulus events (Freq+Rare): {len(events_stim)}[/green]")

    # ISI stats on stimulus events
    isi_stats = compute_isi_statistics(events_stim, sfreq)

    picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True)

    # Apply gradient compensation (required for source reconstruction)
    logger.info("Applying gradient compensation (grade 3)")
    console.print(f"[yellow]\u23f3 Applying gradient compensation (grade 3)...[/yellow]")
    raw = raw.apply_gradient_compensation(grade=3)
    console.print(f"[green]\u2713 Gradient compensation applied[/green]")

    # ================================================================
    # FILTERING
    # ================================================================
    notch_freqs = np.arange(
        raw.info["line_freq"],
        filter_cfg["highcut"] + 1,
        raw.info["line_freq"],
    )

    console.print(f"[yellow]\u23f3 Filtering data ({filter_cfg['lowcut']}-{filter_cfg['highcut']} Hz)...[/yellow]")
    preproc, raw_filt = apply_filtering(
        raw,
        filter_cfg["lowcut"],
        filter_cfg["highcut"],
        notch_freqs,
        picks,
    )
    console.print(f"[green]\u2713 Filtering complete (two versions: {filter_cfg['lowcut']} Hz and 1 Hz highpass)[/green]")

    # ================================================================
    # AR1 EPOCHING (stimulus events only, no reject_dict)
    # ================================================================
    logger.info("Creating epochs for AR1 (Freq+Rare only, no reject_dict)")
    console.print(f"[yellow]\u23f3 Creating stimulus epochs ({tmin} to {tmax} s, Freq+Rare only)...[/yellow]")
    epochs_filt = create_epochs(raw_filt, events_stim, event_id_stim, tmin, tmax, picks)
    console.print(f"[green]\u2713 Created {len(epochs_filt)} stimulus epochs (no reject_dict)[/green]")

    # ================================================================
    # FIRST AUTOREJECT PASS (fit only, for ICA)
    # ================================================================
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]FIRST AUTOREJECT PASS (FOR ICA)[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    ar_first, reject_log_first = run_autoreject(
        epochs_filt,
        n_interpolate=ar_cfg["n_interpolate"],
        consensus=ar_cfg["consensus"],
        n_jobs=config["computing"]["n_jobs"],
        random_state=42,
    )

    good_mask = get_good_epochs_mask(reject_log_first)
    n_good = np.sum(good_mask)
    console.print(f"[green]\u2713 Will use {n_good}/{len(epochs_filt)} good epochs for ICA fitting[/green]")

    # ================================================================
    # ICA
    # ================================================================
    console.print(f"[yellow]\u23f3 Computing/loading noise covariance for sub-{subject}...[/yellow]")
    noise_cov = compute_or_load_noise_cov(subject, bids_root, derivatives_root)
    console.print(f"[green]\u2713 Noise covariance ready[/green]")

    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]ICA ARTIFACT REMOVAL[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    cleaned_raw, _, ica, ecg_inds, eog_inds, ecg_scores, eog_scores = run_ica_pipeline(
        preproc,
        epochs_filt[good_mask],
        noise_cov,
        n_components=ica_cfg.get("n_components", 20),
        random_state=ica_cfg.get("random_state", 42),
    )

    # ================================================================
    # EPOCH FROM ICA-CLEANED CONTINUOUS (same events, no reject_dict)
    # ================================================================
    logger.info("Creating epochs from ICA-cleaned continuous data (Freq+Rare, no reject_dict)")
    console.print(f"\n[yellow]\u23f3 Creating epochs from ICA-cleaned data (no reject_dict)...[/yellow]")
    epochs_preproc = create_epochs(cleaned_raw, events_stim, event_id_stim, tmin, tmax, picks)
    console.print(f"[green]\u2713 Created {len(epochs_preproc)} ICA-cleaned epochs (1:1 with AR1)[/green]")

    # AR1 flags now map 1:1 to epochs_preproc (same events, same count)
    assert len(epochs_preproc) == len(epochs_filt), (
        f"Epoch count mismatch: epochs_preproc={len(epochs_preproc)} vs epochs_filt={len(epochs_filt)}"
    )

    # ================================================================
    # PRE-AR2 SAFETY FILTER
    # ================================================================
    pre_ar2_stats = None
    outlier_mask = None
    if pre_ar2_cfg.get("enabled", False):
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]PRE-AR2 OUTLIER FILTER[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        ptp_multiplier = pre_ar2_cfg.get("ptp_multiplier", 10)
        outlier_mask, pre_ar2_stats = pre_ar2_outlier_filter(epochs_preproc, ptp_multiplier)
        n_outliers = pre_ar2_stats["n_outliers"]
        if n_outliers > 0:
            epochs_for_ar2 = epochs_preproc[~outlier_mask]
            console.print(f"[green]\u2713 Removed {n_outliers} extreme outlier epochs before AR2[/green]")
        else:
            epochs_for_ar2 = epochs_preproc
            console.print(f"[green]\u2713 No extreme outliers found[/green]")
    else:
        epochs_for_ar2 = epochs_preproc

    # ================================================================
    # SECOND AUTOREJECT PASS (fit + reject_log + transform)
    # ================================================================
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]SECOND AUTOREJECT PASS (FIT + REJECT_LOG + TRANSFORM)[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    logger.info("Running second AutoReject pass (fit + reject_log + transform)")

    ar_second, reject_log_second_raw, epochs_interp_subset = run_autoreject_both(
        epochs_for_ar2.copy(),
        n_interpolate=ar_cfg["n_interpolate"],
        consensus=ar_cfg["consensus"],
        n_jobs=config["computing"]["n_jobs"],
        random_state=42,
    )

    # Map AR2 flags back to full epoch array if pre-AR2 filter removed some
    if outlier_mask is not None and np.any(outlier_mask):
        # Build full-size reject log flags: outlier epochs are marked bad
        ar2_flags_full = np.ones(len(epochs_preproc), dtype=bool)  # start all bad
        non_outlier_indices = np.where(~outlier_mask)[0]
        ar2_flags_full[non_outlier_indices] = reject_log_second_raw.bad_epochs
        reject_log_second = reject_log_second_raw  # keep raw for saving

        # Build full-size interpolated epochs: use originals for outlier positions
        # (outliers get flagged as bad, not interpolated)
        epochs_interpolated = epochs_preproc.copy()
        # Copy interpolated data into non-outlier positions
        epochs_interp_data = epochs_interpolated.get_data()
        epochs_interp_data[non_outlier_indices] = epochs_interp_subset.get_data()
        epochs_interpolated = mne.EpochsArray(
            epochs_interp_data, epochs_interpolated.info,
            events=epochs_interpolated.events,
            event_id=epochs_interpolated.event_id,
            tmin=epochs_interpolated.tmin,
        )
    else:
        ar2_flags_full = reject_log_second_raw.bad_epochs
        reject_log_second = reject_log_second_raw
        epochs_interpolated = epochs_interp_subset

    n_ar2_bad = int(np.sum(ar2_flags_full))
    console.print(f"[green]\u2713 AR2: {n_ar2_bad}/{len(epochs_preproc)} bad epochs[/green]")

    # ================================================================
    # THRESHOLD-BASED DETECTION (data-driven or fixed)
    # ================================================================
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]THRESHOLD-BASED EPOCH DETECTION[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    threshold_mode = threshold_cfg.get("mode", "data_driven")
    if threshold_mode == "data_driven":
        mad_mult = threshold_cfg.get("mad_multiplier", 4.0)
        flat_mad_mult = threshold_cfg.get("flat_mad_multiplier", 4.0)
        threshold_bad_mask, threshold_stats = detect_bad_epochs_data_driven(
            epochs_preproc, mad_mult, flat_mad_mult
        )
    else:
        reject_threshold = threshold_cfg.get("reject", {"mag": 4000e-15})
        flat_threshold = threshold_cfg.get("flat", {"mag": 1e-15})
        threshold_bad_mask, threshold_stats = detect_bad_epochs_threshold(
            epochs_preproc,
            reject_threshold=reject_threshold,
            flat_threshold=flat_threshold,
        )
    console.print(
        f"[green]\u2713 Threshold detection ({threshold_mode}): {threshold_stats['n_bad']}/{threshold_stats['n_total']} "
        f"bad epochs ({threshold_stats['pct_bad']:.1f}%)[/green]"
    )

    # ================================================================
    # ADD BAD_AR2 ANNOTATIONS TO CONTINUOUS RAW
    # ================================================================
    add_bad_epoch_annotations(cleaned_raw, epochs_preproc, ar2_flags_full, "BAD_AR2")

    # Also add event annotations
    event_annots = mne.annotations_from_events(
        events_all,
        sfreq=raw.info["sfreq"],
        event_desc={v: k for k, v in event_id_all.items()},
    )
    # Match orig_time so annotations can be concatenated
    event_annots._orig_time = cleaned_raw.annotations.orig_time
    cleaned_raw.set_annotations(cleaned_raw.annotations + event_annots)

    # ================================================================
    # GENERATE REPORT
    # ================================================================
    report = generate_preprocessing_report(
        raw,
        cleaned_raw,
        raw_filt,
        epochs_filt,
        epochs_preproc,
        epochs_interpolated,
        reject_log_first,
        reject_log_second,
        threshold_bad_mask,
        threshold_stats,
        ica,
        ecg_inds,
        eog_inds,
        noise_cov,
        picks,
        event_counts=event_counts,
        isi_stats=isi_stats,
        events_stim=events_stim,
        sfreq=sfreq,
        pre_ar2_stats=pre_ar2_stats,
    )

    # ================================================================
    # SAVE OUTPUTS
    # ================================================================
    logger.info("=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)

    # 1. Save continuous ICA-cleaned data with BAD_AR2 annotations
    logger.info("Saving continuous data (ICA-cleaned + BAD_AR2 annotations)...")
    write_raw_bids(
        cleaned_raw,
        paths["preproc"],
        format="FIF",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )
    logger.info(f"\u2713 Continuous (ICA + annotations): {paths['preproc'].fpath}")

    # 2. Save all ICA epochs (with AR2 reject log for downstream filtering)
    logger.info(f"Saving epochs (ICA, all, n={len(epochs_preproc)})...")
    write_raw_bids(
        cleaned_raw,
        paths["epoch_ica"],
        format="FIF",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )  # Init BIDS structure
    epochs_preproc.save(paths["epoch_ica"].fpath, overwrite=True)
    logger.info(f"\u2713 Epochs (ICA, all): {paths['epoch_ica'].fpath}")

    # 3. Save AR2-interpolated epochs
    logger.info(f"Saving epochs (AR2 interpolated, n={len(epochs_interpolated)})...")
    write_raw_bids(
        cleaned_raw,
        paths["epoch_ar2_interp"],
        format="FIF",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )  # Init BIDS structure
    epochs_interpolated.save(paths["epoch_ar2_interp"].fpath, overwrite=True)
    logger.info(f"\u2713 Epochs (AR2 interpolated): {paths['epoch_ar2_interp'].fpath}")

    # Save first AutoReject log
    ar_log_first_path = Path(str(paths["ARlog_first"].fpath) + ".pkl")
    with open(ar_log_first_path, "wb") as f:
        pickle.dump(reject_log_first, f)
    logger.info(f"Saved first AutoReject log to {ar_log_first_path}")

    # Save second AutoReject log
    ar_log_second_path = Path(str(paths["ARlog_second"].fpath) + ".pkl")
    with open(ar_log_second_path, "wb") as f:
        pickle.dump(reject_log_second, f)
    logger.info(f"Saved second AutoReject log to {ar_log_second_path}")

    # Save HTML report
    logger.info("Saving HTML report...")
    report_path = Path(str(paths["report"].fpath) + ".html")
    report.save(report_path, open_browser=False, overwrite=True)
    logger.info(f"\u2713 Report: {report_path}")

    # Save text summary
    logger.info("Saving text summary...")
    summary_path = Path(str(paths["report"].fpath) + "_summary.txt")
    save_preprocessing_summary(
        summary_path,
        subject,
        run,
        len(epochs_preproc),
        reject_log_first,
        reject_log_second,
        threshold_bad_mask,
        threshold_stats,
        ecg_inds,
        eog_inds,
        event_counts=event_counts,
        isi_stats=isi_stats,
        pre_ar2_stats=pre_ar2_stats,
    )
    logger.info(f"\u2713 Summary: {summary_path}")

    # ================================================================
    # SAVE METADATA
    # ================================================================
    metadata_path = Path(str(paths["preproc"].fpath).replace("_meg.fif", "_params.json"))

    # Retention: based on AR2 flags on full epoch set
    n_total = len(epochs_preproc)
    n_ar2_good = int(np.sum(~ar2_flags_full))
    retention_rate = 100 * n_ar2_good / n_total if n_total > 0 else 100.0

    params = {
        "tmin": tmin,
        "tmax": tmax,
        "filter": filter_cfg,
        "event_counts": event_counts,
        "isi_statistics": isi_stats,
        "ica": {
            "n_components": ica_cfg.get("n_components", 20),
            "ecg_components": ecg_inds,
            "eog_components": eog_inds,
            "ecg_scores": ecg_scores.tolist() if isinstance(ecg_scores, np.ndarray) else ecg_scores,
            "eog_scores": eog_scores.tolist() if isinstance(eog_scores, np.ndarray) else eog_scores,
            "ecg_threshold": float(ica_cfg.get("ecg_threshold", 0.50)),
            "eog_threshold": float(ica_cfg.get("eog_threshold", 4.0)),
            "ecg_forced": bool(np.max(np.abs(ecg_scores)) < ica_cfg.get("ecg_threshold", 0.50)) if len(ecg_scores) > 0 else False,
            "eog_forced": bool(np.max(np.abs(eog_scores)) < ica_cfg.get("eog_threshold", 4.0)) if len(eog_scores) > 0 else False,
        },
        "autoreject_first_pass": {
            "description": "First pass (fit only) for ICA",
            "n_bad_epochs": int(np.sum(reject_log_first.bad_epochs)),
            "n_total_epochs": len(epochs_filt),
            "pct_bad": float(100 * np.sum(reject_log_first.bad_epochs) / len(epochs_filt)),
        },
        "autoreject_second_pass": {
            "description": "Second pass (fit + reject_log + transform) after ICA",
            "n_bad_epochs": n_ar2_bad,
            "n_total_epochs": n_total,
            "pct_bad": float(100 * n_ar2_bad / n_total) if n_total > 0 else 0.0,
            "mode": ar_cfg.get("second_pass_mode", "both"),
        },
        "threshold_detection": {
            "description": f"Threshold-based detection after ICA ({threshold_mode})",
            "mode": threshold_mode,
            "n_bad_epochs": threshold_stats["n_bad"],
            "n_total_epochs": threshold_stats["n_total"],
            "pct_bad": threshold_stats["pct_bad"],
            "reject_threshold": {k: float(v) for k, v in threshold_stats["reject_threshold"].items()},
            "flat_threshold": {k: float(v) for k, v in threshold_stats["flat_threshold"].items()},
        },
        "retention": {
            "n_total_stimulus_epochs": n_total,
            "n_good_after_ar2": n_ar2_good,
            "retention_rate_pct": retention_rate,
        },
    }

    # Pre-AR2 filter stats
    if pre_ar2_stats is not None:
        params["pre_ar2_filter"] = pre_ar2_stats

    # AR2 vs threshold comparison
    params["ar2_vs_threshold"] = {
        "agreement_pct": float(np.mean(ar2_flags_full == threshold_bad_mask) * 100),
        "both_reject": int(np.sum(ar2_flags_full & threshold_bad_mask)),
        "only_ar_rejects": int(np.sum(ar2_flags_full & ~threshold_bad_mask)),
        "only_threshold_rejects": int(np.sum(threshold_bad_mask & ~ar2_flags_full)),
    }

    # 3-way comparison (1:1, no mapping needed)
    ar1_m = reject_log_first.bad_epochs
    ar2_m = ar2_flags_full
    thr_m = threshold_bad_mask
    M = len(ar1_m)

    if M > 0 and len(ar2_m) == M and len(thr_m) == M:
        kappa_ar1_ar2 = float(_cohen_kappa(ar1_m, ar2_m))
        kappa_ar1_thr = float(_cohen_kappa(ar1_m, thr_m))
        kappa_ar2_thr = float(_cohen_kappa(ar2_m, thr_m))

        n_ar1_bad = int(ar1_m.sum())
        n_still_bad = int((ar1_m & ar2_m).sum())
        n_rescued = n_ar1_bad - n_still_bad

        params["three_way_comparison"] = {
            "n_epochs_compared": M,
            "ar1_bad": n_ar1_bad,
            "ar2_bad": int(ar2_m.sum()),
            "threshold_bad": int(thr_m.sum()),
            "ica_effectiveness": {
                "bad_before_ica": n_ar1_bad,
                "still_bad_after_ica": n_still_bad,
                "rescued_by_ica": n_rescued,
                "rescue_rate_pct": float(100 * n_rescued / n_ar1_bad) if n_ar1_bad > 0 else 0.0,
            },
            "agreement_categories": {
                "all_3_bad": int((ar1_m & ar2_m & thr_m).sum()),
                "all_3_good": int((~ar1_m & ~ar2_m & ~thr_m).sum()),
                "ar1_ar2_only": int((ar1_m & ar2_m & ~thr_m).sum()),
                "ar1_thr_only": int((ar1_m & ~ar2_m & thr_m).sum()),
                "ar2_thr_only": int((~ar1_m & ar2_m & thr_m).sum()),
                "only_ar1": int((ar1_m & ~ar2_m & ~thr_m).sum()),
                "only_ar2": int((~ar1_m & ar2_m & ~thr_m).sum()),
                "only_threshold": int((~ar1_m & ~ar2_m & thr_m).sum()),
            },
            "pairwise_kappa": {
                "ar1_vs_ar2": kappa_ar1_ar2,
                "ar1_vs_threshold": kappa_ar1_thr,
                "ar2_vs_threshold": kappa_ar2_thr,
            },
            "pairwise_agreement_pct": {
                "ar1_vs_ar2": float(100 * np.mean(ar1_m == ar2_m)),
                "ar1_vs_threshold": float(100 * np.mean(ar1_m == thr_m)),
                "ar2_vs_threshold": float(100 * np.mean(ar2_m == thr_m)),
            },
        }

    save_preprocessing_metadata(metadata_path, config, subject, run, params)

    # Log final summary
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Subject: {subject}, Run: {run}")
    logger.info(f"Stimulus epochs: {n_total} (Freq+Rare only)")
    logger.info(f"AR2 retention: {retention_rate:.1f}% ({n_ar2_good}/{n_total})")
    logger.info(f"ICA components removed: {len(ecg_inds) + len(eog_inds)} (ECG: {len(ecg_inds)}, EOG: {len(eog_inds)})")
    logger.info("=" * 60)

    # Cleanup
    del raw, raw_filt, preproc, cleaned_raw, epochs_filt
    del epochs_preproc, epochs_interpolated, epochs_for_ar2
    del ar_first, ar_second, ica, report


def main():
    """Main preprocessing workflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="MEG preprocessing pipeline")
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        required=True,
        help="Subject ID to process",
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        default=None,
        help="Run numbers to process (default: all task runs from config)",
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=None,
        help="Override BIDS root from config",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip runs that are already preprocessed",
    )
    parser.add_argument(
        "--crop",
        type=float,
        default=None,
        help="Crop raw data to first N seconds (for quick testing)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        default=False,
        help="Skip generating subject-level aggregate report after preprocessing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Setup logging
    log_dir = Path(config["paths"]["logs"]) / "preprocessing"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"preprocessing_sub-{args.subject}_{timestamp}.log"

    setup_logging(__name__, log_file=log_file, level=args.log_level)

    logger.info("=" * 80)
    logger.info("Preprocessing - Stage 1")
    logger.info("=" * 80)

    # Determine paths
    if args.bids_root:
        bids_root = args.bids_root
        logger.info(f"Using BIDS root from CLI: {bids_root}")
    else:
        bids_root = Path(config["paths"]["data_root"]) / "bids"
        logger.info(f"Using BIDS root from config: {bids_root}")

    derivatives_root = Path(config["paths"]["data_root"]) / config["paths"]["derivatives"]

    # Determine runs to process
    if args.runs:
        runs = args.runs
        logger.info(f"Processing runs from CLI: {runs}")
    else:
        runs = config["bids"]["task_runs"]
        logger.info(f"Processing runs from config: {runs}")

    # Process each run
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Preprocessing sub-{args.subject}...", total=len(runs))

        for run in runs:
            progress.update(task, description=f"Processing run {run}")

            try:
                preprocess_run(
                    args.subject,
                    run,
                    bids_root,
                    derivatives_root,
                    config,
                    args.skip_existing,
                    args.crop,
                )
            except Exception as e:
                logger.error(f"Failed to process run {run}: {e}", exc_info=True)

            progress.advance(task)

    console.print(f"\n[bold green]\u2713 Preprocessing complete for sub-{args.subject}![/bold green]")
    console.print(f"  Processed {len(runs)} runs")
    console.print(f"\n[bold]Output locations:[/bold]")
    console.print(f"  Continuous (ICA + BAD_AR2): {derivatives_root}/preprocessed/sub-{args.subject}/")
    console.print(f"  Epochs (ICA, all): {derivatives_root}/epochs/sub-{args.subject}/*proc-ica*")
    console.print(f"  Epochs (AR2 interp): {derivatives_root}/epochs/sub-{args.subject}/*proc-ar2interp*")
    console.print(f"  AR logs: {derivatives_root}/preprocessed/sub-{args.subject}/*ARlog*.pkl")
    console.print(f"  Reports: {derivatives_root}/preprocessed/sub-{args.subject}/*_report_meg.html")
    console.print(f"\n[bold]Logs:[/bold] {log_file}")
    console.print(f"\n[dim]Use ARlog2.pkl or BAD_AR2 annotations to filter bad epochs downstream[/dim]")

    # Generate subject-level aggregate report
    if not args.skip_report:
        try:
            from code.preprocessing.aggregate_reports import generate_subject_report
            console.print(f"\n[yellow]Generating subject-level report...[/yellow]")
            report_path = generate_subject_report(
                args.subject, derivatives_root, runs, config
            )
            console.print(f"[green]Subject report: {report_path}[/green]")
        except Exception as e:
            logger.warning(f"Could not generate subject-level report: {e}")

    logger.info("Preprocessing complete")
    return 0


if __name__ == "__main__":
    exit(main())
