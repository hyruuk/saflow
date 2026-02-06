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
    run_autoreject_transform,
)
from code.preprocessing.ica_pipeline import run_ica_pipeline
from code.preprocessing.utils import (
    apply_filtering,
    compute_or_load_noise_cov,
    create_epochs,
    create_preprocessing_paths,
    detect_bad_epochs_threshold,
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
    reject_threshold: dict,
) -> plt.Figure:
    """Create PTP amplitude distribution figure with two side-by-side subplots.

    Args:
        epochs_pre_ica: Pre-ICA epochs (1Hz filtered, N epochs).
        epochs_post_ica: Post-ICA epochs (M epochs).
        ar1_flags: Boolean array (N,) — AR1 flags on pre-ICA epochs.
        ar2_flags: Boolean array (M,) — AR2 flags on post-ICA epochs.
        threshold_flags: Boolean array (M,) — Threshold flags on post-ICA epochs.
        reject_threshold: Dict with threshold values (e.g. {'mag': 4000e-15}).

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

    # Add threshold line
    if "mag" in reject_threshold:
        ax2.axvline(reject_threshold["mag"] * 1e15, color="red", linestyle="--",
                     linewidth=2, label=f"Threshold ({reject_threshold['mag']*1e15:.0f} fT)")

    ax2.set_xlabel("Max PTP amplitude (fT)")
    ax2.set_title("Post-ICA Epochs")
    ax2.legend()

    fig.suptitle("Peak-to-Peak Amplitude Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def generate_preprocessing_report(
    raw: mne.io.Raw,
    preproc: mne.io.Raw,
    raw_filt: mne.io.Raw,
    epochs_filt: mne.Epochs,
    epochs: mne.Epochs,
    epochs_ica_only: mne.Epochs,
    epochs_ica_ar: mne.Epochs,
    reject_log_first: object,
    reject_log_second: object,
    threshold_bad_mask: np.ndarray,
    threshold_stats: dict,
    ica,
    ecg_inds: list,
    eog_inds: list,
    noise_cov: mne.Covariance,
    picks: list,
    ar1_flags_mapped: np.ndarray = None,
) -> mne.Report:
    """Generate HTML report for preprocessing with three-way comparison.

    Args:
        raw: Original raw data.
        preproc: Filtered data (ICA-cleaned continuous).
        raw_filt: Filtered data for ICA (1 Hz highpass).
        epochs_filt: Pre-ICA epochs (1 Hz filtered, N epochs).
        epochs: Epochs before ICA (with reject_dict applied, M epochs).
        epochs_ica_only: Epochs after ICA only (no second AR pass).
        epochs_ica_ar: Epochs after ICA + second AR pass.
        reject_log_first: AutoReject log from first pass (for ICA).
        reject_log_second: AutoReject log from second pass (with interpolation).
        threshold_bad_mask: Boolean mask from threshold-based detection.
        threshold_stats: Statistics from threshold-based detection.
        ica: Fitted ICA object.
        ecg_inds: ECG component indices.
        eog_inds: EOG component indices.
        noise_cov: Noise covariance matrix.
        picks: Channel picks.
        ar1_flags_mapped: AR1 flags mapped to surviving (post-reject_dict) epochs.

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
    report.add_raw(preproc, title="Filtered data")
    fig = preproc.plot(duration=20, start=plot_start, show=False)
    report.add_figure(fig, title="Time series (filtered)")
    fig = preproc.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (filtered)")

    # Filtered data for AR
    report.add_raw(raw_filt, title="Filtered data (for AR)")
    fig = raw_filt.plot(duration=20, start=plot_start, show=False)
    report.add_figure(fig, title="Time series (filtered for AR)")
    fig = raw_filt.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (filtered for AR)")

    # Evoked responses before cleaning
    for cond in ["Freq", "Rare", "Resp"]:
        if cond in epochs.event_id:
            evoked = epochs[cond].average()
            fig = evoked.plot(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - before cleaning")
            fig = evoked.plot_joint(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - Joint - before cleaning")

    # First AutoReject pass (for ICA fitting) — uses epochs_filt (N epochs)
    if np.sum(reject_log_first.bad_epochs) > 0:
        try:
            fig = epochs_filt[reject_log_first.bad_epochs].plot(show=False)
            report.add_figure(fig, title="Bad epochs (1st AR pass)")
        except Exception as e:
            logger.warning(f"Could not plot bad epochs: {e}")

    fig = reject_log_first.plot("horizontal", show=False)
    report.add_figure(fig, title="AutoReject decisions (1st pass - for ICA)")

    # Second AutoReject pass (after ICA, fit only for bad epoch detection)
    if reject_log_second is not None:
        if np.sum(reject_log_second.bad_epochs) > 0:
            try:
                fig = epochs_ica_only[reject_log_second.bad_epochs].plot(show=False)
                report.add_figure(fig, title="Bad epochs (2nd AR pass - post-ICA)")
            except Exception as e:
                logger.warning(f"Could not plot bad epochs: {e}")

        fig = reject_log_second.plot("horizontal", show=False)
        report.add_figure(fig, title="AutoReject decisions (2nd pass - post-ICA bad epoch detection)")

    # Comprehensive summary with detailed metrics
    n_bad_first = np.sum(reject_log_first.bad_epochs)
    n_total_filt = len(epochs_filt)  # N: total pre-ICA epochs (1 Hz filtered)
    n_total = len(epochs)  # M: surviving epochs after reject_dict
    n_good_first = n_total_filt - n_bad_first

    # Second AR pass stats (if performed)
    if reject_log_second is not None:
        n_bad_second = np.sum(reject_log_second.bad_epochs)
        ar2_flags = reject_log_second.bad_epochs
    else:
        n_bad_second = 0
        ar2_flags = np.zeros(len(epochs_ica_only), dtype=bool)

    # Threshold detection stats
    n_bad_threshold = threshold_stats["n_bad"]
    reject_thresh = threshold_stats["reject_threshold"].get("mag", "N/A")
    flat_thresh = threshold_stats["flat_threshold"].get("mag", "N/A")

    # --- 3-Way Comparison Figures ---
    has_3way = (ar1_flags_mapped is not None
                and reject_log_second is not None
                and len(ar1_flags_mapped) == len(ar2_flags))

    if has_3way:
        ar1_m = ar1_flags_mapped
        ar2_m = ar2_flags
        thr_m = threshold_bad_mask
        M = len(ar1_m)

        # 3-way heatmap
        fig_heatmap = _create_epoch_comparison_figure(ar1_m, ar2_m, thr_m)
        report.add_figure(fig_heatmap, title="3-Way Bad Epoch Comparison")
        plt.close(fig_heatmap)

        # PTP distribution
        fig_ptp = _create_ptp_distribution_figure(
            epochs_filt, epochs_ica_only,
            reject_log_first.bad_epochs, ar2_m, thr_m,
            threshold_stats["reject_threshold"],
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

        # ICA effectiveness: how many AR1-flagged epochs are still bad after ICA?
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
        <td>{int(ar2_m.sum())}</td><td>{100*ar2_m.sum()/M:.1f}%</td>
    </tr>
    <tr>
        <td>Threshold</td><td>Post-ICA</td><td>{M}</td>
        <td>{int(thr_m.sum())}</td><td>{100*thr_m.sum()/M:.1f}%</td>
    </tr>
    </table>

    <h3>Table 2: ICA Effectiveness on Bad Epochs</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Metric</th><th>Count</th>
    </tr>
    <tr><td>Bad before ICA (AR1, mapped)</td><td>{n_ar1_bad}</td></tr>
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
    <p><i>3-way comparison not available (requires second AR pass and epoch alignment).</i></p>
    """

    summary_text = f"""
    <h2>Preprocessing Summary</h2>

    <h3>Data Processing Flow</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Stage</th><th>Epochs</th><th>Change</th><th>Description</th>
    </tr>
    <tr>
        <td><b>Original (1 Hz filt)</b></td>
        <td>{n_total_filt}</td>
        <td>-</td>
        <td>Total epochs before preprocessing</td>
    </tr>
    <tr style="background-color: #fff9e6;">
        <td><b>After AR Pass 1</b></td>
        <td>{n_good_first}</td>
        <td>-{int(n_bad_first)} ({100*n_bad_first/n_total_filt:.1f}%)</td>
        <td>Bad epochs removed for ICA fitting</td>
    </tr>
    <tr style="background-color: #e6f3ff;">
        <td><b>After reject_dict + ICA</b></td>
        <td>{len(epochs_ica_only)}</td>
        <td>-</td>
        <td>Surviving epochs after amplitude rejection + ICA cleaning</td>
    </tr>
    <tr style="background-color: #d4edda;">
        <td><b>Output</b></td>
        <td colspan="3"><b>{len(epochs_ica_only)} epochs saved</b> (use ARlog2 / threshold mask to filter in analysis)</td>
    </tr>
    </table>

    {three_way_html}

    <h3>ICA Components Removed</h3>
    <ul>
    <li><b>ECG components:</b> {len(ecg_inds)} (ICs: {ecg_inds})</li>
    <li><b>EOG components:</b> {len(eog_inds)} (ICs: {eog_inds})</li>
    <li><b>Total components removed:</b> {len(ecg_inds) + len(eog_inds)}</li>
    </ul>

    <h3>Output Files</h3>
    <ul>
    <li><b>Continuous (ICA-cleaned):</b> derivatives/preprocessed/.../proc-clean_meg.fif</li>
    <li><b>Epochs (ICA):</b> derivatives/epochs/.../proc-ica_meg.fif (n={len(epochs_ica_only)})</li>
    <li><b>AR logs:</b> derivatives/preprocessed/.../ARlog1.pkl (for ICA), ARlog2.pkl (post-ICA)</li>
    </ul>

    <p><i>Note: Use ARlog2.pkl to filter out bad epochs in downstream analysis (stats/ML).
    A detailed text summary is saved alongside this report (*_summary.txt)</i></p>
    """
    report.add_html(summary_text, title="Preprocessing Summary")

    # Noise covariance
    report.add_covariance(
        noise_cov, info=preproc.info, title="Noise covariance matrix"
    )

    # ICA sources (timeseries)
    fig = ica.plot_sources(preproc, show=False)
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
            figs_props = ica.plot_properties(epochs_ica_only, picks=excluded_inds, show=False)
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
        picks_mag = mne.pick_types(raw_filt.info, meg="mag")

        # Pre-ICA PSD (from raw_filt which is the 1 Hz highpass filtered data)
        psd_pre = epochs_filt.compute_psd(picks="mag")
        psd_data_pre = psd_pre.get_data().mean(axis=(0, 1))  # average over epochs and channels
        freqs_pre = psd_pre.freqs

        # Post-ICA PSD
        psd_post = epochs_ica_only.compute_psd(picks="mag")
        psd_data_post = psd_post.get_data().mean(axis=(0, 1))
        freqs_post = psd_post.freqs

        ax_psd.semilogy(freqs_pre, psd_data_pre, color="blue", alpha=0.8, label="Pre-ICA")
        ax_psd.semilogy(freqs_post, psd_data_post, color="red", alpha=0.8, label="Post-ICA")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD (T²/Hz)")
        ax_psd.set_title("Average PSD: Pre-ICA vs Post-ICA")
        ax_psd.legend()
        ax_psd.grid(True, alpha=0.3)
        fig_psd_overlay.tight_layout()
        report.add_figure(fig_psd_overlay, title="PSD Before vs After ICA")
        plt.close(fig_psd_overlay)
    except Exception as e:
        logger.warning(f"Could not create PSD overlay figure: {e}")

    # Cleaned data - ICA only
    fig = preproc.plot(duration=20, start=plot_start, show=False)
    report.add_figure(fig, title="Time series (ICA cleaned)")
    fig = preproc.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (ICA cleaned)")
    fig = epochs_ica_only.plot(show=False)
    report.add_figure(fig, title="Epochs (ICA only)")
    fig = epochs_ica_only.plot_psd(average=False, picks="mag", show=False)
    report.add_figure(fig, title="PSD epochs (ICA only)")

    # Cleaned data - ICA + AR (only if second AR was performed)
    if reject_log_second is not None:
        fig = epochs_ica_ar.plot(show=False)
        report.add_figure(fig, title="Epochs (ICA + AR)")
        fig = epochs_ica_ar.plot_psd(average=False, picks="mag", show=False)
        report.add_figure(fig, title="PSD epochs (ICA + AR)")

    # Evoked responses - ICA only version
    evokeds_ica = []
    titles_ica = []
    for cond in ["Freq", "Rare", "Resp"]:
        if cond in epochs_ica_only.event_id:
            evoked = epochs_ica_only[cond].average()
            fig = evoked.plot(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - ICA only")
            fig = evoked.plot_joint(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - Joint - ICA only")
            evokeds_ica.append(evoked)
            titles_ica.append(f"Evoked ({cond}) - ICA")

    # Evoked responses - ICA + AR version (only if second AR was performed)
    evokeds_ica_ar = []
    titles_ica_ar = []
    if reject_log_second is not None:
        for cond in ["Freq", "Rare", "Resp"]:
            if cond in epochs_ica_ar.event_id:
                evoked = epochs_ica_ar[cond].average()
                fig = evoked.plot(show=False)
                report.add_figure(fig, title=f"Evoked ({cond}) - ICA + AR")
                fig = evoked.plot_joint(show=False)
                report.add_figure(fig, title=f"Evoked ({cond}) - Joint - ICA + AR")
                evokeds_ica_ar.append(evoked)
                titles_ica_ar.append(f"Evoked ({cond}) - ICA+AR")

    # Difference waves: Rare - Freq
    if "Rare" in epochs_ica_only.event_id and "Freq" in epochs_ica_only.event_id:
        # ICA only
        evoked_diff_ica = mne.combine_evoked(
            [epochs_ica_only["Rare"].average(), -epochs_ica_only["Freq"].average()],
            weights="equal",
        )
        fig = evoked_diff_ica.plot(show=False)
        report.add_figure(fig, title="Evoked (Rare - Freq) - ICA only")
        fig = evoked_diff_ica.plot_joint(show=False)
        report.add_figure(fig, title="Evoked (Rare - Freq) - Joint - ICA only")
        evokeds_ica.append(evoked_diff_ica)
        titles_ica.append("Evoked (Rare - Freq) - ICA")

        # ICA + AR (only if second AR was performed)
        if reject_log_second is not None:
            evoked_diff_ica_ar = mne.combine_evoked(
                [epochs_ica_ar["Rare"].average(), -epochs_ica_ar["Freq"].average()],
                weights="equal",
            )
            fig = evoked_diff_ica_ar.plot(show=False)
            report.add_figure(fig, title="Evoked (Rare - Freq) - ICA + AR")
            fig = evoked_diff_ica_ar.plot_joint(show=False)
            report.add_figure(fig, title="Evoked (Rare - Freq) - Joint - ICA + AR")
            evokeds_ica_ar.append(evoked_diff_ica_ar)
            titles_ica_ar.append("Evoked (Rare - Freq) - ICA+AR")

    if evokeds_ica:
        report.add_evokeds(evokeds_ica, titles=titles_ica)
    if evokeds_ica_ar:
        report.add_evokeds(evokeds_ica_ar, titles=titles_ica_ar)

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
    n_epochs_ica_only: int,
    n_epochs_ica_ar: int,
    ar1_flags_mapped: np.ndarray = None,
):
    """Save text-based preprocessing summary for easy parsing.

    Args:
        output_path: Path to save summary text file.
        subject: Subject ID.
        run: Run number.
        n_total_epochs: Total number of epochs before cleaning.
        reject_log_first: First AR reject log.
        reject_log_second: Second AR reject log (can be None).
        threshold_bad_mask: Boolean mask from threshold detection.
        threshold_stats: Statistics from threshold detection.
        ecg_inds: ECG component indices removed.
        eog_inds: EOG component indices removed.
        n_epochs_ica_only: Number of epochs after ICA only.
        n_epochs_ica_ar: Number of epochs after ICA+AR.
        ar1_flags_mapped: AR1 flags mapped to surviving epochs (optional).
    """
    n_bad_first = int(np.sum(reject_log_first.bad_epochs))

    # Build second AR section conditionally
    if reject_log_second is not None:
        n_bad_second = int(np.sum(reject_log_second.bad_epochs))
        ar_bad = reject_log_second.bad_epochs
        both_bad = int(np.sum(ar_bad & threshold_bad_mask))
        only_ar = int(np.sum(ar_bad & ~threshold_bad_mask))
        only_thresh = int(np.sum(threshold_bad_mask & ~ar_bad))
        agreement = float(np.mean(ar_bad == threshold_bad_mask) * 100)

        ar2_section = f"""
Second AutoReject Pass (post-ICA, fit only):
  - Purpose: Detect bad epochs for downstream filtering (stats/ML)
  - Bad epochs detected: {n_bad_second} ({100*n_bad_second/n_epochs_ica_only:.1f}%)
  - Mode: Fit only (no interpolation)
  - Output: ARlog2.pkl (use to filter epochs in downstream analysis)

Threshold vs AutoReject Comparison
----------------------------------
  - Agreement: {agreement:.1f}%
  - Both methods reject: {both_bad} epochs
  - Only AutoReject rejects: {only_ar} epochs
  - Only Threshold rejects: {only_thresh} epochs
"""
        ar2_output = f"AR Log (2nd pass): derivatives/preprocessed/sub-{subject}/meg/*ARlog2.pkl"
        retention_flow = f"{n_total_epochs} → {n_total_epochs - n_bad_first} (ICA fit) → {n_epochs_ica_only} saved ({n_bad_second} flagged for downstream filtering)"
    else:
        ar2_section = "\nSecond AutoReject Pass: SKIPPED (run without --skip-second-ar to enable)\n"
        ar2_output = "(Second AR pass skipped)"
        retention_flow = f"{n_total_epochs} → {n_total_epochs - n_bad_first} (-{n_bad_first}) → {n_epochs_ica_only}"

    # 3-way comparison section
    three_way_section = ""
    if (ar1_flags_mapped is not None and reject_log_second is not None
            and len(ar1_flags_mapped) == len(ar_bad)):
        ar1_m = ar1_flags_mapped
        ar2_m = ar_bad
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
3-Way Bad Epoch Comparison (on {M} post-ICA epochs)
----------------------------------------------------
  AR1 (pre-ICA, mapped):  {n_ar1_bad} bad ({100*n_ar1_bad/M:.1f}%)
  AR2 (post-ICA):         {int(ar2_m.sum())} bad ({100*ar2_m.sum()/M:.1f}%)
  Threshold (post-ICA):   {int(thr_m.sum())} bad ({100*thr_m.sum()/M:.1f}%)

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

Data Quality Metrics
-------------------
Total epochs (before cleaning): {n_total_epochs}

First AutoReject Pass (for ICA fitting):
  - Bad epochs detected: {n_bad_first} ({100*n_bad_first/n_total_epochs:.1f}%)
  - Good epochs for ICA: {n_total_epochs - n_bad_first} ({100*(n_total_epochs-n_bad_first)/n_total_epochs:.1f}%)

ICA Artifact Removal:
  - ECG components removed: {len(ecg_inds)} (ICs: {ecg_inds})
  - EOG components removed: {len(eog_inds)} (ICs: {eog_inds})
  - Total ICA components removed: {len(ecg_inds) + len(eog_inds)}

Threshold-Based Detection (post-ICA):
  - Bad epochs detected: {threshold_stats['n_bad']} ({threshold_stats['pct_bad']:.1f}%)
  - Reject threshold: {threshold_stats['reject_threshold']}
  - Flat threshold: {threshold_stats['flat_threshold']}
{ar2_section}{three_way_section}
Output Files
-----------
Continuous (after ICA): derivatives/preprocessed/sub-{subject}/meg/*_proc-clean_meg.fif
Epochs (ICA only): derivatives/epochs/sub-{subject}/meg/*_proc-ica_meg.fif (n={n_epochs_ica_only})
{ar2_output}

Data Retention Summary
---------------------
Original → After AR1 → After ICA{" → After AR2" if reject_log_second else ""}
{retention_flow}

Overall retention: {100*n_epochs_ica_ar/n_total_epochs:.1f}%
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
    skip_second_ar: bool = False,
):
    """Preprocess a single MEG run.

    Args:
        subject: Subject ID.
        run: Run number.
        bids_root: BIDS dataset root directory.
        derivatives_root: Derivatives directory.
        config: Configuration dictionary.
        skip_existing: Skip if preprocessed file already exists.
        crop: If set, crop raw data to first N seconds (for quick testing).
        skip_second_ar: If True, skip second AutoReject pass after ICA.
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

    # Epoch timing from config
    # Event marker (t=0) is at stimulus onset (0% intensity).
    # Epoch window captures high-visibility portion:
    # - tmin: 50% intensity (rising phase)
    # - midpoint: 100% intensity (peak)
    # - tmax: 50% intensity (falling phase)
    epochs_cfg = config["analysis"]["epochs"]
    tmin = epochs_cfg["tmin"]
    tmax = epochs_cfg["tmax"]
    logger.info(f"Epoch timing: tmin={tmin}s, tmax={tmax}s (duration={tmax-tmin:.3f}s)")

    # Load raw BIDS data (FIF format with renamed channels)
    logger.info(f"Loading raw data: {paths['raw']}")
    console.print(f"[yellow]⏳ Loading raw BIDS data...[/yellow]")
    raw = mne.io.read_raw_fif(str(paths["raw"].fpath), preload=True, verbose=False)
    raw.info["line_freq"] = 60  # Set line frequency for notch filtering
    console.print(f"[green]✓ Loaded {raw.n_times} samples ({len(raw.ch_names)} channels)[/green]")

    # Rename and set channel types for ECG/EOG (recorded on EEG channels in CTF)
    # This matches what the BIDS conversion does in code/bids/utils.py
    channels_cfg = config["preprocessing"].get("channels", {})
    ecg_ch = channels_cfg.get("ecg_channel", "EEG059")
    veog_ch = channels_cfg.get("veog_channel", "EEG057")
    heog_ch = channels_cfg.get("heog_channel", "EEG058")

    # Rename channels to BIDS-compliant names
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

    # Set channel types
    channel_types = {}
    if "ECG" in raw.ch_names:
        channel_types["ECG"] = "ecg"
    if "vEOG" in raw.ch_names:
        channel_types["vEOG"] = "eog"
    if "hEOG" in raw.ch_names:
        channel_types["hEOG"] = "eog"

    if channel_types:
        raw.set_channel_types(channel_types)
        console.print(f"[green]✓ Renamed and set channel types: ECG, vEOG, hEOG[/green]")

    # Crop if requested (for quick testing)
    if crop is not None:
        original_duration = raw.times[-1]
        logger.info(f"Cropping raw data to first {crop} seconds (original: {original_duration:.1f}s)")
        console.print(f"[yellow]⏳ Cropping to first {crop} seconds (testing mode)...[/yellow]")
        raw.crop(tmin=0, tmax=crop)
        console.print(f"[green]✓ Cropped to {raw.times[-1]:.1f}s ({raw.n_times} samples)[/green]")

    # Resample if configured (do this early, before computing event samples)
    resample_sfreq = config["preprocessing"].get("resample_sfreq")
    if resample_sfreq is not None and raw.info["sfreq"] != resample_sfreq:
        original_sfreq = raw.info["sfreq"]
        logger.info(f"Resampling from {original_sfreq} Hz to {resample_sfreq} Hz")
        console.print(f"[yellow]⏳ Resampling from {original_sfreq} Hz to {resample_sfreq} Hz...[/yellow]")
        raw.resample(resample_sfreq, verbose=False)
        console.print(f"[green]✓ Resampled to {raw.info['sfreq']} Hz ({raw.n_times} samples)[/green]")

    # Load events from BIDS events.tsv file
    events_tsv_path = str(paths["raw"].fpath).replace("_meg.fif", "_events.tsv")
    events_df = pd.read_csv(events_tsv_path, sep="\t")

    # Convert to MNE events array: [sample, 0, event_id]
    # Map trial_type to event IDs (1=rare, 2=frequent based on value column)
    sfreq = raw.info["sfreq"]
    events_list = []
    event_id = {}
    for _, row in events_df.iterrows():
        sample = int(row["onset"] * sfreq)
        trial_type = str(row.get("trial_type", row.get("value", "unknown")))
        if trial_type not in event_id:
            event_id[trial_type] = len(event_id) + 1
        events_list.append([sample, 0, event_id[trial_type]])
    events = np.array(events_list)
    picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True)
    console.print(f"[green]✓ Detected {len(events)} events[/green]")

    # Apply gradient compensation (required for source reconstruction)
    logger.info("Applying gradient compensation (grade 3)")
    console.print(f"[yellow]⏳ Applying gradient compensation (grade 3)...[/yellow]")
    raw = raw.apply_gradient_compensation(grade=3)
    console.print(f"[green]✓ Gradient compensation applied[/green]")

    # Filtering
    notch_freqs = np.arange(
        raw.info["line_freq"],
        filter_cfg["highcut"] + 1,
        raw.info["line_freq"],
    )

    console.print(f"[yellow]⏳ Filtering data ({filter_cfg['lowcut']}-{filter_cfg['highcut']} Hz)...[/yellow]")
    preproc, raw_filt = apply_filtering(
        raw,
        filter_cfg["lowcut"],
        filter_cfg["highcut"],
        notch_freqs,
        picks,
    )
    console.print(f"[green]✓ Filtering complete (two versions: 0.1 Hz and 1 Hz highpass)[/green]")

    # Epoching for AutoReject
    logger.info("Creating epochs for AutoReject")
    console.print(f"[yellow]⏳ Creating epochs ({tmin} to {tmax} s)...[/yellow]")
    epochs_filt = create_epochs(raw_filt, events, event_id, tmin, tmax, picks)
    console.print(f"[green]✓ Created {len(epochs_filt)} epochs[/green]")

    # First AutoReject pass (fit only, for ICA)
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

    # Get good epochs mask for ICA fitting
    good_mask = get_good_epochs_mask(reject_log_first)
    n_good = np.sum(good_mask)
    console.print(f"[green]✓ Will use {n_good}/{len(epochs_filt)} good epochs for ICA fitting[/green]")

    # Compute noise covariance from subject's empty-room recording
    console.print(f"[yellow]⏳ Computing/loading noise covariance for sub-{subject}...[/yellow]")
    noise_cov = compute_or_load_noise_cov(subject, bids_root, derivatives_root)
    console.print(f"[green]✓ Noise covariance ready[/green]")

    # Run ICA pipeline (fit on good epochs only)
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]ICA ARTIFACT REMOVAL[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    cleaned_raw, _, ica, ecg_inds, eog_inds = run_ica_pipeline(
        preproc,
        epochs_filt[good_mask],
        noise_cov,
        n_components=ica_cfg.get("n_components", 20),
        random_state=ica_cfg.get("random_state", 42),
    )

    # Create final epochs from ICA-cleaned data
    logger.info("Creating epochs from ICA-cleaned data")
    console.print(f"\n[yellow]⏳ Creating final epochs from ICA-cleaned data...[/yellow]")
    reject_dict = dict(mag=3000e-15)
    epochs = create_epochs(preproc, events, event_id, tmin, tmax, picks, reject_dict)
    epochs_ica_only = ica.apply(epochs.copy())
    console.print(f"[green]✓ Created {len(epochs_ica_only)} ICA-cleaned epochs[/green]")

    # Map AR1 flags to surviving epoch indices for 3-way comparison
    # epochs uses reject_dict which may drop some of the N epochs_filt epochs
    surviving_indices = [i for i, dl in enumerate(epochs.drop_log) if len(dl) == 0]
    ar1_flags_mapped = reject_log_first.bad_epochs[np.array(surviving_indices)]
    n_dropped = len(reject_log_first.bad_epochs) - len(surviving_indices)
    if n_dropped > 0:
        logger.info(f"Note: {n_dropped} epochs dropped by amplitude threshold before 3-way comparison")
    console.print(f"[green]✓ Mapped AR1 flags: {int(ar1_flags_mapped.sum())} bad out of {len(ar1_flags_mapped)} surviving epochs[/green]")

    # Threshold-based bad epoch detection (for comparison with AutoReject)
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]THRESHOLD-BASED EPOCH DETECTION[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    logger.info("Running threshold-based bad epoch detection")

    # Get threshold parameters from config or use defaults
    threshold_cfg = config["preprocessing"].get("threshold_rejection", {})
    reject_threshold = threshold_cfg.get("reject", {"mag": 4000e-15})
    flat_threshold = threshold_cfg.get("flat", {"mag": 1e-15})

    threshold_bad_mask, threshold_stats = detect_bad_epochs_threshold(
        epochs_ica_only,
        reject_threshold=reject_threshold,
        flat_threshold=flat_threshold,
    )
    console.print(
        f"[green]✓ Threshold detection: {threshold_stats['n_bad']}/{threshold_stats['n_total']} "
        f"bad epochs ({threshold_stats['pct_bad']:.1f}%)[/green]"
    )

    # Second AutoReject pass (fit only, for bad epoch detection post-ICA)
    # This runs by default to identify which epochs are still bad after ICA cleaning
    # The epochs are NOT transformed (no interpolation) - we just get the reject log
    if not skip_second_ar:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]SECOND AUTOREJECT PASS (FIT ONLY - BAD EPOCH DETECTION)[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        logger.info("Running second AutoReject pass (fit only, for bad epoch detection)")
        ar_second, reject_log_second = run_autoreject(
            epochs_ica_only.copy(),
            n_interpolate=ar_cfg["n_interpolate"],
            consensus=ar_cfg["consensus"],
            n_jobs=config["computing"]["n_jobs"],
            random_state=42,
        )
        # epochs_ica_ar is the same as epochs_ica_only (no transformation)
        # but we now have reject_log_second to know which epochs are bad post-ICA
        epochs_ica_ar = epochs_ica_only.copy()
    else:
        console.print(f"\n[yellow]Skipping second AutoReject pass (use default to enable)[/yellow]")
        logger.info("Skipping second AutoReject pass")
        # Use ICA-only epochs as final output
        epochs_ica_ar = epochs_ica_only.copy()
        ar_second = None
        reject_log_second = None

    # Generate report comparing both versions
    report = generate_preprocessing_report(
        raw,
        cleaned_raw,
        raw_filt,
        epochs_filt,
        epochs,
        epochs_ica_only,
        epochs_ica_ar,
        reject_log_first,
        reject_log_second,
        threshold_bad_mask,
        threshold_stats,
        ica,
        ecg_inds,
        eog_inds,
        noise_cov,
        picks,
        ar1_flags_mapped=ar1_flags_mapped,
    )

    # Set annotations for cleaned raw
    cleaned_raw.set_annotations(
        mne.annotations_from_events(
            events,
            sfreq=raw.info["sfreq"],
            event_desc={v: k for k, v in event_id.items()},
        )
    )

    # Save outputs
    logger.info("=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)

    # Save preprocessed continuous data (after ICA, before second AR)
    logger.info("Saving continuous data (ICA-cleaned)...")
    write_raw_bids(
        cleaned_raw,
        paths["preproc"],
        format="FIF",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )
    logger.info(f"✓ Continuous (ICA): {paths['preproc'].fpath}")

    # Save epochs (ICA only version)
    logger.info(f"Saving epochs (ICA only, n={len(epochs_ica_only)})...")
    write_raw_bids(
        cleaned_raw,
        paths["epoch_ica"],
        format="FIF",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )  # Init BIDS structure
    epochs_ica_only.save(paths["epoch_ica"].fpath, overwrite=True)
    logger.info(f"✓ Epochs (ICA): {paths['epoch_ica'].fpath}")

    # Note: We no longer save separate ICA+AR epochs since second AR is fit-only
    # The reject_log_second is saved below and can be used for downstream filtering

    # Save first AutoReject log (for ICA)
    ar_log_first_path = Path(str(paths["ARlog_first"].fpath) + ".pkl")
    with open(ar_log_first_path, "wb") as f:
        pickle.dump(reject_log_first, f)
    logger.info(f"Saved first AutoReject log to {ar_log_first_path}")

    # Save second AutoReject log (with interpolation) - only if performed
    if reject_log_second is not None:
        ar_log_second_path = Path(str(paths["ARlog_second"].fpath) + ".pkl")
        with open(ar_log_second_path, "wb") as f:
            pickle.dump(reject_log_second, f)
        logger.info(f"Saved second AutoReject log to {ar_log_second_path}")

    # Save HTML report
    logger.info("Saving HTML report...")
    report_path = Path(str(paths["report"].fpath) + ".html")
    report.save(report_path, open_browser=False, overwrite=True)
    logger.info(f"✓ Report: {report_path}")

    # Save text summary
    logger.info("Saving text summary...")
    summary_path = Path(str(paths["report"].fpath) + "_summary.txt")
    save_preprocessing_summary(
        summary_path,
        subject,
        run,
        len(epochs_filt),
        reject_log_first,
        reject_log_second,
        threshold_bad_mask,
        threshold_stats,
        ecg_inds,
        eog_inds,
        len(epochs_ica_only),
        len(epochs_ica_ar),
        ar1_flags_mapped=ar1_flags_mapped,
    )
    logger.info(f"✓ Summary: {summary_path}")

    # Save metadata
    metadata_path = Path(str(paths["preproc"].fpath).replace("_meg.fif", "_params.json"))

    # Build metadata params
    params = {
        "tmin": tmin,
        "tmax": tmax,
        "filter": filter_cfg,
        "ica": {
            "n_components": ica_cfg.get("n_components", 20),
            "ecg_components": ecg_inds,
            "eog_components": eog_inds,
        },
        "autoreject_first_pass": {
            "description": "First pass (fit only) for ICA",
            "n_bad_epochs": int(np.sum(reject_log_first.bad_epochs)),
            "n_total_epochs": len(epochs_filt),
            "pct_bad": float(100 * np.sum(reject_log_first.bad_epochs) / len(epochs_filt)),
        },
        "threshold_detection": {
            "description": "Threshold-based detection after ICA",
            "n_bad_epochs": threshold_stats["n_bad"],
            "n_total_epochs": threshold_stats["n_total"],
            "pct_bad": threshold_stats["pct_bad"],
            "reject_threshold": {k: float(v) for k, v in threshold_stats["reject_threshold"].items()},
            "flat_threshold": {k: float(v) for k, v in threshold_stats["flat_threshold"].items()},
        },
    }

    # Add second AR pass info if performed
    if reject_log_second is not None:
        ar_bad = reject_log_second.bad_epochs
        params["autoreject_second_pass"] = {
            "description": "Second pass (fit only) after ICA for bad epoch detection",
            "n_bad_epochs": int(np.sum(reject_log_second.bad_epochs)),
            "n_total_epochs": len(epochs_ica_only),
            "pct_bad": float(100 * np.sum(reject_log_second.bad_epochs) / len(epochs_ica_only)),
            "mode": "fit_only",
            "note": "Use ARlog2.pkl to filter bad epochs in downstream analysis",
        }
        params["threshold_vs_autoreject"] = {
            "agreement_pct": float(np.mean(ar_bad == threshold_bad_mask) * 100),
            "both_reject": int(np.sum(ar_bad & threshold_bad_mask)),
            "only_ar_rejects": int(np.sum(ar_bad & ~threshold_bad_mask)),
            "only_threshold_rejects": int(np.sum(threshold_bad_mask & ~ar_bad)),
        }

    # Add 3-way comparison to metadata
    if (ar1_flags_mapped is not None and reject_log_second is not None
            and len(ar1_flags_mapped) == len(reject_log_second.bad_epochs)):
        ar1_m = ar1_flags_mapped
        ar2_m = reject_log_second.bad_epochs
        thr_m = threshold_bad_mask
        M = len(ar1_m)

        kappa_ar1_ar2 = float(_cohen_kappa(ar1_m, ar2_m)) if M > 0 else 0.0
        kappa_ar1_thr = float(_cohen_kappa(ar1_m, thr_m)) if M > 0 else 0.0
        kappa_ar2_thr = float(_cohen_kappa(ar2_m, thr_m)) if M > 0 else 0.0

        n_ar1_bad = int(ar1_m.sum())
        n_still_bad = int((ar1_m & ar2_m).sum())
        n_rescued = n_ar1_bad - n_still_bad

        params["three_way_comparison"] = {
            "n_epochs_compared": M,
            "ar1_mapped_bad": n_ar1_bad,
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
    logger.info(f"Data retention: {len(epochs_ica_ar)}/{len(epochs_filt)} epochs ({100*len(epochs_ica_ar)/len(epochs_filt):.1f}%)")
    logger.info(f"ICA components removed: {len(ecg_inds) + len(eog_inds)} (ECG: {len(ecg_inds)}, EOG: {len(eog_inds)})")
    if reject_log_second is not None:
        logger.info(f"Channels interpolated (AR pass 2): {int(np.sum(reject_log_second.labels == 2))}")
    logger.info("=" * 60)

    # Cleanup
    del raw, raw_filt, preproc, cleaned_raw, epochs, epochs_filt
    del epochs_ica_only, epochs_ica_ar
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
        "--skip-second-ar",
        action="store_true",
        default=False,
        help="Skip second AutoReject pass after ICA (default: run second AR pass for bad epoch detection)",
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
                    args.skip_second_ar,
                )
            except Exception as e:
                logger.error(f"Failed to process run {run}: {e}", exc_info=True)

            progress.advance(task)

    console.print(f"\n[bold green]✓ Preprocessing complete for sub-{args.subject}![/bold green]")
    console.print(f"  Processed {len(runs)} runs")
    console.print(f"\n[bold]Output locations:[/bold]")
    console.print(f"  Continuous (ICA): {derivatives_root}/preprocessed/sub-{args.subject}/")
    console.print(f"  Epochs (ICA): {derivatives_root}/epochs/sub-{args.subject}/")
    console.print(f"  AR logs (for filtering): {derivatives_root}/preprocessed/sub-{args.subject}/*ARlog*.pkl")
    console.print(f"  Reports: {derivatives_root}/preprocessed/sub-{args.subject}/*_report_meg.html")
    console.print(f"  Summaries: {derivatives_root}/preprocessed/sub-{args.subject}/*_summary.txt")
    console.print(f"\n[bold]Logs:[/bold] {log_file}")
    console.print(f"\n[dim]Read *_summary.txt files for detailed preprocessing metrics[/dim]")
    console.print(f"[dim]Use ARlog2.pkl in downstream analysis to filter out bad epochs post-ICA[/dim]")

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
