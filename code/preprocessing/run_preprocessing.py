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

import mne
import numpy as np
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
)
from code.utils.config import load_config
from code.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
console = Console()


def generate_preprocessing_report(
    raw: mne.io.Raw,
    preproc: mne.io.Raw,
    raw_filt: mne.io.Raw,
    epochs: mne.Epochs,
    epochs_ica_only: mne.Epochs,
    epochs_ica_ar: mne.Epochs,
    reject_log_first: object,
    reject_log_second: object,
    ica,
    ecg_inds: list,
    eog_inds: list,
    noise_cov: mne.Covariance,
    picks: list,
) -> mne.Report:
    """Generate HTML report for preprocessing with two-pass comparison.

    Args:
        raw: Original raw data.
        preproc: Filtered data.
        raw_filt: Filtered data for ICA.
        epochs: Epochs before cleaning.
        epochs_ica_only: Epochs after ICA only (no second AR pass).
        epochs_ica_ar: Epochs after ICA + second AR pass.
        reject_log_first: AutoReject log from first pass (for ICA).
        reject_log_second: AutoReject log from second pass (with interpolation).
        ica: Fitted ICA object.
        ecg_inds: ECG component indices.
        eog_inds: EOG component indices.
        noise_cov: Noise covariance matrix.
        picks: Channel picks.

    Returns:
        MNE Report object.
    """
    logger.info("Generating HTML report")

    report = mne.Report(verbose=False)

    # Raw data
    report.add_raw(raw, title="Raw data")
    fig = raw.plot(duration=20, start=50, show=False)
    report.add_figure(fig, title="Time series (raw)")
    fig = raw.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (raw)")

    # Filtered data
    report.add_raw(preproc, title="Filtered data")
    fig = preproc.plot(duration=20, start=50, show=False)
    report.add_figure(fig, title="Time series (filtered)")
    fig = preproc.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (filtered)")

    # Filtered data for AR
    report.add_raw(raw_filt, title="Filtered data (for AR)")
    fig = raw_filt.plot(duration=20, start=50, show=False)
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

    # First AutoReject pass (for ICA fitting)
    if np.sum(reject_log_first.bad_epochs) > 0:
        try:
            fig = epochs[reject_log_first.bad_epochs].plot(show=False)
            report.add_figure(fig, title="Bad epochs (1st AR pass)")
        except Exception as e:
            logger.warning(f"Could not plot bad epochs: {e}")

    fig = reject_log_first.plot("horizontal", show=False)
    report.add_figure(fig, title="AutoReject decisions (1st pass - for ICA)")

    # Second AutoReject pass (after ICA, with interpolation)
    if np.sum(reject_log_second.bad_epochs) > 0:
        try:
            fig = epochs_ica_only[reject_log_second.bad_epochs].plot(show=False)
            report.add_figure(fig, title="Bad epochs (2nd AR pass)")
        except Exception as e:
            logger.warning(f"Could not plot bad epochs: {e}")

    fig = reject_log_second.plot("horizontal", show=False)
    report.add_figure(fig, title="AutoReject decisions (2nd pass - with interpolation)")

    # Comprehensive summary with detailed metrics
    n_bad_first = np.sum(reject_log_first.bad_epochs)
    n_bad_second = np.sum(reject_log_second.bad_epochs)
    n_total = len(epochs)
    n_interpolated = np.sum(reject_log_second.labels == 2)
    n_good_first = n_total - n_bad_first

    summary_text = f"""
    <h2>Preprocessing Summary</h2>

    <h3>Data Processing Flow</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Stage</th><th>Epochs</th><th>Change</th><th>Description</th>
    </tr>
    <tr>
        <td><b>Original</b></td>
        <td>{n_total}</td>
        <td>-</td>
        <td>Total epochs before preprocessing</td>
    </tr>
    <tr style="background-color: #fff9e6;">
        <td><b>After AR Pass 1</b></td>
        <td>{n_good_first}</td>
        <td>-{n_bad_first} ({100*n_bad_first/n_total:.1f}%)</td>
        <td>Bad epochs removed for ICA fitting</td>
    </tr>
    <tr style="background-color: #e6f3ff;">
        <td><b>After ICA</b></td>
        <td>{len(epochs_ica_only)}</td>
        <td>{len(epochs_ica_only) - n_good_first:+d}</td>
        <td>Physiological artifacts (ECG, EOG) removed</td>
    </tr>
    <tr style="background-color: #e6ffe6;">
        <td><b>After AR Pass 2</b></td>
        <td>{len(epochs_ica_ar)}</td>
        <td>-{len(epochs_ica_only) - len(epochs_ica_ar)} ({100*(len(epochs_ica_only) - len(epochs_ica_ar))/len(epochs_ica_only):.1f}%)</td>
        <td>Bad channels interpolated, remaining bad epochs removed</td>
    </tr>
    <tr style="background-color: #d4edda;">
        <td><b>Final Retention</b></td>
        <td colspan="3"><b>{len(epochs_ica_ar)}/{n_total} ({100*len(epochs_ica_ar)/n_total:.1f}% retained)</b></td>
    </tr>
    </table>

    <h3>AutoReject Details</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Metric</th><th>First Pass (for ICA)</th><th>Second Pass (after ICA)</th>
    </tr>
    <tr>
        <td>Input epochs</td>
        <td>{n_total}</td>
        <td>{len(epochs_ica_only)}</td>
    </tr>
    <tr>
        <td>Bad epochs detected</td>
        <td>{n_bad_first} ({100*n_bad_first/n_total:.1f}%)</td>
        <td>{n_bad_second} ({100*n_bad_second/len(epochs_ica_only):.1f}%)</td>
    </tr>
    <tr>
        <td>Channels interpolated</td>
        <td>N/A (fit only)</td>
        <td>{n_interpolated}</td>
    </tr>
    <tr>
        <td>Output epochs</td>
        <td>{n_good_first} (for ICA fitting)</td>
        <td>{len(epochs_ica_ar)} (final)</td>
    </tr>
    </table>

    <h3>ICA Components Removed</h3>
    <ul>
    <li><b>ECG components:</b> {len(ecg_inds)} (ICs: {ecg_inds})</li>
    <li><b>EOG components:</b> {len(eog_inds)} (ICs: {eog_inds})</li>
    <li><b>Total components removed:</b> {len(ecg_inds) + len(eog_inds)}</li>
    </ul>

    <h3>Output Files</h3>
    <ul>
    <li><b>Continuous (ICA-cleaned):</b> derivatives/preprocessed/.../proc-clean_meg.fif</li>
    <li><b>Epochs (ICA only):</b> derivatives/epochs/.../proc-ica_meg.fif (n={len(epochs_ica_only)})</li>
    <li><b>Epochs (ICA+AR):</b> derivatives/epochs/.../proc-icaar_meg.fif (n={len(epochs_ica_ar)})</li>
    </ul>

    <p><i>Note: A detailed text summary is saved alongside this report (*_summary.txt)</i></p>
    """
    report.add_html(summary_text, title="Preprocessing Summary")

    # Noise covariance
    report.add_covariance(
        noise_cov, info=preproc.info, title="Noise covariance matrix"
    )

    # ICA
    fig = ica.plot_sources(preproc, show=False)
    report.add_figure(fig, title="ICA sources")

    # Cleaned data - ICA only
    fig = preproc.plot(duration=20, start=50, show=False)
    report.add_figure(fig, title="Time series (ICA cleaned)")
    fig = preproc.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (ICA cleaned)")
    fig = epochs_ica_only.plot(show=False)
    report.add_figure(fig, title="Epochs (ICA only)")
    fig = epochs_ica_only.plot_psd(average=False, picks="mag", show=False)
    report.add_figure(fig, title="PSD epochs (ICA only)")

    # Cleaned data - ICA + AR
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

    # Evoked responses - ICA + AR version
    evokeds_ica_ar = []
    titles_ica_ar = []
    for cond in ["Freq", "Rare", "Resp"]:
        if cond in epochs_ica_ar.event_id:
            evoked = epochs_ica_ar[cond].average()
            fig = evoked.plot(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - ICA + AR")
            fig = evoked.plot_joint(show=False)
            report.add_figure(fig, title=f"Evoked ({cond}) - Joint - ICA + AR")
            evokeds_ica_ar.append(evoked)
            titles_ica_ar.append(f"Evoked ({cond}) - ICA+AR")

    # Difference waves: Rare - Freq (both versions)
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

        # ICA + AR
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

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.debug(f"Saved preprocessing metadata to {output_path}")


def save_preprocessing_summary(
    output_path: Path,
    subject: str,
    run: str,
    n_total_epochs: int,
    reject_log_first,
    reject_log_second,
    ecg_inds: list,
    eog_inds: list,
    n_epochs_ica_only: int,
    n_epochs_ica_ar: int,
):
    """Save text-based preprocessing summary for easy parsing.

    Args:
        output_path: Path to save summary text file.
        subject: Subject ID.
        run: Run number.
        n_total_epochs: Total number of epochs before cleaning.
        reject_log_first: First AR reject log.
        reject_log_second: Second AR reject log.
        ecg_inds: ECG component indices removed.
        eog_inds: EOG component indices removed.
        n_epochs_ica_only: Number of epochs after ICA only.
        n_epochs_ica_ar: Number of epochs after ICA+AR.
    """
    n_bad_first = int(np.sum(reject_log_first.bad_epochs))
    n_bad_second = int(np.sum(reject_log_second.bad_epochs))
    n_interpolated = int(np.sum(reject_log_second.labels == 2))

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

Second AutoReject Pass (after ICA):
  - Bad epochs detected: {n_bad_second} ({100*n_bad_second/n_epochs_ica_only:.1f}%)
  - Channels interpolated: {n_interpolated}
  - Final epochs retained: {n_epochs_ica_ar} ({100*n_epochs_ica_ar/n_total_epochs:.1f}% of original)

Output Files
-----------
Continuous (after ICA): derivatives/preprocessed/sub-{subject}/meg/*_proc-clean_meg.fif
Epochs (ICA only): derivatives/epochs/sub-{subject}/meg/*_proc-ica_meg.fif (n={n_epochs_ica_only})
Epochs (ICA+AR): derivatives/epochs/sub-{subject}/meg/*_proc-icaar_meg.fif (n={n_epochs_ica_ar})

Data Retention Summary
---------------------
Original → After AR1 → After ICA → After AR2
{n_total_epochs} → {n_total_epochs - n_bad_first} (-{n_bad_first}) → {n_epochs_ica_only} → {n_epochs_ica_ar} (-{n_epochs_ica_only - n_epochs_ica_ar})

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
):
    """Preprocess a single MEG run.

    Args:
        subject: Subject ID.
        run: Run number.
        bids_root: BIDS dataset root directory.
        derivatives_root: Derivatives directory.
        config: Configuration dictionary.
        skip_existing: Skip if preprocessed file already exists.
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

    # Epoch timing (from original cc_saflow)
    tmin = 0.426
    tmax = 1.278

    # Load raw data
    logger.info(f"Loading raw data: {paths['raw']}")
    console.print(f"[yellow]⏳ Loading raw BIDS data...[/yellow]")
    raw = read_raw_bids(paths["raw"], verbose=False)
    raw.load_data()
    console.print(f"[green]✓ Loaded {raw.n_times} samples ({len(raw.ch_names)} channels)[/green]")

    events, event_id = mne.events_from_annotations(raw)
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

    # Compute noise covariance
    record_date = raw.info["meas_date"].strftime("%Y%m%d")
    console.print(f"[yellow]⏳ Computing/loading noise covariance (date: {record_date})...[/yellow]")
    noise_cov = compute_or_load_noise_cov(record_date, bids_root, derivatives_root)
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

    # Second AutoReject pass (fit_transform with interpolation)
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]SECOND AUTOREJECT PASS (WITH INTERPOLATION)[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    logger.info("Running second AutoReject pass (with interpolation)")
    epochs_ica_ar, ar_second, reject_log_second = run_autoreject_transform(
        epochs_ica_only.copy(),
        n_interpolate=ar_cfg["n_interpolate"],
        consensus=ar_cfg["consensus"],
        n_jobs=config["computing"]["n_jobs"],
        random_state=42,
    )

    # Generate report comparing both versions
    report = generate_preprocessing_report(
        raw,
        cleaned_raw,
        raw_filt,
        epochs,
        epochs_ica_only,
        epochs_ica_ar,
        reject_log_first,
        reject_log_second,
        ica,
        ecg_inds,
        eog_inds,
        noise_cov,
        picks,
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

    # Save epochs (ICA + AR version)
    logger.info(f"Saving epochs (ICA+AR, n={len(epochs_ica_ar)})...")
    write_raw_bids(
        cleaned_raw,
        paths["epoch_ica_ar"],
        format="FIF",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )  # Init BIDS structure
    epochs_ica_ar.save(paths["epoch_ica_ar"].fpath, overwrite=True)
    logger.info(f"✓ Epochs (ICA+AR): {paths['epoch_ica_ar'].fpath}")

    # Save first AutoReject log (for ICA)
    ar_log_first_path = Path(str(paths["ARlog_first"].fpath) + ".pkl")
    with open(ar_log_first_path, "wb") as f:
        pickle.dump(reject_log_first, f)
    logger.info(f"Saved first AutoReject log to {ar_log_first_path}")

    # Save second AutoReject log (with interpolation)
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
        ecg_inds,
        eog_inds,
        len(epochs_ica_only),
        len(epochs_ica_ar),
    )
    logger.info(f"✓ Summary: {summary_path}")

    # Save metadata
    metadata_path = Path(str(paths["preproc"].fpath).replace("_meg.fif", "_params.json"))
    save_preprocessing_metadata(
        metadata_path,
        config,
        subject,
        run,
        {
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
            "autoreject_second_pass": {
                "description": "Second pass (fit_transform) after ICA",
                "n_bad_epochs": int(np.sum(reject_log_second.bad_epochs)),
                "n_total_epochs": len(epochs_ica_only),
                "pct_bad": float(100 * np.sum(reject_log_second.bad_epochs) / len(epochs_ica_only)),
                "n_channels_interpolated": int(np.sum(reject_log_second.labels == 2)),
                "n_epochs_final": len(epochs_ica_ar),
            },
        },
    )

    # Log final summary
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Subject: {subject}, Run: {run}")
    logger.info(f"Data retention: {len(epochs_ica_ar)}/{len(epochs_filt)} epochs ({100*len(epochs_ica_ar)/len(epochs_filt):.1f}%)")
    logger.info(f"ICA components removed: {len(ecg_inds) + len(eog_inds)} (ECG: {len(ecg_inds)}, EOG: {len(eog_inds)})")
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
                )
            except Exception as e:
                logger.error(f"Failed to process run {run}: {e}", exc_info=True)

            progress.advance(task)

    console.print(f"\n[bold green]✓ Preprocessing complete for sub-{args.subject}![/bold green]")
    console.print(f"  Processed {len(runs)} runs")
    console.print(f"\n[bold]Output locations:[/bold]")
    console.print(f"  Continuous (ICA): {derivatives_root}/preprocessed/sub-{args.subject}/")
    console.print(f"  Epochs (ICA only): {derivatives_root}/epochs/sub-{args.subject}/")
    console.print(f"  Epochs (ICA+AR): {derivatives_root}/epochs/sub-{args.subject}/")
    console.print(f"  Reports: {derivatives_root}/preprocessed/sub-{args.subject}/*_report_meg.html")
    console.print(f"  Summaries: {derivatives_root}/preprocessed/sub-{args.subject}/*_summary.txt")
    console.print(f"\n[bold]Logs:[/bold] {log_file}")
    console.print(f"\n[dim]Read *_summary.txt files for detailed preprocessing metrics[/dim]")

    logger.info("Preprocessing complete")
    return 0


if __name__ == "__main__":
    exit(main())
