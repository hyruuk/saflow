"""ICA artifact removal pipeline for MEG preprocessing.

This module implements ICA-based artifact removal for ECG and EOG artifacts.
"""

import logging
from typing import List, Tuple

import mne
import numpy as np
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def fit_ica(
    epochs: mne.Epochs,
    n_components,
    random_state: int,
    noise_cov: mne.Covariance = None,
    decim: int = 3,
) -> ICA:
    """Fit ICA on epoched data.

    Args:
        epochs: Epoched MEG data.
        n_components: Number of ICA components (int), or float 0-1 for
            explained variance threshold (e.g. 0.99 retains 99% variance).
        random_state: Random seed for reproducibility.
        noise_cov: Noise covariance matrix (optional).
        decim: Decimation factor during ICA fitting.

    Returns:
        Fitted ICA object.

    Examples:
        >>> ica = fit_ica(epochs, n_components=0.99, random_state=42)
    """
    if isinstance(n_components, float) and 0 < n_components < 1:
        logger.info(f"Fitting ICA with {n_components:.2f} explained variance threshold")
        console.print(f"[yellow]⏳ ICA: Fitting with {n_components:.2f} explained variance on {len(epochs)} epochs...[/yellow]")
    else:
        logger.info(f"Fitting ICA with {n_components} components")
        console.print(f"[yellow]⏳ ICA: Fitting {n_components} components on {len(epochs)} epochs...[/yellow]")

    ica = ICA(n_components=n_components, random_state=random_state, noise_cov=noise_cov)
    ica.fit(epochs, decim=decim, verbose=True)  # Enable verbose output

    n_actual = ica.n_components_
    console.print(f"[green]✓ ICA: Fitting complete — {n_actual} components[/green]")
    logger.info(f"ICA fitted on {len(epochs)} epochs → {n_actual} components")
    return ica


def find_ecg_components(
    ica: ICA,
    raw: mne.io.Raw,
    ecg_channel: str = "ECG",
    threshold: float = 0.25,
) -> Tuple[List[int], np.ndarray, bool]:
    """Identify ICA components related to cardiac artifacts.

    Uses union of CTPS and correlation methods for more robust detection.

    Args:
        ica: Fitted ICA object.
        raw: Continuous MEG data.
        ecg_channel: Name of ECG channel.
        threshold: Threshold for both CTPS and correlation methods.

    Returns:
        Tuple of (component_indices, scores, forced) where forced is True
        if no component exceeded the threshold and the best one was
        force-selected. Scores are from the CTPS method.

    Examples:
        >>> ecg_inds, ecg_scores, ecg_forced = find_ecg_components(ica, raw, threshold=0.25)
    """
    logger.info(f"Detecting ECG components (threshold={threshold})")
    console.print(f"[yellow]⏳ ICA: Detecting ECG components (CTPS + correlation)...[/yellow]")

    # Create ECG epochs
    ecg_epochs = create_ecg_epochs(raw, ch_name=ecg_channel, verbose=False)

    # Method 1: CTPS
    ecg_inds_ctps, ecg_scores = ica.find_bads_ecg(
        ecg_epochs, ch_name=ecg_channel, method="ctps", threshold=threshold, verbose=False
    )

    # Method 2: Correlation
    ecg_inds_corr, _ = ica.find_bads_ecg(
        ecg_epochs, ch_name=ecg_channel, method="correlation", threshold=threshold, verbose=False
    )

    # Union of both methods
    ecg_inds = sorted(set(ecg_inds_ctps) | set(ecg_inds_corr))

    if ecg_inds_ctps:
        logger.info(f"CTPS detected: {ecg_inds_ctps}")
    if ecg_inds_corr:
        logger.info(f"Correlation detected: {ecg_inds_corr}")

    # If no components detected by either method, use the one with highest CTPS score
    forced = False
    if not ecg_inds:
        forced = True
        max_idx = int(np.argmax(np.abs(ecg_scores)))
        ecg_inds = [max_idx]
        console.print(f"[yellow]⚠ No ECG components above threshold, using highest score: IC{max_idx}[/yellow]")
        logger.warning(
            f"No ECG components above threshold, using highest score: IC{max_idx}"
        )
    else:
        console.print(f"[green]✓ Detected {len(ecg_inds)} ECG components: {ecg_inds} (CTPS: {ecg_inds_ctps}, corr: {ecg_inds_corr})[/green]")
        logger.info(f"Detected {len(ecg_inds)} ECG components (union): {ecg_inds}")

    return ecg_inds, ecg_scores, forced


def find_eog_components(
    ica: ICA,
    raw: mne.io.Raw,
    eog_channel: str = "vEOG",
    threshold: float = 2.5,
) -> Tuple[List[int], np.ndarray, bool]:
    """Identify ICA components related to ocular artifacts.

    Args:
        ica: Fitted ICA object.
        raw: Continuous MEG data.
        eog_channel: Name of EOG channel.
        threshold: Z-score threshold for component detection.

    Returns:
        Tuple of (component_indices, scores, forced) where forced is True
        if no component exceeded the threshold and the best one was
        force-selected.

    Examples:
        >>> eog_inds, eog_scores, eog_forced = find_eog_components(ica, raw, threshold=2.5)
    """
    logger.info(f"Detecting EOG components (threshold={threshold})")
    console.print(f"[yellow]⏳ ICA: Detecting EOG components...[/yellow]")

    # Create EOG epochs
    eog_epochs = create_eog_epochs(raw, ch_name=eog_channel, verbose=False)

    # Find bad components
    eog_inds, eog_scores = ica.find_bads_eog(
        eog_epochs, ch_name=eog_channel, threshold=threshold, verbose=False
    )

    # If no components detected, use the one with highest score
    forced = False
    if not eog_inds:
        forced = True
        max_idx = int(np.argmax(np.abs(eog_scores)))
        eog_inds = [max_idx]
        console.print(f"[yellow]⚠ No EOG components above threshold, using highest score: IC{max_idx}[/yellow]")
        logger.warning(
            f"No EOG components above threshold, using highest score: IC{max_idx}"
        )
    else:
        console.print(f"[green]✓ Detected {len(eog_inds)} EOG components: {eog_inds}[/green]")
        logger.info(f"Detected {len(eog_inds)} EOG components: {eog_inds}")

    return eog_inds, eog_scores, forced


def apply_ica(
    ica: ICA,
    raw: mne.io.Raw,
    epochs: mne.Epochs = None,
) -> Tuple[mne.io.Raw, mne.Epochs]:
    """Apply ICA artifact removal to data.

    Args:
        ica: Fitted ICA object with excluded components set.
        raw: Continuous MEG data.
        epochs: Epoched data (optional).

    Returns:
        Tuple of (cleaned_raw, cleaned_epochs).

    Examples:
        >>> cleaned_raw, cleaned_epochs = apply_ica(ica, raw, epochs)
    """
    logger.info(f"Applying ICA (removing components: {ica.exclude})")
    console.print(f"[yellow]⏳ ICA: Applying artifact removal (excluding ICs: {ica.exclude})...[/yellow]")

    cleaned_raw = ica.apply(raw.copy())

    cleaned_epochs = None
    if epochs is not None:
        cleaned_epochs = ica.apply(epochs.copy())

    console.print("[green]✓ ICA: Artifact removal applied[/green]")
    logger.debug("ICA application complete")
    return cleaned_raw, cleaned_epochs


def run_ica_pipeline(
    raw: mne.io.Raw,
    epochs: mne.Epochs,
    noise_cov: mne.Covariance,
    n_components=0.99,
    random_state: int = 42,
    ecg_threshold: float = 0.25,
    eog_threshold: float = 2.5,
) -> Tuple[mne.io.Raw, mne.Epochs, ICA, List[int], List[int], np.ndarray, np.ndarray, bool, bool]:
    """Run complete ICA pipeline for artifact removal.

    Args:
        raw: Continuous MEG data (for component detection).
        epochs: Epoched data (for ICA fitting).
        noise_cov: Noise covariance matrix.
        n_components: Number of ICA components (int), or float 0-1 for
            explained variance threshold.
        random_state: Random seed.
        ecg_threshold: ECG detection threshold (CTPS + correlation union).
        eog_threshold: EOG z-score detection threshold.

    Returns:
        Tuple of (cleaned_raw, cleaned_epochs, ica, ecg_inds, eog_inds,
        ecg_scores, eog_scores, ecg_forced, eog_forced).

    Examples:
        >>> cleaned_raw, cleaned_epochs, ica, ecg_inds, eog_inds, ecg_scores, eog_scores, ecg_forced, eog_forced = run_ica_pipeline(
        ...     raw, epochs, noise_cov
        ... )
    """
    logger.info("=" * 60)
    logger.info("Running ICA pipeline")
    logger.info("=" * 60)

    # Fit ICA
    ica = fit_ica(epochs, n_components, random_state, noise_cov)

    # Detect ECG components
    ecg_inds, ecg_scores, ecg_forced = find_ecg_components(ica, raw, threshold=ecg_threshold)

    # Detect EOG components
    eog_inds, eog_scores, eog_forced = find_eog_components(ica, raw, threshold=eog_threshold)

    # Exclude components
    to_remove = ecg_inds + eog_inds
    ica.exclude = to_remove
    logger.info(f"Excluding {len(to_remove)} ICA components: {to_remove}")

    # Apply ICA
    cleaned_raw, cleaned_epochs = apply_ica(ica, raw, epochs)

    logger.info("ICA pipeline complete")
    return cleaned_raw, cleaned_epochs, ica, ecg_inds, eog_inds, ecg_scores, eog_scores, ecg_forced, eog_forced
