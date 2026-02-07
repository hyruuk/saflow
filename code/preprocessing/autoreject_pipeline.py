"""AutoReject pipeline for automated epoch rejection.

This module implements the AutoReject algorithm for identifying bad epochs
and channels in MEG data.
"""

import logging

import mne
import numpy as np
from autoreject import AutoReject
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def run_autoreject(
    epochs: mne.Epochs,
    n_interpolate: list = None,
    consensus: list = None,
    picks: str = "mag",
    n_jobs: int = 1,
    random_state: int = 42,
) -> tuple:
    """Run AutoReject to identify bad epochs and channels.

    Args:
        epochs: Epoched MEG data.
        n_interpolate: Number of channels to interpolate (grid search).
        consensus: Fraction of channels for rejection (grid search).
        picks: Channel types to use.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (autoreject_object, reject_log).

    Examples:
        >>> ar, log = run_autoreject(epochs, n_jobs=12)
        >>> print(f"Bad epochs: {np.sum(log.bad_epochs)}")
    """
    if n_interpolate is None:
        n_interpolate = [1, 4, 32]

    if consensus is None:
        consensus = [0.1, 0.2, 0.3, 0.5]

    logger.info("Running AutoReject (first pass - fit only)")
    logger.info(f"  n_interpolate: {n_interpolate}")
    logger.info(f"  consensus: {consensus}")
    logger.info(f"  n_jobs: {n_jobs}")

    console.print("[yellow]⏳ AutoReject Pass 1: Fitting model (this may take several minutes)...[/yellow]")

    ar = AutoReject(
        n_interpolate=n_interpolate,
        consensus=consensus,
        picks=picks,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose="progressbar",  # Enable progress bar
    )

    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)
    console.print("[green]✓ AutoReject Pass 1: Complete[/green]")

    n_bad = np.sum(reject_log.bad_epochs)
    n_total = len(epochs)
    pct_bad = 100 * n_bad / n_total

    logger.info(f"AutoReject identified {n_bad}/{n_total} bad epochs ({pct_bad:.1f}%)")

    return ar, reject_log


def get_good_epochs_mask(reject_log) -> np.ndarray:
    """Get boolean mask of good (non-rejected) epochs.

    Args:
        reject_log: AutoReject reject log object.

    Returns:
        Boolean array where True indicates good epochs.

    Examples:
        >>> good_mask = get_good_epochs_mask(reject_log)
        >>> epochs_clean = epochs[good_epoch]
    """
    return ~reject_log.bad_epochs


def run_autoreject_transform(
    epochs: mne.Epochs,
    n_interpolate: list = None,
    consensus: list = None,
    picks: str = "mag",
    n_jobs: int = 1,
    random_state: int = 42,
) -> tuple:
    """Run AutoReject with transform (interpolates bad channels).

    This is the second pass of AutoReject, typically run after ICA cleaning.
    Unlike the first pass (fit only), this applies interpolation to bad channels.

    Args:
        epochs: Epoched MEG data (typically after ICA).
        n_interpolate: Number of channels to interpolate (grid search).
        consensus: Fraction of channels for rejection (grid search).
        picks: Channel types to use.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (cleaned_epochs, autoreject_object, reject_log).

    Examples:
        >>> epochs_ar, ar, log = run_autoreject_transform(epochs_ica, n_jobs=12)
        >>> print(f"Interpolated channels in {np.sum(log.bad_epochs)} epochs")
    """
    if n_interpolate is None:
        n_interpolate = [1, 4, 32]

    if consensus is None:
        consensus = [0.1, 0.2, 0.3, 0.5]

    logger.info("Running AutoReject with transform (second pass)")
    logger.info(f"  n_interpolate: {n_interpolate}")
    logger.info(f"  consensus: {consensus}")
    logger.info(f"  n_jobs: {n_jobs}")

    console.print("[yellow]⏳ AutoReject Pass 2: Fitting and transforming (with interpolation)...[/yellow]")

    ar = AutoReject(
        n_interpolate=n_interpolate,
        consensus=consensus,
        picks=picks,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose="progressbar",  # Enable progress bar
    )

    # fit_transform applies interpolation
    epochs_clean = ar.fit_transform(epochs)
    reject_log = ar.get_reject_log(epochs)
    console.print("[green]✓ AutoReject Pass 2: Complete[/green]")

    n_bad = np.sum(reject_log.bad_epochs)
    n_total = len(epochs)
    pct_bad = 100 * n_bad / n_total

    # Count interpolated channels
    n_interpolated = np.sum(reject_log.labels == 2)  # 2 = interpolated
    n_total_channels = reject_log.labels.size

    logger.info(
        f"AutoReject transform: {n_bad}/{n_total} bad epochs ({pct_bad:.1f}%)"
    )
    logger.info(
        f"AutoReject transform: {n_interpolated}/{n_total_channels} channels interpolated"
    )

    return epochs_clean, ar, reject_log


def run_autoreject_both(
    epochs: mne.Epochs,
    n_interpolate: list = None,
    consensus: list = None,
    picks: str = "mag",
    n_jobs: int = 1,
    random_state: int = 42,
) -> tuple:
    """Run AutoReject: fit once, then return both reject_log and interpolated epochs.

    This avoids double-fitting by calling fit() once, then both
    get_reject_log() (for flags) and transform() (for interpolation).

    Args:
        epochs: Epoched MEG data (typically after ICA).
        n_interpolate: Number of channels to interpolate (grid search).
        consensus: Fraction of channels for rejection (grid search).
        picks: Channel types to use.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (autoreject_object, reject_log, epochs_interpolated).
    """
    if n_interpolate is None:
        n_interpolate = [1, 4, 32]

    if consensus is None:
        consensus = [0.1, 0.2, 0.3, 0.5]

    logger.info("Running AutoReject (fit + reject_log + transform)")
    logger.info(f"  n_interpolate: {n_interpolate}")
    logger.info(f"  consensus: {consensus}")
    logger.info(f"  n_jobs: {n_jobs}")

    console.print("[yellow]\u23f3 AutoReject: Fitting model (fit + reject_log + transform)...[/yellow]")

    ar = AutoReject(
        n_interpolate=n_interpolate,
        consensus=consensus,
        picks=picks,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose="progressbar",
    )

    ar.fit(epochs)

    # Get reject log (flags) without modifying epochs
    reject_log = ar.get_reject_log(epochs)

    # Get interpolated epochs via transform
    epochs_interpolated = ar.transform(epochs)

    console.print("[green]\u2713 AutoReject: fit + reject_log + transform complete[/green]")

    n_bad = np.sum(reject_log.bad_epochs)
    n_total = len(epochs)
    pct_bad = 100 * n_bad / n_total

    # Count interpolated channels
    n_interpolated = np.sum(reject_log.labels == 2)  # 2 = interpolated
    n_total_channels = reject_log.labels.size

    logger.info(
        f"AutoReject: {n_bad}/{n_total} bad epochs ({pct_bad:.1f}%), "
        f"{n_interpolated}/{n_total_channels} channel-epochs interpolated"
    )

    return ar, reject_log, epochs_interpolated
