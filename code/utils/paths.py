"""Path utilities for saflow.

This module provides helper functions for constructing BIDS-compliant paths
and managing file organization across the pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Optional


def get_bids_root(config: Dict[str, Any]) -> Path:
    """Get BIDS root directory.

    Args:
        config: Configuration dictionary.

    Returns:
        Path to BIDS root directory.
    """
    return Path(config["paths"]["raw"])


def get_derivatives_root(config: Dict[str, Any]) -> Path:
    """Get derivatives root directory.

    Args:
        config: Configuration dictionary.

    Returns:
        Path to derivatives directory.
    """
    return Path(config["paths"]["derivatives"])


def get_processed_root(config: Dict[str, Any]) -> Path:
    """Get processed data root directory.

    Args:
        config: Configuration dictionary.

    Returns:
        Path to processed directory.
    """
    return Path(config["paths"]["processed"])


def get_subject_dir(
    config: Dict[str, Any], subject: str, session: Optional[str] = None
) -> Path:
    """Get BIDS subject directory.

    Args:
        config: Configuration dictionary.
        subject: Subject ID (without 'sub-' prefix).
        session: Optional session ID (without 'ses-' prefix).

    Returns:
        Path to subject directory.
    """
    bids_root = get_bids_root(config)
    subject_dir = bids_root / f"sub-{subject}"

    if session:
        subject_dir = subject_dir / f"ses-{session}"

    return subject_dir


def get_meg_dir(
    config: Dict[str, Any], subject: str, session: Optional[str] = None
) -> Path:
    """Get BIDS MEG directory for a subject.

    Args:
        config: Configuration dictionary.
        subject: Subject ID (without 'sub-' prefix).
        session: Optional session ID (without 'ses-' prefix).

    Returns:
        Path to MEG directory.
    """
    subject_dir = get_subject_dir(config, subject, session)
    return subject_dir / "meg"


def get_bids_basename(
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    acquisition: Optional[str] = None,
    processing: Optional[str] = None,
    space: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Construct BIDS-compliant filename (without extension).

    Args:
        subject: Subject ID (without 'sub-' prefix).
        session: Optional session ID (without 'ses-' prefix).
        task: Optional task name (without 'task-' prefix).
        run: Optional run ID (without 'run-' prefix).
        acquisition: Optional acquisition label (without 'acq-' prefix).
        processing: Optional processing label (without 'proc-' prefix).
        space: Optional space label (without 'space-' prefix).
        suffix: Optional suffix (e.g., 'meg', 'eeg', 'events').

    Returns:
        BIDS-compliant basename.

    Example:
        >>> get_bids_basename('01', session='recording', task='gradCPT', run='02', suffix='meg')
        'sub-01_ses-recording_task-gradCPT_run-02_meg'
    """
    parts = [f"sub-{subject}"]

    if session:
        parts.append(f"ses-{session}")
    if task:
        parts.append(f"task-{task}")
    if acquisition:
        parts.append(f"acq-{acquisition}")
    if run:
        parts.append(f"run-{run}")
    if processing:
        parts.append(f"proc-{processing}")
    if space:
        parts.append(f"space-{space}")
    if suffix:
        parts.append(suffix)

    return "_".join(parts)


def get_preprocessed_path(
    config: Dict[str, Any],
    subject: str,
    run: str,
    session: Optional[str] = None,
    suffix: str = "meg",
    extension: str = ".fif",
) -> Path:
    """Get path to preprocessed MEG file.

    Args:
        config: Configuration dictionary.
        subject: Subject ID.
        run: Run ID.
        session: Optional session ID.
        suffix: File suffix (default: 'meg').
        extension: File extension (default: '.fif').

    Returns:
        Path to preprocessed file.
    """
    derivatives_root = get_derivatives_root(config)
    task_name = config["bids"]["task_name"]

    if session is None:
        session = config["bids"]["sessions"][0]

    basename = get_bids_basename(
        subject=subject, session=session, task=task_name, run=run, suffix=suffix
    )

    return derivatives_root / f"sub-{subject}" / "meg" / f"{basename}{extension}"


def get_features_path(
    config: Dict[str, Any],
    feature_type: str,
    subject: str,
    run: str,
    analysis_space: str = "sensor",
    session: Optional[str] = None,
    extension: str = ".npy",
) -> Path:
    """Get path to feature file.

    Args:
        config: Configuration dictionary.
        feature_type: Type of feature (e.g., 'psd', 'lzc', 'fooof').
        subject: Subject ID.
        run: Run ID.
        analysis_space: Analysis space ('sensor' or 'source').
        session: Optional session ID.
        extension: File extension (default: '.npy').

    Returns:
        Path to feature file.
    """
    processed_root = get_processed_root(config)
    task_name = config["bids"]["task_name"]

    if session is None:
        session = config["bids"]["sessions"][0]

    # Features are organized by analysis space
    feature_dir = processed_root / f"features_{analysis_space}" / feature_type

    basename = get_bids_basename(
        subject=subject,
        session=session,
        task=task_name,
        run=run,
        space=analysis_space,
        suffix=feature_type,
    )

    return feature_dir / f"{basename}{extension}"


def get_statistics_path(
    config: Dict[str, Any],
    test_type: str,
    feature_type: str,
    analysis_space: str = "sensor",
    extension: str = ".npy",
) -> Path:
    """Get path to statistical results file.

    Args:
        config: Configuration dictionary.
        test_type: Type of statistical test (e.g., 'ttest', 'permutation').
        feature_type: Type of feature being tested.
        analysis_space: Analysis space ('sensor' or 'source').
        extension: File extension (default: '.npy').

    Returns:
        Path to statistics file.
    """
    processed_root = get_processed_root(config)
    stats_dir = processed_root / f"statistics_{analysis_space}" / test_type

    filename = f"{feature_type}_{test_type}{extension}"
    return stats_dir / filename


def get_classification_path(
    config: Dict[str, Any],
    classifier_type: str,
    feature_type: str,
    analysis_space: str = "sensor",
    extension: str = ".pkl",
) -> Path:
    """Get path to classification results file.

    Args:
        config: Configuration dictionary.
        classifier_type: Type of classifier (e.g., 'logistic', 'rf', 'svm').
        feature_type: Type of feature used for classification.
        analysis_space: Analysis space ('sensor' or 'source').
        extension: File extension (default: '.pkl').

    Returns:
        Path to classification results file.
    """
    processed_root = get_processed_root(config)
    classif_dir = processed_root / f"classification_{analysis_space}" / classifier_type

    filename = f"{feature_type}_{classifier_type}{extension}"
    return classif_dir / filename


def get_figure_path(
    config: Dict[str, Any],
    figure_name: str,
    analysis_space: str = "sensor",
    extension: str = ".png",
) -> Path:
    """Get path to figure file.

    Args:
        config: Configuration dictionary.
        figure_name: Name of the figure.
        analysis_space: Analysis space ('sensor' or 'source').
        extension: File extension (default: '.png').

    Returns:
        Path to figure file.
    """
    reports_root = Path(config["paths"]["reports"])
    figures_dir = reports_root / "figures" / analysis_space

    if not figure_name.endswith(extension):
        figure_name = f"{figure_name}{extension}"

    return figures_dir / figure_name


def get_log_path(config: Dict[str, Any], log_name: str, stage: str) -> Path:
    """Get path to log file.

    Args:
        config: Configuration dictionary.
        log_name: Name of the log file (without extension).
        stage: Pipeline stage (e.g., 'preprocessing', 'features').

    Returns:
        Path to log file.
    """
    logs_root = Path(config["paths"]["logs"])
    log_dir = logs_root / stage

    if not log_name.endswith(".log"):
        log_name = f"{log_name}.log"

    return log_dir / log_name


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Path to directory or file (if file, creates parent directory).

    Returns:
        Path to directory.
    """
    if path.suffix:  # It's a file path
        directory = path.parent
    else:  # It's a directory path
        directory = path

    directory.mkdir(parents=True, exist_ok=True)
    return directory
