"""Data loading and feature extraction utilities for saflow.

This module provides utilities for:
- Loading features (PSD, LZC, FOOOF)
- Loading trial-level and subject-level data
- Balancing datasets across conditions
- Epoch selection and filtering
- VTC-based data splitting

All functions use logging and avoid hardcoded paths.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pickle
from scipy.stats import zscore

logger = logging.getLogger(__name__)


def load_features(
    feature_folder: Union[str, Path],
    feature: str = "psd",
    splitby: str = "inout",
    inout: str = "INOUT_2575",
    remove_errors: bool = True,
    get_task: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load feature data from processed files.

    Loads feature data (PSD, LZC, FOOOF) from BIDS derivatives and organizes
    it by condition (IN/OUT) with optional filtering by task type.

    Args:
        feature_folder: Path to folder containing feature files.
        feature: Feature type to load. Options: 'psd', 'lzc', 'slope', 'offset',
            'knee', 'r_squared'. Defaults to 'psd'.
        splitby: How to split conditions. Currently only 'inout' is supported.
            Defaults to 'inout'.
        inout: Which INOUT classification to use. Options: 'INOUT_2575', 'INOUT_50_50',
            'INOUT_10_90'. Defaults to 'INOUT_2575'.
        remove_errors: Whether to remove error trials. Defaults to True.
        get_task: List of task types to include (e.g., ['correct_commission']).
            Defaults to None (all tasks).

    Returns:
        Tuple containing:
        - X: Feature data array, shape (n_features, n_trials, n_channels)
        - y: Labels (0=IN, 1=OUT), shape (n_trials,)
        - groups: Subject indices, shape (n_trials,)
        - metadata: Dictionary with 'vtc' and 'task' arrays

    Examples:
        >>> X, y, groups, meta = load_features(
        ...     "/data/derivatives/psd",
        ...     feature="psd",
        ...     inout="INOUT_2575"
        ... )
        >>> print(f"Loaded {X.shape[1]} trials with {X.shape[0]} features")
    """
    feature_folder = Path(feature_folder)
    if get_task is None:
        get_task = ["correct_commission"]

    X = []
    y = []
    groups = []
    VTC = []
    task = []

    for idx_subj, subj in enumerate(sorted(os.listdir(feature_folder))):
        subj_path = feature_folder / subj / "meg"
        if not subj_path.exists():
            logger.warning(f"MEG folder not found for subject {subj}, skipping")
            continue

        for trial_file in os.listdir(subj_path):
            if "desc-" not in trial_file:
                continue

            filepath = subj_path / trial_file

            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                logger.error(f"Could not load {trial_file}: {e}")
                continue

            # Select epoch based on criteria
            epoch_selected = select_trial(
                data["info"], type_how="lapse", inout=inout, verbose=False
            )

            if epoch_selected:
                try:
                    # Get INOUT classification
                    if splitby == "inout":
                        inout_value = data["info"].get(inout)
                        if isinstance(inout_value, str):
                            if inout_value == "IN":
                                y.append(0)
                            elif inout_value == "OUT":
                                y.append(1)
                            else:
                                continue  # Skip MID or other values

                            groups.append(idx_subj)

                            # Load feature data
                            if feature in ["psd", "lzc"]:
                                X.append(data["data"])

                            elif feature in ["slope", "offset", "r_squared", "knee"]:
                                # Load FOOOF data
                                fooof_file = str(filepath).replace("_desc-", "_fg-")
                                with open(fooof_file, "rb") as f:
                                    fooof_data = pickle.load(f)

                                temp_X = []
                                for fm in fooof_data["fooof"]:
                                    if feature == "slope":
                                        temp_X.append(
                                            fm.get_params("aperiodic_params")[-1]
                                        )
                                    elif feature == "offset":
                                        temp_X.append(
                                            fm.get_params("aperiodic_params")[0]
                                        )
                                    elif feature == "knee":
                                        temp_X.append(
                                            fm.get_params("aperiodic_params")[1]
                                        )
                                    elif feature == "r_squared":
                                        temp_X.append(fm.get_params("r_squared"))
                                X.append(np.array(temp_X))

                            VTC.append(np.nanmean(data["info"]["included_VTC"]))
                            task.append(data["info"]["task"])

                except Exception as e:
                    logger.error(f"Could not process trial in {trial_file}: {e}")
                    continue

    X = np.array(X)
    if X.ndim == 3:
        X = X.transpose(1, 0, 2)  # (features, trials, channels)

    y = np.array(y)
    groups = np.array(groups)
    VTC = np.array(VTC)
    task = np.array(task)

    logger.info(
        f"Loaded {len(X.T)} trials from {len(np.unique(groups))} subjects "
        f"with {X.shape[0]} features"
    )

    return X, y, groups, {"vtc": VTC, "task": task}


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: Optional[int] = 42069,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance dataset by equalizing class counts within each group/subject.

    For each subject, randomly samples the same number of trials from each
    class to balance the dataset.

    Args:
        X: Feature data, shape (n_features, n_trials, n_channels).
        y: Labels, shape (n_trials,).
        groups: Group/subject indices, shape (n_trials,).
        seed: Random seed for reproducibility. Defaults to 42069.

    Returns:
        Tuple containing:
        - balanced_X: Balanced feature data
        - balanced_y: Balanced labels
        - balanced_groups: Balanced group indices

    Examples:
        >>> X, y, groups = balance_dataset(X, y, groups, seed=42)
        >>> print(f"Balanced to {len(y)} trials")
    """
    n_observations = X.shape[1]
    balanced_indices = []

    if seed is not None:
        np.random.seed(seed)

    # Process each group
    for group in np.unique(groups):
        group_indices = np.where(groups == group)[0]

        # Count observations in each class within the group
        class_counts = {
            label: np.sum(y[group_indices] == label)
            for label in np.unique(y[group_indices])
        }

        logger.debug(f"Group {group}: {class_counts}")

        # Find minimum count among classes in this group
        min_count = min(class_counts.values())

        # Sample from each class
        for class_value in np.unique(y[group_indices]):
            class_indices = group_indices[y[group_indices] == class_value]
            sampled_indices = np.random.choice(
                class_indices, size=min_count, replace=False
            )
            balanced_indices.extend(sampled_indices)

    # Sort indices to maintain original order
    balanced_indices = sorted(set(balanced_indices))

    # Select balanced data
    balanced_X = X[:, balanced_indices, :]
    balanced_y = y[balanced_indices]
    balanced_groups = groups[balanced_indices]

    logger.info(
        f"Balanced dataset from {n_observations} to {len(balanced_y)} trials "
        f"({len(np.unique(balanced_groups))} subjects)"
    )

    return balanced_X, balanced_y, balanced_groups


def get_VTC_bounds(
    events_dicts: List[Dict[str, Any]],
    lowbound: int = 25,
    highbound: int = 75,
) -> Tuple[float, float]:
    """Get VTC percentile bounds for IN/OUT classification.

    Computes the lower and upper percentile bounds of VTC (variability of
    reaction times) across all events to define IN and OUT zones.

    Args:
        events_dicts: List of event dictionaries with 'included_VTC' key.
        lowbound: Lower percentile for IN zone. Defaults to 25.
        highbound: Upper percentile for OUT zone. Defaults to 75.

    Returns:
        Tuple containing:
        - inbound: VTC value at lower percentile
        - outbound: VTC value at upper percentile

    Examples:
        >>> inbound, outbound = get_VTC_bounds(events_dicts, 25, 75)
        >>> print(f"IN < {inbound:.2f}, OUT > {outbound:.2f}")
    """
    # Get the averaged VTC for each epoch
    run_VTCs = []
    for info_dict in events_dicts:
        run_VTCs.append(np.mean(info_dict["included_VTC"], axis=0))

    # Obtain the bounds of the VTC for this run
    inbound = np.percentile(run_VTCs, lowbound, axis=0)
    outbound = np.percentile(run_VTCs, highbound, axis=0)

    logger.debug(
        f"VTC bounds: IN < {inbound:.3f} ({lowbound}th percentile), "
        f"OUT > {outbound:.3f} ({highbound}th percentile)"
    )

    return inbound, outbound


def get_inout(
    info_dict: Dict[str, Any],
    inbound: float,
    outbound: float,
) -> str:
    """Classify epoch as IN, OUT, or MID based on VTC value.

    Args:
        info_dict: Event dictionary with 'included_VTC' key.
        inbound: Lower VTC threshold for IN classification.
        outbound: Upper VTC threshold for OUT classification.

    Returns:
        Classification: 'IN', 'OUT', or 'MID'.

    Examples:
        >>> classification = get_inout(event_dict, 0.5, 1.5)
        >>> print(classification)  # 'IN', 'OUT', or 'MID'
    """
    VTC = np.mean(info_dict["included_VTC"], axis=0)

    if VTC <= inbound:
        return "IN"
    elif VTC >= outbound:
        return "OUT"
    else:
        return "MID"


def select_epoch(
    event_dict: Dict[str, Any],
    bad_how: str = "any",
    type_how: str = "alltrials",
    inout_epoch: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Select epoch based on quality and task criteria.

    Determines whether to retain an epoch based on artifact rejection,
    task type, and IN/OUT classification.

    Args:
        event_dict: Event dictionary with metadata.
        bad_how: How to handle bad epochs. 'any' rejects if any bad epoch in window.
            Defaults to 'any'.
        type_how: Task type filter. Options:
            - 'alltrials': Keep all trials
            - 'correct': Keep only correct trials
            - 'rare': Keep rare stimulus trials (hits and errors)
            - 'lapse': Keep only commission error trials
            Defaults to 'alltrials'.
        inout_epoch: Required IN/OUT classification ('IN', 'OUT', or None).
            Defaults to None (no filtering).
        verbose: Whether to log selection details. Defaults to False.

    Returns:
        Whether to retain the epoch.

    Examples:
        >>> keep = select_epoch(event_dict, type_how='correct', inout_epoch='IN')
        >>> if keep:
        ...     # Process epoch
    """
    # Check if bad epoch
    if bad_how == "any":
        bad_epoch = np.sum(event_dict["included_bad_epochs"]) > 0
    else:
        bad_epoch = False

    # Check trial types across the epoch
    if type_how == "alltrials":
        retain_task = True
    elif type_how == "correct":
        correct_task_types = ["correct_omission", "correct_commission"]
        retain_task = all(
            item in correct_task_types for item in event_dict["included_task"]
        )
    elif type_how == "rare":
        rare_task_types = ["correct_omission", "commission_error"]
        retain_task = any(
            item in event_dict["included_task"] for item in rare_task_types
        )
    elif type_how == "lapse":
        retain_task = "commission_error" in event_dict["included_task"]
    else:
        logger.warning(f"Unknown type_how: {type_how}, defaulting to alltrials")
        retain_task = True

    # Check inout type
    retain_inout = False
    if inout_epoch is not None:
        if inout_epoch in ["IN", "OUT"]:
            retain_inout = True
    else:
        retain_inout = True  # No INOUT filtering

    retain_epoch = retain_task & ~bad_epoch & retain_inout

    if verbose:
        logger.debug(
            f"Bad: {bad_epoch}, InOut: {retain_inout}, Type: {retain_task}, "
            f"Retain: {retain_epoch}"
        )
        logger.debug(f"Tasks: {event_dict['included_task']}")
        logger.debug(f"Bad epochs: {event_dict['included_bad_epochs']}")

    return retain_epoch


def select_trial(
    info_dict: Dict[str, Any],
    type_how: str = "lapse",
    inout: str = "INOUT_2575",
    verbose: bool = False,
) -> bool:
    """Select trial based on task type and IN/OUT classification.

    Simplified trial selection wrapper for select_epoch with INOUT from dict.

    Args:
        info_dict: Trial info dictionary with INOUT classification keys.
        type_how: Task type filter (see select_epoch). Defaults to 'lapse'.
        inout: Which INOUT key to use. Defaults to 'INOUT_2575'.
        verbose: Whether to log selection details. Defaults to False.

    Returns:
        Whether to retain the trial.

    Examples:
        >>> keep = select_trial(trial_info, type_how='lapse', inout='INOUT_2575')
    """
    inout_value = info_dict.get(inout, "MID")
    return select_epoch(
        info_dict,
        bad_how="any",
        type_how=type_how,
        inout_epoch=inout_value,
        verbose=verbose,
    )


def get_trial_data(
    data_reshaped: np.ndarray,
    trial_idx: int,
    feat_to_get: str,
    freq_bins: Optional[np.ndarray] = None,
    zscored: bool = False,
) -> np.ndarray:
    """Extract trial data and compute frequency band power.

    Args:
        data_reshaped: Data array, shape (n_trials, n_channels).
        trial_idx: Trial index to extract.
        feat_to_get: Feature to extract ('ksor' or frequency band name).
        freq_bins: Frequency bins for band power calculation. Defaults to None.
        zscored: Whether to z-score the data. Defaults to False.

    Returns:
        Trial data array, shape (n_channels, n_features).

    Examples:
        >>> trial_data = get_trial_data(data, 0, 'psd_corrected')
    """
    from code.utils.signal_processing import average_bands

    trial_data = []
    n_chans = data_reshaped.shape[1]

    for chan_idx in range(n_chans):
        if feat_to_get == "ksor":
            # Extract FOOOF parameters
            feat_data = [
                data_reshaped[trial_idx][chan_idx]["knee"],
                data_reshaped[trial_idx][chan_idx]["exponent"],
                data_reshaped[trial_idx][chan_idx]["offset"],
                data_reshaped[trial_idx][chan_idx]["r_squared"],
            ]
            trial_data.append(np.array(feat_data))
        else:
            # Extract PSD and compute band power
            psd = data_reshaped[trial_idx][chan_idx][feat_to_get]
            if freq_bins is None:
                freq_bins = data_reshaped[trial_idx][chan_idx]["freq_bins"]
            power_bands = average_bands(psd, freq_bins)
            trial_data.append(power_bands)

    trial_data = np.array(trial_data)

    if zscored:
        trial_data = zscore(trial_data, axis=0)

    return trial_data
