"""Unified data loaders for feature extraction (sensor/source/atlas spaces).

This module provides a single interface for loading MEG data across different
analysis spaces, enabling the same feature extraction code to work at all levels.

Author: Claude (Anthropic)
Date: 2026-01-31
"""

import logging
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, Optional

import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids

from code.utils.config import load_config

logger = logging.getLogger(__name__)

# Unified data structure for all spaces
# Returns: (n_spatial, n_times) continuous data + metadata
SpatialData = namedtuple("SpatialData", ["data", "sfreq", "spatial_names"])


def load_data(
    space: str,
    bids_root: Path,
    subject: str,
    run: str,
    input_type: str = "continuous",
    processing: str = "clean",
    config: Optional[Dict[str, Any]] = None,
) -> SpatialData:
    """Load data in any analysis space (sensor/source/atlas).

    Unified loader that returns consistent format regardless of space.

    Args:
        space: Analysis space - one of:
            - "sensor": MEG channel data
            - "source": Vertex-level source data
            - Any atlas name (e.g., "aparc.a2009s", "schaefer_100"): ROI time series
        bids_root: BIDS root directory
        subject: Subject ID (e.g., "04")
        run: Run ID (e.g., "02")
        input_type: "continuous" or "epochs" (default: "continuous")
        processing: Processing state for sensor/source data:
            - "clean": ICA-cleaned continuous (for continuous)
            - "ica": ICA-only epochs (for epochs)
            - "icaar": ICA+AutoReject epochs (for epochs)
        config: Configuration dictionary (optional, loaded if None)

    Returns:
        SpatialData with:
            - data: np.ndarray of shape (n_spatial, n_times)
            - sfreq: float, sampling frequency in Hz
            - spatial_names: List[str], names of spatial units

    Raises:
        FileNotFoundError: If data files don't exist

    Examples:
        >>> from pathlib import Path
        >>> spatial_data = load_data("sensor", Path("/data/bids"), "04", "02")
        >>> print(spatial_data.data.shape)  # (n_channels, n_times)

        >>> # Load atlas data by name
        >>> spatial_data = load_data("aparc.a2009s", Path("/data/bids"), "04", "02")
        >>> spatial_data = load_data("schaefer_100", Path("/data/bids"), "04", "02")
    """
    if config is None:
        config = load_config()

    logger.info(
        f"Loading {space} data: sub-{subject}, run-{run}, "
        f"type={input_type}, processing={processing}"
    )

    # Determine if space is sensor, source, or an atlas name
    if space == "sensor":
        return _load_sensor_data(
            bids_root, subject, run, input_type, processing, config
        )
    elif space == "source":
        return _load_source_data(
            bids_root, subject, run, input_type, processing, config
        )
    else:
        # Assume it's an atlas name (e.g., "aparc.a2009s", "schaefer_100")
        return _load_atlas_data(
            bids_root, subject, run, input_type, processing, config, atlas_name=space
        )


def _load_sensor_data(
    bids_root: Path,
    subject: str,
    run: str,
    input_type: str,
    processing: str,
    config: Dict[str, Any],
) -> SpatialData:
    """Load sensor-level MEG data (channels).

    For continuous: loads ICA-cleaned continuous data (processing="clean")
    For epochs: loads epoched data (processing="ica" or "icaar")
    """
    # Derivatives are at data_root/derivatives, not bids_root/derivatives
    data_root = Path(config["paths"]["data_root"])
    derivatives_root = data_root / config["paths"]["derivatives"]

    if input_type == "continuous":
        # Load continuous ICA-cleaned data
        # Files are stored as: sub-{subject}_task-gradCPT_run-{run}_proc-clean_meg.fif
        # (no session in filename)
        preproc_dir = derivatives_root / "preprocessed" / f"sub-{subject}" / "meg"
        task_name = config["bids"]["task_name"]
        fif_file = preproc_dir / f"sub-{subject}_task-{task_name}_run-{run}_proc-clean_meg.fif"

        if not fif_file.exists():
            raise FileNotFoundError(
                f"Continuous sensor data not found: {fif_file}"
            )

        logger.debug(f"Loading continuous MEG: {fif_file}")
        raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

        # Get MEG channels only
        picks = mne.pick_types(
            raw.info, meg=True, ref_meg=False, eeg=False, eog=False, ecg=False
        )
        data = raw.get_data(picks=picks)  # (n_channels, n_times)
        sfreq = raw.info["sfreq"]
        ch_names = [raw.ch_names[i] for i in picks]

    elif input_type == "epochs":
        # Load epoched data (ICA or ICA+AR)
        # Files are stored as: sub-{subject}_task-gradCPT_run-{run}_proc-{ica|icaar}_meg.fif
        epochs_dir = derivatives_root / "epochs" / f"sub-{subject}" / "meg"
        task_name = config["bids"]["task_name"]
        epoch_file = epochs_dir / f"sub-{subject}_task-{task_name}_run-{run}_proc-{processing}_meg.fif"

        if not epoch_file.exists():
            raise FileNotFoundError(f"Epoched sensor data not found: {epoch_file}")

        logger.debug(f"Loading epochs: {epoch_file}")
        epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)

        # Get MEG channels only
        picks = mne.pick_types(
            epochs.info, meg=True, ref_meg=False, eeg=False, eog=False, ecg=False
        )
        # Concatenate epochs into continuous data: (n_epochs, n_channels, n_times) -> (n_channels, n_times_total)
        epoch_data = epochs.get_data(picks=picks)  # (n_epochs, n_channels, n_times)
        data = np.concatenate(
            epoch_data, axis=1
        ).T  # Transpose to (n_channels, n_times_total)
        sfreq = epochs.info["sfreq"]
        ch_names = [epochs.ch_names[i] for i in picks]

    else:
        raise ValueError(f"Invalid input_type '{input_type}'. Must be 'continuous' or 'epochs'")

    logger.info(
        f"Loaded sensor data: {data.shape[0]} channels, "
        f"{data.shape[1]} samples, {sfreq:.1f} Hz"
    )

    return SpatialData(data=data, sfreq=sfreq, spatial_names=ch_names)


def _load_source_data(
    bids_root: Path,
    subject: str,
    run: str,
    input_type: str,
    processing: str,
    config: Dict[str, Any],
) -> SpatialData:
    """Load source-level data (vertices).

    Loads morphed source estimates from derivatives/morphed_sources/
    """
    data_root = Path(config["paths"]["data_root"])
    derivatives_root = data_root / config["paths"]["derivatives"]
    morph_dir = derivatives_root / "morphed_sources" / f"sub-{subject}" / "meg"

    if not morph_dir.exists():
        raise FileNotFoundError(f"Morphed sources directory not found: {morph_dir}")

    # Find morphed source estimate file
    # Pattern: sub-{subject}_ses-recording_task-gradCPT_run-{run}_desc-morphed_meg-stc.h5
    pattern = f"sub-{subject}_ses-recording_task-*_run-{run}_*morphed*-stc.h5"
    morph_files = list(morph_dir.glob(pattern))

    if not morph_files:
        raise FileNotFoundError(
            f"Source data not found for sub-{subject} run-{run} in {morph_dir}"
        )

    morph_file = morph_files[0]
    logger.debug(f"Loading source data: {morph_file}")

    # Read source estimate (remove -stc.h5 extension for mne.read_source_estimate)
    stc_basename = str(morph_file).replace("-stc.h5", "")
    stc = mne.read_source_estimate(stc_basename, subject=None)

    # Extract data
    data = stc.data  # (n_vertices, n_times)
    sfreq = 1.0 / stc.tstep
    vertex_names = [f"vertex-{i}" for i in range(data.shape[0])]

    logger.info(
        f"Loaded source data: {data.shape[0]} vertices, "
        f"{data.shape[1]} samples, {sfreq:.1f} Hz"
    )

    return SpatialData(data=data, sfreq=sfreq, spatial_names=vertex_names)


def _load_atlas_data(
    bids_root: Path,
    subject: str,
    run: str,
    input_type: str,
    processing: str,
    config: Dict[str, Any],
    atlas_name: str,
) -> SpatialData:
    """Load atlas-level data (ROI-averaged time series).

    Loads pre-computed ROI time series from derivatives/atlas_timeseries_{atlas}/

    Args:
        atlas_name: Short atlas name (e.g., "aparc.a2009s", "schaefer_100")
    """
    data_root = Path(config["paths"]["data_root"])
    derivatives_root = data_root / config["paths"]["derivatives"]

    atlas_dir = derivatives_root / f"atlas_timeseries_{atlas_name}" / f"sub-{subject}"

    if not atlas_dir.exists():
        raise FileNotFoundError(
            f"Atlas directory not found: {atlas_dir}\n"
            f"Run 'invoke pipeline.atlas --subject {subject}' first to generate atlas data."
        )

    # Find atlas file (.npz format)
    pattern = f"sub-{subject}_*_run-{run}_space-{atlas_name}_timeseries.npz"
    atlas_files = list(atlas_dir.glob(pattern))

    if not atlas_files:
        raise FileNotFoundError(
            f"Atlas data not found for sub-{subject} run-{run} atlas={atlas_name} in {atlas_dir}"
        )

    atlas_file = atlas_files[0]
    logger.debug(f"Loading atlas data: {atlas_file}")

    # Load npz file
    atlas_data = np.load(atlas_file, allow_pickle=True)

    # Extract ROI time series
    data = atlas_data["data"]  # (n_rois, n_times)
    sfreq = float(atlas_data["sfreq"])
    roi_names = list(atlas_data["roi_names"])

    logger.info(
        f"Loaded atlas data ({atlas_name}): {len(roi_names)} ROIs, "
        f"{data.shape[1]} samples, {sfreq:.1f} Hz"
    )

    return SpatialData(data=data, sfreq=sfreq, spatial_names=roi_names)
