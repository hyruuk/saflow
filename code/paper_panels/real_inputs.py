"""Load aligned real Schaefer-400 features and corrected behavioral labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

from code.paper_panels.alignment import (
    build_alignment_keys,
    require_exact_alignment,
    validate_schaefer_400,
)
from code.paper_panels.contracts import PAPER_BANDS, PANEL23_FEATURES
from code.paper_panels.labels import (
    LABEL_IN,
    LABEL_OUT,
    OUTCOME_COMMISSION_ERROR,
    OUTCOME_CORRECT_OMISSION,
    build_corrected_window_labels,
)
from code.paper_panels.preflight import (
    _events_path,
    _feature_paths,
    _validate_event_provenance,
)


@dataclass(frozen=True)
class RealFigure3Inputs:
    """Memory-bounded aligned inputs for all three paper panels."""

    feature_tensor: np.ndarray
    panel1_tensor: np.ndarray
    states: np.ndarray
    outcomes: np.ndarray
    cells: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    alignment_keys: np.ndarray
    parcel_order: tuple[str, ...]
    feature_order: tuple[str, ...]
    frequencies: np.ndarray
    raw_spectrum_in: np.ndarray
    raw_spectrum_out: np.ndarray
    corrected_spectrum_in: np.ndarray
    corrected_spectrum_out: np.ndarray
    fooof_state_in: np.ndarray
    fooof_state_out: np.ndarray
    run_label_contexts: tuple["RunLabelContext", ...]
    input_inventory: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RunLabelContext:
    """Behavioral ingredients needed to rebuild one shifted run's labels."""

    start: int
    stop: int
    subject: str
    run: str
    vtc: np.ndarray
    contributing_indices: np.ndarray
    contributing_bad_flags: np.ndarray


def load_real_inputs(
    config: dict[str, Any],
    subjects: Sequence[str],
    runs: Sequence[str],
    *,
    include_spectra: bool = False,
) -> RealFigure3Inputs:
    """Load, align, validate, and concatenate corrected real recordings."""
    recordings = [
        _load_recording(
            config,
            str(subject),
            str(run),
            include_spectra=include_spectra,
        )
        for subject in subjects
        for run in runs
    ]
    if not recordings:
        raise ValueError("at least one subject/run recording is required")
    parcel_order = recordings[0]["parcel_order"]
    frequencies = recordings[0]["frequencies"]
    for recording in recordings[1:]:
        if recording["parcel_order"] != parcel_order:
            raise ValueError("Schaefer-400 parcel order differs across recordings")
        if not np.array_equal(recording["frequencies"], frequencies):
            raise ValueError("PSD frequency grids differ across recordings")
    contexts = []
    offset = 0
    for recording in recordings:
        count = len(recording["states"])
        contexts.append(
            RunLabelContext(
                start=offset,
                stop=offset + count,
                subject=recording["subject"],
                run=recording["run"],
                vtc=recording["vtc"],
                contributing_indices=recording["contributing_indices"],
                contributing_bad_flags=recording["contributing_bad_flags"],
            )
        )
        offset += count
    return RealFigure3Inputs(
        feature_tensor=np.concatenate(
            [recording["feature_tensor"] for recording in recordings]
        ),
        panel1_tensor=np.concatenate(
            [recording["panel1_tensor"] for recording in recordings]
        ),
        states=np.concatenate([recording["states"] for recording in recordings]),
        outcomes=np.concatenate([recording["outcomes"] for recording in recordings]),
        cells=np.concatenate([recording["cells"] for recording in recordings]),
        subjects=np.concatenate([recording["subjects"] for recording in recordings]),
        runs=np.concatenate([recording["runs"] for recording in recordings]),
        alignment_keys=np.concatenate(
            [recording["alignment_keys"] for recording in recordings]
        ),
        parcel_order=parcel_order,
        feature_order=PANEL23_FEATURES,
        frequencies=frequencies,
        raw_spectrum_in=np.stack(
            [recording["raw_spectrum_in"] for recording in recordings]
        ),
        raw_spectrum_out=np.stack(
            [recording["raw_spectrum_out"] for recording in recordings]
        ),
        corrected_spectrum_in=np.stack(
            [recording["corrected_spectrum_in"] for recording in recordings]
        ),
        corrected_spectrum_out=np.stack(
            [recording["corrected_spectrum_out"] for recording in recordings]
        ),
        fooof_state_in=np.stack(
            [recording["fooof_state_in"] for recording in recordings]
        ),
        fooof_state_out=np.stack(
            [recording["fooof_state_out"] for recording in recordings]
        ),
        run_label_contexts=tuple(contexts),
        input_inventory=tuple(
            item
            for recording in recordings
            for item in recording["input_inventory"]
        ),
    )


def _load_recording(
    config: dict[str, Any],
    subject: str,
    run: str,
    *,
    include_spectra: bool,
) -> dict[str, Any]:
    """Load one subject/run and derive strict labels and paper features."""
    paths = _feature_paths(config, subject, run)
    if any(path is None for path in paths.values()):
        raise FileNotFoundError(f"missing Paper panels features for sub-{subject} run-{run}")
    events = pd.read_csv(_events_path(config, subject, run), sep="\t")
    _validate_event_provenance(events)
    raw, frequencies, metadata, parcels = _load_psd(paths["welch"])
    corrected, corrected_frequencies, corrected_metadata, corrected_parcels = (
        _load_psd(paths["corrected_psd"])
    )
    fooof, fooof_metadata, fooof_parcels = _load_fooof(paths["fooof"])
    validate_schaefer_400(parcels)
    _require_recording_alignment(
        subject,
        run,
        metadata,
        corrected_metadata,
        fooof_metadata,
        parcels,
        corrected_parcels,
        fooof_parcels,
    )
    if not np.array_equal(frequencies, corrected_frequencies):
        raise ValueError("raw and corrected PSD frequency grids differ")
    labels, keys = _build_labels(events, metadata, subject, run)
    feature_tensor = np.concatenate(
        [fooof, _band_reduce(corrected, frequencies)], axis=2
    )
    raw_bands = np.log10(np.maximum(_band_reduce(raw, frequencies), np.finfo(float).tiny))
    return {
        "feature_tensor": feature_tensor,
        "panel1_tensor": np.concatenate([raw_bands, feature_tensor], axis=2),
        "states": _state_names(labels["state"]),
        "outcomes": _outcome_names(labels["outcome"]),
        "cells": labels["cell"],
        "subjects": np.repeat(subject, len(feature_tensor)),
        "runs": np.repeat(run, len(feature_tensor)),
        "alignment_keys": keys,
        "parcel_order": tuple(parcels),
        "frequencies": frequencies,
        "raw_spectrum_in": _optional_state_spectrum(
            np.log10(np.maximum(raw, np.finfo(float).tiny)),
            labels["state"],
            LABEL_IN,
            include_spectra,
        ),
        "raw_spectrum_out": _optional_state_spectrum(
            np.log10(np.maximum(raw, np.finfo(float).tiny)),
            labels["state"],
            LABEL_OUT,
            include_spectra,
        ),
        "corrected_spectrum_in": _optional_state_spectrum(
            corrected, labels["state"], LABEL_IN, include_spectra
        ),
        "corrected_spectrum_out": _optional_state_spectrum(
            corrected, labels["state"], LABEL_OUT, include_spectra
        ),
        "fooof_state_in": _optional_state_features(
            fooof, labels["state"], LABEL_IN, include_spectra
        ),
        "fooof_state_out": _optional_state_features(
            fooof, labels["state"], LABEL_OUT, include_spectra
        ),
        "subject": subject,
        "run": run,
        "vtc": trials_vtc(events),
        "contributing_indices": np.asarray(
            list(metadata["included_epoch_indices"]), dtype=int
        ),
        "contributing_bad_flags": np.asarray(
            list(metadata["included_bad_ar2"]), dtype=bool
        ),
        "input_inventory": tuple(
            _stable_inventory(path)
            for path in (_events_path(config, subject, run), *paths.values())
            if path is not None
        ),
    }


def _load_psd(
    path: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    """Load one PSD bundle without retaining its archive handle."""
    if path is None:
        raise FileNotFoundError("PSD path is missing")
    with np.load(path, allow_pickle=True) as archive:
        return (
            np.asarray(archive["psds"], dtype=float),
            np.asarray(archive["freqs"], dtype=float),
            archive["trial_metadata"].item(),
            archive["ch_names"].astype(str).tolist(),
        )


def _load_fooof(
    path: Path | None,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Load exponent, offset, and fit quality in canonical feature order."""
    if path is None:
        raise FileNotFoundError("FOOOF path is missing")
    with np.load(path, allow_pickle=True) as archive:
        tensor = np.stack(
            [archive["exponent"], archive["offset"], archive["r_squared"]], axis=2
        ).astype(float)
        return (
            tensor,
            archive["trial_metadata"].item(),
            archive["ch_names"].astype(str).tolist(),
        )


def _require_recording_alignment(
    subject: str,
    run: str,
    reference: dict[str, Any],
    corrected: dict[str, Any],
    fooof: dict[str, Any],
    parcels: list[str],
    corrected_parcels: list[str],
    fooof_parcels: list[str],
) -> None:
    """Require exact window keys and parcel order across feature families."""
    reference_keys = _metadata_keys(reference, subject, run)
    require_exact_alignment(reference_keys, _metadata_keys(corrected, subject, run))
    require_exact_alignment(reference_keys, _metadata_keys(fooof, subject, run))
    if not (parcels == corrected_parcels == fooof_parcels):
        raise ValueError("feature families have incompatible parcel order")


def _metadata_keys(
    metadata: dict[str, Any], subject: str, run: str
) -> np.ndarray:
    """Build stable alignment keys from embedded window metadata."""
    indices = np.asarray(list(metadata["included_epoch_indices"]), dtype=int)
    return build_alignment_keys(
        [subject] * len(indices), [run] * len(indices), metadata["onset"], indices
    )


def _build_labels(
    events: pd.DataFrame,
    metadata: dict[str, Any],
    subject: str,
    run: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Rebuild strict reflected-boundary labels from authoritative events."""
    trials = events[events["trial_type"].isin(("Freq", "Rare"))].sort_values(
        "trial_idx"
    )
    indices = np.asarray(list(metadata["included_epoch_indices"]), dtype=int)
    bad = np.asarray(list(metadata["included_bad_ar2"]), dtype=bool)
    labels = build_corrected_window_labels(
        trials["VTC_filtered"].to_numpy(float),
        indices,
        bad,
        trials["task"].astype(str).to_numpy(),
    )
    return labels, _metadata_keys(metadata, subject, run)


def trials_vtc(events: pd.DataFrame) -> np.ndarray:
    """Return ordered authoritative filtered VTC for task trials only."""
    return (
        events[events["trial_type"].isin(("Freq", "Rare"))]
        .sort_values("trial_idx")["VTC_filtered"]
        .to_numpy(float)
    )


def _stable_inventory(path: Path) -> dict[str, Any]:
    """Record a stable path/size/mtime inventory without hashing large PSDs."""
    status = path.stat()
    return {
        "path": str(path),
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
    }


def _band_reduce(psd: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Average corrected PSD within the seven canonical non-Delta bands."""
    bands = []
    for band in PAPER_BANDS:
        mask = (frequencies >= band.low_hz) & (frequencies < band.high_hz)
        if not mask.any():
            raise ValueError(f"frequency grid does not cover {band.display_name}")
        bands.append(np.nanmean(psd[..., mask], axis=-1))
    return np.stack(bands, axis=2)


def _state_spectrum(
    psd: np.ndarray, states: np.ndarray, target: int
) -> np.ndarray:
    """Return run-level parcel spectra averaged across valid windows."""
    selected = np.asarray(states) == target
    if not selected.any():
        return np.full(psd.shape[1:], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(psd[selected], axis=0)


def _optional_state_spectrum(
    psd: np.ndarray, states: np.ndarray, target: int, include: bool
) -> np.ndarray:
    """Retain large parcel spectra only for Panel 1 aggregation."""
    return _state_spectrum(psd, states, target) if include else np.empty((0, 0))


def _state_spatial_features(
    values: np.ndarray, states: np.ndarray, target: int
) -> np.ndarray:
    """Return run-level parcel features averaged across valid windows."""
    selected = np.asarray(states) == target
    if not selected.any():
        return np.full(values.shape[1:], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values[selected], axis=0)


def _optional_state_features(
    values: np.ndarray, states: np.ndarray, target: int, include: bool
) -> np.ndarray:
    """Retain state-level FOOOF fits only for Panel 1 spectral rendering."""
    return (
        _state_spatial_features(values, states, target)
        if include
        else np.empty((0, 0))
    )


def _state_names(values: np.ndarray) -> np.ndarray:
    """Convert numeric state constants into public labels."""
    names = np.full(len(values), "MID", dtype="<U3")
    names[values == LABEL_IN] = "IN"
    names[values == LABEL_OUT] = "OUT"
    return names


def _outcome_names(values: np.ndarray) -> np.ndarray:
    """Convert numeric matched-anchor outcomes into public labels."""
    names = np.full(len(values), "other", dtype="<U20")
    names[values == OUTCOME_CORRECT_OMISSION] = "correct_omission"
    names[values == OUTCOME_COMMISSION_ERROR] = "commission_error"
    return names
