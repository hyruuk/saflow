"""Helpers for loading bad-epoch annotations from cleaned raw recordings.

The autoreject second pass writes ``BAD_AR2`` annotations onto the cleaned
continuous recording at ``derivatives/preprocessed/sub-XX/meg/
sub-XX_task-<task>_run-YY_proc-clean_meg.fif``. Comparative analyses
(stats, classification, viz) need those annotations to drop trials that
overlap rejected periods.

To keep this lookup cheap and decoupled from the heavy raw FIFs (which
are commonly only available on the HPC), the preprocessing pipeline
also writes a tiny ``*_annotations.json`` sidecar next to the FIF.
``load_clean_raw_annotations`` reads the sidecar first and falls back
to the FIF when the sidecar is missing — so analyses can run on a
machine that only has the JSON sidecars synced over.

JSON sidecar schema:

    {
      "subject": "04",
      "run": "02",
      "sfreq": 1200.0,
      "n_times": 588000,
      "duration_sec": 489.99,
      "orig_time": null,
      "annotations": [
        {"onset": 12.34, "duration": 0.85, "description": "BAD_AR2"},
        ...
      ]
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


def get_clean_raw_path(subject: str, run: str, config: Dict[str, Any]) -> Path:
    """Return the path of the ICA-cleaned continuous FIF for sub/run."""
    data_root = Path(config["paths"]["data_root"])
    derivatives_root = data_root / config["paths"]["derivatives"]
    task_name = config["bids"]["task_name"]
    return (
        derivatives_root
        / "preprocessed"
        / f"sub-{subject}"
        / "meg"
        / f"sub-{subject}_task-{task_name}_run-{run}_proc-clean_meg.fif"
    )


def get_annotations_sidecar_path(
    subject: str, run: str, config: Dict[str, Any]
) -> Path:
    """Return the path of the cleaned-raw annotations JSON sidecar."""
    fif_path = get_clean_raw_path(subject, run, config)
    return fif_path.with_name(
        fif_path.name.replace("_meg.fif", "_annotations.json")
    )


def annotations_to_dict(
    annotations: mne.Annotations,
    *,
    sfreq: Optional[float] = None,
    n_times: Optional[int] = None,
    subject: Optional[str] = None,
    run: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialise an ``mne.Annotations`` object to a JSON-friendly dict."""
    orig_time = annotations.orig_time
    if hasattr(orig_time, "isoformat"):
        orig_time_repr = orig_time.isoformat()
    elif orig_time is None:
        orig_time_repr = None
    else:
        orig_time_repr = str(orig_time)

    items = [
        {
            "onset": float(o),
            "duration": float(d),
            "description": str(desc),
        }
        for o, d, desc in zip(
            np.asarray(annotations.onset),
            np.asarray(annotations.duration),
            np.asarray(annotations.description),
        )
    ]

    out: Dict[str, Any] = {"annotations": items, "orig_time": orig_time_repr}
    if subject is not None:
        out["subject"] = subject
    if run is not None:
        out["run"] = run
    if sfreq is not None:
        out["sfreq"] = float(sfreq)
    if n_times is not None:
        out["n_times"] = int(n_times)
        if sfreq is not None:
            out["duration_sec"] = float(n_times) / float(sfreq)
    return out


def annotations_from_dict(payload: Dict[str, Any]) -> mne.Annotations:
    """Reconstruct an ``mne.Annotations`` from the sidecar dict."""
    items = payload.get("annotations", [])
    onsets = np.asarray([a["onset"] for a in items], dtype=float)
    durations = np.asarray([a["duration"] for a in items], dtype=float)
    descs = [a["description"] for a in items]
    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descs,
        orig_time=None,
    )


def write_annotations_sidecar(
    annotations: mne.Annotations,
    *,
    subject: str,
    run: str,
    config: Dict[str, Any],
    sfreq: Optional[float] = None,
    n_times: Optional[int] = None,
) -> Path:
    """Write ``annotations`` to the JSON sidecar for sub/run."""
    path = get_annotations_sidecar_path(subject, run, config)
    payload = annotations_to_dict(
        annotations, sfreq=sfreq, n_times=n_times, subject=subject, run=run
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_clean_raw_annotations(
    subject: str,
    run: str,
    config: Dict[str, Any],
    *,
    prefer_sidecar: bool = True,
) -> Optional[mne.Annotations]:
    """Load annotations from the cleaned continuous recording.

    Tries the JSON sidecar first (cheap, sync-friendly). Falls back to
    reading the FIF header when the sidecar does not exist. Returns
    ``None`` only when neither source is available.
    """
    if prefer_sidecar:
        sidecar = get_annotations_sidecar_path(subject, run, config)
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    payload = json.load(f)
                return annotations_from_dict(payload)
            except Exception as exc:
                logger.warning(
                    f"Failed to read annotations sidecar {sidecar}: {exc}"
                )

    fif = get_clean_raw_path(subject, run, config)
    if fif.exists():
        raw = mne.io.read_raw_fif(fif, preload=False, verbose="ERROR")
        return raw.annotations

    logger.warning(
        f"No annotation source found for sub-{subject} run-{run} "
        f"(neither JSON sidecar nor cleaned-raw FIF)."
    )
    return None
