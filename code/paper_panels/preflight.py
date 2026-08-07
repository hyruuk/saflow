"""Blinded input validation and cell-count reporting for paper panels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from code.paper_panels.alignment import build_alignment_keys, validate_schaefer_400
from code.paper_panels.labels import build_corrected_window_labels
from code.utils.behavioral import VTC_FILTER_METHOD, VTC_FILTER_VERSION

CELL_ORDER = (
    "IN_correct_omission",
    "IN_commission_error",
    "OUT_correct_omission",
    "OUT_commission_error",
)


def inspect_inputs(
    config: dict[str, Any], subjects: Sequence[str], runs: Sequence[str]
) -> dict[str, Any]:
    """Inspect actual behavioral and Schaefer inputs without testing effects."""
    records = [
        inspect_recording(config, str(subject), str(run))
        for subject in subjects
        for run in runs
    ]
    status = "passed" if records and all(item["status"] == "passed" for item in records) else "failed"
    return {
        "status": status,
        "blinded": True,
        "subjects": list(subjects),
        "runs": list(runs),
        "recordings": records,
        "summary": summarize_recordings(records),
    }


def inspect_recording(config: dict[str, Any], subject: str, run: str) -> dict[str, Any]:
    """Validate one subject/run's events, windows, bad flags, and parcel order."""
    events_path = _events_path(config, subject, run)
    feature_paths = _feature_paths(config, subject, run)
    feature_path = feature_paths["welch"]
    report: dict[str, Any] = {
        "subject": subject,
        "run": run,
        "events": str(events_path),
        "features": str(feature_path) if feature_path else None,
        "feature_files": {
            name: str(path) if path else None for name, path in feature_paths.items()
        },
        "errors": [],
    }
    if not events_path.exists():
        report["errors"].append("missing_events")
    for family, path in feature_paths.items():
        if path is None:
            report["errors"].append(f"missing_schaefer_400_{family}")
    if report["errors"]:
        report["status"] = "failed"
        return report
    try:
        events = pd.read_csv(events_path, sep="\t")
        _validate_event_provenance(events)
        metadata, parcel_names = _load_feature_metadata(feature_path)
        validate_schaefer_400(parcel_names)
        counts, keys = _label_counts(events, metadata, subject, run)
        _validate_feature_alignment(feature_paths, keys, subject, run)
        report.update(counts)
        report["alignment_key_count"] = len(keys)
    except (KeyError, TypeError, ValueError) as error:
        report["errors"].append(str(error))
    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def summarize_recordings(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize recording and four-cell eligibility counts."""
    cell_totals: Counter[str] = Counter()
    for record in records:
        cell_totals.update(record.get("cell_counts", {}))
    return {
        "expected": len(records),
        "passed": sum(record["status"] == "passed" for record in records),
        "failed": sum(record["status"] != "passed" for record in records),
        "cell_totals": dict(sorted(cell_totals.items())),
        "modulation_eligible_recordings": sum(
            record.get("modulation_eligible", False) for record in records
        ),
        "coupling_eligible_recordings": sum(
            record.get("coupling_eligible", False) for record in records
        ),
    }


def write_reports(report: dict[str, Any], output_directory: Path) -> None:
    """Write JSON and tabular blinded QC reports."""
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "preflight_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(report["recordings"]).to_csv(
        output_directory / "recording_qc.tsv", sep="\t", index=False
    )


def _events_path(config: dict[str, Any], subject: str, run: str) -> Path:
    return (
        Path(config["paths"]["bids"])
        / f"sub-{subject}"
        / "meg"
        / f"sub-{subject}_task-gradCPT_run-{run}_events.tsv"
    )


def _feature_paths(
    config: dict[str, Any], subject: str, run: str
) -> dict[str, Path | None]:
    root = Path(config["paths"]["features"])
    patterns = {
        "welch": (
            "welch_psds_schaefer_400",
            f"sub-{subject}_ses-*_task-gradCPT_run-{run}_space-schaefer_400_desc-welchw8_psds.npz",
        ),
        "fooof": (
            "fooof_schaefer_400",
            f"sub-{subject}_ses-*_task-gradCPT_run-{run}_space-schaefer_400_desc-fooofw8.npz",
        ),
        "corrected_psd": (
            "welch_psds_corrected_schaefer_400",
            f"sub-{subject}_ses-*_task-gradCPT_run-{run}_space-schaefer_400_desc-welch-corrw8_psds.npz",
        ),
    }
    found: dict[str, Path | None] = {}
    for name, (directory, pattern) in patterns.items():
        candidates = sorted((root / directory / f"sub-{subject}").glob(pattern))
        found[name] = candidates[0] if len(candidates) == 1 else None
    return found


def _validate_event_provenance(events: pd.DataFrame) -> None:
    required = {
        "trial_idx", "VTC_raw", "VTC_filtered", "task",
        "VTC_filter_method", "VTC_filter_version",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events missing corrected columns: {missing}")
    methods = set(events["VTC_filter_method"].dropna().astype(str))
    versions = set(events["VTC_filter_version"].dropna().astype(str))
    if methods != {VTC_FILTER_METHOD} or versions != {VTC_FILTER_VERSION}:
        raise ValueError("events have stale or incompatible VTC filter provenance")


def _load_feature_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    with np.load(path, allow_pickle=True) as bundle:
        metadata = bundle["trial_metadata"].item()
        parcel_names = bundle["ch_names"].astype(str).tolist()
    return metadata, parcel_names


def _validate_feature_alignment(
    paths: dict[str, Path | None],
    reference_keys: np.ndarray,
    subject: str,
    run: str,
) -> None:
    """Require identical keys and spatial order across every paper family."""
    from code.paper_panels.alignment import require_exact_alignment

    reference_names: list[str] | None = None
    for path in paths.values():
        if path is None:
            raise ValueError("paper feature family is missing")
        metadata, names = _load_feature_metadata(path)
        indices = np.asarray(list(metadata["included_epoch_indices"]), dtype=int)
        keys = build_alignment_keys(
            [subject] * len(indices), [run] * len(indices), metadata["onset"], indices
        )
        require_exact_alignment(reference_keys, keys)
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError("Schaefer-400 parcel order differs across feature families")


def _label_counts(
    events: pd.DataFrame,
    metadata: dict[str, Any],
    subject: str,
    run: str,
) -> tuple[dict[str, Any], np.ndarray]:
    required = {
        "onset", "anchor_epoch_index", "included_epoch_indices",
        "included_bad_ar2",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"feature metadata missing corrected alignment fields: {missing}")
    trials = events[events["trial_type"].isin(["Freq", "Rare"])].copy()
    trials = trials.sort_values("trial_idx")
    indices = np.asarray(list(metadata["included_epoch_indices"]), dtype=int)
    bad = np.asarray(list(metadata["included_bad_ar2"]), dtype=bool)
    labels = build_corrected_window_labels(
        trials["VTC_filtered"].to_numpy(dtype=float),
        indices,
        bad,
        trials["task"].astype(str).to_numpy(),
    )
    keys = build_alignment_keys(
        [subject] * len(indices),
        [run] * len(indices),
        metadata["onset"],
        indices,
    )
    observed = Counter(labels["cell"][labels["cell"] != ""])
    cell_counts = {cell: int(observed[cell]) for cell in CELL_ORDER}
    return {
        "n_trials": len(trials),
        "n_windows": len(indices),
        "n_bad_windows": int(labels["bad_any"].sum()),
        "n_in_windows": int(np.sum(labels["state"] == -1)),
        "n_out_windows": int(np.sum(labels["state"] == 1)),
        "cell_counts": cell_counts,
        "modulation_eligible": all(cell_counts[cell] >= 5 for cell in CELL_ORDER),
        "coupling_eligible": all(cell_counts[cell] >= 10 for cell in CELL_ORDER),
    }, keys
