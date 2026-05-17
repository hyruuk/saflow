"""Backfill ``BAD_AR1`` so the autoreject first pass propagates through the pipeline.

The autoreject second pass already lives in the cleaned-raw annotations
(``BAD_AR2``), recorded in the per-run annotation JSON sidecar; feature
extraction reads that sidecar so every space (sensor / source / atlas) and
feature family picks up ``bad_ar2`` consistently. The first pass had no such
home — its verdict only existed inside ``*_desc-ARlog1*.pkl``.

This backfill puts AR1 where AR2 lives, in two stages:

  continuous : derive ``BAD_AR1`` annotation intervals from the ARlog1 pickle
               and write them into the annotation JSON sidecar (and, with
               --update-fif, the cleaned-raw FIF). This is the *single source
               of truth* — once it is done, ANY future feature extraction
               (any space, any family) gets ``bad_ar1`` for free via the
               already-updated segmenter, exactly like ``bad_ar2``.

  features   : patch feature files that were extracted *before* the sidecar
               carried ``BAD_AR1``, so existing ``.npz`` outputs gain a
               ``bad_ar1`` column without recomputing feature values. Reads
               ``BAD_AR1`` straight from the (now updated) sidecar, so it goes
               through the same path the pipeline uses.

Run ``continuous`` before ``features`` (the default ``--stage both`` does so).
After a fresh preprocessing run the ``continuous`` stage is unnecessary —
preprocessing writes ``BAD_AR1`` itself — but ``features`` still patches any
feature files extracted in between.

Alignment: the ARlog1 ``bad_epochs`` array is in stimulus-epoch order; the
sidecar's Freq/Rare *event* annotations give those epochs' onsets. Feature
windows are located by exact onset match of their anchor trial. Anything that
fails verification is skipped, never written.

Usage:
    python -m code.utils.backfill_ar1 --dry-run
    python -m code.utils.backfill_ar1 --stage continuous
    python -m code.utils.backfill_ar1 --stage features --space sensor
    python -m code.utils.backfill_ar1                      # both stages
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.features.utils import compute_bad_trial_mask
from code.utils.annotations import (
    get_annotations_sidecar_path,
    get_clean_raw_path,
    load_clean_raw_annotations,
)
from code.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_FOLDER_PATTERNS = (
    "welch_psds_*",
    "welch_psds_corrected_*",
    "fooof_*",
    "complexity_*",
)

FILENAME_RE = re.compile(
    r"^sub-(?P<subject>[A-Za-z0-9]+)_"
    r"ses-[A-Za-z0-9]+_"
    r"task-(?P<task>[A-Za-z0-9]+)_"
    r"run-(?P<run>[A-Za-z0-9]+)_"
    r".*\.npz$"
)

STIM_DESCRIPTIONS = {"Freq", "Rare"}
AR1_DESC = "BAD_AR1"
ONSET_TOL = 0.005  # s; far below the ~1.3 s gradCPT ISI


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def find_ar1_log(subject: str, run: str, config: dict) -> Optional[Path]:
    """Locate the first-pass autoreject log pickle for a sub/run."""
    data_root = Path(config["paths"]["data_root"])
    meg_dir = (
        data_root / config["paths"]["derivatives"] / "preprocessed"
        / f"sub-{subject}" / "meg"
    )
    if not meg_dir.is_dir():
        return None
    hits = sorted(meg_dir.glob(f"sub-{subject}_*run-{run}_*ARlog1*.pkl"))
    return hits[0] if hits else None


def load_ar1_flags(pkl_path: Path) -> Optional[np.ndarray]:
    """Return the AR1 ``bad_epochs`` boolean array from a reject-log pickle."""
    try:
        with open(pkl_path, "rb") as f:
            reject_log = pickle.load(f)
    except Exception as exc:
        logger.warning(f"Could not unpickle {pkl_path.name}: {exc}")
        return None
    flags = getattr(reject_log, "bad_epochs", None)
    if flags is None:
        logger.warning(f"{pkl_path.name} has no 'bad_epochs' attribute")
        return None
    return np.asarray(flags, dtype=bool)


def stim_onsets_from_annotations(annotations) -> Optional[np.ndarray]:
    """Sorted onsets (s) of the Freq/Rare stimulus event annotations."""
    if annotations is None or len(annotations) == 0:
        return None
    onsets = [
        float(o)
        for o, d in zip(annotations.onset, annotations.description)
        if d in STIM_DESCRIPTIONS
    ]
    if not onsets:
        return None
    return np.sort(np.asarray(onsets, dtype=float))


# --------------------------------------------------------------------------
# Stage 1 — continuous data (annotation sidecar)
# --------------------------------------------------------------------------

def backfill_sidecar(
    subject: str,
    run: str,
    config: dict,
    tmin: float,
    tmax: float,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[str, int]:
    """Write BAD_AR1 intervals into the annotation sidecar. Returns (status, n_bad)."""
    sidecar = get_annotations_sidecar_path(subject, run, config)
    if not sidecar.exists():
        return "skip_no_sidecar", 0

    payload = json.loads(sidecar.read_text())
    items = payload.get("annotations", [])
    has_ar1 = any(a.get("description") == AR1_DESC for a in items)
    if has_ar1 and not overwrite:
        return "already_done", sum(a.get("description") == AR1_DESC for a in items)

    stim_onsets = np.sort(
        np.asarray(
            [float(a["onset"]) for a in items if a.get("description") in STIM_DESCRIPTIONS],
            dtype=float,
        )
    )
    if stim_onsets.size == 0:
        return "skip_no_stim_events", 0

    pkl = find_ar1_log(subject, run, config)
    if pkl is None:
        return "skip_no_ar1log", 0
    ar1 = load_ar1_flags(pkl)
    if ar1 is None:
        return "skip_bad_ar1log", 0
    if len(ar1) > len(stim_onsets):
        logger.warning(
            f"sub-{subject} run-{run}: AR1 log ({len(ar1)}) longer than stim "
            f"events ({len(stim_onsets)}); skipping."
        )
        return "skip_len_mismatch", 0

    # AR1 epochs are stimulus epochs in time order; boundary drops (if any)
    # are at the tail, so ar1[i] maps to stim_onsets[i].
    bad_idx = np.where(ar1)[0]
    epoch_dur = float(tmax - tmin)
    new_items = [
        {
            "onset": float(stim_onsets[i]) + float(tmin),
            "duration": epoch_dur,
            "description": AR1_DESC,
        }
        for i in bad_idx
    ]
    if dry_run:
        return "dry_run", len(new_items)

    kept = [a for a in items if a.get("description") != AR1_DESC]
    payload["annotations"] = kept + new_items
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(sidecar)
    return "updated", len(new_items)


def update_fif_annotations(subject: str, run: str, config: dict, dry_run: bool) -> str:
    """Mirror the sidecar's BAD_AR1 annotations into the cleaned-raw FIF."""
    import mne

    fif = get_clean_raw_path(subject, run, config)
    if not fif.exists():
        return "skip_no_fif"
    annotations = load_clean_raw_annotations(subject=subject, run=run, config=config)
    if annotations is None:
        return "skip_no_annotations"
    if dry_run:
        return "dry_run"
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    raw.set_annotations(annotations)
    raw.save(fif, overwrite=True, verbose="ERROR")
    return "updated"


# --------------------------------------------------------------------------
# Stage 2 — existing feature files
# --------------------------------------------------------------------------

def iter_feature_files(features_root: Path, space_filter: Optional[str]):
    for pattern in FEATURE_FOLDER_PATTERNS:
        for folder in sorted(features_root.glob(pattern)):
            if not folder.is_dir():
                continue
            if space_filter is not None and not folder.name.endswith(f"_{space_filter}"):
                continue
            for npz_file in sorted(folder.rglob("sub-*.npz")):
                yield npz_file


def parse_filename(npz_file: Path) -> Optional[Tuple[str, str]]:
    m = FILENAME_RE.match(npz_file.name)
    if m is None:
        return None
    return m.group("subject"), m.group("run")


def per_trial_ar1_from_sidecar(
    subject: str, run: str, config: dict, tmin: float, tmax: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (stim_onsets, per_trial_bad_ar1) computed from the sidecar.

    Goes through the exact path the segmenter uses: BAD_AR1 annotations +
    compute_bad_trial_mask. Returns None when the sidecar has no BAD_AR1.
    """
    annotations = load_clean_raw_annotations(subject=subject, run=run, config=config)
    stim_onsets = stim_onsets_from_annotations(annotations)
    if stim_onsets is None:
        return None
    if not any(d == AR1_DESC for d in annotations.description):
        return None  # stage 1 has not run for this run
    per_trial = compute_bad_trial_mask(
        onsets=stim_onsets,
        tmin=tmin,
        tmax=tmax,
        annotations=annotations,
        bad_prefix=AR1_DESC,
    )
    return stim_onsets, per_trial


def _anchor_indices(
    anchor_onsets: np.ndarray, stim_onsets: np.ndarray
) -> Optional[np.ndarray]:
    """Map window anchor onsets to indices in the stim-onset sequence."""
    idx = np.searchsorted(stim_onsets, anchor_onsets)
    out = np.empty(len(anchor_onsets), dtype=int)
    for k, (a, i) in enumerate(zip(anchor_onsets, idx)):
        cands = [j for j in (i - 1, i, i + 1) if 0 <= j < len(stim_onsets)]
        if not cands:
            return None
        best = min(cands, key=lambda j: abs(stim_onsets[j] - a))
        if abs(stim_onsets[best] - a) > ONSET_TOL:
            return None
        out[k] = best
    if len(out) > 1 and not np.all(np.diff(out) == 1):
        return None
    return out


def backfill_feature_file(
    npz_file: Path,
    stim_onsets: np.ndarray,
    per_trial_ar1: np.ndarray,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[str, int, int]:
    """Add bad_ar1 to one feature file. Returns (status, n_windows, n_bad_ar1)."""
    with np.load(npz_file, allow_pickle=True) as npz:
        if "trial_metadata" not in npz.files:
            return "skip_no_metadata", 0, 0
        meta = npz["trial_metadata"].item()
        arrays = None if dry_run else {k: npz[k] for k in npz.files if k != "trial_metadata"}

    if "onset" not in meta:
        return "skip_no_onset", 0, 0
    if "bad_ar1" in meta and not overwrite:
        return "already_tagged", len(meta["onset"]), int(np.sum(meta["bad_ar1"]))

    anchor_onsets = np.asarray(meta["onset"], dtype=float)
    n_win = len(anchor_onsets)
    # Window size = length of the per-window included arrays. Any of these
    # windowed fields works; single-trial files have none of them (W=1).
    window = 1
    for key in ("included_bad_ar2", "included_task", "included_VTC"):
        if key in meta and len(meta[key]):
            window = int(len(np.asarray(meta[key][0])))
            break

    anchor_idx = _anchor_indices(anchor_onsets, stim_onsets)
    if anchor_idx is None:
        return "skip_unaligned", n_win, 0
    if anchor_idx.min() - window + 1 < 0 or anchor_idx.max() >= len(per_trial_ar1):
        return "skip_out_of_range", n_win, 0

    bad_ar1 = np.zeros(n_win, dtype=bool)
    included: List[List[bool]] = []
    for k, a in enumerate(anchor_idx):
        win = per_trial_ar1[a - window + 1 : a + 1].astype(bool)
        included.append(win.tolist())
        bad_ar1[k] = bool(win.any())

    n_bad = int(bad_ar1.sum())
    if dry_run:
        return "dry_run", n_win, n_bad

    meta["bad_ar1"] = bad_ar1.tolist()
    if window > 1:
        meta["included_bad_ar1"] = included

    assert arrays is not None
    save_kwargs = dict(arrays)
    save_kwargs["trial_metadata"] = meta
    tmp = npz_file.with_name(npz_file.stem + ".tmp.npz")
    np.savez_compressed(tmp, **save_kwargs)
    tmp.replace(npz_file)
    return "updated", n_win, n_bad


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _bump(counts: Dict[str, int], status: str) -> None:
    counts[status] = counts.get(status, 0) + 1


def run_continuous_stage(config: dict, tmin: float, tmax: float, args) -> None:
    subjects = config["bids"]["subjects"]
    runs = config["bids"]["task_runs"]
    counts: Dict[str, int] = {}
    n_bad_total = 0
    logger.info("STAGE continuous — writing BAD_AR1 into annotation sidecars")
    for subject in subjects:
        for run in runs:
            status, n_bad = backfill_sidecar(
                subject, run, config, tmin, tmax, args.overwrite, args.dry_run
            )
            _bump(counts, status)
            if status in ("updated", "dry_run", "already_done"):
                n_bad_total += n_bad
            if status == "updated":
                logger.info(f"updated sidecar  sub-{subject} run-{run}  ({n_bad} BAD_AR1)")
            elif status == "dry_run":
                logger.info(f"would update     sub-{subject} run-{run}  ({n_bad} BAD_AR1)")
            elif status not in ("already_done",):
                logger.warning(f"{status}  sub-{subject} run-{run}")

            if args.update_fif and status in ("updated", "already_done"):
                fif_status = update_fif_annotations(subject, run, config, args.dry_run)
                _bump(counts, f"fif_{fif_status}")

    logger.info("Continuous stage summary:")
    for status, c in sorted(counts.items()):
        logger.info(f"  {status}: {c}")
    logger.info(f"BAD_AR1 annotations written: {n_bad_total}")


def run_features_stage(config: dict, tmin: float, tmax: float, args) -> None:
    features_root = Path(config["paths"]["data_root"]) / config["paths"]["features"]
    logger.info(f"STAGE features — patching feature files under {features_root}")

    cache: Dict[Tuple[str, str], Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    counts: Dict[str, int] = {}
    n_win_total = 0
    n_bad_total = 0

    for npz_file in iter_feature_files(features_root, args.space):
        parsed = parse_filename(npz_file)
        if parsed is None:
            _bump(counts, "skip_unparsed")
            continue
        subject, run = parsed
        key = (subject, run)
        if key not in cache:
            cache[key] = per_trial_ar1_from_sidecar(subject, run, config, tmin, tmax)
        resolved = cache[key]
        if resolved is None:
            _bump(counts, "skip_no_ar1_in_sidecar")
            continue
        stim_onsets, per_trial_ar1 = resolved

        status, n_win, n_bad = backfill_feature_file(
            npz_file, stim_onsets, per_trial_ar1, args.overwrite, args.dry_run
        )
        _bump(counts, status)
        if status in ("updated", "dry_run", "already_tagged"):
            n_win_total += n_win
            n_bad_total += n_bad
        rel = npz_file.relative_to(features_root)
        if status == "updated":
            logger.info(f"updated  {rel}  ({n_bad}/{n_win} bad_ar1)")
        elif status == "dry_run":
            logger.info(f"would update  {rel}  ({n_bad}/{n_win} bad_ar1)")
        elif status not in ("already_tagged",):
            logger.warning(f"{status}  {rel}")

    logger.info("Features stage summary:")
    for status, c in sorted(counts.items()):
        logger.info(f"  {status}: {c}")
    logger.info(f"Windows seen: {n_win_total} (bad_ar1: {n_bad_total})")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill BAD_AR1 into annotation sidecars and feature files."
    )
    parser.add_argument(
        "--stage", choices=("continuous", "features", "both"), default="both",
        help="continuous = sidecars (source of truth); features = patch npz files.",
    )
    parser.add_argument("--space", default=None, help="features stage: restrict to one space.")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if present.")
    parser.add_argument(
        "--update-fif", action="store_true",
        help="continuous stage: also rewrite the cleaned-raw FIF annotations "
             "(heavy; not needed — feature extraction reads the sidecar).",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    tmin = float(config["analysis"]["epochs"]["tmin"])
    tmax = float(config["analysis"]["epochs"]["tmax"])
    if args.dry_run:
        logger.info("DRY RUN — no files will be modified.")

    if args.stage in ("continuous", "both"):
        run_continuous_stage(config, tmin, tmax, args)
    if args.stage in ("features", "both"):
        run_features_stage(config, tmin, tmax, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
