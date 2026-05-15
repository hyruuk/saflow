"""Backfill ``bad_ar1`` into existing feature ``trial_metadata``.

Feature ``.npz`` files already carry ``bad_ar2`` (autoreject second pass,
post-ICA). They do *not* carry ``bad_ar1`` (first pass, pre-ICA) because the
preprocessing pipeline only annotated the cleaned raw with ``BAD_AR2`` until
the BAD_AR1 change. This script adds ``bad_ar1`` so the configurable
``analysis.bad_trial_rule`` (ar1 / ar2 / union) can be used on data that was
extracted before that change — without recomputing any feature values.

How it works (no raw FIF / no feature recompute):
  1. The first-pass autoreject log (``*_desc-ARlog1*.pkl``) stores a per-epoch
     ``bad_epochs`` boolean over the stimulus epochs, in time order.
  2. The annotations JSON sidecar carries the Freq/Rare *event* annotations,
     i.e. the ordered onsets of every stimulus epoch.
  3. ``trial_metadata['onset']`` is each window's anchor (last trial) onset.
     Matching it against the ordered stimulus onsets locates the window in the
     epoch sequence; ``bad_ar1`` for the window is then ANY of its W trials.

Safety: alignment is verified per file (exact onset match, stride-1 anchors,
in-range indices). A file that fails verification is skipped, never rewritten.

NOTE — interpolation counts (``n_interp``) are deliberately NOT backfilled.
The pre-AR2 outlier mask is not persisted and the pre-interpolation epoch data
is overwritten by channel interpolation, so the AR2 reject log's interpolation
labels cannot be re-aligned from saved artifacts. ``n_interp`` must instead be
emitted by a future preprocessing run.

Usage:
    python -m code.utils.backfill_ar1 --dry-run
    python -m code.utils.backfill_ar1 --space sensor
    python -m code.utils.backfill_ar1
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.utils.annotations import load_clean_raw_annotations
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
    r"space-(?P<space>[^_]+)_"
    r".*\.npz$"
)

STIM_DESCRIPTIONS = {"Freq", "Rare"}
ONSET_TOL = 0.005  # seconds; ~6 samples at 1200 Hz, far below the ~1.3 s ISI


def iter_feature_files(features_root: Path, space_filter: Optional[str]):
    for pattern in FEATURE_FOLDER_PATTERNS:
        for folder in sorted(features_root.glob(pattern)):
            if not folder.is_dir():
                continue
            if space_filter is not None and not folder.name.endswith(f"_{space_filter}"):
                continue
            for npz_file in sorted(folder.rglob("sub-*.npz")):
                yield npz_file


def parse_filename(npz_file: Path) -> Optional[Tuple[str, str, str]]:
    m = FILENAME_RE.match(npz_file.name)
    if m is None:
        return None
    return m.group("subject"), m.group("run"), m.group("space")


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


def _anchor_indices(
    anchor_onsets: np.ndarray, stim_onsets: np.ndarray
) -> Optional[np.ndarray]:
    """Map each window anchor onset to its index in the stim-onset sequence.

    Returns None if any anchor fails to match a stim onset within tolerance,
    or the matched indices are not strictly consecutive (stride 1).
    """
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


def backfill_one(
    npz_file: Path,
    ar1_flags: np.ndarray,
    stim_onsets: np.ndarray,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[str, int, int]:
    """Backfill a single feature file. Returns (status, n_windows, n_bad_ar1)."""
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

    # Window size: length of the per-window included arrays (1 if single-trial).
    if "included_bad_ar2" in meta and len(meta["included_bad_ar2"]):
        window = int(len(np.asarray(meta["included_bad_ar2"][0])))
    else:
        window = 1

    anchor_idx = _anchor_indices(anchor_onsets, stim_onsets)
    if anchor_idx is None:
        return "skip_unaligned", n_win, 0

    # Each window covers stim epochs [anchor - window + 1, anchor].
    if anchor_idx.min() - window + 1 < 0 or anchor_idx.max() >= len(ar1_flags):
        return "skip_out_of_range", n_win, 0

    bad_ar1 = np.zeros(n_win, dtype=bool)
    included_bad_ar1: List[np.ndarray] = []
    for k, a in enumerate(anchor_idx):
        win_slice = ar1_flags[a - window + 1 : a + 1]
        included_bad_ar1.append(win_slice.astype(bool))
        bad_ar1[k] = bool(win_slice.any())

    n_bad = int(bad_ar1.sum())
    if dry_run:
        return "dry_run", n_win, n_bad

    meta["bad_ar1"] = bad_ar1.tolist()
    if window > 1:
        meta["included_bad_ar1"] = [arr.tolist() for arr in included_bad_ar1]

    assert arrays is not None
    save_kwargs = dict(arrays)
    save_kwargs["trial_metadata"] = meta
    tmp_path = npz_file.with_name(npz_file.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **save_kwargs)
    tmp_path.replace(npz_file)
    return "updated", n_win, n_bad


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add bad_ar1 column to existing feature trial_metadata."
    )
    parser.add_argument("--space", default=None, help="Restrict to one analysis space.")
    parser.add_argument("--dry-run", action="store_true", help="Report only; rewrite nothing.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute bad_ar1 if present.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    features_root = Path(config["paths"]["data_root"]) / config["paths"]["features"]

    logger.info(f"Backfilling bad_ar1 under {features_root}")
    if args.dry_run:
        logger.info("DRY RUN — no files will be modified.")

    # Per-run caches so the 8 feature files of one run reuse one pkl + sidecar read.
    ar1_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    onset_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    counts: Dict[str, int] = {}
    n_win_total = 0
    n_bad_total = 0

    for npz_file in iter_feature_files(features_root, args.space):
        parsed = parse_filename(npz_file)
        if parsed is None:
            counts["skip_unparsed"] = counts.get("skip_unparsed", 0) + 1
            continue
        subject, run, _space = parsed
        key = (subject, run)

        if key not in ar1_cache:
            pkl = find_ar1_log(subject, run, config)
            ar1_cache[key] = load_ar1_flags(pkl) if pkl else None
            ann = load_clean_raw_annotations(subject=subject, run=run, config=config)
            onset_cache[key] = stim_onsets_from_annotations(ann)

        ar1_flags = ar1_cache[key]
        stim_onsets = onset_cache[key]
        if ar1_flags is None:
            counts["skip_no_ar1log"] = counts.get("skip_no_ar1log", 0) + 1
            continue
        if stim_onsets is None:
            counts["skip_no_onsets"] = counts.get("skip_no_onsets", 0) + 1
            continue
        if len(ar1_flags) > len(stim_onsets):
            logger.warning(
                f"sub-{subject} run-{run}: AR1 log ({len(ar1_flags)}) longer than "
                f"stim onsets ({len(stim_onsets)}); skipping run."
            )
            counts["skip_len_mismatch"] = counts.get("skip_len_mismatch", 0) + 1
            continue

        status, n_win, n_bad = backfill_one(
            npz_file, ar1_flags, stim_onsets, args.overwrite, args.dry_run
        )
        counts[status] = counts.get(status, 0) + 1
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

    logger.info("Summary by status:")
    for status, c in sorted(counts.items()):
        logger.info(f"  {status}: {c}")
    logger.info(f"Windows seen: {n_win_total} (bad_ar1: {n_bad_total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
