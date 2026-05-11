"""One-off backfill: add ``bad_ar2`` to existing feature ``trial_metadata``.

Existing feature ``.npz`` files (welch PSDs, FOOOF, FOOOF-corrected PSDs,
complexity) were written before the segmenter started tagging trials whose
epoch window overlaps the autoreject second-pass ``BAD_AR2`` annotations on
the cleaned continuous recording. This script walks every feature file,
re-loads the matching cleaned-raw annotations, recomputes the bad mask from
the per-trial ``onset`` already saved in ``trial_metadata``, and rewrites
the file in place with a ``bad_ar2`` boolean column.

Feature *values* are not recomputed — only metadata. Trials that were
previously stored stay in the same row order, which is critical because
PSD/FOOOF/complexity arrays are indexed by row.

Usage:
    python -m code.utils.backfill_bad_trials                    # all spaces
    python -m code.utils.backfill_bad_trials --space sensor     # one space
    python -m code.utils.backfill_bad_trials --dry-run          # report only
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from code.features.utils import compute_bad_trial_mask
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


def iter_feature_files(
    features_root: Path,
    space_filter: Optional[str] = None,
) -> Iterable[Path]:
    """Yield every feature npz file under ``features_root``."""
    for pattern in FEATURE_FOLDER_PATTERNS:
        for folder in sorted(features_root.glob(pattern)):
            if not folder.is_dir():
                continue
            if space_filter is not None and not folder.name.endswith(f"_{space_filter}"):
                continue
            for npz_file in sorted(folder.rglob("sub-*.npz")):
                yield npz_file


def parse_filename(npz_file: Path) -> Optional[Tuple[str, str, str]]:
    """Extract (subject, run, space) from a feature filename."""
    m = FILENAME_RE.match(npz_file.name)
    if not m:
        return None
    return m.group("subject"), m.group("run"), m.group("space")


def backfill_one(
    npz_file: Path,
    config: dict,
    tmin: float,
    tmax: float,
    annotations_cache: dict,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Tuple[str, int, int]:
    """Backfill a single npz file. Returns (status, n_trials, n_bad)."""
    parsed = parse_filename(npz_file)
    if parsed is None:
        return "skip_unparsed", 0, 0
    subject, run, _space = parsed

    # On dry-run we only need to read trial_metadata — skip the heavy
    # PSD/complexity arrays, which are large and compressed.
    if dry_run:
        with np.load(npz_file, allow_pickle=True) as npz:
            if "trial_metadata" not in npz.files:
                return "skip_no_metadata", 0, 0
            meta_dict = npz["trial_metadata"].item()
        arrays = None  # not needed for dry-run
    else:
        with np.load(npz_file, allow_pickle=True) as npz:
            if "trial_metadata" not in npz.files:
                return "skip_no_metadata", 0, 0
            meta_dict = npz["trial_metadata"].item()
            arrays = {k: npz[k] for k in npz.files if k != "trial_metadata"}

    if "onset" not in meta_dict:
        return "skip_no_onset", 0, 0

    if "bad_ar2" in meta_dict and not overwrite:
        return "already_tagged", len(meta_dict["onset"]), int(np.sum(meta_dict["bad_ar2"]))

    # Cache annotations per (subject, run) so 8 feature files for the same
    # run only trigger one FIF read.
    cache_key = (subject, run)
    if cache_key not in annotations_cache:
        annotations_cache[cache_key] = load_clean_raw_annotations(
            subject=subject,
            run=run,
            config=config,
        )
    annotations = annotations_cache[cache_key]
    if annotations is None:
        return "skip_no_raw", len(meta_dict["onset"]), 0

    onsets = np.asarray(meta_dict["onset"], dtype=float)
    bad_mask = compute_bad_trial_mask(
        onsets=onsets,
        tmin=tmin,
        tmax=tmax,
        annotations=annotations,
    )
    meta_dict["bad_ar2"] = bad_mask.tolist()
    n_bad = int(bad_mask.sum())

    if dry_run:
        return "dry_run", len(onsets), n_bad

    # Rewrite the npz preserving every other key. ``trial_metadata`` is
    # stored as an object scalar (the dict) — match how it was originally
    # serialised. Write to a temp path and rename so an interrupted run
    # cannot leave a half-written feature file behind.
    assert arrays is not None
    save_kwargs = dict(arrays)
    save_kwargs["trial_metadata"] = meta_dict
    # ``np.savez_compressed`` auto-appends ``.npz`` if the path does not
    # already end with it, so build a tmp path that already ends in ``.npz``.
    tmp_path = npz_file.with_name(npz_file.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **save_kwargs)
    tmp_path.replace(npz_file)

    return "updated", len(onsets), n_bad


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add bad_ar2 column to existing feature trial_metadata."
    )
    parser.add_argument(
        "--space",
        type=str,
        default=None,
        help="Restrict to one analysis space (e.g. 'sensor', 'schaefer_400').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without rewriting any file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute bad_ar2 even when it already exists.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml).",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    features_root = (
        Path(config["paths"]["data_root"]) / config["paths"]["features"]
    )
    tmin = float(config["analysis"]["epochs"]["tmin"])
    tmax = float(config["analysis"]["epochs"]["tmax"])

    logger.info(f"Backfilling under {features_root}")
    logger.info(f"Epoch window: tmin={tmin} tmax={tmax}")
    if args.space:
        logger.info(f"Filtering to space: {args.space}")
    if args.dry_run:
        logger.info("DRY RUN — no files will be modified.")

    annotations_cache: dict = {}
    counts: dict = {}
    n_trials_total = 0
    n_bad_total = 0

    for npz_file in iter_feature_files(features_root, space_filter=args.space):
        status, n_trials, n_bad = backfill_one(
            npz_file=npz_file,
            config=config,
            tmin=tmin,
            tmax=tmax,
            annotations_cache=annotations_cache,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        counts[status] = counts.get(status, 0) + 1
        if status in ("updated", "dry_run", "already_tagged"):
            n_trials_total += n_trials
            n_bad_total += n_bad
        if status == "updated":
            logger.info(
                f"updated  {npz_file.relative_to(features_root)}  "
                f"({n_bad}/{n_trials} bad)"
            )
        elif status == "dry_run":
            logger.info(
                f"would update  {npz_file.relative_to(features_root)}  "
                f"({n_bad}/{n_trials} bad)"
            )
        elif status not in ("already_tagged",):
            logger.warning(
                f"{status}  {npz_file.relative_to(features_root)}"
            )

    logger.info("Summary by status:")
    for status, c in sorted(counts.items()):
        logger.info(f"  {status}: {c}")
    logger.info(f"Trials seen: {n_trials_total} (bad: {n_bad_total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
