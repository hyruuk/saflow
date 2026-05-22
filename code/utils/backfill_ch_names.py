"""Backfill ch_names into existing fooof / welch_psds / welch_psds_corrected npz files.

Older feature outputs predate the convention of carrying ``ch_names`` in every
feature npz (only ``complexity_*`` had it). Downstream analyses now expect a
consistent spatial-axis label across feature families, so this utility opens
each existing file, derives the spatial names from the source data (via
``code.features.loaders.load_data`` — one call per (subject, run, space)),
and rewrites the npz with the extra key.

Idempotent: files that already carry ``ch_names`` are skipped.

Atomic: each file is written to a sibling ``.tmp`` and renamed in place.

Usage:
    python -m code.utils.backfill_ch_names --space sensor
    python -m code.utils.backfill_ch_names --space sensor schaefer_400 \
        --families fooof welch_psds welch_psds_corrected
    python -m code.utils.backfill_ch_names --dry-run
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.features.loaders import load_data
from code.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


FAMILY_FOLDERS = {
    "fooof": "fooof_{space}",
    "welch_psds": "welch_psds_{space}",
    "welch_psds_corrected": "welch_psds_corrected_{space}",
}

# Regex to pull (subject, run) out of any of the feature file names. All three
# families use the same ``sub-{S}_..._run-{R}_..._desc-...npz`` convention.
_FILE_RE = re.compile(
    r"^sub-(?P<subject>[A-Za-z0-9]+)_ses-[^_]+_task-[^_]+_run-(?P<run>[A-Za-z0-9]+)_"
)


def _parse_subject_run(name: str) -> Optional[Tuple[str, str]]:
    m = _FILE_RE.match(name)
    if not m:
        return None
    return m.group("subject"), m.group("run")


def _needs_patch(path: Path) -> bool:
    """Cheap check — does this npz lack ch_names?"""
    try:
        with np.load(path, allow_pickle=True) as npz:
            return "ch_names" not in npz.files
    except Exception:
        return False  # treat unreadable files as not-needing-patch (we'll error later)


def _patch_one(path: Path, ch_names: List[str], dry_run: bool) -> str:
    """Return one of: 'skipped', 'patched', 'mismatch'."""
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "ch_names" in npz.files:
                return "skipped"
            payload: Dict[str, np.ndarray] = {k: npz[k] for k in npz.files}
    except Exception as exc:
        logger.error(f"  cannot read {path.name}: {exc}")
        return "skipped"

    # Check spatial-axis length agrees so we don't write a mismatched label set.
    n_spatial_target = len(ch_names)
    inferred_n: Optional[int] = None
    if "psds" in payload:
        inferred_n = int(payload["psds"].shape[1])
    elif "exponent" in payload:
        inferred_n = int(payload["exponent"].shape[1])

    if inferred_n is not None and inferred_n != n_spatial_target:
        logger.warning(
            f"  {path.name}: n_spatial mismatch — file has {inferred_n} but "
            f"derived ch_names has {n_spatial_target}. SKIPPED to avoid "
            f"writing a wrong index."
        )
        return "mismatch"

    if dry_run:
        return "patched"

    payload["ch_names"] = np.asarray(list(ch_names))
    # np.savez_compressed always writes a ".npz" suffix — use a tmp basename
    # that already ends in ".npz" so the renamed-in path is correct.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return "patched"


def backfill_space(
    space: str,
    families: List[str],
    bids_root: Path,
    features_root: Path,
    config: Dict,
    subjects: Optional[List[str]] = None,
    runs: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Backfill all files for one space. Returns counts dict."""
    counts = {"patched": 0, "skipped": 0, "mismatch": 0, "errors": 0}

    ch_names_cache: Dict[Tuple[str, str], List[str]] = {}

    def _get_ch_names(subject: str, run: str) -> Optional[List[str]]:
        key = (subject, run)
        if key in ch_names_cache:
            return ch_names_cache[key]
        try:
            sd = load_data(
                space=space, bids_root=bids_root, subject=subject, run=run,
                input_type="continuous", processing="clean", config=config,
            )
            names = list(sd.spatial_names)
        except FileNotFoundError as exc:
            logger.warning(f"  sub-{subject} run-{run}: source data missing ({exc})")
            return None
        except Exception as exc:
            logger.error(f"  sub-{subject} run-{run}: load_data failed: {exc}")
            return None
        ch_names_cache[key] = names
        return names

    for family in families:
        folder = features_root / FAMILY_FOLDERS[family].format(space=space)
        if not folder.exists():
            logger.info(f"[{space}/{family}] folder absent, skipping: {folder}")
            continue
        npz_files = sorted(folder.rglob("*.npz"))
        if not npz_files:
            logger.info(f"[{space}/{family}] no npz files in {folder}")
            continue
        logger.info(f"[{space}/{family}] scanning {len(npz_files)} files in {folder}")

        for npz_path in npz_files:
            sr = _parse_subject_run(npz_path.name)
            if sr is None:
                logger.warning(f"  cannot parse subject/run from {npz_path.name}")
                counts["errors"] += 1
                continue
            subject, run = sr
            if subjects and subject not in subjects:
                continue
            if runs and run not in runs:
                continue
            if not _needs_patch(npz_path):
                counts["skipped"] += 1
                continue
            ch_names = _get_ch_names(subject, run)
            if ch_names is None:
                counts["errors"] += 1
                continue
            result = _patch_one(npz_path, ch_names, dry_run=dry_run)
            counts[result] += 1
            if result == "patched":
                logger.info(
                    f"  {'(dry-run) ' if dry_run else ''}patched {npz_path.name}"
                )

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Backfill ch_names into existing fooof/welch_psd feature files."
    )
    parser.add_argument(
        "--space", nargs="+", default=None,
        help=(
            "Space(s) to backfill (e.g. sensor schaefer_400). "
            "Default: all spaces found under <features>/<family>_*."
        ),
    )
    parser.add_argument(
        "--families", nargs="+", default=list(FAMILY_FOLDERS.keys()),
        choices=list(FAMILY_FOLDERS.keys()),
        help="Which feature families to backfill (default: all three).",
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Restrict to these subject IDs.")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Restrict to these run IDs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; do not write.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    data_root = Path(config["paths"]["data_root"])
    bids_root = data_root / config["paths"]["bids"]
    features_root = data_root / config["paths"]["features"]

    # Discover spaces if not given: glob the first family folder
    if args.space:
        spaces = args.space
    else:
        spaces: List[str] = []
        for family in args.families:
            for sub in features_root.glob(f"{family}_*"):
                if sub.is_dir():
                    space = sub.name[len(family) + 1:]
                    if space and space not in spaces:
                        spaces.append(space)
        if not spaces:
            raise SystemExit(f"No feature folders found under {features_root}")

    logger.info(f"Spaces: {spaces}")
    logger.info(f"Families: {args.families}")
    if args.dry_run:
        logger.info("DRY RUN — no files will be modified")

    totals = {"patched": 0, "skipped": 0, "mismatch": 0, "errors": 0}
    for space in spaces:
        logger.info("=" * 70)
        logger.info(f"SPACE: {space}")
        logger.info("=" * 70)
        counts = backfill_space(
            space=space,
            families=args.families,
            bids_root=bids_root,
            features_root=features_root,
            config=config,
            subjects=args.subjects,
            runs=args.runs,
            dry_run=args.dry_run,
        )
        logger.info(f"[{space}] {counts}")
        for k, v in counts.items():
            totals[k] += v

    logger.info("=" * 70)
    logger.info(f"TOTAL: {totals}")


if __name__ == "__main__":
    main()
