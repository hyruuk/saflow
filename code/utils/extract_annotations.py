"""Extract BAD_* annotations from existing cleaned-raw FIFs to JSON sidecars.

Run this once on the HPC (or wherever the cleaned-raw FIFs live) so that
downstream comparative analyses can drop bad trials by reading the tiny
JSON sidecar instead of the heavy raw FIF. Subsequent preprocessing runs
write the sidecar automatically — this script is purely for backfilling
already-processed runs.

Usage:
    python -m code.utils.extract_annotations              # all subjects/runs in config
    python -m code.utils.extract_annotations --subject 04 # one subject
    python -m code.utils.extract_annotations --overwrite  # rewrite existing sidecars
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import mne

from code.utils.annotations import (
    get_annotations_sidecar_path,
    get_clean_raw_path,
    write_annotations_sidecar,
)
from code.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_one(
    subject: str, run: str, config: dict, overwrite: bool = False
) -> str:
    """Return a status string for sub/run."""
    fif = get_clean_raw_path(subject, run, config)
    sidecar = get_annotations_sidecar_path(subject, run, config)

    if not fif.exists():
        return "no_fif"
    if sidecar.exists() and not overwrite:
        return "already_present"

    raw = mne.io.read_raw_fif(fif, preload=False, verbose="ERROR")
    write_annotations_sidecar(
        raw.annotations,
        subject=subject,
        run=run,
        config=config,
        sfreq=raw.info["sfreq"],
        n_times=raw.n_times,
    )
    return "extracted"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill annotations JSON sidecars from cleaned-raw FIFs."
    )
    parser.add_argument(
        "--subject", type=str, default=None, help="Limit to one subject."
    )
    parser.add_argument(
        "--run", type=str, default=None, help="Limit to one run."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing sidecars.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml).",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    subjects = [args.subject] if args.subject else config["bids"]["subjects"]
    runs = [args.run] if args.run else config["bids"]["task_runs"]

    counts: dict = {}
    for subject in subjects:
        for run in runs:
            status = extract_one(subject, run, config, overwrite=args.overwrite)
            counts[status] = counts.get(status, 0) + 1
            if status == "extracted":
                logger.info(f"extracted  sub-{subject} run-{run}")
            elif status == "no_fif":
                logger.warning(f"no_fif     sub-{subject} run-{run}")
    logger.info("Summary:")
    for status, c in sorted(counts.items()):
        logger.info(f"  {status}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
