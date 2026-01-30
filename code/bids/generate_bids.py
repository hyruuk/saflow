"""Convert raw MEG data to BIDS format.

Stage 0 of the saflow pipeline: Raw → BIDS conversion.

This script:
1. Finds CTF MEG datasets (.ds files) in raw directory
2. Converts to BIDS format using mne-bids
3. Enriches gradCPT task events with behavioral data (VTC, RT, performance)
4. Adds IN/OUT zone classifications for different percentile bounds
5. Writes empty-room noise recordings

Usage:
    # Use paths from config
    python code/bids/generate_bids.py

    # Override paths
    python code/bids/generate_bids.py -i /path/to/raw -o /path/to/bids

    # Process specific subjects only
    python code/bids/generate_bids.py --subjects 04 05 06

Author: Claude (Anthropic)
Date: 2026-01-30
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import mne
import pandas as pd
from mne import Annotations
from mne_bids import write_raw_bids
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from code.bids.utils import (
    add_behavioral_info,
    add_inout_zones,
    add_trial_indices,
    detect_events,
    load_meg_recording,
    parse_info_from_name,
)
from code.utils.behavioral import get_VTC_from_file
from code.utils.config import load_config
from code.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
console = Console()


def get_git_hash() -> Optional[str]:
    """Get current git commit hash for provenance tracking.

    Returns:
        Git commit hash, or None if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not retrieve git hash")
        return None


def save_provenance(output_dir: Path, config: dict, subjects_processed: List[str]):
    """Save provenance information to JSON.

    Args:
        output_dir: BIDS root directory.
        config: Configuration dictionary.
        subjects_processed: List of subject IDs that were processed.
    """
    provenance = {
        "script": "code/bids/generate_bids.py",
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "subjects_processed": subjects_processed,
        "config": {
            "task_name": config["bids"]["task_name"],
            "subjects": config["bids"]["subjects"],
            "task_runs": config["bids"]["task_runs"],
            "rest_runs": config["bids"]["rest_runs"],
        },
    }

    provenance_file = output_dir / "code" / "provenance_bids.json"
    provenance_file.parent.mkdir(parents=True, exist_ok=True)

    with open(provenance_file, "w") as f:
        json.dump(provenance, f, indent=2)

    logger.info(f"Saved provenance to {provenance_file}")


def process_noise_file(ds_path: Path, bids_root: Path):
    """Convert empty-room noise recording to BIDS.

    Args:
        ds_path: Path to noise .ds file.
        bids_root: BIDS dataset root directory.
    """
    logger.info(f"Processing noise file: {ds_path.name}")

    try:
        raw = mne.io.read_raw_ctf(str(ds_path), verbose=False)
        raw.info["line_freq"] = 60

        # Create BIDS path for empty-room recording
        er_date = raw.info["meas_date"].strftime("%Y%m%d")
        from mne_bids import BIDSPath

        er_bids_path = BIDSPath(
            subject="emptyroom",
            session=er_date,
            task="noise",
            datatype="meg",
            extension=".ds",
            root=str(bids_root),
        )

        logger.info(f"Writing to {er_bids_path.basename}")
        write_raw_bids(raw, er_bids_path, format="auto", overwrite=True, verbose=False)

    except Exception as e:
        logger.error(f"Failed to process noise file {ds_path.name}: {e}")


def enrich_gradcpt_events(
    events_path: Path,
    subject: str,
    run: str,
    behav_files: List[str],
    logs_dir: Path,
    config: dict,
):
    """Enrich gradCPT events file with behavioral data.

    Adds trial indices, VTC (raw and filtered), RT, and task performance to
    the BIDS events file. VTC is computed using filter parameters from config.

    Zone classifications (IN/OUT/MID) are NOT pre-computed here - they will be
    computed on-demand during feature extraction using the bounds specified in
    config['analysis']['inout_bounds'].

    Args:
        events_path: Path to BIDS events.tsv file.
        subject: Subject ID.
        run: Run number.
        behav_files: List of behavioral logfile names.
        logs_dir: Directory containing behavioral logfiles.
        config: Configuration dictionary.
    """
    logger.info(f"Enriching events for sub-{subject} run-{run}")

    # Load BIDS events
    events_df = pd.read_csv(events_path, sep="\t")

    # Add trial indices
    events_df = add_trial_indices(events_df)

    # Get filter parameters from config
    filter_config = config["behavioral"]["vtc"]["filter"]
    filt_type = filter_config["type"]

    if filt_type == "gaussian":
        fwhm = filter_config["gaussian_fwhm"]
        logger.info(f"Using Gaussian filter with FWHM={fwhm}")
    elif filt_type == "butterworth":
        filt_order = filter_config["butterworth_order"]
        filt_cutoff = filter_config["butterworth_cutoff"]
        logger.info(f"Using Butterworth filter (order={filt_order}, cutoff={filt_cutoff})")
    else:
        logger.warning(f"Unknown filter type '{filt_type}', using Gaussian with FWHM=9")
        filt_type = "gaussian"
        fwhm = 9

    # Get behavioral data from logfiles
    # Note: inout_bounds parameter is ignored in new architecture
    # (zones computed on-demand during feature extraction)
    (
        _,  # IN_idx (not needed)
        _,  # OUT_idx (not needed)
        VTC_raw,
        VTC_filtered,
        _,  # IN_mask (not needed)
        _,  # OUT_mask (not needed)
        performance_dict,
        df_response,
        RT_to_VTC,
    ) = get_VTC_from_file(
        subject=subject,
        run=run,
        files_list=behav_files,
        logs_dir=logs_dir,
        cpt_blocs=["2", "3", "4", "5", "6", "7"],
        filt_type=filt_type,
        filt_config=filter_config,
    )

    # Add behavioral info to events (VTC_raw, VTC_filtered, RT, task)
    events_df = add_behavioral_info(
        events_df,
        VTC_raw,
        VTC_filtered,
        RT_to_VTC,
        performance_dict
    )

    # Save enriched events
    events_df.to_csv(events_path, sep="\t", index=False)
    logger.info(f"Saved enriched events with VTC_raw, VTC_filtered, RT, task to {events_path}")


def process_subject_recording(
    ds_path: Path,
    bids_root: Path,
    behav_dir: Path,
    subject_list: List[str],
):
    """Convert subject MEG recording to BIDS.

    Args:
        ds_path: Path to subject .ds file.
        bids_root: BIDS dataset root directory.
        behav_dir: Directory containing behavioral logfiles.
        subject_list: List of subjects to process.
    """
    fname = ds_path.name

    # Check if this subject should be processed
    try:
        subject_id = parse_info_from_name(fname)[0]
    except Exception as e:
        logger.warning(f"Could not parse filename {fname}: {e}")
        return

    if subject_id not in subject_list:
        logger.debug(f"Skipping subject {subject_id} (not in subject list)")
        return

    logger.info(f"Processing: {fname}")

    try:
        # Load recording
        raw, bidspath, task = load_meg_recording(ds_path, bids_root)

        if task == "gradCPT":
            # Detect events
            events, event_id = detect_events(raw)

            # Clear annotations (will be in events file instead)
            raw.set_annotations(Annotations([], [], []))

            # Write to BIDS
            logger.info(f"Writing to {bidspath.basename}")
            write_raw_bids(
                raw,
                bidspath,
                events=events,
                event_id=event_id,
                format="auto",
                overwrite=True,
                verbose=False,
            )

            # Enrich events with behavioral data
            events_path = bidspath.copy().update(suffix="events", extension=".tsv")
            behav_files = [f.name for f in behav_dir.iterdir() if f.is_file()]

            # Load config for behavioral parameters
            from code.utils.config import load_config
            config = load_config()

            enrich_gradcpt_events(
                events_path,
                bidspath.subject,
                bidspath.run,
                behav_files,
                behav_dir,
                config,
            )

        else:
            # Resting state - no events
            logger.info(f"Writing resting state run: {bidspath.basename}")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No events found.*resting state")
                write_raw_bids(
                    raw,
                    bidspath,
                    format="auto",
                    overwrite=True,
                    verbose=False,
                )

    except Exception as e:
        logger.error(f"Failed to process {fname}: {e}", exc_info=True)


def main():
    """Main BIDS conversion workflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Convert raw MEG data to BIDS format"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Path to raw data directory (overrides config)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to BIDS output directory (overrides config)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Process specific subjects only (e.g., --subjects 04 05)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without processing data",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Setup logging
    log_dir = Path(config["paths"]["logs"]) / "bids"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"bids_conversion_{timestamp}.log"

    setup_logging(__name__, log_file=log_file, level=args.log_level)

    logger.info("=" * 80)
    logger.info("BIDS Conversion - Stage 0")
    logger.info("=" * 80)

    # Determine paths
    if args.input:
        raw_dir = args.input
        logger.info(f"Using input directory from CLI: {raw_dir}")
    else:
        raw_dir = Path(config["paths"]["data_root"]) / config["paths"]["raw"]
        logger.info(f"Using input directory from config: {raw_dir}")

    if args.output:
        bids_root = args.output
        logger.info(f"Using output directory from CLI: {bids_root}")
    else:
        bids_root = Path(config["paths"]["data_root"]) / "bids"
        logger.info(f"Using output directory from config: {bids_root}")

    # Determine subjects to process
    if args.subjects:
        subject_list = args.subjects
        logger.info(f"Processing subjects from CLI: {subject_list}")
    else:
        subject_list = config["bids"]["subjects"]
        logger.info(f"Processing subjects from config: {len(subject_list)} subjects")

    # Validate paths
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return 1

    meg_dir = raw_dir / "meg"
    behav_dir = raw_dir / "behav"

    if not meg_dir.exists():
        logger.error(f"MEG directory not found: {meg_dir}")
        return 1

    if not behav_dir.exists():
        logger.error(f"Behavioral directory not found: {behav_dir}")
        return 1

    # Create BIDS root
    bids_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"BIDS root: {bids_root}")

    # Find all .ds files
    ds_files = sorted(meg_dir.glob("*/*.ds"))
    logger.info(f"Found {len(ds_files)} .ds files")

    if not ds_files:
        logger.warning(f"No .ds files found in {meg_dir}")
        return 1

    # Dry run mode - just validate and exit
    if args.dry_run:
        console.print("\n[bold yellow]DRY RUN MODE - No files will be processed[/bold yellow]\n")

        # Count files by type
        noise_count = 0
        subject_files = {}

        for ds_path in ds_files:
            fname = ds_path.name
            if "NOISE1Trial5min" in fname:
                noise_count += 1
            elif "SA" in fname and "procedure" not in fname:
                try:
                    subj_id = parse_info_from_name(fname)[0]
                    if subj_id in subject_list:
                        if subj_id not in subject_files:
                            subject_files[subj_id] = []
                        subject_files[subj_id].append(fname)
                except Exception:
                    pass

        console.print(f"[bold]Files to process:[/bold]")
        console.print(f"  Noise files: {noise_count}")
        console.print(f"  Subjects: {len(subject_files)}")
        console.print(f"  Total recordings: {sum(len(f) for f in subject_files.values())}")

        console.print(f"\n[bold]Output directory:[/bold] {bids_root}")
        console.print(f"[bold]Log directory:[/bold] {log_dir}")

        console.print("\n[bold green]✓ Validation complete - ready to process[/bold green]")
        console.print("\nRun without --dry-run to process files.")
        return 0

    # Process files with progress bar
    subjects_processed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting to BIDS...", total=len(ds_files))

        for ds_path in ds_files:
            fname = ds_path.name
            progress.update(task, description=f"Processing {fname}")

            try:
                if "NOISE1Trial5min" in fname:
                    # Empty-room noise recording
                    process_noise_file(ds_path, bids_root)
                elif "SA" in fname and "procedure" not in fname:
                    # Subject recording
                    process_subject_recording(
                        ds_path, bids_root, behav_dir, subject_list
                    )

                    # Track subjects
                    try:
                        subj_id = parse_info_from_name(fname)[0]
                        if subj_id not in subjects_processed:
                            subjects_processed.append(subj_id)
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Error processing {fname}: {e}", exc_info=True)

            progress.advance(task)

    # Save provenance
    save_provenance(bids_root, config, sorted(subjects_processed))

    # Summary
    console.print("\n[bold green]✓ BIDS conversion complete![/bold green]")
    console.print(f"  Processed {len(subjects_processed)} subjects")
    console.print(f"  BIDS dataset: {bids_root}")
    console.print(f"  Logs: {log_dir}")

    logger.info("BIDS conversion complete")
    return 0


if __name__ == "__main__":
    exit(main())
