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
import fcntl
import json
import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

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
from code.utils.behavioral import (
    VTC_FILTER_METHOD,
    VTC_FILTER_VERSION,
    get_VTC_from_file,
)
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
            "vtc_filter": {
                "method": VTC_FILTER_METHOD,
                "version": VTC_FILTER_VERSION,
                "fwhm_trials": config["behavioral"]["vtc"]["filter"]["gaussian_fwhm"],
                "boundary": "reflect",
            },
        },
    }

    provenance_file = output_dir / "code" / "provenance_bids.json"
    provenance_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=provenance_file.parent,
        prefix=f".{provenance_file.name}.",
        delete=False,
    ) as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
        temporary_path = Path(stream.name)
    os.replace(temporary_path, provenance_file)

    logger.info(f"Saved provenance to {provenance_file}")


def get_noise_recordings(meg_dir: Path) -> Dict[str, Path]:
    """Find all noise recordings and index by date.

    Args:
        meg_dir: Directory containing MEG .ds files.

    Returns:
        Dictionary mapping recording dates (YYYYMMDD) to noise file paths.
    """
    noise_files = {}
    for ds_path in meg_dir.glob("*/*.ds"):
        if "NOISE1Trial5min" in ds_path.name:
            try:
                raw = mne.io.read_raw_ctf(str(ds_path), verbose=False)
                er_date = raw.info["meas_date"].strftime("%Y%m%d")
                noise_files[er_date] = ds_path
                logger.debug(f"Found noise recording for date {er_date}: {ds_path.name}")
            except Exception as e:
                logger.warning(f"Could not read noise file {ds_path.name}: {e}")
    return noise_files


@contextmanager
def _subject_noise_lock(bids_root: Path, subject: str) -> Iterator[None]:
    """Serialize empty-room writes targeting one BIDS subject.

    Lock files live under ``bids/code`` so concurrent run-array cells cannot
    overwrite the same split FIF files. The lock is advisory and is released
    automatically when the process exits.
    """
    lock_dir = bids_root / "code" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"sub-{subject}_task-noise.lock"
    with lock_path.open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _noise_derivative_is_complete(bids_root: Path, subject: str) -> bool:
    """Return whether a reusable empty-room BIDS recording is complete."""
    meg_dir = bids_root / f"sub-{subject}" / "meg"
    data_files = list(meg_dir.glob(f"sub-{subject}_task-noise*_meg.fif"))
    required_sidecars = (
        meg_dir / f"sub-{subject}_task-noise_meg.json",
        meg_dir / f"sub-{subject}_task-noise_channels.tsv",
    )
    return bool(data_files) and all(path.is_file() and path.stat().st_size for path in required_sidecars)


def _write_noise_recording(
    subject: str,
    recording_date: str,
    noise_files: Dict[str, Path],
    bids_root: Path,
) -> None:
    """Write the appropriate empty-room recording without locking."""
    noise_path = noise_files[recording_date]
    raw = mne.io.read_raw_ctf(str(noise_path), verbose=False)
    raw.info["line_freq"] = 60

    rename_map = {}
    for old_name, new_name in (
        ("EEG057", "vEOG"),
        ("EEG058", "hEOG"),
        ("EEG059", "ECG"),
    ):
        if old_name in raw.ch_names:
            rename_map[old_name] = new_name
    if rename_map:
        mne.rename_channels(raw.info, rename_map)

    from mne_bids import BIDSPath

    noise_bids_path = BIDSPath(
        subject=subject,
        task="noise",
        datatype="meg",
        root=str(bids_root),
    )
    logger.info("Writing noise to %s", noise_bids_path.basename)
    write_raw_bids(raw, noise_bids_path, format="FIF", overwrite=True, verbose=False)


def copy_noise_to_subject(
    subject: str,
    recording_date: str,
    noise_files: Dict[str, Path],
    bids_root: Path,
) -> None:
    """Copy one subject's empty-room recording using a filesystem lock.

    Args:
        subject: Subject ID (e.g., '04').
        recording_date: Date of the subject's recording (YYYYMMDD).
        noise_files: Dictionary mapping dates to noise file paths.
        bids_root: BIDS dataset root directory.
    """
    if recording_date not in noise_files:
        logger.warning(f"No noise recording found for date {recording_date} (sub-{subject})")
        return

    try:
        with _subject_noise_lock(bids_root, subject):
            if _noise_derivative_is_complete(bids_root, subject):
                logger.info("Reusing complete noise recording for sub-%s", subject)
                return
            logger.info(
                "Copying noise recording for sub-%s (date: %s)",
                subject,
                recording_date,
            )
            _write_noise_recording(subject, recording_date, noise_files, bids_root)
    except Exception as e:
        logger.error(f"Failed to copy noise recording for sub-{subject}: {e}")


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

    events_df["VTC_filter_method"] = VTC_FILTER_METHOD
    events_df["VTC_filter_version"] = VTC_FILTER_VERSION
    events_df["VTC_filter_fwhm_trials"] = float(filter_config["gaussian_fwhm"])

    # Save enriched events
    events_df.to_csv(events_path, sep="\t", index=False)
    _write_events_sidecar(events_path, float(filter_config["gaussian_fwhm"]))
    logger.info("Saved reflected-boundary VTC enrichment to %s", events_path)


def _write_events_sidecar(events_path: Path, fwhm: float) -> None:
    """Write BIDS event-column metadata for corrected VTC provenance."""
    path = Path(events_path.fpath) if hasattr(events_path, "fpath") else Path(events_path)
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    metadata.update({
        "VTC_filtered": {
            "Description": "Run-wise Gaussian-smoothed variability time course",
            "FilterMethod": VTC_FILTER_METHOD,
            "FilterVersion": VTC_FILTER_VERSION,
            "BoundaryMode": "reflect",
            "FWHMTrials": fwhm,
        },
        "VTC_filter_method": {"Description": "VTC filtering implementation"},
        "VTC_filter_version": {"Description": "VTC filter contract version"},
        "VTC_filter_fwhm_trials": {"Description": "Gaussian FWHM in trials"},
    })
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def events_have_current_vtc(events_path: Path, fwhm: float) -> bool:
    """Return whether an events file has compatible corrected VTC provenance."""
    if not events_path.exists():
        return False
    try:
        events = pd.read_csv(events_path, sep="\t")
    except (OSError, pd.errors.ParserError):
        return False
    required = {
        "VTC_filtered", "VTC_filter_method", "VTC_filter_version",
        "VTC_filter_fwhm_trials",
    }
    if not required.issubset(events.columns):
        return False
    return (
        set(events["VTC_filter_method"].dropna().astype(str)) == {VTC_FILTER_METHOD}
        and set(events["VTC_filter_version"].dropna().astype(str)) == {VTC_FILTER_VERSION}
        and set(events["VTC_filter_fwhm_trials"].dropna().astype(float)) == {float(fwhm)}
    )


def process_subject_recording(
    ds_path: Path,
    bids_root: Path,
    behav_dir: Path,
    subject_list: List[str],
    run_list: List[str],
    config: dict,
    skip_valid: bool,
) -> bool:
    """Convert subject MEG recording to BIDS.

    Args:
        ds_path: Path to subject .ds file.
        bids_root: BIDS dataset root directory.
        behav_dir: Directory containing behavioral logfiles.
        subject_list: List of subjects to process.
        run_list: List of run IDs to process.
        config: Validated project configuration.
        skip_valid: Skip a cell only when corrected provenance matches.

    Returns:
        Whether the selected recording completed successfully.
    """
    fname = ds_path.name

    # Check if this subject should be processed
    try:
        subject_id = parse_info_from_name(fname)[0]
    except Exception as e:
        logger.warning(f"Could not parse filename {fname}: {e}")
        return False

    _, run_id, task = parse_info_from_name(fname)
    if subject_id not in subject_list or run_id not in run_list:
        logger.debug("Skipping sub-%s run-%s (not selected)", subject_id, run_id)
        return True
    expected_events = (
        bids_root / f"sub-{subject_id}" / "meg"
        / f"sub-{subject_id}_task-gradCPT_run-{run_id}_events.tsv"
    )
    fwhm = config["behavioral"]["vtc"]["filter"]["gaussian_fwhm"]
    if task == "gradCPT" and skip_valid and events_have_current_vtc(expected_events, fwhm):
        logger.info("Skipping provenance-compatible sub-%s run-%s", subject_id, run_id)
        return True

    logger.info(f"Processing: {fname}")

    try:
        # Load recording
        raw, bidspath, task = load_meg_recording(ds_path, bids_root)

        if task == "gradCPT":
            # Detect events
            events, event_id = detect_events(raw)

            # Clear annotations (will be in events file instead)
            raw.set_annotations(Annotations([], [], []))

            # Write to BIDS (FIF format to preserve renamed channels)
            logger.info(f"Writing to {bidspath.basename}")
            write_raw_bids(
                raw,
                bidspath,
                events=events,
                event_id=event_id,
                format="FIF",
                overwrite=True,
                verbose=False,
            )

            # Enrich events with behavioral data
            events_path = bidspath.copy().update(suffix="events", extension=".tsv")
            behav_files = [f.name for f in behav_dir.iterdir() if f.is_file()]

            enrich_gradcpt_events(
                events_path,
                bidspath.subject,
                bidspath.run,
                behav_files,
                behav_dir,
                config,
            )

        else:
            # Resting state - no events (FIF format to preserve renamed channels)
            logger.info(f"Writing resting state run: {bidspath.basename}")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No events found.*resting state")
                write_raw_bids(
                    raw,
                    bidspath,
                    format="FIF",
                    overwrite=True,
                    verbose=False,
                )
        return True

    except Exception as e:
        logger.error(f"Failed to process {fname}: {e}", exc_info=True)
        return False


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
        "--runs",
        nargs="+",
        default=None,
        help="Process specific run IDs only (e.g., --runs 02 03)",
    )
    parser.add_argument(
        "--skip-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip recordings with compatible corrected VTC provenance",
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
        bids_root = Path(config["paths"]["bids"])
        logger.info(f"Using output directory from config: {bids_root}")

    # Determine subjects to process
    if args.subjects:
        subject_list = args.subjects
        logger.info(f"Processing subjects from CLI: {subject_list}")
    else:
        subject_list = config["bids"]["subjects"]
        logger.info(f"Processing subjects from config: {len(subject_list)} subjects")
    run_list = args.runs or [*config["bids"]["task_runs"], *config["bids"]["rest_runs"]]

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

    # Index all noise recordings by date first
    logger.info("Indexing noise recordings...")
    noise_files = get_noise_recordings(meg_dir)
    logger.info(f"Found {len(noise_files)} noise recordings")

    # Track subjects and their recording dates
    subjects_processed = []
    subject_dates = {}  # subject_id -> recording_date
    failed_recordings = []

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
                    # Skip noise files here - we copy them to subject folders later
                    pass
                elif "SA" in fname and "procedure" not in fname:
                    # Subject recording
                    succeeded = process_subject_recording(
                        ds_path, bids_root, behav_dir, subject_list, run_list,
                        config, args.skip_valid,
                    )
                    if not succeeded:
                        failed_recordings.append(fname)

                    # Track subjects and their recording dates
                    try:
                        subj_id = parse_info_from_name(fname)[0]
                        if subj_id in subject_list:
                            if subj_id not in subjects_processed:
                                subjects_processed.append(subj_id)
                            # Get recording date from file
                            raw = mne.io.read_raw_ctf(str(ds_path), verbose=False)
                            rec_date = raw.info["meas_date"].strftime("%Y%m%d")
                            subject_dates[subj_id] = rec_date
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Error processing {fname}: {e}", exc_info=True)

            progress.advance(task)

    # Copy noise recordings to each subject's folder
    console.print("\n[bold]Copying noise recordings to subject folders...[/bold]")
    for subj_id, rec_date in subject_dates.items():
        copy_noise_to_subject(subj_id, rec_date, noise_files, bids_root)

    # Save provenance
    save_provenance(bids_root, config, sorted(subjects_processed))

    # Summary
    console.print("\n[bold green]✓ BIDS conversion complete![/bold green]")
    console.print(f"  Processed {len(subjects_processed)} subjects")
    console.print(f"  BIDS dataset: {bids_root}")
    console.print(f"  Logs: {log_dir}")

    logger.info("BIDS conversion complete")
    if failed_recordings:
        logger.error("Failed recordings: %s", failed_recordings)
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
