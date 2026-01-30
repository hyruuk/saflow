"""Input data validation utilities for saflow.

This module provides utilities for validating that required input data
is present and complete before running pipeline stages.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table

from code.utils.config import load_config

logger = logging.getLogger(__name__)
console = Console()


def check_raw_data_structure(
    data_root: Path,
    expected_subjects: List[str],
    verbose: bool = False,
) -> Tuple[bool, Dict[str, any]]:
    """Validate raw data directory structure.

    Args:
        data_root: Path to raw data directory.
        expected_subjects: List of subject IDs to check for.
        verbose: Whether to show detailed file listings.

    Returns:
        Tuple containing:
        - valid: Whether structure is valid
        - report: Dictionary with validation results
    """
    report = {
        "meg_dir_exists": False,
        "behav_dir_exists": False,
        "noise_files": [],
        "subject_files": {},
        "missing_subjects": [],
        "behav_files": [],
        "errors": [],
    }

    # Check MEG directory
    meg_dir = data_root / "meg"
    if not meg_dir.exists():
        report["errors"].append(f"MEG directory not found: {meg_dir}")
        return False, report

    report["meg_dir_exists"] = True

    # Check behavioral directory
    behav_dir = data_root / "behav"
    if not behav_dir.exists():
        report["errors"].append(f"Behavioral directory not found: {behav_dir}")
        return False, report

    report["behav_dir_exists"] = True

    # Find all .ds files
    ds_files = sorted(meg_dir.glob("*/*.ds"))

    if not ds_files:
        report["errors"].append(f"No .ds files found in {meg_dir}")
        return False, report

    # Categorize files
    for ds_path in ds_files:
        fname = ds_path.name

        if "NOISE1Trial5min" in fname:
            report["noise_files"].append(fname)
        elif "SA" in fname and "procedure" not in fname:
            # Extract subject ID
            try:
                subj_id = fname.split("SA")[1][:2]
                run_id = fname.split("_")[-1][:2]

                if subj_id not in report["subject_files"]:
                    report["subject_files"][subj_id] = []

                report["subject_files"][subj_id].append(f"run-{run_id}")

            except (IndexError, ValueError):
                report["errors"].append(f"Could not parse subject ID from: {fname}")

    # Check for missing subjects
    found_subjects = set(report["subject_files"].keys())
    expected_set = set(expected_subjects)
    report["missing_subjects"] = sorted(expected_set - found_subjects)

    # Check behavioral files
    behav_files = sorted(behav_dir.glob("*.mat"))
    report["behav_files"] = [f.name for f in behav_files]

    if not behav_files:
        report["errors"].append(f"No .mat behavioral files found in {behav_dir}")
        return False, report

    # Validation passed if no errors
    is_valid = len(report["errors"]) == 0

    return is_valid, report


def print_validation_report(
    report: Dict[str, any],
    expected_subjects: List[str],
    verbose: bool = False,
):
    """Print validation report to console.

    Args:
        report: Validation report dictionary.
        expected_subjects: List of expected subject IDs.
        verbose: Whether to show detailed listings.
    """
    console.print("\n[bold]Data Structure Validation[/bold]")
    console.print("=" * 80)

    # Directory checks
    meg_status = "✓" if report["meg_dir_exists"] else "✗"
    behav_status = "✓" if report["behav_dir_exists"] else "✗"

    console.print(f"{meg_status} MEG directory: {'exists' if report['meg_dir_exists'] else 'missing'}")
    console.print(f"{behav_status} Behavioral directory: {'exists' if report['behav_dir_exists'] else 'missing'}")

    # Noise files
    console.print(f"\n[bold]Noise files:[/bold] {len(report['noise_files'])} found")
    if verbose and report["noise_files"]:
        for fname in report["noise_files"]:
            console.print(f"  - {fname}")

    # Subject files
    console.print(f"\n[bold]Subject data:[/bold]")
    console.print(f"  Expected subjects: {len(expected_subjects)}")
    console.print(f"  Found subjects:    {len(report['subject_files'])}")

    if report["missing_subjects"]:
        console.print(f"\n[yellow]⚠️  Missing subjects ({len(report['missing_subjects'])}):[/yellow]")
        for subj in report["missing_subjects"]:
            console.print(f"  - sub-{subj}")

    # Subject table
    if report["subject_files"]:
        table = Table(title="\nSubject Runs")
        table.add_column("Subject", style="cyan")
        table.add_column("Runs", style="green")
        table.add_column("Count", justify="right")

        for subj_id in sorted(report["subject_files"].keys()):
            runs = report["subject_files"][subj_id]
            table.add_row(
                f"sub-{subj_id}",
                ", ".join(sorted(runs)),
                str(len(runs))
            )

        console.print(table)

    # Behavioral files
    console.print(f"\n[bold]Behavioral files:[/bold] {len(report['behav_files'])} found")
    if verbose and report["behav_files"]:
        for fname in report["behav_files"][:10]:  # Show first 10
            console.print(f"  - {fname}")
        if len(report["behav_files"]) > 10:
            console.print(f"  ... and {len(report['behav_files']) - 10} more")

    # Errors
    if report["errors"]:
        console.print(f"\n[bold red]Errors ({len(report['errors'])}):[/bold red]")
        for error in report["errors"]:
            console.print(f"  ✗ {error}")

    # Summary
    console.print("\n" + "=" * 80)
    if report["errors"]:
        console.print("[bold red]✗ Validation FAILED[/bold red]")
        console.print("\nPlease fix the errors above before proceeding.")
    else:
        console.print("[bold green]✓ Validation PASSED[/bold green]")
        console.print("\nInput data is ready for BIDS conversion.")

    console.print("=" * 80 + "\n")


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate saflow input data"
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Check raw input data structure",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override data root from config",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed file listings",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        return 1

    # Determine data root
    if args.data_root:
        data_root = args.data_root
    else:
        data_root = Path(config["paths"]["data_root"]) / config["paths"]["raw"]

    console.print(f"Data root: {data_root}")

    # Get expected subjects
    expected_subjects = config["bids"]["subjects"]

    # Validate inputs
    if args.check_inputs:
        is_valid, report = check_raw_data_structure(
            data_root,
            expected_subjects,
            verbose=args.verbose,
        )

        print_validation_report(report, expected_subjects, verbose=args.verbose)

        return 0 if is_valid else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
