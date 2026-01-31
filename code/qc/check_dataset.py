"""Dataset completeness checker for saflow.

Scans all data directories and reports which files exist/are missing
for each subject across the entire pipeline:

- Sourcedata: Raw MEG (.ds), behavioral logs (.mat), MRI
- BIDS: Converted MEG files
- Derivatives: Preprocessed data
- Processed: Features (FOOOF, PSD, complexity)

Usage:
    python -m code.qc.check_dataset
    python -m code.qc.check_dataset --verbose
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class SubjectFiles:
    """Files found for a single subject."""
    subject: str = ""
    # Sourcedata
    meg_runs: List[str] = field(default_factory=list)
    behav_runs: List[str] = field(default_factory=list)
    behav_extra: List[str] = field(default_factory=list)  # run 0, duplicates, typos
    has_mri: bool = False
    # BIDS
    bids_runs: List[str] = field(default_factory=list)
    # Derivatives (preprocessed)
    preproc_runs: List[str] = field(default_factory=list)
    # Features
    has_fooof: bool = False
    has_psd: bool = False
    has_complexity: bool = False


@dataclass
class DatasetSummary:
    """Summary of entire dataset."""
    subjects: Dict[str, SubjectFiles] = field(default_factory=dict)
    expected_subjects: List[str] = field(default_factory=list)
    expected_runs: List[str] = field(default_factory=list)


# ==============================================================================
# Scanning Functions
# ==============================================================================

def get_subject_date_mapping(sourcedata_meg: Path) -> Dict[str, str]:
    """Build mapping of subjects to their date folders in sourcedata/meg."""
    mapping = {}
    if not sourcedata_meg.exists():
        return mapping

    for date_folder in sorted(sourcedata_meg.iterdir()):
        if not date_folder.is_dir():
            continue
        for ds_file in date_folder.glob("SA*_SAflow*.ds"):
            fname = ds_file.name
            if fname.startswith("SA") and "_SAflow" in fname:
                try:
                    subject = fname.split("SA")[1][:2]
                    if subject not in mapping:
                        mapping[subject] = date_folder.name
                except (IndexError, ValueError):
                    pass
    return mapping


def scan_sourcedata_meg(sourcedata_meg: Path, subject: str, date_folder: str) -> List[str]:
    """Find MEG runs for a subject in sourcedata."""
    runs = []
    folder = sourcedata_meg / date_folder
    if not folder.exists():
        return runs

    for ds_file in folder.glob(f"SA{subject}_SAflow*.ds"):
        try:
            # Parse run from: SA04_SAflow-yharel_20190411_01.ds
            run = ds_file.name.replace(".ds", "").split("_")[-1]
            runs.append(run)
        except (IndexError, ValueError):
            pass
    return sorted(runs)


def scan_sourcedata_behav(sourcedata_behav: Path, subject: str) -> Dict[str, List[str]]:
    """Find behavioral runs for a subject.

    Returns dict with:
        - 'valid': runs 1-6 (expected task runs)
        - 'extra': run 0, duplicates, or unexpected run numbers
    """
    result = {"valid": [], "extra": []}
    if not sourcedata_behav.exists():
        return result

    seen_runs = {}  # run -> list of files (to detect duplicates)

    for mat_file in sorted(sourcedata_behav.glob(f"Data_0_{subject}_*.mat")):
        try:
            # Parse run from: Data_0_04_1_11-Apr-2019_...
            run = mat_file.name.split("_")[3]

            if run not in seen_runs:
                seen_runs[run] = []
            seen_runs[run].append(mat_file.name)
        except (IndexError, ValueError):
            pass

    # Categorize runs
    expected_runs = {"1", "2", "3", "4", "5", "6"}

    for run, files in sorted(seen_runs.items()):
        if run in expected_runs:
            if len(files) == 1:
                result["valid"].append(run)
            else:
                # Multiple files for same run - use first, mark extras
                result["valid"].append(run)
                result["extra"].extend([f"dup:{run}" for _ in files[1:]])
        else:
            # Run 0, typos, etc.
            result["extra"].extend([run] * len(files))

    return result


def scan_sourcedata_mri(sourcedata_mri: Path, subject: str) -> bool:
    """Check if MRI exists for subject."""
    if not sourcedata_mri.exists():
        return False
    # Look for subject folder or files
    subj_folder = sourcedata_mri / f"sub-{subject}"
    if subj_folder.exists():
        return True
    # Also check for numbered folders
    for folder in sourcedata_mri.iterdir():
        if folder.is_dir() and subject in folder.name:
            return True
    return False


def scan_bids(bids_root: Path, subject: str, task: str = "gradCPT") -> List[str]:
    """Find BIDS runs for a subject."""
    runs = []
    meg_dir = bids_root / f"sub-{subject}" / "ses-recording" / "meg"
    if not meg_dir.exists():
        return runs

    for ds_file in meg_dir.glob(f"*task-{task}*.ds"):
        try:
            # Parse run from: sub-04_ses-recording_task-gradCPT_run-02_meg.ds
            for part in ds_file.name.split("_"):
                if part.startswith("run-"):
                    runs.append(part.replace("run-", ""))
                    break
        except (IndexError, ValueError):
            pass
    return sorted(runs)


def scan_derivatives(derivatives_root: Path, subject: str) -> List[str]:
    """Find preprocessed runs for a subject."""
    runs = []
    subj_dir = derivatives_root / f"sub-{subject}" / "meg"
    if not subj_dir.exists():
        return runs

    for fif_file in subj_dir.glob("*_meg.fif"):
        try:
            for part in fif_file.name.split("_"):
                if part.startswith("run-"):
                    runs.append(part.replace("run-", ""))
                    break
        except (IndexError, ValueError):
            pass
    return sorted(set(runs))


def scan_features(features_root: Path, subject: str) -> Dict[str, bool]:
    """Check which feature types exist for a subject."""
    features = {"fooof": False, "psd": False, "complexity": False}

    # Check each feature type (using new folder naming convention)
    feature_dirs = {
        "fooof": features_root / "fooof_sensor" / f"sub-{subject}",
        "psd": features_root / "welch_psds_sensor" / f"sub-{subject}",
        "complexity": features_root / "complexity_sensor" / f"sub-{subject}",
    }

    for feature_type, feature_dir in feature_dirs.items():
        if feature_dir.exists() and any(feature_dir.glob("*.npz")):
            features[feature_type] = True

    return features


# ==============================================================================
# Main Scanner
# ==============================================================================

def scan_dataset(config: Dict) -> DatasetSummary:
    """Scan entire dataset and return summary."""
    summary = DatasetSummary(
        expected_subjects=config["bids"]["subjects"],
        expected_runs=config["bids"]["task_runs"],
    )

    data_root = Path(config["paths"]["data_root"])
    sourcedata = data_root / "sourcedata"
    bids_root = data_root / "bids"
    derivatives_root = data_root / config["paths"]["derivatives"]
    features_root = data_root / config["paths"]["features"]

    # Get subject-date mapping
    subj_date_map = get_subject_date_mapping(sourcedata / "meg")

    # Scan each expected subject
    for subject in summary.expected_subjects:
        sf = SubjectFiles(subject=subject)

        # Sourcedata
        if subject in subj_date_map:
            sf.meg_runs = scan_sourcedata_meg(
                sourcedata / "meg", subject, subj_date_map[subject]
            )
        behav_result = scan_sourcedata_behav(sourcedata / "behav", subject)
        sf.behav_runs = behav_result["valid"]
        sf.behav_extra = behav_result["extra"]
        sf.has_mri = scan_sourcedata_mri(sourcedata / "mri", subject)

        # BIDS
        sf.bids_runs = scan_bids(bids_root, subject)

        # Derivatives
        sf.preproc_runs = scan_derivatives(derivatives_root, subject)

        # Features
        features = scan_features(features_root, subject)
        sf.has_fooof = features["fooof"]
        sf.has_psd = features["psd"]
        sf.has_complexity = features["complexity"]

        summary.subjects[subject] = sf

    return summary


# ==============================================================================
# Report Generation
# ==============================================================================

def format_runs(runs: List[str], expected: List[str], all_runs: bool = False) -> str:
    """Format run list with indicators for missing runs."""
    if not runs:
        return "-"

    if all_runs:
        # Show all 8 runs
        expected_set = {"01", "02", "03", "04", "05", "06", "07", "08"}
    else:
        expected_set = set(expected)

    found_set = set(runs)

    if found_set >= expected_set:
        return f"{len(runs)}/8" if all_runs else f"{len(runs)}/{len(expected)}"
    else:
        missing = sorted(expected_set - found_set)
        return f"{len(runs)}/{len(expected_set)} (miss: {','.join(missing)})"


def bool_to_symbol(value: bool) -> str:
    """Convert boolean to symbol."""
    return "Y" if value else "-"


def generate_report(summary: DatasetSummary, verbose: bool = False) -> str:
    """Generate text report."""
    lines = [
        "=" * 100,
        "DATASET COMPLETENESS REPORT",
        "=" * 100,
        "",
    ]

    # Count totals
    n_subjects = len(summary.subjects)
    n_with_meg = sum(1 for s in summary.subjects.values() if s.meg_runs)
    n_with_bids = sum(1 for s in summary.subjects.values() if s.bids_runs)
    n_with_preproc = sum(1 for s in summary.subjects.values() if s.preproc_runs)
    n_with_features = sum(
        1 for s in summary.subjects.values()
        if s.has_fooof or s.has_psd or s.has_complexity
    )

    lines.extend([
        "SUMMARY",
        "-" * 50,
        f"Expected subjects:     {n_subjects}",
        f"With raw MEG:          {n_with_meg}/{n_subjects}",
        f"With BIDS:             {n_with_bids}/{n_subjects}",
        f"With preprocessed:     {n_with_preproc}/{n_subjects}",
        f"With features:         {n_with_features}/{n_subjects}",
        "",
    ])

    # Header
    header = (
        f"{'Subj':<6} | "
        f"{'MEG Runs':<18} | "
        f"{'Behav':<8} | "
        f"{'MRI':<4} | "
        f"{'BIDS':<18} | "
        f"{'Preproc':<18} | "
        f"{'FOOOF':<6} | "
        f"{'PSD':<4} | "
        f"{'Cmplx':<5}"
    )

    lines.extend([
        "PER-SUBJECT STATUS",
        "-" * 100,
        header,
        "-" * 100,
    ])

    # Each subject
    for subject in sorted(summary.subjects.keys()):
        sf = summary.subjects[subject]

        meg_str = format_runs(sf.meg_runs, [], all_runs=True)
        # Show valid/6, add warning marker if there are extra files
        if sf.behav_runs:
            behav_str = f"{len(sf.behav_runs)}/6"
            if sf.behav_extra:
                behav_str += f" (+{len(sf.behav_extra)})"
        else:
            behav_str = "-"
        mri_str = bool_to_symbol(sf.has_mri)
        bids_str = format_runs(sf.bids_runs, summary.expected_runs)
        preproc_str = format_runs(sf.preproc_runs, summary.expected_runs)
        fooof_str = bool_to_symbol(sf.has_fooof)
        psd_str = bool_to_symbol(sf.has_psd)
        cmplx_str = bool_to_symbol(sf.has_complexity)

        line = (
            f"{subject:<6} | "
            f"{meg_str:<18} | "
            f"{behav_str:<8} | "
            f"{mri_str:<4} | "
            f"{bids_str:<18} | "
            f"{preproc_str:<18} | "
            f"{fooof_str:<6} | "
            f"{psd_str:<4} | "
            f"{cmplx_str:<5}"
        )
        lines.append(line)

    lines.extend([
        "-" * 100,
        "",
        "Legend: Y = present, - = missing/none",
        "        Runs show: found/expected (miss: missing run numbers)",
        "=" * 100,
    ])

    # Verbose: show missing details
    if verbose:
        lines.extend(["", "MISSING DATA DETAILS", "-" * 50])

        for subject in sorted(summary.subjects.keys()):
            sf = summary.subjects[subject]
            issues = []

            # Check MEG
            expected_meg = {"01", "02", "03", "04", "05", "06", "07", "08"}
            missing_meg = expected_meg - set(sf.meg_runs)
            if missing_meg:
                issues.append(f"Missing MEG runs: {sorted(missing_meg)}")

            # Check behavioral
            expected_behav = {"1", "2", "3", "4", "5", "6"}
            missing_behav = expected_behav - set(sf.behav_runs)
            if missing_behav:
                issues.append(f"Missing behav runs: {sorted(missing_behav)}")
            if sf.behav_extra:
                issues.append(f"Extra behav files: {sf.behav_extra}")

            # Check BIDS
            expected_bids = set(summary.expected_runs)
            missing_bids = expected_bids - set(sf.bids_runs)
            if missing_bids:
                issues.append(f"Missing BIDS runs: {sorted(missing_bids)}")

            # Check preproc
            missing_preproc = expected_bids - set(sf.preproc_runs)
            if sf.bids_runs and missing_preproc:
                issues.append(f"Missing preproc runs: {sorted(missing_preproc)}")

            if issues:
                lines.append(f"sub-{subject}:")
                for issue in issues:
                    lines.append(f"  - {issue}")

    return "\n".join(lines)


# ==============================================================================
# CLI
# ==============================================================================

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration."""
    if config_path is None:
        config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Check dataset completeness across all pipeline stages"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed missing data information"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    logger.info("Scanning dataset...")

    # Scan dataset
    summary = scan_dataset(config)

    # Generate and print report
    report = generate_report(summary, verbose=args.verbose)
    print(report)


if __name__ == "__main__":
    main()
