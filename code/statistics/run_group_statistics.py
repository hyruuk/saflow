"""Group-level statistical analysis for IN vs OUT attentional states.

This script performs group-level statistical comparisons between IN and OUT
attentional states using various statistical tests and multiple comparison
corrections.

Correction methods:
- none: No correction (not recommended)
- fdr: Benjamini-Hochberg FDR (controls false discovery rate)
- bonferroni: Bonferroni correction (controls FWER, conservative)
- tmax: Maximum statistic permutation (controls FWER, recommended)

Usage:
    python -m code.statistics.run_group_statistics \
        --feature-type fooof_exponent \
        --space sensor \
        --test paired_ttest \
        --correction tmax --n-permutations 10000
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from code.statistics.corrections import (
    apply_fdr_correction,
    apply_bonferroni_correction,
    apply_tmax_correction,
)
from code.statistics.effect_sizes import (
    compute_cohens_d,
    compute_hedges_g,
    compute_eta_squared,
)
from code.utils.data_loading import load_features, balance_dataset
from code.utils.statistics import subject_contrast, simple_contrast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to ./config.yaml

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path("config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_git_hash() -> str:
    """Get current git commit hash for provenance tracking.

    Returns:
        Git commit hash, or 'unknown' if not in a git repo.
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
        return "unknown"


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for reproducibility tracking.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm ('md5', 'sha256'). Defaults to 'sha256'.

    Returns:
        Hexadecimal hash string.
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def inout_bounds_to_string(bounds: Tuple[int, int]) -> str:
    """Convert INOUT bounds to string format for filenames.

    Args:
        bounds: Tuple of (lower_percentile, upper_percentile).

    Returns:
        String like "2575" or "5050".
    """
    return f"{bounds[0]}{bounds[1]}"


def get_feature_folder(
    config: Dict,
    feature_type: str,
    space: str,
) -> Path:
    """Get the path to feature folder based on feature type and space.

    Args:
        config: Configuration dictionary.
        feature_type: Feature type (e.g., 'fooof_exponent', 'psd_alpha', 'lzc', 'complexity').
        space: Analysis space ('sensor', 'source', 'atlas').

    Returns:
        Path to feature folder.
    """
    data_root = Path(config["paths"]["data_root"])
    processed = data_root / config["paths"]["features"]

    # Map feature types to folder names
    # Actual folder structure: fooof_sensor, complexity_sensor, welch_psds_sensor, etc.
    if feature_type.startswith("fooof_"):
        folder_name = f"fooof_{space}"
    elif feature_type.startswith("psd_corrected_"):
        # Aperiodic-corrected PSDs (periodic component from FOOOF)
        folder_name = f"welch_psds_corrected_{space}"
    elif feature_type.startswith("psd_"):
        folder_name = f"welch_psds_{space}"
    elif feature_type == "complexity" or feature_type.startswith("complexity_"):
        folder_name = f"complexity_{space}"
    elif feature_type.startswith("lzc"):
        folder_name = f"complexity_{space}"
    else:
        # Default to treating it as the feature type name
        folder_name = f"{feature_type}_{space}"

    return processed / folder_name


def get_file_pattern(feature_type: str) -> Tuple[str, str]:
    """Get file pattern and data key for a feature type.

    Returns:
        Tuple of (glob_pattern_suffix, npz_key)
    """
    if feature_type.startswith("fooof_"):
        return "desc-fooof.npz", feature_type.replace("fooof_", "")
    elif feature_type.startswith("psd_corrected_"):
        return "desc-welch-corrected_psds.npz", "psds"  # Corrected PSDs in separate folder
    elif feature_type.startswith("psd_"):
        return "desc-welch_psds.npz", "psds"
    elif feature_type.startswith("complexity_"):
        # All complexity sub-metrics live in the same desc-complexity.npz file;
        # the npz key is the part after 'complexity_' (e.g. 'lzc_median',
        # 'entropy_permutation', 'fractal_higuchi').
        return "desc-complexity.npz", feature_type.replace("complexity_", "")
    else:
        return f"desc-{feature_type}.npz", feature_type


def load_all_features_batched(
    feature_types: List[str],
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict,
    subjects: Optional[List[str]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
    """Load several features sharing the same source files in one pass.

    All `feature_types` must come from the same feature family (same folder,
    same file suffix) — e.g. all PSD bands, or all FOOOF parameters. Each
    (subject, run) npz is opened exactly once; per-feature slices are
    extracted in memory.

    Returns a dict mapping `feature_type` -> `(X, y, groups, metadata)`,
    matching the shape returned by `load_all_features`.
    """
    if not feature_types:
        raise ValueError("feature_types must be non-empty")

    # All requested types must share folder + file suffix.
    folders = {get_feature_folder(config, ft, space) for ft in feature_types}
    if len(folders) != 1:
        raise ValueError(
            f"Batched loading requires a single feature folder; "
            f"got {len(folders)}: {folders}. Call separately per family."
        )
    feature_folder = folders.pop()
    file_patterns = {get_file_pattern(ft)[0] for ft in feature_types}
    if len(file_patterns) != 1:
        raise ValueError(
            f"Batched loading requires a single file pattern; got: {file_patterns}"
        )
    file_pattern = file_patterns.pop()

    if not feature_folder.exists():
        raise FileNotFoundError(f"Feature folder not found: {feature_folder}")

    if subjects is None:
        subjects = config["bids"]["subjects"]
    runs = config["bids"]["task_runs"]
    freq_bands = config.get("features", {}).get("frequency_bands", {})

    # Per requested feature, what to extract from each npz.
    extract_specs: Dict[str, Tuple[str, Optional[Tuple[str, Tuple[float, float]]]]] = {}
    for ft in feature_types:
        _, data_key = get_file_pattern(ft)
        if ft.startswith("psd_corrected_"):
            band = ft.replace("psd_corrected_", "")
            if band not in freq_bands:
                raise ValueError(f"Band '{band}' not in config frequency_bands")
            extract_specs[ft] = (data_key, ("band", freq_bands[band]))
        elif ft.startswith("psd_"):
            band = ft.replace("psd_", "")
            if band not in freq_bands:
                raise ValueError(f"Band '{band}' not in config frequency_bands")
            extract_specs[ft] = (data_key, ("band", freq_bands[band]))
        else:
            extract_specs[ft] = (data_key, None)

    logger.info(
        f"Batched-loading {len(feature_types)} feature(s) from {feature_folder.name}"
    )

    # Per-feature collected lists, plus shared y/groups bookkeeping.
    per_feat_X: Dict[str, List[np.ndarray]] = {ft: [] for ft in feature_types}
    all_y: List[int] = []
    all_groups: List[int] = []
    input_git_hashes: set = set()
    total_in = 0
    total_out = 0

    for subj_idx, subject in enumerate(
        tqdm(subjects, desc="Loading subjects", unit="subj")
    ):
        subj_dir = feature_folder / f"sub-{subject}"
        if not subj_dir.exists():
            continue

        # Per-feature list of per-run arrays for this subject
        subj_data: Dict[str, List[np.ndarray]] = {ft: [] for ft in feature_types}
        subj_vtc: List[np.ndarray] = []
        subj_task: List[np.ndarray] = []

        for run in runs:
            files = list(subj_dir.glob(f"sub-{subject}_*_run-{run}_*_{file_pattern}"))
            if not files:
                continue
            file_path = files[0]

            try:
                npz_data = np.load(file_path, allow_pickle=True)
            except Exception as exc:
                logger.warning(f"Could not open {file_path.name}: {exc}")
                continue

            try:
                meta = npz_data["trial_metadata"].item()

                # Provenance
                params_file = file_path.with_name(file_path.stem + "_params.json")
                if params_file.exists():
                    try:
                        params = json.loads(params_file.read_text())
                        if "git_hash" in params:
                            input_git_hashes.add(params["git_hash"])
                    except Exception:
                        pass

                # If this file has 'psds', pre-decode it once and reuse for every band.
                psd_block = npz_data["psds"] if "psds" in npz_data.files else None
                freqs = npz_data["freqs"] if "freqs" in npz_data.files else None

                ok = True
                feat_blocks = {}
                for ft in feature_types:
                    data_key, transform = extract_specs[ft]
                    if transform is not None and transform[0] == "band":
                        if psd_block is None or freqs is None:
                            logger.warning(
                                f"{file_path.name}: missing 'psds'/'freqs' for {ft}; skipping subject/run"
                            )
                            ok = False
                            break
                        fmin, fmax = transform[1]
                        fmask = (freqs >= fmin) & (freqs <= fmax)
                        feat_blocks[ft] = np.mean(psd_block[:, :, fmask], axis=2)
                    else:
                        if data_key not in npz_data.files:
                            logger.warning(
                                f"{file_path.name}: key '{data_key}' missing for {ft}; skipping subject/run"
                            )
                            ok = False
                            break
                        feat_blocks[ft] = np.asarray(npz_data[data_key])

                if not ok:
                    continue

                for ft in feature_types:
                    subj_data[ft].append(feat_blocks[ft])
                subj_vtc.append(np.array(meta["VTC_filtered"]))
                subj_task.append(np.array(meta["task"]))

            finally:
                npz_data.close()

        if not subj_vtc:
            continue

        # Trial-level concat per feature (same trial order across features).
        for ft in feature_types:
            subj_data[ft] = np.concatenate(subj_data[ft], axis=0)
        subj_vtc_arr = np.concatenate(subj_vtc)
        subj_task_arr = np.concatenate(subj_task)

        # Per-subject IN/OUT bounds (same as load_all_features)
        inbound = np.nanpercentile(subj_vtc_arr, inout_bounds[0])
        outbound = np.nanpercentile(subj_vtc_arr, inout_bounds[1])
        task_mask = subj_task_arr == "correct_commission"
        in_mask = task_mask & (subj_vtc_arr <= inbound)
        out_mask = task_mask & (subj_vtc_arr >= outbound)
        n_in = int(in_mask.sum())
        n_out = int(out_mask.sum())
        if n_in == 0 or n_out == 0:
            continue

        # Append IN then OUT for every feature (same trial order across features).
        for ft in feature_types:
            arr = subj_data[ft]
            per_feat_X[ft].append(arr[in_mask])
            per_feat_X[ft].append(arr[out_mask])
        all_y.extend([0] * n_in + [1] * n_out)
        all_groups.extend([subj_idx] * (n_in + n_out))
        total_in += n_in
        total_out += n_out

    if not all_y:
        raise ValueError("No data loaded from any subject")

    y = np.array(all_y)
    groups = np.array(all_groups)

    # Spatial names (one lookup is enough)
    spatial_names = None
    if space not in ("sensor", "source"):
        try:
            import mne
            from code.source_reconstruction.apply_atlas import get_mne_atlas_name
            fsaverage_path = mne.datasets.fetch_fsaverage(verbose=False)
            subjects_dir = str(Path(fsaverage_path).parent)
            mne_atlas = get_mne_atlas_name(space)
            labels = mne.read_labels_from_annot(
                "fsaverage", parc=mne_atlas, subjects_dir=subjects_dir, verbose=False
            )
            spatial_names = sorted(label.name for label in labels)
        except Exception as exc:
            logger.warning(f"Could not load ROI names for space '{space}': {exc}")

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]] = {}
    for ft in feature_types:
        X = np.concatenate(per_feat_X[ft], axis=0)[np.newaxis, :, :]
        meta_dict = {
            "feature_type": ft,
            "space": space,
            "inout_bounds": inout_bounds,
            "n_subjects": int(len(np.unique(groups))),
            "n_trials": int(len(y)),
            "n_in": total_in,
            "n_out": total_out,
            "input_git_hashes": sorted(input_git_hashes),
        }
        if spatial_names is not None:
            meta_dict["spatial_names"] = spatial_names
        if meta_dict["n_in"] < 20 or meta_dict["n_out"] < 20:
            logger.warning(
                f"{ft}: low trial counts (IN={meta_dict['n_in']}, OUT={meta_dict['n_out']})"
            )
        out[ft] = (X, y, groups, meta_dict)

    logger.info(
        f"Batched load done: {len(feature_types)} feature(s), "
        f"{len(np.unique(groups))} subjects, {len(y)} trials each "
        f"(IN={total_in}, OUT={total_out})"
    )
    return out


def load_all_features(
    feature_type: str,
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict,
    subjects: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load features from all subjects directly from npz files.

    Args:
        feature_type: Feature type to load (e.g., 'fooof_exponent', 'psd_alpha').
        space: Analysis space ('sensor', 'source', 'atlas').
        inout_bounds: Tuple of (lower_percentile, upper_percentile) for IN/OUT zones.
        config: Configuration dictionary.
        subjects: List of subject IDs to include. Defaults to None (all from config).

    Returns:
        Tuple containing:
        - X: Feature data, shape (1, n_trials, n_spatial) for single features
        - y: Labels (0=IN, 1=OUT), shape (n_trials,)
        - groups: Subject indices, shape (n_trials,)
        - metadata: Dictionary with trial info and loading parameters
    """
    from scipy import stats as scipy_stats

    # Get feature folder and file pattern
    feature_folder = get_feature_folder(config, feature_type, space)
    file_pattern, data_key = get_file_pattern(feature_type)

    if not feature_folder.exists():
        raise FileNotFoundError(f"Feature folder not found: {feature_folder}")

    logger.info(f"Loading features from: {feature_folder}")
    logger.info(f"Feature type: {feature_type}, data key: {data_key}")

    # Get subjects list
    if subjects is None:
        subjects = config["bids"]["subjects"]
    runs = config["bids"]["task_runs"]

    # Get frequency bands for PSD
    freq_bands = config.get("features", {}).get("frequency_bands", {})

    # Collect data from all subjects
    all_X = []
    all_y = []
    all_groups = []
    total_in, total_out = 0, 0
    input_git_hashes = set()  # Track git hashes of input files for reproducibility

    for subj_idx, subject in enumerate(tqdm(subjects, desc="Loading subjects", unit="subj")):
        subj_dir = feature_folder / f"sub-{subject}"
        if not subj_dir.exists():
            logger.debug(f"Subject folder not found: {subj_dir}")
            continue

        # Load all runs for this subject
        subj_data = []
        subj_vtc = []
        subj_task = []

        for run in runs:
            # Find file matching pattern
            pattern = f"sub-{subject}_*_run-{run}_*_{file_pattern}"
            files = list(subj_dir.glob(pattern))
            if not files:
                continue

            try:
                file_path = files[0]
                npz_data = np.load(file_path, allow_pickle=True)
                meta = npz_data["trial_metadata"].item()

                # Extract git hash from params.json if it exists
                params_file = file_path.with_name(file_path.stem + "_params.json")
                if params_file.exists():
                    with open(params_file) as f:
                        params = json.load(f)
                        if "git_hash" in params:
                            input_git_hashes.add(params["git_hash"])

                # Get the feature data
                if data_key in npz_data:
                    feat_data = npz_data[data_key]
                else:
                    logger.warning(f"Key '{data_key}' not found in {file_path.name}")
                    continue

                # For PSD or corrected PSD, extract band power
                if feature_type.startswith("psd_corrected_"):
                    band_name = feature_type.replace("psd_corrected_", "")
                elif feature_type.startswith("psd_"):
                    band_name = feature_type.replace("psd_", "")
                else:
                    band_name = None

                if band_name is not None:
                    if band_name in freq_bands:
                        freqs = npz_data["freqs"]
                        fmin, fmax = freq_bands[band_name]
                        freq_mask = (freqs >= fmin) & (freqs <= fmax)
                        # feat_data is (n_trials, n_channels, n_freqs) -> average over freq band
                        feat_data = np.mean(feat_data[:, :, freq_mask], axis=2)
                    else:
                        logger.warning(f"Band '{band_name}' not in config frequency_bands")
                        continue

                subj_data.append(feat_data)
                subj_vtc.append(np.array(meta["VTC_filtered"]))
                subj_task.append(np.array(meta["task"]))

            except Exception as e:
                logger.warning(f"Error loading {files[0]}: {e}")
                continue

        if not subj_data:
            continue

        # Concatenate runs
        subj_data = np.concatenate(subj_data, axis=0)
        subj_vtc = np.concatenate(subj_vtc)
        subj_task = np.concatenate(subj_task)

        # Compute IN/OUT bounds for this subject
        inbound = np.nanpercentile(subj_vtc, inout_bounds[0])
        outbound = np.nanpercentile(subj_vtc, inout_bounds[1])

        # Create masks
        task_mask = subj_task == "correct_commission"
        in_mask = task_mask & (subj_vtc <= inbound)
        out_mask = task_mask & (subj_vtc >= outbound)

        # Get IN and OUT trials
        in_data = subj_data[in_mask]
        out_data = subj_data[out_mask]

        n_in = len(in_data)
        n_out = len(out_data)

        if n_in == 0 or n_out == 0:
            logger.debug(f"Subject {subject}: no IN or OUT trials")
            continue

        # Add to collection
        all_X.append(in_data)
        all_X.append(out_data)
        all_y.extend([0] * n_in)  # IN = 0
        all_y.extend([1] * n_out)  # OUT = 1
        all_groups.extend([subj_idx] * (n_in + n_out))

        total_in += n_in
        total_out += n_out
        logger.debug(f"Subject {subject}: {n_in} IN, {n_out} OUT trials")

    if not all_X:
        raise ValueError("No data loaded from any subject")

    # Concatenate all data
    X = np.concatenate(all_X, axis=0)  # (n_trials, n_channels)
    y = np.array(all_y)
    groups = np.array(all_groups)

    # Reshape X to (1, n_trials, n_channels) for compatibility
    X = X[np.newaxis, :, :]

    # Check for multiple git hashes (data generated at different commits)
    input_git_hashes_list = sorted(input_git_hashes)
    if len(input_git_hashes_list) > 1:
        logger.warning(
            f"Input files were generated with {len(input_git_hashes_list)} different git commits: "
            f"{input_git_hashes_list}. This may indicate inconsistent preprocessing."
        )

    # Load spatial names for non-sensor spaces
    spatial_names = None
    if space not in ("sensor", "source"):
        # Atlas space: load ROI names from atlas definition
        try:
            import mne
            from code.source_reconstruction.apply_atlas import get_mne_atlas_name

            fsaverage_path = mne.datasets.fetch_fsaverage(verbose=False)
            subjects_dir = str(Path(fsaverage_path).parent)
            mne_atlas = get_mne_atlas_name(space)
            labels = mne.read_labels_from_annot(
                "fsaverage", parc=mne_atlas, subjects_dir=subjects_dir, verbose=False
            )
            spatial_names = sorted(label.name for label in labels)
            logger.info(f"Loaded {len(spatial_names)} ROI names from atlas '{space}'")
        except Exception as e:
            logger.warning(f"Could not load ROI names for space '{space}': {e}")

    # Create metadata
    metadata = {
        "feature_type": feature_type,
        "space": space,
        "inout_bounds": inout_bounds,
        "n_subjects": len(np.unique(groups)),
        "n_trials": len(y),
        "n_in": total_in,
        "n_out": total_out,
        "input_git_hashes": input_git_hashes_list,
    }
    if spatial_names is not None:
        metadata["spatial_names"] = spatial_names

    logger.info(
        f"Loaded {len(y)} trials from {metadata['n_subjects']} subjects "
        f"(IN: {total_in}, OUT: {total_out})"
    )

    # Validate sufficient trials
    if metadata["n_in"] < 20 or metadata["n_out"] < 20:
        logger.warning(
            f"Low trial counts: IN={metadata['n_in']}, OUT={metadata['n_out']}. "
            "Results may be unreliable."
        )

    return X, y, groups, metadata


def run_statistical_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_type: str = "paired_ttest",
    n_permutations: int = 10000,
    avg: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run statistical test, either on subject-averaged or trial-level data.

    Convention: OUT - IN direction. Positive t-values mean OUT > IN (red),
    negative t-values mean IN > OUT (blue).

    Two modes controlled by the `avg` parameter:

    - avg=False (default): Trial-level independent t-test (ttest_ind) with
      optional permutations. All trials are treated as independent observations.
      This matches the cc_saflow default behavior.

    - avg=True: Subject-level averaging first, then paired or independent t-test
      depending on test_type. This is the statistically correct approach for
      repeated-measures designs.

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels (0=IN, 1=OUT), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).
        test_type: Type of test ('paired_ttest', 'independent_ttest').
            When avg=False, ttest_ind is always used regardless of this setting.
        n_permutations: Number of permutations for trial-level ttest_ind
            (used only when avg=False). Set to 0 to disable.
        avg: If True, average trials within each subject before testing
            (subject-level statistics). If False, run trial-level statistics
            with ttest_ind (default, matches cc_saflow).

    Returns:
        Tuple containing:
        - contrast: Normalized contrast (IN-OUT)/OUT, shape (n_features, n_spatial)
        - tvals: T-values, shape (n_features, n_spatial)
        - pvals: P-values, shape (n_features, n_spatial)
    """
    from scipy import stats

    n_features = X.shape[0]

    if not avg:
        # --- Trial-level statistics (no subject averaging) ---
        logger.info("Running trial-level independent t-test (avg=False)")

        in_mask = y == 0
        out_mask = y == 1
        n_in = np.sum(in_mask)
        n_out = np.sum(out_mask)
        logger.info(f"Trial counts: IN={n_in}, OUT={n_out}")

        # Compute contrast from trial means
        # Convention: OUT - IN, so positive = OUT > IN (red), negative = IN > OUT (blue)
        grand_in = np.nanmean(X[:, in_mask, :], axis=1)
        grand_out = np.nanmean(X[:, out_mask, :], axis=1)
        contrast = (grand_out - grand_in) / np.abs(grand_in)

        tvals = np.zeros((n_features, X.shape[2]))
        pvals = np.zeros((n_features, X.shape[2]))

        perm_kwarg = {}
        if n_permutations > 0:
            perm_kwarg["permutations"] = n_permutations
            logger.info(f"Using {n_permutations} permutations")

        for feat_idx in range(n_features):
            # OUT first: positive t = OUT > IN (red), negative t = IN > OUT (blue)
            t, p = stats.ttest_ind(
                X[feat_idx, out_mask, :],
                X[feat_idx, in_mask, :],
                axis=0,
                **perm_kwarg,
            )
            tvals[feat_idx, :] = t
            pvals[feat_idx, :] = p

    else:
        # --- Subject-level statistics (average then test) ---
        logger.info(f"Running subject-level {test_type} (avg=True)")

        unique_subjects = np.unique(groups)

        subj_in = []
        subj_out = []

        for subj in unique_subjects:
            subj_mask = groups == subj
            in_mask = subj_mask & (y == 0)
            out_mask = subj_mask & (y == 1)

            if np.sum(in_mask) > 0 and np.sum(out_mask) > 0:
                mean_in = np.nanmean(X[:, in_mask, :], axis=1)
                mean_out = np.nanmean(X[:, out_mask, :], axis=1)
                subj_in.append(mean_in)
                subj_out.append(mean_out)

        subj_in = np.array(subj_in)
        subj_out = np.array(subj_out)

        logger.info(f"Computed subject-level averages for {len(subj_in)} subjects")

        # Convention: OUT - IN, so positive = OUT > IN (red), negative = IN > OUT (blue)
        grand_in = np.nanmean(subj_in, axis=0)
        grand_out = np.nanmean(subj_out, axis=0)
        contrast = (grand_out - grand_in) / np.abs(grand_in)

        tvals = np.zeros((n_features, X.shape[2]))
        pvals = np.zeros((n_features, X.shape[2]))

        if test_type == "paired_ttest":
            for feat_idx in range(n_features):
                # OUT first: positive t = OUT > IN (red), negative t = IN > OUT (blue)
                t, p = stats.ttest_rel(
                    subj_out[:, feat_idx, :],
                    subj_in[:, feat_idx, :],
                    axis=0,
                )
                tvals[feat_idx, :] = t
                pvals[feat_idx, :] = p

        elif test_type == "independent_ttest":
            for feat_idx in range(n_features):
                # OUT first: positive t = OUT > IN (red), negative t = IN > OUT (blue)
                t, p = stats.ttest_ind(
                    subj_out[:, feat_idx, :],
                    subj_in[:, feat_idx, :],
                    axis=0,
                )
                tvals[feat_idx, :] = t
                pvals[feat_idx, :] = p

        else:
            raise ValueError(f"Unknown test type: {test_type}")

    logger.info(
        f"Test complete: {np.sum(pvals < 0.05)} significant tests at alpha=0.05 (uncorrected)"
    )

    return contrast, tvals, pvals


def compute_all_effect_sizes(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute multiple effect size measures.

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels (0=IN, 1=OUT), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).

    Returns:
        Dictionary containing effect size arrays:
        - 'cohens_d': Cohen's d, shape (n_features, n_spatial)
        - 'hedges_g': Hedges' g, shape (n_features, n_spatial)
        - 'eta_squared': Eta-squared, shape (n_features, n_spatial)
    """
    logger.info("Computing effect sizes")

    effect_sizes = {}

    # Cohen's d
    effect_sizes["cohens_d"] = compute_cohens_d(X, y, groups)
    logger.debug(f"Cohen's d range: [{np.nanmin(effect_sizes['cohens_d']):.3f}, {np.nanmax(effect_sizes['cohens_d']):.3f}]")

    # Hedges' g (bias-corrected Cohen's d)
    effect_sizes["hedges_g"] = compute_hedges_g(X, y, groups)
    logger.debug(f"Hedges' g range: [{np.nanmin(effect_sizes['hedges_g']):.3f}, {np.nanmax(effect_sizes['hedges_g']):.3f}]")

    # Eta-squared
    effect_sizes["eta_squared"] = compute_eta_squared(X, y, groups)
    logger.debug(f"Eta-squared range: [{np.nanmin(effect_sizes['eta_squared']):.3f}, {np.nanmax(effect_sizes['eta_squared']):.3f}]")

    return effect_sizes


def apply_corrections(
    pvals: np.ndarray,
    tvals: np.ndarray,
    correction: str,
    alpha: float = 0.05,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    n_permutations: int = 1000,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multiple comparison correction.

    Args:
        pvals: P-values, shape (n_features, n_spatial).
        tvals: T-values, shape (n_features, n_spatial).
        correction: Correction method ('none', 'fdr', 'bonferroni', 'tmax').
        alpha: Significance threshold.
        X: Feature data for tmax (required if correction='tmax').
        y: Labels for tmax (required if correction='tmax').
        groups: Subject indices for tmax (required if correction='tmax').
        n_permutations: Number of permutations for tmax.
        n_jobs: Number of parallel jobs (-1 = all cores).

    Returns:
        Tuple of (corrected_pvals, sig_mask).
    """
    from scipy import stats

    logger.info(f"Applying {correction} correction")

    if correction == "none":
        corrected = pvals.copy()

    elif correction == "fdr":
        corrected = apply_fdr_correction(pvals, alpha, method="bh")

    elif correction == "bonferroni":
        corrected = apply_bonferroni_correction(pvals, alpha)

    elif correction == "tmax":
        if X is None or y is None or groups is None:
            logger.warning("Tmax correction requires X, y, groups. Falling back to FDR.")
            corrected = apply_fdr_correction(pvals, alpha, method="bh")
        else:
            # Tmax permutation test
            logger.info(f"Running tmax with {n_permutations} permutations (n_jobs={n_jobs})...")

            # Get unique subjects and compute subject-level means
            unique_subjects = np.unique(groups)
            n_subjects = len(unique_subjects)

            # Compute subject-level differences (OUT - IN)
            subj_diffs = []
            for subj in unique_subjects:
                subj_mask = groups == subj
                in_mask = subj_mask & (y == 0)
                out_mask = subj_mask & (y == 1)
                if np.sum(in_mask) > 0 and np.sum(out_mask) > 0:
                    mean_in = np.nanmean(X[:, in_mask, :], axis=1)
                    mean_out = np.nanmean(X[:, out_mask, :], axis=1)
                    subj_diffs.append(mean_out - mean_in)

            subj_diffs = np.array(subj_diffs)  # (n_subjects, n_features, n_spatial)

            # Helper function for single permutation
            def _single_permutation(seed):
                rng = np.random.RandomState(seed)
                flip = rng.choice([-1, 1], size=n_subjects)
                perm_diffs = subj_diffs * flip[:, np.newaxis, np.newaxis]
                perm_t, _ = stats.ttest_1samp(perm_diffs, 0, axis=0)
                return np.nanmax(np.abs(perm_t))

            # Build null distribution of max |t| with parallelization
            max_t_perm = Parallel(n_jobs=n_jobs)(
                delayed(_single_permutation)(seed)
                for seed in tqdm(range(n_permutations), desc="Permutations", unit="perm")
            )
            max_t_perm = np.array(max_t_perm)

            # Compute corrected p-values
            corrected = np.zeros_like(tvals)
            for i in range(tvals.shape[0]):
                for j in range(tvals.shape[1]):
                    if not np.isnan(tvals[i, j]):
                        corrected[i, j] = np.mean(max_t_perm >= np.abs(tvals[i, j]))
                    else:
                        corrected[i, j] = 1.0

    else:
        logger.warning(f"Unknown correction method: {correction}. Using none.")
        corrected = pvals.copy()

    sig_mask = corrected < alpha
    n_sig = np.sum(sig_mask)
    logger.info(f"{correction}: {n_sig} significant tests at alpha={alpha}")

    return corrected, sig_mask


def save_statistical_results(
    output_dir: Path,
    feature_type: str,
    inout_bounds: Tuple[int, int],
    test_type: str,
    contrast: np.ndarray,
    tvals: np.ndarray,
    pvals: np.ndarray,
    corrected_pvals: Dict[str, np.ndarray],
    effect_sizes: Dict[str, np.ndarray],
    metadata: Dict,
    config: Dict,
) -> None:
    """Save statistical results with provenance metadata.

    Saves two files:
    - {base_name}_results.npz: All numerical results (contrast, tvals, pvals, effect sizes)
    - {base_name}_metadata.json: Human-readable metadata and provenance

    Args:
        output_dir: Directory to save results.
        feature_type: Feature type.
        inout_bounds: IN/OUT bounds.
        test_type: Statistical test type.
        contrast: Contrast array.
        tvals: T-values.
        pvals: Uncorrected p-values.
        corrected_pvals: Dictionary of corrected p-values.
        effect_sizes: Dictionary of effect sizes.
        metadata: Metadata from data loading.
        config: Configuration dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    inout_str = inout_bounds_to_string(inout_bounds)
    base_name = f"feature-{feature_type}_inout-{inout_str}_test-{test_type}"

    # Build dictionary of all numerical results
    results_dict = {
        "contrast": contrast,
        "tvals": tvals,
        "pvals": pvals,
    }

    # Add corrected p-values with prefixed keys
    for correction, corrected in corrected_pvals.items():
        results_dict[f"pvals_corrected_{correction}"] = corrected

    # Add effect sizes with prefixed keys
    for effect_name, effect_array in effect_sizes.items():
        results_dict[f"effectsize_{effect_name}"] = effect_array

    # Save all results in a single npz file
    results_file = output_dir / f"{base_name}_results.npz"
    np.savez_compressed(results_file, **results_dict)
    logger.info(f"Saved results to {results_file}")

    # Save metadata as separate JSON for easy reading
    script_path = Path(__file__)
    metadata_out = {
        "feature_type": feature_type,
        "inout_bounds": list(inout_bounds),
        "test_type": test_type,
        "timestamp": datetime.now().isoformat(),
        "provenance": {
            "git_hash": get_git_hash(),
            "script_path": str(script_path.name),
            "script_hash": compute_file_hash(script_path),
            "input_data_git_hashes": metadata.get("input_git_hashes", []),
        },
        "data_metadata": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metadata.items()
            if k not in ["vtc", "task", "input_git_hashes"]  # Exclude large arrays and already-saved fields
        },
        "n_features": int(contrast.shape[0]),
        "n_spatial": int(contrast.shape[1]),
        "corrections_applied": list(corrected_pvals.keys()),
        "effect_sizes_computed": list(effect_sizes.keys()),
        "results_file": str(results_file.name),
    }

    metadata_file = output_dir / f"{base_name}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata_out, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run group-level statistical analysis for IN vs OUT states"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more feature types (e.g., 'psd_alpha psd_theta'). "
            "Multiple types from the same family (e.g. all PSD bands) load "
            "their shared source files only once."
        ),
    )
    parser.add_argument(
        "--space",
        type=str,
        default="sensor",
        help="Analysis space: 'sensor', 'source', or an atlas name (e.g., 'schaefer_400', 'aparc.a2009s')",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="paired_ttest",
        choices=["paired_ttest", "independent_ttest"],
        help="Statistical test type",
    )
    parser.add_argument(
        "--correction",
        type=str,
        default="fdr",
        choices=["none", "fdr", "bonferroni", "tmax"],
        help="Multiple comparison correction method",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for permutation test",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (-1 = all cores, 1 = no parallelization)",
    )
    parser.add_argument(
        "--average-trials",
        action="store_true",
        default=False,
        help="Average trials within each subject before testing (subject-level "
             "statistics with paired t-test). Default: trial-level ttest_ind.",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Get INOUT bounds from config
    inout_bounds = tuple(config["analysis"]["inout_bounds"])

    # Set up output directory
    data_root = Path(config["paths"]["data_root"])
    output_dir = data_root / config["paths"]["features"] / f"statistics_{args.space}"

    feature_types: List[str] = list(args.feature_type)

    logger.info("=" * 80)
    logger.info("GROUP-LEVEL STATISTICAL ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Feature types ({len(feature_types)}): {feature_types}")
    logger.info(f"Space: {args.space}")
    logger.info(f"Averaging: {'subject-level' if args.average_trials else 'trial-level (no averaging)'}")
    logger.info(f"Test: {args.test if args.average_trials else 'ttest_ind (trial-level)'}")
    logger.info(f"IN/OUT bounds: {inout_bounds}")
    logger.info(f"Corrections: {args.correction}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info("=" * 80)

    # Load all requested features in a single pass over the source files.
    logger.info("Loading features (batched)...")
    feature_blocks = load_all_features_batched(
        feature_types=feature_types,
        space=args.space,
        inout_bounds=inout_bounds,
        config=config,
    )

    for feat_idx, ft in enumerate(feature_types, start=1):
        X, y, groups, metadata = feature_blocks[ft]
        logger.info("")
        logger.info(f"[{feat_idx}/{len(feature_types)}] Testing: {ft}")
        logger.info("-" * 78)

        contrast, tvals, pvals = run_statistical_test(
            X=X,
            y=y,
            groups=groups,
            test_type=args.test,
            n_permutations=args.n_permutations,
            avg=args.average_trials,
        )

        corrected_pvals, sig_mask = apply_corrections(
            pvals=pvals,
            tvals=tvals,
            correction=args.correction,
            alpha=args.alpha,
            X=X,
            y=y,
            groups=groups,
            n_permutations=args.n_permutations,
            n_jobs=args.n_jobs,
        )
        corrected_pvals_dict = {args.correction: corrected_pvals}

        effect_sizes = compute_all_effect_sizes(X=X, y=y, groups=groups)

        save_statistical_results(
            output_dir=output_dir,
            feature_type=ft,
            inout_bounds=inout_bounds,
            test_type=args.test,
            contrast=contrast,
            tvals=tvals,
            pvals=pvals,
            corrected_pvals=corrected_pvals_dict,
            effect_sizes=effect_sizes,
            metadata=metadata,
            config=config,
        )

    # Visualization
    if args.visualize:
        logger.info("Generating visualizations...")
        from code.statistics.visualize_statistics import (
            plot_contrast_topomap,
            plot_pvalue_topomap,
            plot_effect_size_topomap,
        )

        plots_dir = data_root / config["paths"]["features"] / f"statistics_{args.space}" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots (placeholder for now)
        logger.info("Visualization not yet implemented")

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
