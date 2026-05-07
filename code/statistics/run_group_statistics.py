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
    drop_bad_trials: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
    """Load several features sharing the same source files in one pass.

    All `feature_types` must come from the same feature family (same folder,
    same file suffix) — e.g. all PSD bands, or all FOOOF parameters. Each
    (subject, run) npz is opened exactly once; per-feature slices are
    extracted in memory.

    IN/OUT thresholds are computed per subject from VTC over **all** trials
    (including those flagged ``bad_ar2``). Bad trials are dropped *after*
    masking so the percentile cuts stay anchored to the full distribution
    even though the noisy trials never enter the t-test.

    Returns a dict mapping `feature_type` -> `(X, y, groups, metadata)`,
    matching the shape returned by `load_all_features`. ``metadata``
    includes a ``per_subject`` breakdown of trial counts (n_in, n_out,
    n_bad_in, n_bad_out, n_mid, n_total) plus aggregate ``n_bad_excluded``.
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
    total_bad_excluded = 0
    per_subject: Dict[str, Dict[str, int]] = {}
    bad_metadata_present = False

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
        subj_bad: List[np.ndarray] = []

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

                # bad_ar2 is an opt-in tag added by the segmenter / backfill.
                # If absent, treat all trials as good and warn once.
                run_n = len(meta["VTC_filtered"])
                if "bad_ar2" in meta:
                    subj_bad.append(np.asarray(meta["bad_ar2"], dtype=bool))
                else:
                    subj_bad.append(np.zeros(run_n, dtype=bool))

            finally:
                npz_data.close()

        if not subj_vtc:
            continue

        # Trial-level concat per feature (same trial order across features).
        for ft in feature_types:
            subj_data[ft] = np.concatenate(subj_data[ft], axis=0)
        subj_vtc_arr = np.concatenate(subj_vtc)
        subj_task_arr = np.concatenate(subj_task)
        subj_bad_arr = np.concatenate(subj_bad) if subj_bad else np.zeros_like(subj_vtc_arr, dtype=bool)
        if subj_bad_arr.any():
            bad_metadata_present = True

        # Per-subject IN/OUT bounds — computed on ALL trials (including bads)
        # so the percentile cut is anchored to the full VTC distribution.
        # Bad trials are dropped *after* masking.
        inbound = np.nanpercentile(subj_vtc_arr, inout_bounds[0])
        outbound = np.nanpercentile(subj_vtc_arr, inout_bounds[1])
        task_mask = subj_task_arr == "correct_commission"
        in_mask_full = task_mask & (subj_vtc_arr <= inbound)
        out_mask_full = task_mask & (subj_vtc_arr >= outbound)
        mid_mask_full = task_mask & ~in_mask_full & ~out_mask_full

        # Bookkeeping before bad-filter
        n_in_pre = int(in_mask_full.sum())
        n_out_pre = int(out_mask_full.sum())
        n_mid = int(mid_mask_full.sum())
        n_bad_in = int((in_mask_full & subj_bad_arr).sum()) if drop_bad_trials else 0
        n_bad_out = int((out_mask_full & subj_bad_arr).sum()) if drop_bad_trials else 0

        if drop_bad_trials:
            in_mask = in_mask_full & ~subj_bad_arr
            out_mask = out_mask_full & ~subj_bad_arr
        else:
            in_mask = in_mask_full
            out_mask = out_mask_full

        n_in = int(in_mask.sum())
        n_out = int(out_mask.sum())
        per_subject[subject] = {
            "n_total": int(len(subj_vtc_arr)),
            "n_in": n_in,
            "n_out": n_out,
            "n_mid": n_mid,
            "n_bad_in": n_bad_in,
            "n_bad_out": n_bad_out,
            "n_in_before_bad_filter": n_in_pre,
            "n_out_before_bad_filter": n_out_pre,
        }

        if n_in == 0 or n_out == 0:
            logger.warning(
                f"sub-{subject}: dropped from analysis — n_in={n_in}, n_out={n_out} "
                f"(after bad-filter: bad_in={n_bad_in}, bad_out={n_bad_out})"
            )
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
        total_bad_excluded += n_bad_in + n_bad_out

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

    if drop_bad_trials and not bad_metadata_present:
        logger.warning(
            "drop_bad_trials=True but no bad_ar2 column was found in any "
            "trial_metadata — feature files predate the bad-trial flag. "
            "Run `python -m code.utils.backfill_bad_trials` to backfill, "
            "or recompute features. Falling back to keeping all trials."
        )

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
            "n_bad_excluded": total_bad_excluded,
            "drop_bad_trials": bool(drop_bad_trials),
            "bad_ar2_metadata_present": bool(bad_metadata_present),
            "per_subject": per_subject,
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
        f"(IN={total_in}, OUT={total_out}, bad_excluded={total_bad_excluded})"
    )
    return out


def load_all_features(
    feature_type: str,
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict,
    subjects: Optional[List[str]] = None,
    drop_bad_trials: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Single-feature wrapper around :func:`load_all_features_batched`.

    Behaviour matches the batched loader (drops trials flagged ``bad_ar2``
    by default, computes per-subject IN/OUT thresholds on all trials).
    """
    blocks = load_all_features_batched(
        feature_types=[feature_type],
        space=space,
        inout_bounds=inout_bounds,
        config=config,
        subjects=subjects,
        drop_bad_trials=drop_bad_trials,
    )
    return blocks[feature_type]


def _load_all_features_legacy_unused(
    feature_type: str,
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict,
    subjects: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Legacy single-feature loader. Kept only as documentation for the
    pre-bad-filter behaviour; not used anywhere. Will be removed once the
    git history confirms no external imports."""
    raise NotImplementedError(
        "load_all_features now delegates to load_all_features_batched. "
        "If you need the pre-filter behaviour, pass drop_bad_trials=False."
    )

    # --- unreachable original implementation kept below for reference ---
    from scipy import stats as scipy_stats  # type: ignore[unused-ignore]

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


def _subject_aggregate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    aggregate: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce trial-level X to (n_subjects, n_features, n_spatial) per condition.

    ``aggregate`` is ``"median"`` (default) or ``"mean"``. Subjects missing
    one of the conditions are dropped.
    """
    if aggregate not in ("median", "mean"):
        raise ValueError(f"Unknown aggregate '{aggregate}'")
    reducer = np.nanmedian if aggregate == "median" else np.nanmean

    unique_subjects = np.unique(groups)
    subj_in: List[np.ndarray] = []
    subj_out: List[np.ndarray] = []
    kept = []
    for subj in unique_subjects:
        subj_mask = groups == subj
        in_mask = subj_mask & (y == 0)
        out_mask = subj_mask & (y == 1)
        if not (np.any(in_mask) and np.any(out_mask)):
            continue
        subj_in.append(reducer(X[:, in_mask, :], axis=1))
        subj_out.append(reducer(X[:, out_mask, :], axis=1))
        kept.append(subj)
    return np.array(subj_in), np.array(subj_out), np.array(kept)


def run_statistical_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_type: str = "paired_ttest",
    n_permutations: int = 10000,
    single_trials: bool = False,
    aggregate: str = "median",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run group-level statistical test.

    Convention: OUT - IN direction. Positive t-values mean OUT > IN (red),
    negative t-values mean IN > OUT (blue).

    Default mode: subject-level paired t-test on subject **medians** per
    condition. This matches cc_saflow's `simple_contrast` family of tests
    and is statistically appropriate for the repeated-measures design.

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels (0=IN, 1=OUT), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).
        test_type: 'paired_ttest' (default) or 'independent_ttest'. Only
            applies in subject-level mode.
        n_permutations: Permutations for trial-level ttest_ind. Only used
            when ``single_trials=True``. Set to 0 to disable.
        single_trials: If True, treat all trials as independent observations
            and run ttest_ind (legacy behaviour). Default False.
        aggregate: 'median' (default) or 'mean'. The per-subject aggregation
            statistic used in subject-level mode.

    Returns:
        contrast, tvals, pvals — each of shape (n_features, n_spatial).
    """
    from scipy import stats

    n_features = X.shape[0]

    if single_trials:
        # --- Trial-level statistics (no subject aggregation, legacy) ---
        logger.info(
            "Running trial-level independent t-test (single_trials=True). "
            "Note: this mixes within- and between-subject variance and "
            "is statistically NOT recommended for repeated-measures designs."
        )

        in_mask = y == 0
        out_mask = y == 1
        n_in = int(np.sum(in_mask))
        n_out = int(np.sum(out_mask))
        logger.info(f"Trial counts: IN={n_in}, OUT={n_out}")

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
            t, p = stats.ttest_ind(
                X[feat_idx, out_mask, :],
                X[feat_idx, in_mask, :],
                axis=0,
                **perm_kwarg,
            )
            tvals[feat_idx, :] = t
            pvals[feat_idx, :] = p

    else:
        # --- Subject-level statistics (median per subject + paired test) ---
        logger.info(
            f"Running subject-level {test_type} on per-subject {aggregate}s "
            f"(single_trials=False)"
        )

        subj_in, subj_out, kept_subjects = _subject_aggregate(
            X, y, groups, aggregate=aggregate
        )
        # subj_in / subj_out shape: (n_subjects, n_features, n_spatial)
        logger.info(
            f"Computed subject-level {aggregate}s for {len(kept_subjects)} "
            f"of {len(np.unique(groups))} subjects"
        )

        # Convention: OUT - IN
        grand_in = np.nanmedian(subj_in, axis=0) if aggregate == "median" else np.nanmean(subj_in, axis=0)
        grand_out = np.nanmedian(subj_out, axis=0) if aggregate == "median" else np.nanmean(subj_out, axis=0)
        # Guard the (corrected-PSD) case where grand_in lives near zero
        # (residual log-power) — divide by max(|in|, eps) to avoid blow-ups.
        denom = np.maximum(np.abs(grand_in), 1e-12)
        contrast = (grand_out - grand_in) / denom

        tvals = np.zeros((n_features, X.shape[2]))
        pvals = np.zeros((n_features, X.shape[2]))

        if test_type == "paired_ttest":
            for feat_idx in range(n_features):
                t, p = stats.ttest_rel(
                    subj_out[:, feat_idx, :],
                    subj_in[:, feat_idx, :],
                    axis=0,
                )
                tvals[feat_idx, :] = t
                pvals[feat_idx, :] = p
        elif test_type == "independent_ttest":
            for feat_idx in range(n_features):
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
    aggregate: str = "median",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multiple comparison correction.

    For ``tmax`` the null distribution is built on subject-level paired
    differences of the per-subject ``aggregate`` (median by default), so
    it is the matched FWER control for the default subject-level paired
    t-test in :func:`run_statistical_test`. (When the main test is run in
    ``single_trials`` mode, prefer FDR — the tmax null built here is on a
    different scale than a trial-level ``ttest_ind`` t.)

    Args:
        pvals, tvals: shape (n_features, n_spatial).
        correction: 'none', 'fdr', 'bonferroni', 'tmax'.
        X, y, groups: trial-level data + labels + subject indices.
        n_permutations, n_jobs: tmax knobs.
        aggregate: 'median' (default) or 'mean' — must match the main
            test's aggregation to keep the test stat and the null
            distribution on the same scale.

    Returns:
        (corrected_pvals, sig_mask).
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
            logger.info(
                f"Running tmax with {n_permutations} permutations on per-subject "
                f"paired diffs of {aggregate}s (n_jobs={n_jobs})..."
            )

            subj_in, subj_out, _kept = _subject_aggregate(
                X, y, groups, aggregate=aggregate
            )
            subj_diffs = subj_out - subj_in  # (n_subjects, n_features, n_spatial)
            n_subjects = subj_diffs.shape[0]

            def _single_permutation(seed):
                rng = np.random.RandomState(seed)
                flip = rng.choice([-1, 1], size=n_subjects)
                perm_diffs = subj_diffs * flip[:, np.newaxis, np.newaxis]
                perm_t, _ = stats.ttest_1samp(perm_diffs, 0, axis=0)
                return np.nanmax(np.abs(perm_t))

            max_t_perm = Parallel(n_jobs=n_jobs)(
                delayed(_single_permutation)(seed)
                for seed in tqdm(range(n_permutations), desc="Permutations", unit="perm")
            )
            max_t_perm = np.array(max_t_perm)

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
    analysis_mode: str = "subject-trial-median",
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
    # Bake the analysis path into the filename so Path-1 (subject-spectrum)
    # and Path-2 (subject-trial-median / single-trials) results never
    # overwrite each other for the same feature_type.
    mode_tag = analysis_mode.replace("subject-", "subj-")
    base_name = (
        f"feature-{feature_type}_inout-{inout_str}_test-{test_type}_path-{mode_tag}"
    )

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
        "--analysis-mode",
        type=str,
        default="subject-spectrum",
        choices=["subject-spectrum", "subject-trial-median", "single-trials"],
        help=(
            "Analysis path:\n"
            "  subject-spectrum (default, Path 1): per (subj, run, cond) "
            "median PSD across good trials, fit FOOOF on the aggregated "
            "spectrum, derive psd_<band> / psd_corrected_<band> / "
            "fooof_<param>; mean across runs; paired t-test. Cleaner FOOOF "
            "fits than per-trial. Required for psd_corrected_*.\n"
            "  subject-trial-median: per-trial features (precomputed), "
            "median-aggregate per subject × condition, paired t-test. Use "
            "for complexity (always) and for cross-checking PSD-derived "
            "stats against per-trial FOOOF.\n"
            "  single-trials: legacy trial-level ttest_ind. Mixes within- "
            "and between-subject variance. Not recommended."
        ),
    )
    parser.add_argument(
        "--single-trials",
        action="store_true",
        default=False,
        help="DEPRECATED alias for --analysis-mode single-trials.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Per-subject aggregation statistic for subject-trial-median "
             "mode (ignored otherwise). 'median' (default) matches cc_saflow.",
    )
    parser.add_argument(
        "--keep-bad-trials",
        action="store_true",
        default=False,
        help="Skip the bad_ar2 filter (keeps trials inside autoreject-rejected "
             "BAD_AR2 windows). Default is to drop them.",
    )

    args = parser.parse_args()

    # Honour the deprecated alias.
    if args.single_trials and args.analysis_mode == "subject-spectrum":
        args.analysis_mode = "single-trials"

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
    logger.info(f"Analysis mode: {args.analysis_mode}")
    if args.analysis_mode == "single-trials":
        logger.info("  (trial-level ttest_ind — legacy)")
    elif args.analysis_mode == "subject-trial-median":
        logger.info(f"  per-subject {args.aggregate} of per-trial features → paired {args.test}")
    elif args.analysis_mode == "subject-spectrum":
        logger.info("  per (subj, run, cond) median PSD → FOOOF on aggregate → mean across runs → paired t-test")
    logger.info(f"IN/OUT bounds: {inout_bounds}")
    logger.info(f"Drop bad_ar2 trials: {not args.keep_bad_trials}")
    logger.info(f"Corrections: {args.correction}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info("=" * 80)

    if args.analysis_mode == "subject-spectrum":
        # Path 1: only PSD-derived families are meaningful here. Reject
        # complexity_* up front so the user is not silently shifted off-mode.
        unsupported = [
            ft for ft in feature_types
            if not (ft.startswith("psd_") or ft.startswith("fooof_"))
        ]
        if unsupported:
            raise SystemExit(
                f"--analysis-mode subject-spectrum only supports psd_*, "
                f"psd_corrected_* and fooof_*. Got non-PSD: {unsupported}. "
                f"Re-run those with --analysis-mode subject-trial-median."
            )
        from code.statistics.subject_spectrum import load_subject_spectrum_features
        logger.info("Loading features (subject-spectrum)...")
        feature_blocks = load_subject_spectrum_features(
            feature_types=feature_types,
            space=args.space,
            inout_bounds=inout_bounds,
            config=config,
            drop_bad_trials=not args.keep_bad_trials,
            n_jobs=args.n_jobs,
        )
    else:
        logger.info("Loading features (batched)...")
        feature_blocks = load_all_features_batched(
            feature_types=feature_types,
            space=args.space,
            inout_bounds=inout_bounds,
            config=config,
            drop_bad_trials=not args.keep_bad_trials,
        )

    for feat_idx, ft in enumerate(feature_types, start=1):
        X, y, groups, metadata = feature_blocks[ft]
        logger.info("")
        logger.info(f"[{feat_idx}/{len(feature_types)}] Testing: {ft}")
        logger.info("-" * 78)

        # In subject-spectrum mode, X already carries one row per
        # (subject, condition); aggregate is a no-op (each "trial" is
        # already a subject-level value).
        is_subject_spectrum = args.analysis_mode == "subject-spectrum"
        eff_aggregate = "mean" if is_subject_spectrum else args.aggregate
        eff_single_trials = args.analysis_mode == "single-trials"

        contrast, tvals, pvals = run_statistical_test(
            X=X,
            y=y,
            groups=groups,
            test_type=args.test,
            n_permutations=args.n_permutations,
            single_trials=eff_single_trials,
            aggregate=eff_aggregate,
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
            aggregate=eff_aggregate,
        )
        corrected_pvals_dict = {args.correction: corrected_pvals}

        effect_sizes = compute_all_effect_sizes(X=X, y=y, groups=groups)

        # Surface the test configuration in the sidecar so each results
        # file is self-describing (mode, aggregator, bad-filter status).
        metadata = dict(metadata)
        metadata["analysis_mode"] = args.analysis_mode
        metadata["aggregate"] = eff_aggregate if not eff_single_trials else None
        metadata["test"] = (
            "ttest_ind" if eff_single_trials else args.test
        )
        metadata["drop_bad_trials"] = bool(not args.keep_bad_trials)

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
            analysis_mode=args.analysis_mode,
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
