"""Group-level statistical analysis for IN vs OUT attentional states.

This script performs group-level statistical comparisons between IN and OUT
attentional states using various statistical tests and multiple comparison
corrections.

Usage:
    python -m code.statistics.run_group_statistics \
        --feature-type fooof_exponent \
        --space sensor \
        --test paired_ttest \
        --correction fdr bonferroni
"""

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

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
        feature_type: Feature type (e.g., 'fooof_exponent', 'psd_alpha', 'lzc').
        space: Analysis space ('sensor', 'source', 'atlas').

    Returns:
        Path to feature folder.
    """
    data_root = Path(config["paths"]["data_root"])
    processed = data_root / config["paths"]["processed"]

    # Map feature types to folder names
    if feature_type.startswith("fooof_"):
        folder_name = f"features_fooof_{space}"
    elif feature_type.startswith("psd_"):
        folder_name = f"features_psd_{space}"
    elif feature_type.startswith("lzc"):
        folder_name = f"features_lzc_{space}"
    else:
        # Default to treating it as the feature type name
        folder_name = f"features_{feature_type}_{space}"

    return processed / folder_name


def load_all_features(
    feature_type: str,
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict,
    subjects: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load features from all subjects using utils/data_loading.py.

    Args:
        feature_type: Feature type to load (e.g., 'fooof_exponent', 'psd_alpha', 'lzc').
        space: Analysis space ('sensor', 'source', 'atlas').
        inout_bounds: Tuple of (lower_percentile, upper_percentile) for IN/OUT zones.
        config: Configuration dictionary.
        subjects: List of subject IDs to include. Defaults to None (all subjects).

    Returns:
        Tuple containing:
        - X: Feature data, shape (n_features, n_trials, n_spatial)
        - y: Labels (0=IN, 1=OUT), shape (n_trials,)
        - groups: Subject indices, shape (n_trials,)
        - metadata: Dictionary with trial info and loading parameters
    """
    # Get feature folder
    feature_folder = get_feature_folder(config, feature_type, space)

    if not feature_folder.exists():
        raise FileNotFoundError(f"Feature folder not found: {feature_folder}")

    logger.info(f"Loading features from: {feature_folder}")

    # Map INOUT bounds to string format used in data_loading
    inout_str = f"INOUT_{inout_bounds[0]}{inout_bounds[1]}"

    # Determine which feature to load from the feature_type
    # feature_type examples: 'fooof_exponent', 'psd_alpha', 'lzc_antropy_median'
    if feature_type.startswith("fooof_"):
        # FOOOF features: 'exponent', 'offset', 'knee', 'r_squared'
        feature_name = feature_type.replace("fooof_", "")
        if feature_name == "exponent":
            feature_key = "slope"
        else:
            feature_key = feature_name
    elif feature_type.startswith("psd_"):
        # PSD features: 'alpha', 'theta', 'beta', etc.
        feature_key = "psd"
    elif feature_type.startswith("lzc"):
        feature_key = "lzc"
    else:
        feature_key = feature_type

    # Load features using existing utility
    X, y, groups, metadata = load_features(
        feature_folder=feature_folder,
        feature=feature_key,
        splitby="inout",
        inout=inout_str,
        remove_errors=True,
        get_task=["correct_commission"],
    )

    # Add loading parameters to metadata
    metadata["feature_type"] = feature_type
    metadata["space"] = space
    metadata["inout_bounds"] = inout_bounds
    metadata["n_subjects"] = len(np.unique(groups))
    metadata["n_trials"] = len(y)
    metadata["n_in"] = np.sum(y == 0)
    metadata["n_out"] = np.sum(y == 1)

    logger.info(
        f"Loaded {len(y)} trials from {len(np.unique(groups))} subjects "
        f"(IN: {np.sum(y == 0)}, OUT: {np.sum(y == 1)})"
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run statistical test using existing utils/statistics.py functions.

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels (0=IN, 1=OUT), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).
        test_type: Type of test ('paired_ttest', 'independent_ttest', 'permutation').
        n_permutations: Number of permutations for permutation test.

    Returns:
        Tuple containing:
        - contrast: Normalized contrast (IN-OUT)/OUT, shape (n_features, n_spatial)
        - tvals: T-values, shape (n_features, n_spatial)
        - pvals: P-values, shape (n_features, n_spatial)
    """
    logger.info(f"Running {test_type} statistical test")

    if test_type == "paired_ttest":
        # Within-subject paired t-test
        contrast, tvals, pvals = subject_contrast(X, y)

    elif test_type == "independent_ttest":
        # Independent samples t-test (between subjects)
        contrast, tvals, pvals = simple_contrast(X, y, groups)

    elif test_type == "permutation":
        # Permutation test (not yet implemented in utils/statistics.py)
        logger.warning(
            "Permutation test not fully implemented. Using paired t-test instead."
        )
        contrast, tvals, pvals = subject_contrast(X, y)

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
    corrections: List[str],
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Apply multiple comparison corrections.

    Args:
        pvals: P-values, shape (n_features, n_spatial).
        corrections: List of correction methods ('fdr', 'bonferroni', 'tmax').
        alpha: Significance threshold.

    Returns:
        Dictionary mapping correction method to corrected p-values.
    """
    corrected_pvals = {}

    for correction in corrections:
        logger.info(f"Applying {correction} correction")

        if correction == "fdr":
            corrected = apply_fdr_correction(pvals, alpha, method="bh")
        elif correction == "bonferroni":
            corrected = apply_bonferroni_correction(pvals, alpha)
        elif correction == "tmax":
            logger.warning("Tmax correction requires permutation distribution. Skipping.")
            continue
        else:
            logger.warning(f"Unknown correction method: {correction}")
            continue

        n_sig = np.sum(corrected < alpha)
        corrected_pvals[correction] = corrected
        logger.info(f"{correction}: {n_sig} significant tests at alpha={alpha}")

    return corrected_pvals


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

    # Save contrast
    contrast_file = output_dir / f"{base_name}_contrast.npz"
    np.savez_compressed(contrast_file, contrast=contrast)
    logger.info(f"Saved contrast to {contrast_file}")

    # Save t-values
    tvals_file = output_dir / f"{base_name}_tvals.npz"
    np.savez_compressed(tvals_file, tvals=tvals)
    logger.info(f"Saved t-values to {tvals_file}")

    # Save uncorrected p-values
    pvals_file = output_dir / f"{base_name}_pvals.npz"
    np.savez_compressed(pvals_file, pvals=pvals)
    logger.info(f"Saved uncorrected p-values to {pvals_file}")

    # Save corrected p-values
    for correction, corrected in corrected_pvals.items():
        corrected_file = output_dir / f"{base_name}_pvals-corrected-{correction}.npz"
        np.savez_compressed(corrected_file, pvals=corrected)
        logger.info(f"Saved {correction}-corrected p-values to {corrected_file}")

    # Save effect sizes
    for effect_name, effect_array in effect_sizes.items():
        effect_file = output_dir / f"{base_name}_effectsize-{effect_name}.npz"
        np.savez_compressed(effect_file, effect_size=effect_array)
        logger.info(f"Saved {effect_name} to {effect_file}")

    # Save metadata
    metadata_out = {
        "feature_type": feature_type,
        "inout_bounds": list(inout_bounds),
        "test_type": test_type,
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "data_metadata": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metadata.items()
            if k not in ["vtc", "task"]  # Exclude large arrays
        },
        "n_features": int(contrast.shape[0]),
        "n_spatial": int(contrast.shape[1]),
        "corrections_applied": list(corrected_pvals.keys()),
        "effect_sizes_computed": list(effect_sizes.keys()),
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
        required=True,
        help="Feature type (e.g., fooof_exponent, psd_alpha, lzc)",
    )
    parser.add_argument(
        "--space",
        type=str,
        default="sensor",
        choices=["sensor", "source", "atlas"],
        help="Analysis space",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="paired_ttest",
        choices=["paired_ttest", "independent_ttest", "permutation"],
        help="Statistical test type",
    )
    parser.add_argument(
        "--correction",
        type=str,
        nargs="+",
        default=["fdr", "bonferroni"],
        help="Correction methods (can specify multiple)",
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

    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Get INOUT bounds from config
    inout_bounds = tuple(config["analysis"]["inout_bounds"])

    # Set up output directory
    data_root = Path(config["paths"]["data_root"])
    output_dir = data_root / config["paths"]["processed"] / f"statistics_{args.space}" / "group"

    logger.info("=" * 80)
    logger.info("GROUP-LEVEL STATISTICAL ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Feature type: {args.feature_type}")
    logger.info(f"Space: {args.space}")
    logger.info(f"Test: {args.test}")
    logger.info(f"IN/OUT bounds: {inout_bounds}")
    logger.info(f"Corrections: {args.correction}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info("=" * 80)

    # Load features
    logger.info("Loading features...")
    X, y, groups, metadata = load_all_features(
        feature_type=args.feature_type,
        space=args.space,
        inout_bounds=inout_bounds,
        config=config,
    )

    # Run statistical test
    logger.info("Running statistical test...")
    contrast, tvals, pvals = run_statistical_test(
        X=X,
        y=y,
        groups=groups,
        test_type=args.test,
        n_permutations=args.n_permutations,
    )

    # Apply corrections
    logger.info("Applying multiple comparison corrections...")
    corrected_pvals = apply_corrections(
        pvals=pvals,
        corrections=args.correction,
        alpha=args.alpha,
    )

    # Compute effect sizes
    logger.info("Computing effect sizes...")
    effect_sizes = compute_all_effect_sizes(
        X=X,
        y=y,
        groups=groups,
    )

    # Save results
    logger.info("Saving results...")
    save_statistical_results(
        output_dir=output_dir,
        feature_type=args.feature_type,
        inout_bounds=inout_bounds,
        test_type=args.test,
        contrast=contrast,
        tvals=tvals,
        pvals=pvals,
        corrected_pvals=corrected_pvals,
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

        plots_dir = data_root / config["paths"]["processed"] / f"statistics_{args.space}" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots (placeholder for now)
        logger.info("Visualization not yet implemented")

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
