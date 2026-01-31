"""Visualization functions for statistical results.

This module provides plotting functions for:
- Contrast topomaps
- P-value maps with significance markers
- Effect size topomaps
- Statistical result summaries

All functions use existing visualization utilities and follow saflow conventions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_contrast_topomap(
    contrast: np.ndarray,
    title: str = "IN vs OUT Contrast",
    output_file: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
) -> None:
    """Plot contrast as topomap using existing visualization utilities.

    Args:
        contrast: Contrast array, shape (n_features, n_spatial).
        title: Plot title.
        output_file: Path to save figure. If None, display interactively.
        vmin: Minimum value for colormap. Defaults to None (auto).
        vmax: Maximum value for colormap. Defaults to None (auto).
        cmap: Colormap name. Defaults to 'RdBu_r'.

    Examples:
        >>> plot_contrast_topomap(contrast, title="FOOOF Exponent")
    """
    logger.info(f"Plotting contrast topomap: {title}")

    # Placeholder implementation
    # TODO: Integrate with code.utils.visualization.grid_topoplot
    logger.warning("Contrast topomap plotting not yet fully implemented")

    if output_file:
        logger.info(f"Would save to: {output_file}")


def plot_pvalue_topomap(
    pvals: np.ndarray,
    title: str = "P-values",
    alpha: float = 0.05,
    output_file: Optional[Path] = None,
    mark_significant: bool = True,
) -> None:
    """Plot p-values as topomap with significance markers.

    Args:
        pvals: P-value array, shape (n_features, n_spatial).
        title: Plot title.
        alpha: Significance threshold for markers.
        output_file: Path to save figure. If None, display interactively.
        mark_significant: Whether to mark significant tests.

    Examples:
        >>> plot_pvalue_topomap(pvals, alpha=0.05, mark_significant=True)
    """
    logger.info(f"Plotting p-value topomap: {title}")

    # Placeholder implementation
    logger.warning("P-value topomap plotting not yet fully implemented")

    n_significant = np.sum(pvals < alpha)
    logger.info(f"Significant tests: {n_significant}/{pvals.size}")

    if output_file:
        logger.info(f"Would save to: {output_file}")


def plot_effect_size_topomap(
    effect_size: np.ndarray,
    effect_name: str = "Cohen's d",
    title: Optional[str] = None,
    output_file: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Plot effect size as topomap.

    Args:
        effect_size: Effect size array, shape (n_features, n_spatial).
        effect_name: Name of effect size measure.
        title: Plot title. Defaults to effect_name.
        output_file: Path to save figure. If None, display interactively.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.

    Examples:
        >>> plot_effect_size_topomap(cohens_d, effect_name="Cohen's d")
    """
    if title is None:
        title = effect_name

    logger.info(f"Plotting effect size topomap: {title}")

    # Placeholder implementation
    logger.warning("Effect size topomap plotting not yet fully implemented")

    logger.info(
        f"Effect size range: [{np.nanmin(effect_size):.3f}, {np.nanmax(effect_size):.3f}]"
    )

    if output_file:
        logger.info(f"Would save to: {output_file}")


def plot_statistical_summary(
    contrast: np.ndarray,
    pvals: np.ndarray,
    corrected_pvals: Dict[str, np.ndarray],
    effect_sizes: Dict[str, np.ndarray],
    alpha: float = 0.05,
    output_file: Optional[Path] = None,
) -> None:
    """Create a summary figure with multiple statistical visualizations.

    Args:
        contrast: Contrast array.
        pvals: Uncorrected p-values.
        corrected_pvals: Dictionary of corrected p-values.
        effect_sizes: Dictionary of effect sizes.
        alpha: Significance threshold.
        output_file: Path to save figure.

    Examples:
        >>> plot_statistical_summary(
        ...     contrast, pvals, corrected_pvals, effect_sizes
        ... )
    """
    logger.info("Creating statistical summary figure")

    # Placeholder implementation
    logger.warning("Statistical summary plotting not yet fully implemented")

    # Log summary statistics
    logger.info(f"Uncorrected: {np.sum(pvals < alpha)} significant tests")
    for method, corr_pvals in corrected_pvals.items():
        n_sig = np.sum(corr_pvals < alpha)
        logger.info(f"{method}: {n_sig} significant tests")

    if output_file:
        logger.info(f"Would save to: {output_file}")


def plot_correction_comparison(
    pvals: np.ndarray,
    corrected_pvals: Dict[str, np.ndarray],
    alpha: float = 0.05,
    output_file: Optional[Path] = None,
) -> None:
    """Compare different multiple comparison correction methods.

    Creates a bar plot showing the number of significant tests for each
    correction method.

    Args:
        pvals: Uncorrected p-values.
        corrected_pvals: Dictionary of corrected p-values.
        alpha: Significance threshold.
        output_file: Path to save figure.

    Examples:
        >>> plot_correction_comparison(pvals, corrected_pvals, alpha=0.05)
    """
    logger.info("Creating correction method comparison plot")

    # Count significant tests for each method
    counts = {"uncorrected": np.sum(pvals < alpha)}
    for method, corr_pvals in corrected_pvals.items():
        counts[method] = np.sum(corr_pvals < alpha)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(counts.keys())
    values = list(counts.values())

    ax.bar(methods, values, color="steelblue", alpha=0.7)
    ax.set_xlabel("Correction Method", fontsize=12)
    ax.set_ylabel("Number of Significant Tests", fontsize=12)
    ax.set_title(f"Comparison of Correction Methods (α={alpha})", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Rotate x-labels if needed
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved correction comparison to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_effect_size_histogram(
    effect_sizes: Dict[str, np.ndarray],
    output_file: Optional[Path] = None,
) -> None:
    """Plot histograms of effect sizes.

    Args:
        effect_sizes: Dictionary mapping effect size name to array.
        output_file: Path to save figure.

    Examples:
        >>> plot_effect_size_histogram(effect_sizes)
    """
    logger.info("Creating effect size histograms")

    n_effects = len(effect_sizes)
    fig, axes = plt.subplots(1, n_effects, figsize=(6 * n_effects, 5))

    if n_effects == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, effect_sizes.items()):
        # Flatten and remove NaNs
        values_flat = values.flatten()
        values_clean = values_flat[~np.isnan(values_flat)]

        # Plot histogram
        ax.hist(values_clean, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero effect")
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{name} Distribution", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        # Add summary statistics
        mean_val = np.nanmean(values_clean)
        median_val = np.nanmedian(values_clean)
        ax.text(
            0.02,
            0.98,
            f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved effect size histograms to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_pvalue_histogram(
    pvals: np.ndarray,
    corrected_pvals: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 0.05,
    output_file: Optional[Path] = None,
) -> None:
    """Plot histogram of p-values.

    A uniform distribution indicates proper null hypothesis behavior.
    Enrichment near zero suggests true effects.

    Args:
        pvals: Uncorrected p-values.
        corrected_pvals: Optional dictionary of corrected p-values.
        alpha: Significance threshold to mark.
        output_file: Path to save figure.

    Examples:
        >>> plot_pvalue_histogram(pvals, alpha=0.05)
    """
    logger.info("Creating p-value histogram")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Flatten and remove NaNs
    pvals_flat = pvals.flatten()
    pvals_clean = pvals_flat[~np.isnan(pvals_flat)]

    # Plot histogram
    ax.hist(pvals_clean, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(alpha, color="red", linestyle="--", linewidth=2, label=f"α={alpha}")
    ax.set_xlabel("P-value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("P-value Distribution", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # Add expected uniform distribution line
    n_total = len(pvals_clean)
    expected_per_bin = n_total / 50
    ax.axhline(expected_per_bin, color="gray", linestyle=":", linewidth=2, label="Uniform")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved p-value histogram to {output_file}")
    else:
        plt.show()

    plt.close()
