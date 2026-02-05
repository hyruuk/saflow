"""Visualization utilities for saflow.

This module provides utilities for:
- Topographic plotting (grid topoplots)
- MEG/EEG data visualization
- Statistical result visualization

All functions use matplotlib and MNE visualization tools.
"""

import logging
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mne
import numpy as np

logger = logging.getLogger(__name__)


def grid_topoplot(
    array_data: np.ndarray,
    chan_info: mne.Info,
    titles_x: List[str],
    titles_y: List[str],
    row_titles: Optional[List[str]] = None,
    masks: Optional[np.ndarray] = None,
    mask_params: Optional[dict] = None,
    cmap: Optional[Union[str, List[str]]] = None,
    vlims: Optional[List[Tuple[float, float]]] = None,
    title: Optional[str] = None,
    letters: Optional[List[str]] = None,
    axes: Optional[np.ndarray] = None,
    fig: Optional[plt.Figure] = None,
    cb_frac: float = 0.005,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a grid of topoplots from array data.

    Plots a 2D grid of topographic maps for MEG/EEG data visualization.
    First dimension of array_data is used for rows, second for columns.

    Args:
        array_data: Data to plot, shape (n_rows, n_cols, n_channels).
        chan_info: MNE Info object with channel locations.
        titles_x: Column titles, length n_cols.
        titles_y: Row titles for y-axis labels, length n_rows.
        row_titles: Row titles displayed to the left of each row. Defaults to None.
        masks: Boolean masks for significance, shape (n_rows, n_cols, n_channels).
            Defaults to None.
        mask_params: Parameters for mask visualization. Defaults to None.
        cmap: Colormap(s) for each row. Can be single string or list of strings.
            Defaults to 'magma'.
        vlims: Color limits for each row, list of (vmin, vmax) tuples.
            Defaults to symmetric limits based on data.
        title: Figure title. Defaults to None.
        letters: Letters (A, B, C) for panel labeling. Defaults to None.
        axes: Pre-existing axes array. Defaults to None (creates new figure).
        fig: Pre-existing figure. Defaults to None (creates new figure).
        cb_frac: Colorbar fraction. Defaults to 0.005.

    Returns:
        Tuple containing:
        - fig: Matplotlib Figure object
        - axes: Array of Axes objects

    Examples:
        >>> fig, axes = grid_topoplot(
        ...     data, info, ['Alpha', 'Beta'], ['IN', 'OUT'],
        ...     masks=sig_mask, vlims=[(-0.1, 0.1), (-0.2, 0.2)]
        ... )
        >>> plt.savefig('topoplots.png', dpi=300)

    Notes:
        - If array_data is 2D, it will be reshaped to (1, n_cols, n_channels)
        - Color limits (vlims) should be provided for publication-quality figures
        - Use masks to highlight statistically significant channels
    """
    # Handle 2D input
    if len(array_data.shape) == 2:
        array_data = array_data.reshape((1, array_data.shape[0], array_data.shape[1]))

        if axes is None:
            fig, axes = plt.subplots(
                array_data.shape[0],
                array_data.shape[1],
                figsize=(3 * array_data.shape[1], 3 * array_data.shape[0]),
                dpi=300,
            )
            axes = axes.reshape(1, len(axes))
    else:
        if axes is None:
            fig, axes = plt.subplots(
                array_data.shape[0],
                array_data.shape[1],
                figsize=(3 * array_data.shape[1], 3 * array_data.shape[0]),
                dpi=300,
            )

    # Default vlims if not provided
    if vlims is None:
        vlims = [
            (-np.max(np.abs(row_data)), np.max(np.abs(row_data)))
            for row_data in array_data
        ]

    # Default row titles if not provided
    if row_titles is None:
        row_titles = titles_y

    # Plot each topomap
    for idx_row, row in enumerate(axes):
        for idx_col, ax in enumerate(row):
            # Select colormap
            if cmap is not None:
                if isinstance(cmap, list):
                    current_cmap = cmap[idx_row]
                else:
                    current_cmap = cmap
            else:
                current_cmap = "magma"

            # Select mask
            if masks is not None:
                current_mask = masks[idx_row, idx_col]
            else:
                current_mask = None

            # Set default mask_params for white circles with black borders
            if current_mask is not None and mask_params is None:
                current_mask_params = dict(
                    marker="o",
                    markerfacecolor="w",
                    markeredgecolor="k",
                    linewidth=0,
                    markersize=5,
                )
            else:
                current_mask_params = mask_params

            # Plot topomap
            mne.viz.plot_topomap(
                array_data[idx_row, idx_col],
                chan_info,
                axes=ax,
                show=False,
                cmap=current_cmap,
                mask=current_mask,
                mask_params=current_mask_params,
                vlim=vlims[idx_row] if vlims is not None else None,
                extrapolate="local",
                outlines="head",
                sphere=0.15,
                contours=0,
            )

            # Add column titles
            if idx_row == 0:
                ax.set_title(titles_x[idx_col])

            # Add row labels and letters
            if idx_col == 0:
                ax.set_ylabel(titles_y[idx_row], fontsize=12, rotation=90, labelpad=10)

                if letters is not None:
                    ax.text(
                        -0.02,
                        1.01,
                        letters[idx_row],
                        transform=ax.transAxes,
                        size=14,
                        weight="bold",
                    )

                # Add row title to the left
                ax.text(
                    -0.3,
                    0.5,
                    row_titles[idx_row],
                    fontsize=14,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation="horizontal",
                    transform=ax.transAxes,
                )

    # Add colorbars
    for row_idx in range(array_data.shape[0]):
        fig.colorbar(
            axes[row_idx][0].images[-1],
            ax=axes[row_idx],
            orientation="vertical",
            fraction=cb_frac,
        )

    # Add figure title
    if title is not None:
        if len(axes) > 2:
            y = 1.0
        else:
            y = 1.1
        fig.suptitle(title, y=y, fontsize=16)

    logger.info(
        f"Created grid topoplot: {array_data.shape[0]} rows × "
        f"{array_data.shape[1]} columns"
    )

    return fig, axes


def plot_psd_topomap(
    psd_data: np.ndarray,
    chan_info: mne.Info,
    freq_bands: List[str],
    conditions: Optional[List[str]] = None,
    vlims: Optional[List[Tuple[float, float]]] = None,
    masks: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot PSD data as grid of topographic maps.

    Convenience wrapper around grid_topoplot for PSD visualization.

    Args:
        psd_data: PSD data, shape (n_conditions, n_freq_bands, n_channels).
        chan_info: MNE Info object with channel locations.
        freq_bands: Frequency band names (e.g., ['Delta', 'Theta', 'Alpha']).
        conditions: Condition names (e.g., ['IN', 'OUT']). Defaults to None.
        vlims: Color limits for each condition. Defaults to None.
        masks: Significance masks. Defaults to None.
        title: Figure title. Defaults to None.

    Returns:
        Tuple containing:
        - fig: Matplotlib Figure object
        - axes: Array of Axes objects

    Examples:
        >>> fig, axes = plot_psd_topomap(
        ...     psd_data, info, ['Delta', 'Theta', 'Alpha', 'Beta'],
        ...     conditions=['IN', 'OUT'], title='PSD Topomaps'
        ... )
    """
    if conditions is None:
        conditions = [f"Condition {i+1}" for i in range(psd_data.shape[0])]

    fig, axes = grid_topoplot(
        array_data=psd_data,
        chan_info=chan_info,
        titles_x=freq_bands,
        titles_y=conditions,
        row_titles=conditions,
        masks=masks,
        vlims=vlims,
        title=title,
    )

    return fig, axes


def plot_contrast_topomap(
    contrast_data: np.ndarray,
    chan_info: mne.Info,
    freq_bands: List[str],
    pval_mask: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot statistical contrast as topographic maps.

    Visualizes contrasts (e.g., condition A - condition B) with optional
    significance masking.

    Args:
        contrast_data: Contrast data, shape (n_freq_bands, n_channels).
        chan_info: MNE Info object with channel locations.
        freq_bands: Frequency band names.
        pval_mask: P-value array for masking, shape (n_freq_bands, n_channels).
            Defaults to None.
        alpha: Significance threshold for masking. Defaults to 0.05.
        title: Figure title. Defaults to None.
        cmap: Colormap. Defaults to 'RdBu_r' (diverging).

    Returns:
        Tuple containing:
        - fig: Matplotlib Figure object
        - axes: Array of Axes objects

    Examples:
        >>> fig, axes = plot_contrast_topomap(
        ...     contrast, info, ['Delta', 'Theta'], pval_mask=pvals,
        ...     alpha=0.05, title='IN vs OUT'
        ... )
    """
    # Create significance mask
    if pval_mask is not None:
        masks = pval_mask < alpha
    else:
        masks = None

    # Reshape for grid_topoplot (expects 3D: rows, cols, channels)
    contrast_data = contrast_data.reshape(1, *contrast_data.shape)
    if masks is not None:
        masks = masks.reshape(1, *masks.shape)

    fig, axes = grid_topoplot(
        array_data=contrast_data,
        chan_info=chan_info,
        titles_x=freq_bands,
        titles_y=["Contrast"],
        row_titles=["Contrast"],
        masks=masks,
        cmap=cmap,
        title=title,
    )

    return fig, axes
