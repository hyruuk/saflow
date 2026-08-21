"""Shared 16:9 slide geometry, typography, and export helpers.

Manuscript composites are read at page scale, where 6--8 pt annotations stay
legible. The same drawing code exported to a slide is read from the back of a
room, so every slide export routes its text through the sizes defined here
instead of reusing composite-scale values. ``scale_typography`` retrofits the
same treatment onto figures drawn by composite-scale plotters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

SLIDE_FIGSIZE = (16.0, 9.0)
SLIDE_DPI = 160

TITLE_SIZE = 26.0
SUBTITLE_SIZE = 16.0
CELL_TITLE_SIZE = 17.0
LARGE_CELL_TITLE_SIZE = 21.0
AXIS_LABEL_SIZE = 16.0
TICK_LABEL_SIZE = 14.0
LEGEND_SIZE = 15.0
ANNOTATION_SIZE = 14.0
FOOTER_SIZE = 14.0
MARK_SIZE = 20.0

SUBTITLE_COLOR = "#444444"
FOOTER_COLOR = "#444444"
POSITIVE_COLOR = "#a40000"
NEGATIVE_COLOR = "#00408a"

# Composite plotters annotate at 6--8 pt; 2.3x lifts that to 14--18 pt while the
# floor rescues anything already sized for a slide from staying small.
TEXT_SCALE = 2.3
TEXT_FLOOR = 14.0
LINE_SCALE = 1.6
LINE_FLOOR = 1.8
MARKER_AREA_SCALE = 4.0


def new_slide(title: str, subtitle: str | None = None) -> plt.Figure:
    """Create one white 16:9 slide carrying a self-contained heading."""
    figure = plt.figure(figsize=SLIDE_FIGSIZE, dpi=SLIDE_DPI, facecolor="white")
    figure.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=0.965)
    if subtitle:
        figure.text(
            0.5, 0.905, subtitle, ha="center", va="center",
            fontsize=SUBTITLE_SIZE, color=SUBTITLE_COLOR,
        )
    return figure


def add_footer(figure: plt.Figure, text: str, y: float = 0.04) -> None:
    """Print one interpretation note along the bottom of a slide."""
    figure.text(0.5, y, text, ha="center", fontsize=FOOTER_SIZE, color=FOOTER_COLOR)


def grid_shape(count: int) -> tuple[int, int]:
    """Return the (rows, columns) slide grid used for map families."""
    if count <= 4:
        return 1, count
    columns = int(np.ceil(count / 2))
    return int(np.ceil(count / columns)), columns


def add_map_grid(
    figure: plt.Figure,
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    left: float = 0.035,
    right: float = 0.90,
    top: float = 0.84,
    bottom: float = 0.12,
) -> list[plt.Axes]:
    """Lay out one family of pre-rendered cortical composites on a slide."""
    rows, columns = grid_shape(len(images))
    grid = figure.add_gridspec(
        rows, columns, left=left, right=right, top=top, bottom=bottom,
        hspace=0.24, wspace=0.08,
    )
    size = CELL_TITLE_SIZE if len(images) > 4 else LARGE_CELL_TITLE_SIZE
    axes = []
    for index, (image, title) in enumerate(zip(images, titles)):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        axis.imshow(image, interpolation="bilinear", aspect="equal")
        axis.set_axis_off()
        axis.set_title(title, fontsize=size, pad=7, fontweight="semibold")
        axes.append(axis)
    return axes


def add_colorbar(
    figure: plt.Figure,
    minimum: float,
    maximum: float,
    color_map: str,
    label: str,
    *,
    bounds: tuple[float, float, float, float] = (0.925, 0.22, 0.018, 0.52),
    above: str | None = None,
    below: str | None = None,
    above_color: str = POSITIVE_COLOR,
    below_color: str = NEGATIVE_COLOR,
) -> plt.Axes:
    """Add a presentation-scale colorbar with optional direction labels."""
    axis = figure.add_axes(bounds)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(minimum, maximum), cmap=color_map),
        cax=axis,
    )
    colorbar.set_label(label, fontsize=AXIS_LABEL_SIZE, labelpad=10)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if above:
        axis.text(0.5, 1.04, above, transform=axis.transAxes, ha="center",
                  fontsize=ANNOTATION_SIZE, color=above_color)
    if below:
        axis.text(0.5, -0.04, below, transform=axis.transAxes, ha="center", va="top",
                  fontsize=ANNOTATION_SIZE, color=below_color)
    return axis


def add_heatmap_row(
    figure: plt.Figure,
    panels: Sequence[tuple[str, np.ndarray, np.ndarray]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    *,
    color_map: str,
    limit: float,
    bottom: float = 0.24,
    height: float = 0.55,
) -> list[plt.Axes]:
    """Draw side-by-side matrices sharing one scale, at slide typography."""
    count = len(panels)
    span = (0.86 - 0.05 * (count - 1)) / count
    axes = []
    for index, (title, values, p_values) in enumerate(panels):
        axis = figure.add_axes((0.055 + index * (span + 0.05), bottom, span, height))
        axis.imshow(values, cmap=color_map, vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(
            np.arange(len(column_labels)), list(column_labels),
            rotation=45, ha="right", fontsize=TICK_LABEL_SIZE,
        )
        axis.set_yticks(np.arange(len(row_labels)), list(row_labels), fontsize=TICK_LABEL_SIZE)
        axis.set_title(title, fontsize=CELL_TITLE_SIZE, fontweight="semibold", pad=10)
        for row, column in np.argwhere(np.asarray(p_values) < 0.05):
            axis.text(column, row, "•", ha="center", va="center", fontsize=MARK_SIZE)
        axes.append(axis)
    return axes


def brain_image(axis: plt.Axes) -> np.ndarray:
    """Return the cortical composite already rasterized into one axes."""
    if not axis.images:
        raise ValueError("axes carries no rendered cortical composite")
    return np.asarray(axis.images[0].get_array())


def slide_directory(reports_root: Path, stem: str) -> Path:
    """Return the slide directory paired with one manuscript figure stem."""
    directory = reports_root / "figures" / "slides" / stem
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_slide(
    figure: plt.Figure, directory: Path, index: int, name: str, sidecar: dict[str, Any]
) -> Path:
    """Save one exact 2560x1440 slide PNG beside its provenance sidecar."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{index:02d}_{name}.png"
    figure.savefig(path, dpi=SLIDE_DPI, facecolor="white")
    plt.close(figure)
    payload = {**sidecar, "component": name, "path": str(path)}
    path.with_name(f"{path.name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return path


def scale_typography(
    figure: plt.Figure, scale: float = TEXT_SCALE, floor: float = TEXT_FLOOR
) -> None:
    """Enlarge every text, line, and marker drawn at composite scale."""
    for text in figure.texts:
        _resize(text, scale, floor)
    for axis in figure.axes:
        for text in (axis.title, axis.xaxis.label, axis.yaxis.label, *axis.texts):
            _resize(text, scale, floor)
        for name in ("x", "y"):
            _scale_tick_labels(axis, name, scale, floor)
        legend = axis.get_legend()
        if legend is not None:
            for text in (legend.get_title(), *legend.get_texts()):
                _resize(text, scale, floor)
        _thicken(axis)


def _resize(text: Any, scale: float, floor: float) -> None:
    """Grow one text object without ever shrinking it."""
    if text is None or not str(text.get_text()):
        return
    text.set_fontsize(max(float(text.get_fontsize()) * scale, floor))


def _scale_tick_labels(axis: plt.Axes, name: str, scale: float, floor: float) -> None:
    """Resize ticks through tick_params so regenerated labels keep the size."""
    labels = (axis.xaxis if name == "x" else axis.yaxis).get_ticklabels()
    if not labels:
        return
    current = float(labels[0].get_fontsize())
    axis.tick_params(axis=name, which="both", labelsize=max(current * scale, floor))


def _thicken(axis: plt.Axes) -> None:
    """Give hairline strokes and small markers presentation weight."""
    for line in axis.lines:
        line.set_linewidth(max(line.get_linewidth() * LINE_SCALE, LINE_FLOOR))
        if line.get_marker() not in (None, "None", "", " "):
            line.set_markersize(line.get_markersize() * 1.4)
    for collection in axis.collections:
        if isinstance(collection, PathCollection):
            collection.set_sizes(np.asarray(collection.get_sizes()) * MARKER_AREA_SCALE)
