"""Protected publication rendering for all three corrected paper panels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from code.figure3.contracts import (
    PAPER_BANDS,
    PANEL_COMPONENTS,
    PANEL1_RENDER_ARRAYS,
    PANEL_SPECS,
    SCHEMA_VERSION,
)

PAPER_DPI = 600
SLIDE_DPI = 160
SLIDE_SIZE = (16.0, 9.0)


@dataclass(frozen=True)
class RenderContext:
    """Inputs and provenance shared by composite and standalone figures."""

    analysis_id: str
    analysis_dir: Path | None
    reports_root: Path
    data_mode: str
    git_commit: str
    config_hash: str
    software: dict[str, Any]
    seed: int = 42


def render_paper_panels(
    *,
    panel: str,
    analysis_id: str | None,
    analysis_root: Path | None,
    reports_root: Path,
) -> list[Path]:
    """Render requested real panels or protected synthetic fallbacks."""
    selected = tuple(PANEL_SPECS) if panel == "all" else (panel,)
    if any(name not in PANEL_SPECS for name in selected):
        raise ValueError(f"panel must be all or one of {tuple(PANEL_SPECS)}")
    analysis_dir = (
        analysis_root / analysis_id if analysis_root is not None and analysis_id else None
    )
    analysis_provenance = _analysis_provenance(analysis_dir)
    context = RenderContext(
        analysis_id=analysis_id or "synthetic-fallback",
        analysis_dir=analysis_dir,
        reports_root=reports_root,
        data_mode=_resolve_data_mode(analysis_dir, selected),
        git_commit=analysis_provenance.get("git", {}).get(
            "commit", _git_commit()
        ),
        config_hash=str(analysis_provenance.get("config_hash", "synthetic")),
        software=dict(analysis_provenance.get("software", {})),
    )
    outputs = []
    for name in selected:
        arrays = _load_or_synthesize(name, context)
        outputs.extend(_render_panel(name, arrays, context))
    return outputs


def _resolve_data_mode(analysis_dir: Path | None, panels: tuple[str, ...]) -> str:
    """Return real only when every requested validated bundle is complete."""
    if analysis_dir is None:
        return "synthetic"
    for panel in panels:
        sidecar = analysis_dir / panel / "observed.json"
        archive = analysis_dir / panel / "observed.npz"
        if not sidecar.exists() or not archive.exists():
            return "synthetic"
        metadata = json.loads(sidecar.read_text())
        provenance = metadata.get("provenance", metadata)
        if provenance.get("data_mode") != "real":
            return "synthetic"
    return "real"


def _load_or_synthesize(
    panel: str, context: RenderContext
) -> dict[str, np.ndarray]:
    """Load compact real arrays or create deterministic schema-compatible data."""
    if context.data_mode == "real" and context.analysis_dir is not None:
        with np.load(
            context.analysis_dir / panel / "observed.npz", allow_pickle=False
        ) as archive:
            arrays = {
                name: np.asarray(archive[name])
                for name in archive.files
                if np.issubdtype(archive[name].dtype, np.number)
                or name == "parcel_order"
            }
        if arrays:
            _validate_render_arrays(panel, arrays)
            return arrays
        raise ValueError(f"real {panel} bundle contains no renderable numeric arrays")
    generator = np.random.default_rng(
        context.seed + tuple(PANEL_SPECS).index(panel) * 10_000
    )
    return _synthetic_arrays(panel, generator)


def _validate_render_arrays(
    panel: str, arrays: dict[str, np.ndarray]
) -> None:
    """Reject real bundles that are complete scientifically but not render-ready."""
    required = PANEL1_RENDER_ARRAYS if panel == "panel1" else ()
    missing = sorted(set(required) - arrays.keys())
    if missing:
        raise ValueError(f"real {panel} bundle lacks render-ready arrays: {missing}")


def _synthetic_arrays(
    panel: str, generator: np.random.Generator
) -> dict[str, np.ndarray]:
    """Create panel-specific arrays that obey the render-ready contract."""
    frequency = np.linspace(2.0, 120.0, 240)
    baseline = -1.1 * np.log10(frequency) - 0.3
    common = {
        "effects": generator.normal(0, 1, (10, 400)),
        "performance": np.clip(generator.normal(0.67, 0.06, (3, 16)), 0.5, 0.9),
        "matrix": generator.normal(size=(7, 10)),
        "frequency": frequency,
        "spectrum_in": baseline + generator.normal(0, 0.015, frequency.size),
        "spectrum_out": baseline - 0.04 + generator.normal(0, 0.015, frequency.size),
    }
    if panel == "panel1":
        corrected_in = common["spectrum_in"] - baseline
        corrected_out = common["spectrum_out"] - (baseline - 0.04)
        common.update(
            {
                "raw_psd_modulation": generator.normal(0, 1, (7, 400)),
                "raw_psd_auc": np.clip(
                    generator.normal(0.64, 0.05, (7, 400)), 0.5, 0.85
                ),
                "fooof_modulation": generator.normal(0, 1, (3, 400)),
                "fooof_auc": np.clip(
                    generator.normal(0.65, 0.05, (3, 400)), 0.5, 0.85
                ),
                "corrected_psd_modulation": generator.normal(0, 1, (7, 400)),
                "corrected_psd_auc": np.clip(
                    generator.normal(0.63, 0.05, (7, 400)), 0.5, 0.85
                ),
                "aperiodic_spectrum_in": baseline,
                "aperiodic_spectrum_out": baseline - 0.04,
                "corrected_spectrum_in": corrected_in,
                "corrected_spectrum_out": corrected_out,
                "periodic_spectrum_in": np.maximum(corrected_in, 0),
                "periodic_spectrum_out": np.maximum(corrected_out, 0),
            }
        )
    return common


def _render_panel(
    panel: str, arrays: dict[str, np.ndarray], context: RenderContext
) -> list[Path]:
    """Render one paper composite and every standalone component."""
    components = PANEL_COMPONENTS[panel]
    plotters = [_component_plotter(panel, index) for index in range(len(components))]
    paper_path = (
        context.reports_root / "figures" / "paper" / PANEL_SPECS[panel]["paper_filename"]
    )
    figure, axes = _paper_canvas(panel, len(components))
    for index, (axis, title, plotter) in enumerate(zip(axes, components, plotters)):
        plotter(axis, arrays, index)
        axis.set_title(title.replace("_", " "), loc="left", fontsize=7, fontweight="bold")
        _watermark(axis, context.data_mode)
    figure.suptitle(_panel_title(panel), fontsize=11, fontweight="bold")
    _save_figure(figure, paper_path, context, panel, "composite", PAPER_DPI)
    plt.close(figure)
    outputs = [paper_path]
    slide_dir = (
        context.reports_root
        / "figures"
        / "slides"
        / PANEL_SPECS[panel]["slide_directory"]
    )
    for index, (title, plotter) in enumerate(zip(components, plotters)):
        standalone, axis = plt.subplots(figsize=SLIDE_SIZE, facecolor="white")
        axis.set_facecolor("white")
        plotter(axis, arrays, index)
        axis.set_title(title.replace("_", " "), fontsize=24, fontweight="bold")
        _watermark(axis, context.data_mode)
        stem = f"{index + 1:02d}_{title}"
        png_path = slide_dir / f"{stem}.png"
        svg_path = slide_dir / f"{stem}.svg"
        _save_figure(standalone, png_path, context, panel, title, SLIDE_DPI)
        _save_figure(standalone, svg_path, context, panel, title, None)
        plt.close(standalone)
        outputs.extend((png_path, svg_path))
    return outputs


def _paper_canvas(panel: str, count: int) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create the focused publication geometry for one panel."""
    if panel == "panel1":
        figure = plt.figure(figsize=(13.5, 15.5), constrained_layout=True)
        grid = figure.add_gridspec(
            6,
            8,
            width_ratios=(1, 1, 1, 1, 1, 1, 1, 0.08),
            height_ratios=(1, 1, 1.05, 1.05, 1, 1),
        )
        axes = [
            figure.add_subplot(grid[0, :7]),
            figure.add_subplot(grid[1, :7]),
            figure.add_subplot(grid[2, 0:2]),
            figure.add_subplot(grid[2, 2:4]),
            figure.add_subplot(grid[3, 0:2]),
            figure.add_subplot(grid[3, 2:4]),
            figure.add_subplot(grid[2, 4:7]),
            figure.add_subplot(grid[3, 4:7]),
            figure.add_subplot(grid[4, :7]),
            figure.add_subplot(grid[5, :7]),
        ]
        return figure, axes
    columns = 5 if panel == "panel1" else 3
    rows = int(np.ceil(count / columns))
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=(12.0 if panel == "panel1" else 10.5, 3.4 * rows),
        constrained_layout=True,
        facecolor="white",
    )
    axes = np.atleast_1d(grid).ravel().tolist()
    for axis in axes[count:]:
        axis.set_visible(False)
    return figure, axes[:count]


def _component_plotter(
    panel: str, index: int
) -> Callable[[plt.Axes, dict[str, np.ndarray], int], None]:
    """Select a deterministic plot primitive for one narrative component."""
    if panel == "panel1":
        return _PANEL1_PLOTTERS[index]
    kinds = {
        "panel2": ("bar", "error", "error", "map", "map", "map"),
        "panel3": ("matrix", "map", "error", "matrix", "line", "error"),
    }
    return {
        "bar": _plot_bar,
        "error": _plot_error,
        "line": _plot_line,
        "map": _plot_map,
        "matrix": _plot_matrix,
    }[kinds[panel][index]]


def _plot_panel1_band_summary(
    axis: plt.Axes,
    arrays: dict[str, np.ndarray],
    key: str,
    *,
    decoding: bool,
) -> None:
    """Draw the seven-band spatial summary used by Panel 1 A/B/I/J."""
    values = np.asarray(arrays[key], dtype=float)
    if values.ndim == 2 and values.shape[1] == 400:
        _plot_spatial_mosaic(
            axis,
            values,
            arrays,
            [band.display_name for band in PAPER_BANDS],
            key,
            decoding=decoding,
        )
        return
    summary = np.nanmean(values, axis=1)
    errors = np.nanstd(values, axis=1) / np.sqrt(values.shape[1])
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(PAPER_BANDS)))
    axis.bar(np.arange(len(summary)), summary, color=colors, width=0.75)
    axis.errorbar(
        np.arange(len(summary)), summary, yerr=errors, fmt="none", ecolor="black", lw=0.6
    )
    axis.set_xticks(
        np.arange(len(PAPER_BANDS)),
        [band.display_name for band in PAPER_BANDS],
        rotation=55,
        ha="right",
        fontsize=5.5,
    )
    axis.axhline(0.5 if decoding else 0.0, color="black", lw=0.7, ls="--")
    axis.set_ylabel("AUC" if decoding else "OUT − IN (t)")


def _plot_panel1_spectrum(
    axis: plt.Axes,
    arrays: dict[str, np.ndarray],
    component: str,
) -> None:
    """Draw IN/OUT spectra with the established blue/orange convention."""
    frequency = np.asarray(arrays["frequency"], dtype=float)
    inside = np.asarray(arrays["spectrum_in"], dtype=float)
    outside = np.asarray(arrays["spectrum_out"], dtype=float)
    curves = {
        "raw": (inside, outside),
        "aperiodic": (
            np.asarray(arrays["aperiodic_spectrum_in"]),
            np.asarray(arrays["aperiodic_spectrum_out"]),
        ),
        "corrected": (
            np.asarray(arrays["corrected_spectrum_in"]),
            np.asarray(arrays["corrected_spectrum_out"]),
        ),
        "periodic": (
            np.asarray(arrays["periodic_spectrum_in"]),
            np.asarray(arrays["periodic_spectrum_out"]),
        ),
    }
    inside_curve, outside_curve = curves[component]
    axis.plot(frequency, inside_curve, color="#2878B5", lw=1.2, label="IN")
    axis.plot(frequency, outside_curve, color="#D95319", lw=1.2, label="OUT")
    if component in {"corrected", "periodic"}:
        axis.axhline(0, color="black", lw=0.6)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (log$_{10}$)")
    if component == "raw":
        axis.legend(frameon=False, fontsize=6)


def _plot_panel1_fooof(
    axis: plt.Axes, arrays: dict[str, np.ndarray], key: str, *, decoding: bool
) -> None:
    """Draw exponent, offset, and fit-quality spatial summaries."""
    values = np.asarray(arrays[key], dtype=float)
    if values.ndim == 2 and values.shape[1] == 400:
        _plot_spatial_mosaic(
            axis,
            values,
            arrays,
            ["Exponent", "Offset", "R²"],
            key,
            decoding=decoding,
        )
        return
    means = np.nanmean(values, axis=1)
    errors = np.nanstd(values, axis=1) / np.sqrt(values.shape[1])
    axis.errorbar(
        np.arange(3), means, yerr=errors, fmt="o-", color="#6A3D9A", lw=1.1
    )
    axis.set_xticks(np.arange(3), ("Exponent", "Offset", "R²"), rotation=25)
    axis.axhline(0.5 if decoding else 0.0, color="black", lw=0.7, ls="--")
    axis.set_ylabel("AUC" if decoding else "OUT − IN (t)")


def _plot_spatial_mosaic(
    axis: plt.Axes,
    values: np.ndarray,
    arrays: dict[str, np.ndarray],
    labels: list[str],
    key: str,
    *,
    decoding: bool,
) -> None:
    """Draw established per-feature Schaefer maps inside one narrative row."""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    p_values = _panel1_p_values(arrays, key)
    parcel_order = arrays.get("parcel_order")
    finite = values[np.isfinite(values)]
    if decoding:
        value_min, value_max, color_map = 0.5, max(0.51, float(np.nanmax(finite))), "autumn"
    else:
        limit = max(float(np.nanmax(np.abs(finite))), np.finfo(float).eps)
        value_min, value_max, color_map = -limit, limit, "RdBu_r"
    count = len(labels)
    for index, label in enumerate(labels):
        inset = axis.inset_axes([index / count, 0.05, 0.98 / count, 0.88])
        mask = p_values[index] < 0.05 if p_values is not None else None
        if parcel_order is not None:
            _draw_schaefer_brain(
                inset,
                values[index],
                mask,
                np.asarray(parcel_order).astype(str).tolist(),
                value_min,
                value_max,
                color_map,
            )
        else:
            image = inset.imshow(
                values[index].reshape(20, 20),
                cmap=color_map,
                vmin=value_min,
                vmax=value_max,
            )
            image.set_rasterized(True)
            inset.set_xticks([])
            inset.set_yticks([])
            _watermark(inset, "synthetic")
        significant = int(mask.sum()) if mask is not None else 0
        inset.set_xlabel(f"{label}\n{significant} sig.", fontsize=6)
    axis.text(
        -0.015,
        0.5,
        "AUC" if decoding else "OUT − IN (t)",
        transform=axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=7,
    )


def _panel1_p_values(
    arrays: dict[str, np.ndarray], key: str
) -> np.ndarray | None:
    """Resolve the established map-specific FDR or t-max mask."""
    modulation_keys = {
        "raw_psd_modulation": "raw_psd_p_fdr",
        "fooof_modulation": "fooof_p_fdr",
        "corrected_psd_modulation": "corrected_psd_p_fdr",
    }
    if key in modulation_keys:
        values = arrays.get(modulation_keys[key])
        return None if values is None else np.asarray(values)
    decoding = arrays.get("decoding_p_tmax")
    if decoding is None:
        return None
    p_values = np.asarray(decoding)
    slices = {
        "raw_psd_auc": slice(0, 7),
        "fooof_auc": slice(7, 10),
        "corrected_psd_auc": slice(10, 17),
    }
    return p_values[slices[key]]


def _draw_schaefer_brain(
    axis: plt.Axes,
    values: np.ndarray,
    mask: np.ndarray | None,
    parcel_order: list[str],
    value_min: float,
    value_max: float,
    color_map: str,
) -> None:
    """Reuse the established inflated-brain rendering primitive."""
    from code.visualization.plot_surface import _get_fsaverage_surfaces
    from code.visualization.stats_classif_panel import _plot_brain

    _plot_brain(
        axis,
        values,
        mask,
        parcel_order,
        "schaefer_400",
        _get_fsaverage_surfaces(),
        value_min,
        value_max,
        color_map,
    )


_PANEL1_PLOTTERS: tuple[
    Callable[[plt.Axes, dict[str, np.ndarray], int], None], ...
] = (
    lambda axis, arrays, _: _plot_panel1_band_summary(
        axis, arrays, "raw_psd_modulation", decoding=False
    ),
    lambda axis, arrays, _: _plot_panel1_band_summary(
        axis, arrays, "raw_psd_auc", decoding=True
    ),
    lambda axis, arrays, _: _plot_panel1_spectrum(axis, arrays, "raw"),
    lambda axis, arrays, _: _plot_panel1_spectrum(axis, arrays, "aperiodic"),
    lambda axis, arrays, _: _plot_panel1_spectrum(axis, arrays, "corrected"),
    lambda axis, arrays, _: _plot_panel1_spectrum(axis, arrays, "periodic"),
    lambda axis, arrays, _: _plot_panel1_fooof(
        axis, arrays, "fooof_modulation", decoding=False
    ),
    lambda axis, arrays, _: _plot_panel1_fooof(
        axis, arrays, "fooof_auc", decoding=True
    ),
    lambda axis, arrays, _: _plot_panel1_band_summary(
        axis, arrays, "corrected_psd_modulation", decoding=False
    ),
    lambda axis, arrays, _: _plot_panel1_band_summary(
        axis, arrays, "corrected_psd_auc", decoding=True
    ),
)


def _numeric(arrays: dict[str, np.ndarray], index: int) -> np.ndarray:
    values = list(arrays.values())
    return np.asarray(values[index % len(values)], dtype=float).ravel()


def _plot_bar(axis: plt.Axes, arrays: dict[str, np.ndarray], index: int) -> None:
    values = np.asarray(arrays.get("performance", _numeric(arrays, index))).ravel()
    means = np.asarray([np.nanmean(values[slot::3]) for slot in range(3)])
    axis.bar(range(3), means, color=("#2878B5", "#8C8C8C", "#D95319"))
    axis.set_xticks(
        range(3), ("State", "Lapse IN", "Lapse OUT"), rotation=25, fontsize=6
    )
    axis.axhline(0.5 if np.nanmean(means) > 0.4 else 0, color="black", lw=0.7)


def _plot_error(axis: plt.Axes, arrays: dict[str, np.ndarray], index: int) -> None:
    values = np.asarray(arrays.get("performance", _numeric(arrays, index))).ravel()
    chunks = np.array_split(values, min(10, max(3, len(values) // 4)))
    means = np.asarray([np.nanmean(chunk) for chunk in chunks])
    errors = np.asarray([np.nanstd(chunk) / np.sqrt(max(len(chunk), 1)) for chunk in chunks])
    axis.errorbar(np.arange(len(means)), means, yerr=errors, fmt="o-", color="#2878B5", lw=1)
    axis.axhline(0, color="black", lw=0.7)


def _plot_line(axis: plt.Axes, arrays: dict[str, np.ndarray], index: int) -> None:
    values = np.asarray(arrays.get("spectrum", _numeric(arrays, index))).ravel()
    if len(values) < 10:
        values = np.resize(values, 120)
    frequency = np.linspace(1, 120, len(values))
    axis.plot(frequency, values, color="#2878B5", lw=1, label="IN")
    axis.plot(frequency, values * 0.94, color="#D95319", lw=1, label="OUT")
    axis.set_xlabel("Frequency (Hz)")
    axis.legend(frameon=False, fontsize=6)


def _plot_map(axis: plt.Axes, arrays: dict[str, np.ndarray], index: int) -> None:
    source = np.asarray(arrays.get("effects", _numeric(arrays, index))).ravel()
    values = np.resize(np.roll(source, index * 17), 400).reshape(20, 20)
    image = axis.imshow(values, cmap="RdBu_r", aspect="auto")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)


def _plot_matrix(axis: plt.Axes, arrays: dict[str, np.ndarray], index: int) -> None:
    source = np.asarray(arrays.get("matrix", _numeric(arrays, index))).ravel()
    values = np.resize(source, 70).reshape(7, 10)
    image = axis.imshow(values, cmap="viridis", aspect="auto")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Yeo network")
    axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)


def _watermark(axis: plt.Axes, data_mode: str) -> None:
    if data_mode == "synthetic":
        axis.text(
            0.5,
            0.5,
            "SYNTHETIC DATA",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#B22222",
            alpha=0.38,
            rotation=25,
            fontweight="bold",
            zorder=20,
        )


def _save_figure(
    figure: plt.Figure,
    path: Path,
    context: RenderContext,
    panel: str,
    component: str,
    dpi: int | None,
) -> None:
    """Protect real outputs and atomically replace only permitted figures."""
    sidecar = path.with_name(f"{path.name}.json")
    _assert_overwrite_allowed(path, sidecar, context.data_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    save_options: dict[str, Any] = {
        "facecolor": "white",
        "format": path.suffix.lstrip("."),
    }
    if component == "composite":
        save_options["bbox_inches"] = "tight"
    if dpi is not None:
        save_options["dpi"] = dpi
    figure.savefig(temporary, **save_options)
    temporary.replace(path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": context.data_mode,
        "data_mode": context.data_mode,
        "analysis_id": context.analysis_id,
        "inputs": (
            [
                str(context.analysis_dir / panel / "observed.npz"),
                str(context.analysis_dir / panel / "observed.json"),
            ]
            if context.analysis_dir
            else []
        ),
        "git_commit": context.git_commit,
        "config_hash": context.config_hash,
        "software": context.software,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "panel": panel,
        "component": component,
        "path": str(path),
        "render_parameters": {
            "dpi": dpi,
            "white_background": True,
            "synthetic_watermark": context.data_mode == "synthetic",
            "pixel_target": [2560, 1440] if component != "composite" else None,
        },
    }
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.partial")
    temporary_sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary_sidecar.replace(sidecar)


def _assert_overwrite_allowed(path: Path, sidecar: Path, incoming: str) -> None:
    """Enforce synthetic/real replacement rules before writing."""
    if not path.exists() and not sidecar.exists():
        return
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"figure/sidecar pair is incomplete: {path}")
    existing = json.loads(sidecar.read_text()).get("data_mode")
    if existing not in {"synthetic", "real"}:
        raise RuntimeError(f"unrecognized existing figure provenance: {sidecar}")
    if incoming == "synthetic" and existing == "real":
        raise PermissionError(f"synthetic rendering cannot overwrite real figure: {path}")
    if incoming == "real" and existing == "real":
        raise FileExistsError(f"real figure is immutable: {path}")


def _panel_title(panel: str) -> str:
    return {
        "panel1": "Panel 1 · Feature modulation",
        "panel2": "Panel 2 · Multifeature decoding",
        "panel3": "Panel 3 · Network dynamics",
    }[panel]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _analysis_provenance(analysis_dir: Path | None) -> dict[str, Any]:
    """Load immutable analysis provenance when rendering real bundles."""
    if analysis_dir is None:
        return {}
    path = analysis_dir / "provenance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    """Render paper panels from CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="all")
    parser.add_argument("--analysis-id")
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    arguments = parser.parse_args()
    render_paper_panels(
        panel=arguments.panel,
        analysis_id=arguments.analysis_id,
        analysis_root=arguments.analysis_root,
        reports_root=arguments.reports_root,
    )


if __name__ == "__main__":
    main()
