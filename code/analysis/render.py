"""Protected publication rendering for all three corrected panel analysis."""

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

from code.analysis.contracts import (
    CANONICAL_BANDS,
    CORRECTED_FEATURES,
    FEATURE_DISPLAY_NAMES,
    PANEL_COMPONENTS,
    PANEL_SPECS,
    PANEL_ANALYSES,
    SCHEMA_VERSION,
)
from code.analysis.provenance import active_analysis_id, resolve_analysis_directory
from code.utils.yeo_networks import YEO7_NETWORKS, network_display_name

PANEL2_CAPTION = """Figure X. Trial-level multifeature decoding of IN versus OUT attentional state. Models combined FOOOF exponent and offset with seven aperiodic-corrected PSD bands across 400 Schaefer parcels; FOOOF R² was excluded. (A) Population leave-one-subject-out performance. (B) Individual leave-one-run-out performance. (C) Population-versus-individual performance comparison. (D) Held-out feature reliance. (E) Held-out Yeo-7 network reliance. (F) Feature-by-network reliance. State performance used run-wise circular VTC shifts; reliance used synchronized subject sign flips with separate maximum-statistic families. Population and individual summaries included {state_n} participants.\n"""
PANEL3_CAPTION = """Figure X. Network dynamics across attentional state and behavioral outcome. Neural features were summarized within the seven Yeo networks for four state-by-outcome cells (IN-correct, IN-lapse, OUT-correct, and OUT-lapse). Analyses included two FOOOF parameters (exponent and offset) and seven aperiodic-corrected PSD features; FOOOF R² was retained only as a fit-quality metric and was not analyzed. (A) Four-cell profile for the network–feature pair with the largest absolute interaction t statistic. (B) Complete network-by-feature state–outcome interaction map. (C) Prespecified simple effects, showing for each feature the signed t statistic from its strongest Yeo-network expression. (D) Standardized interaction effect sizes and bootstrap 95% confidence intervals for the seven largest absolute interaction t statistics. (E) Four-cell default-mode–dorsal-attention coupling profile for the feature with the largest absolute coupling-interaction t statistic. (F) Complete prespecified DMN–DAN coupling contrasts. Dots in heatmaps mark synchronized maximum-|t| family-wise p < 0.05. Selection in A, C, D, and E is descriptive; complete inferential families remain visible in B and F. Primary inference was corrected separately across the FOOOF modulation, corrected-PSD modulation, and DMN–DAN coupling families. Complete-case analyses included {modulation_n} participants for modulation and {coupling_n} participants for coupling; secondary mixed-effects models used all available observations.\n"""

COMPOSITE_DPI = 600
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


def render_panels(
    *,
    panel: str,
    analysis_id: str | None,
    analysis_root: Path | None,
    reports_root: Path,
) -> list[Path]:
    """Render requested real panels or protected synthetic fallbacks."""
    available = ("panel2", "panel3")
    selected = available if panel == "all" else (panel,)
    if any(name not in available for name in selected):
        raise ValueError(
            "generic rendering supports panel2, panel3, or all; "
            "render the canonical Panel 1 with `invoke viz.panel1`"
        )
    if analysis_root is not None:
        resolved_id = analysis_id or active_analysis_id(analysis_root)
        analysis_dir = resolve_analysis_directory(analysis_root, resolved_id)
        analysis_id = resolved_id
    else:
        analysis_dir = None
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
        result_directory = analysis_dir / PANEL_ANALYSES[panel]
        sidecar = result_directory / "observed.json"
        archive = result_directory / "observed.npz"
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
            context.analysis_dir
            / PANEL_ANALYSES[panel]
            / "observed.npz",
            allow_pickle=False,
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
    required_by_panel = {
        "panel2": (
            "population_auc", "population_null", "within_subject_auc",
            "population_feature_reliance", "population_feature_reliance_p_fwer",
            "population_network_reliance", "population_network_reliance_p_fwer",
            "population_cell_reliance", "population_cell_reliance_p_fwer",
        ),
        "panel3": (
            "network_cell_means",
            "interaction",
            "fooof_t_values",
            "fooof_p_fwer",
            "corrected_psd_t_values",
            "corrected_psd_p_fwer",
            "coupling",
            "coupling_t_values",
            "coupling_p_fwer",
        ),
    }
    required = required_by_panel.get(panel, ())
    missing = sorted(set(required) - arrays.keys())
    if missing:
        raise ValueError(f"real {panel} bundle lacks render-ready arrays: {missing}")
    if panel == "panel3" and arrays["network_cell_means"].shape[-1] != len(
        CORRECTED_FEATURES
    ):
        raise ValueError(
            "Panel 3 bundle uses a stale feature schema; rerun network dynamics "
            f"with the {len(CORRECTED_FEATURES)}-feature contract"
        )


def _synthetic_arrays(
    panel: str, generator: np.random.Generator
) -> dict[str, np.ndarray]:
    """Create panel-specific arrays that obey the render-ready contract."""
    frequency = np.linspace(2.0, 120.0, 240)
    baseline = -1.1 * np.log10(frequency) - 0.3
    common = {
        "effects": generator.normal(0, 1, (len(CORRECTED_FEATURES), 400)),
        "performance": np.clip(generator.normal(0.67, 0.06, (3, 16)), 0.5, 0.9),
        "matrix": generator.normal(size=(7, len(CORRECTED_FEATURES))),
        "frequency": frequency,
        "spectrum_in": baseline + generator.normal(0, 0.015, frequency.size),
        "spectrum_out": baseline - 0.04 + generator.normal(0, 0.015, frequency.size),
    }
    if panel == "panel2":
        subject_count = 20
        feature_count = len(CORRECTED_FEATURES)
        network_count = len(YEO7_NETWORKS)
        common.update({
            "population_auc": np.asarray(0.64),
            "population_null": generator.normal(0.5, 0.02, 1000),
            "within_subject_auc": generator.normal(0.61, 0.06, subject_count),
            "population_feature_reliance": generator.normal(
                0.015, 0.01, (subject_count, feature_count)
            ),
            "population_feature_reliance_p_fwer": generator.uniform(
                0.01, 1, feature_count
            ),
            "population_network_reliance": generator.normal(
                0.012, 0.01, (subject_count, network_count)
            ),
            "population_network_reliance_p_fwer": generator.uniform(
                0.01, 1, network_count
            ),
            "population_cell_reliance": generator.normal(
                0.008, 0.01, (subject_count, feature_count * network_count)
            ),
            "population_cell_reliance_p_fwer": generator.uniform(
                0.01, 1, feature_count * network_count
            ),
        })
    if panel == "panel3":
        subject_count = 20
        feature_count = len(CORRECTED_FEATURES)
        cells = generator.normal(0, 0.45, (subject_count, 4, 7, feature_count))
        cells[:, 3, 6, 1] += 0.9
        cells[:, 1, 6, 1] -= 0.35
        interactions = cells[:, 0] - cells[:, 1] - cells[:, 2] + cells[:, 3]
        coupling = generator.normal(0, 0.3, (subject_count, 4, feature_count))
        coupling[:, 3, 0] += 0.65
        t_values = generator.normal(0, 1.3, (5, 7, feature_count))
        t_values[0, 6, 1] = 4.8
        p_values = np.clip(generator.uniform(0.08, 1, t_values.shape), 0, 1)
        p_values[0, 6, 1] = 0.012
        coupling_t = generator.normal(0, 1.2, (5, feature_count))
        coupling_t[0, 0] = 4.2
        coupling_p = np.clip(generator.uniform(0.08, 1, coupling_t.shape), 0, 1)
        coupling_p[0, 0] = 0.018
        common.update(
            {
                "network_cell_means": cells,
                "interaction": interactions,
                "fooof_t_values": t_values[:, :, :2].reshape(5, -1),
                "fooof_p_fwer": p_values[:, :, :2].reshape(5, -1),
                "corrected_psd_t_values": t_values[:, :, 2:].reshape(5, -1),
                "corrected_psd_p_fwer": p_values[:, :, 2:].reshape(5, -1),
                "coupling": coupling,
                "coupling_t_values": coupling_t,
                "coupling_p_fwer": coupling_p,
            }
        )
    return common


def _render_panel(
    panel: str, arrays: dict[str, np.ndarray], context: RenderContext
) -> list[Path]:
    """Render one final composite and every standalone component."""
    components = PANEL_COMPONENTS[panel]
    plotters = [_component_plotter(panel, index) for index in range(len(components))]
    composite_path = (
        context.reports_root
        / "figures"
        / PANEL_SPECS[panel].get("composite_directory", "paper")
        / PANEL_SPECS[panel]["composite_filename"]
    )
    figure, axes = _composite_canvas(panel, len(components))
    for index, (axis, title, plotter) in enumerate(zip(axes, components, plotters)):
        plotter(axis, arrays, index)
        axis.set_title(_component_title(panel, title), loc="left", fontsize=7, fontweight="bold")
        _watermark(axis, context.data_mode)
    figure.suptitle(_panel_title(panel), fontsize=11, fontweight="bold")
    _save_figure(figure, composite_path, context, panel, "composite", COMPOSITE_DPI)
    if panel == "panel2":
        _write_panel2_captions(composite_path, context)
    elif panel == "panel3":
        _write_panel3_captions(composite_path, context)
    plt.close(figure)
    outputs = [composite_path]
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
        axis.set_title(_component_title(panel, title), fontsize=24, fontweight="bold")
        _watermark(axis, context.data_mode)
        stem = f"{index + 1:02d}_{title}"
        png_path = slide_dir / f"{stem}.png"
        svg_path = slide_dir / f"{stem}.svg"
        _save_figure(standalone, png_path, context, panel, title, SLIDE_DPI)
        _save_figure(standalone, svg_path, context, panel, title, None)
        plt.close(standalone)
        outputs.extend((png_path, svg_path))
    return outputs


def _write_panel2_captions(composite_path: Path, context: RenderContext) -> None:
    """Write the Panel 2 caption beside manuscript and slide artifacts."""
    counts = _panel2_subject_counts(context.analysis_dir)
    caption = PANEL2_CAPTION.format(
        state_n=counts.get("state", "N/A"),
    )
    _write_captions(composite_path, context.reports_root, "panel2", caption)


def _write_panel3_captions(composite_path: Path, context: RenderContext) -> None:
    """Write the Panel 3 caption beside manuscript and slide artifacts."""
    summary = _panel3_summary(context.analysis_dir)
    caption = PANEL3_CAPTION.format(
        modulation_n=summary.get("modulation_complete_subject_n", "N/A"),
        coupling_n=summary.get("coupling_complete_subject_n", "N/A"),
    )
    _write_captions(composite_path, context.reports_root, "panel3", caption)


def _write_captions(
    composite_path: Path, reports_root: Path, panel: str, caption: str
) -> None:
    """Write one caption beside its composite and slide exports."""
    composite_path.with_suffix(".txt").write_text(caption)
    slide_caption = (
        reports_root
        / "figures"
        / "slides"
        / PANEL_SPECS[panel]["slide_directory"]
        / f"{Path(PANEL_SPECS[panel]['composite_filename']).stem}.txt"
    )
    slide_caption.parent.mkdir(parents=True, exist_ok=True)
    slide_caption.write_text(caption)


def _panel2_subject_counts(analysis_dir: Path | None) -> dict[str, int]:
    """Read per-model subject counts from the compact Panel 2 bundle."""
    if analysis_dir is None:
        return {}
    archive = analysis_dir / PANEL_ANALYSES["panel2"] / "observed.npz"
    with np.load(archive, allow_pickle=False) as arrays:
        return {"state": int(arrays["within_subject_auc"].size)}


def _panel3_summary(analysis_dir: Path | None) -> dict[str, Any]:
    """Load sample-size metadata for the Panel 3 caption."""
    if analysis_dir is None:
        return {}
    metadata = analysis_dir / PANEL_ANALYSES["panel3"] / "observed.json"
    return json.loads(metadata.read_text()).get("summary", {})


def _composite_canvas(panel: str, count: int) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create the focused publication geometry for one panel."""
    columns = 3
    rows = int(np.ceil(count / columns))
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=(10.5, 3.4 * rows),
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
    if panel == "panel3":
        return _PANEL3_PLOTTERS[index]
    if panel == "panel2":
        return _PANEL2_PLOTTERS[index]
    kinds = {}
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
            [band.display_name for band in CANONICAL_BANDS],
            key,
            decoding=decoding,
        )
        return
    summary = np.nanmean(values, axis=1)
    errors = np.nanstd(values, axis=1) / np.sqrt(values.shape[1])
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(CANONICAL_BANDS)))
    axis.bar(np.arange(len(summary)), summary, color=colors, width=0.75)
    axis.errorbar(
        np.arange(len(summary)), summary, yerr=errors, fmt="none", ecolor="black", lw=0.6
    )
    axis.set_xticks(
        np.arange(len(CANONICAL_BANDS)),
        [band.display_name for band in CANONICAL_BANDS],
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
        "fooof_auc": slice(7, 9),
        "corrected_psd_auc": slice(9, 16),
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
    values = np.resize(source, 7 * len(CORRECTED_FEATURES)).reshape(
        7, len(CORRECTED_FEATURES)
    )
    image = axis.imshow(values, cmap="viridis", aspect="auto")
    feature_labels = [FEATURE_DISPLAY_NAMES[name] for name in CORRECTED_FEATURES]
    network_labels = [network_display_name(name) for name in YEO7_NETWORKS]
    axis.set_xticks(np.arange(len(feature_labels)), feature_labels)
    axis.set_yticks(np.arange(len(network_labels)), network_labels)
    axis.tick_params(axis="x", labelrotation=45, labelsize=6)
    axis.tick_params(axis="y", labelsize=6)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Yeo network")
    axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)


def _plot_panel2_population(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    """Show population LOSO performance against its circular-shift null."""
    null = np.asarray(arrays["population_null"], dtype=float)
    observed = float(np.asarray(arrays["population_auc"]))
    axis.hist(null, bins=30, color="#B8C5D1", edgecolor="white")
    axis.axvline(observed, color="#D95319", lw=2, label=f"Observed {observed:.3f}")
    axis.set_xlabel("Population LOSO AUC")
    axis.set_ylabel("Permutation count")
    axis.legend(frameon=False, fontsize=7)


def _plot_panel2_within(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    """Show individual leave-one-run-out AUC values."""
    values = np.asarray(arrays["within_subject_auc"], dtype=float)
    ordered = np.sort(values)
    axis.scatter(np.arange(len(ordered)), ordered, color="#2878B5", s=16)
    axis.axhline(0.5, color="black", lw=0.7, ls="--")
    axis.set_xlabel("Participant (sorted)")
    axis.set_ylabel("Individual AUC")


def _plot_panel2_comparison(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    """Compare population and mean individual state decoding."""
    within = np.asarray(arrays["within_subject_auc"], dtype=float)
    values = [float(np.asarray(arrays["population_auc"])), float(np.nanmean(within))]
    error = [0.0, float(np.nanstd(within, ddof=1) / np.sqrt(len(within)))]
    axis.bar((0, 1), values, yerr=error, color=("#D95319", "#2878B5"), capsize=3)
    axis.set_xticks((0, 1), ("Population", "Individual mean"))
    axis.axhline(0.5, color="black", lw=0.7, ls="--")
    axis.set_ylabel("AUC")


def _plot_reliance_bar(
    axis: plt.Axes, arrays: dict[str, np.ndarray], family: str, labels: list[str]
) -> None:
    """Plot held-out reliance means and corrected significance."""
    values = np.asarray(arrays[f"population_{family}_reliance"], dtype=float)
    p_values = np.asarray(arrays[f"population_{family}_reliance_p_fwer"], dtype=float)
    means = np.nanmean(values, axis=0)
    errors = np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    positions = np.arange(len(means))
    axis.bar(positions, means, yerr=errors, color="#4C78A8", capsize=2)
    for position in positions[p_values < 0.05]:
        axis.text(position, means[position] + errors[position], "•", ha="center")
    axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=6)
    axis.set_ylabel("Held-out AUC decrease")


def _plot_panel2_feature_reliance(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    _plot_reliance_bar(axis, arrays, "feature", _feature_labels())


def _plot_panel2_network_reliance(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    _plot_reliance_bar(axis, arrays, "network", _network_labels())


def _plot_panel2_cell_reliance(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    """Plot the complete network-by-feature reliance matrix."""
    values = np.asarray(arrays["population_cell_reliance"], dtype=float)
    matrix = np.nanmean(values, axis=0).reshape(len(YEO7_NETWORKS), -1)
    p_values = np.asarray(
        arrays["population_cell_reliance_p_fwer"], dtype=float
    ).reshape(matrix.shape)
    _symmetric_heatmap(
        axis, matrix, row_labels=_network_labels(), column_labels=_feature_labels(),
        p_values=p_values, colorbar_label="Held-out AUC decrease",
    )


_PANEL2_PLOTTERS: tuple[
    Callable[[plt.Axes, dict[str, np.ndarray], int], None], ...
] = (
    _plot_panel2_population,
    _plot_panel2_within,
    _plot_panel2_comparison,
    _plot_panel2_feature_reliance,
    _plot_panel2_network_reliance,
    _plot_panel2_cell_reliance,
)


def _feature_labels() -> list[str]:
    """Return concise publication labels in canonical feature order."""
    return [FEATURE_DISPLAY_NAMES[name] for name in CORRECTED_FEATURES]


def _network_labels() -> list[str]:
    """Return full canonical Yeo-7 names."""
    return [network_display_name(name) for name in YEO7_NETWORKS]


def _network_inference(
    arrays: dict[str, np.ndarray], value: str
) -> np.ndarray:
    """Reassemble FOOOF and corrected-PSD network inference matrices."""
    fooof = np.asarray(arrays[f"fooof_{value}"], dtype=float).reshape(5, 7, 2)
    corrected = np.asarray(
        arrays[f"corrected_psd_{value}"], dtype=float
    ).reshape(5, 7, 7)
    return np.concatenate((fooof, corrected), axis=2)


def _symmetric_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    row_labels: list[str],
    column_labels: list[str],
    p_values: np.ndarray | None = None,
    colorbar_label: str = "t statistic",
) -> None:
    """Draw a centered heatmap with optional corrected-significance marks."""
    limit = max(float(np.nanmax(np.abs(values))), 1e-6)
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)), column_labels)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.tick_params(axis="x", labelrotation=45, labelsize=6)
    axis.tick_params(axis="y", labelsize=6)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    if p_values is not None:
        for row, column in np.argwhere(np.asarray(p_values) < 0.05):
            axis.text(column, row, "•", ha="center", va="center", fontsize=8)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    colorbar.set_label(colorbar_label, fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def _plot_factorial_profile(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    annotation: str,
    ylabel: str,
) -> None:
    """Plot correct and lapse IN-to-OUT profiles with 95% normal CIs."""
    means = np.nanmean(values, axis=0)
    counts = np.sum(np.isfinite(values), axis=0)
    errors = 1.96 * np.nanstd(values, axis=0, ddof=1) / np.sqrt(counts)
    styles = (
        ("Correct", [0, 2], "#2F4858", "o"),
        ("Lapse", [1, 3], "#E76F51", "s"),
    )
    for label, indices, color, marker in styles:
        axis.errorbar(
            (0, 1), means[indices], yerr=errors[indices], label=label,
            color=color, marker=marker, lw=2, capsize=3,
        )
    axis.set_xticks((0, 1), ("IN", "OUT"))
    axis.set_ylabel(ylabel, fontsize=7)
    axis.axhline(0, color="#B8B8B8", lw=0.7, zorder=0)
    axis.legend(frameon=False, fontsize=7, ncols=2)
    axis.text(0.02, 0.98, annotation, transform=axis.transAxes, va="top", fontsize=7)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_panel3_profile(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    t_values = _network_inference(arrays, "t_values")[0]
    network_index, feature_index = np.unravel_index(
        np.nanargmax(np.abs(t_values)), t_values.shape
    )
    values = np.asarray(arrays["network_cell_means"], dtype=float)[
        :, :, network_index, feature_index
    ]
    annotation = f"{_network_labels()[network_index]} · {_feature_labels()[feature_index]}"
    _plot_factorial_profile(axis, values, annotation=annotation, ylabel="Feature value")


def _plot_panel3_interactions(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    _symmetric_heatmap(
        axis,
        _network_inference(arrays, "t_values")[0],
        row_labels=_network_labels(),
        column_labels=_feature_labels(),
        p_values=_network_inference(arrays, "p_fwer")[0],
    )


def _plot_panel3_simple_effects(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    t_values = _network_inference(arrays, "t_values")[1:]
    p_values = _network_inference(arrays, "p_fwer")[1:]
    strongest = np.nanargmax(np.abs(t_values), axis=1)
    rows = np.arange(t_values.shape[0])[:, None]
    columns = np.arange(t_values.shape[2])[None, :]
    summary = t_values[rows, strongest, columns]
    summary_p = p_values[rows, strongest, columns]
    labels = ["Lapse−correct | IN", "Lapse−correct | OUT", "OUT−IN | correct", "OUT−IN | lapse"]
    _symmetric_heatmap(
        axis, summary, row_labels=labels, column_labels=_feature_labels(),
        p_values=summary_p,
    )
    axis.set_ylabel("Strongest Yeo-network expression", fontsize=7)


def _bootstrap_dz_interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Return Cohen's dz and deterministic percentile bootstrap interval."""
    clean = values[np.isfinite(values)]
    estimate = float(np.mean(clean) / np.std(clean, ddof=1))
    generator = np.random.default_rng(seed)
    samples = generator.choice(clean, (500, len(clean)), replace=True)
    bootstrap = np.mean(samples, axis=1) / np.std(samples, axis=1, ddof=1)
    lower, upper = np.nanpercentile(bootstrap, (2.5, 97.5))
    return estimate, float(lower), float(upper)


def _plot_panel3_forest(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    interactions = np.asarray(arrays["interaction"], dtype=float)
    t_values = _network_inference(arrays, "t_values")[0]
    top = np.argsort(np.abs(t_values).ravel())[-7:]
    labels: list[str] = []
    estimates, lower, upper = [], [], []
    for rank, flat_index in enumerate(top):
        network_index, feature_index = np.unravel_index(flat_index, t_values.shape)
        estimate, low, high = _bootstrap_dz_interval(
            interactions[:, network_index, feature_index], 4200 + rank
        )
        estimates.append(estimate)
        lower.append(low)
        upper.append(high)
        labels.append(f"{_network_labels()[network_index]} · {_feature_labels()[feature_index]}")
    positions = np.arange(len(top))
    estimates_array = np.asarray(estimates)
    axis.errorbar(
        estimates_array, positions,
        xerr=(estimates_array - lower, np.asarray(upper) - estimates_array),
        fmt="o", color="#315A7D", ecolor="#8AA6B8", capsize=2,
    )
    axis.axvline(0, color="#777777", lw=0.8)
    axis.set_yticks(positions, labels, fontsize=6)
    axis.set_xlabel("Interaction effect (Cohen's $d_z$)", fontsize=7)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_panel3_coupling_profile(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    t_values = np.asarray(arrays["coupling_t_values"], dtype=float)[0]
    feature_index = int(np.nanargmax(np.abs(t_values)))
    values = np.asarray(arrays["coupling"], dtype=float)[:, :, feature_index]
    _plot_factorial_profile(
        axis, values, annotation=_feature_labels()[feature_index],
        ylabel="DMN–DAN coupling (Fisher z)",
    )


def _plot_panel3_coupling_contrasts(
    axis: plt.Axes, arrays: dict[str, np.ndarray], _: int
) -> None:
    labels = [
        "Interaction", "Lapse−correct | IN", "Lapse−correct | OUT",
        "OUT−IN | correct", "OUT−IN | lapse",
    ]
    _symmetric_heatmap(
        axis, np.asarray(arrays["coupling_t_values"], dtype=float),
        row_labels=labels, column_labels=_feature_labels(),
        p_values=np.asarray(arrays["coupling_p_fwer"], dtype=float),
    )


_PANEL3_PLOTTERS: tuple[
    Callable[[plt.Axes, dict[str, np.ndarray], int], None], ...
] = (
    _plot_panel3_profile,
    _plot_panel3_interactions,
    _plot_panel3_simple_effects,
    _plot_panel3_forest,
    _plot_panel3_coupling_profile,
    _plot_panel3_coupling_contrasts,
)


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
    sidecar = _figure_sidecar(path, component)
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
                str(
                    context.analysis_dir
                    / PANEL_ANALYSES[panel]
                    / "observed.npz"
                ),
                str(
                    context.analysis_dir
                    / PANEL_ANALYSES[panel]
                    / "observed.json"
                ),
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


def _figure_sidecar(path: Path, component: str) -> Path:
    """Use a clean composite JSON name while distinguishing slide formats."""
    return path.with_suffix(".json") if component == "composite" else path.with_name(
        f"{path.name}.json"
    )


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


def _component_title(panel: str, component: str) -> str:
    """Return concise narrative titles while preserving stable artifact IDs."""
    if panel != "panel3":
        return component.replace("_", " ")
    return {
        "A_four_cell_overview": "A  Strongest factorial profile",
        "B_interaction": "B  State × outcome interactions",
        "C_simple_effects": "C  Prespecified simple effects",
        "D_network_summary": "D  Largest interaction effect sizes",
        "E_dmn_dan_coupling": "E  DMN–DAN coupling profile",
        "F_coupling_contrasts": "F  DMN–DAN coupling contrasts",
    }[component]


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
    """Render panel analysis from CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="all")
    parser.add_argument("--analysis-id")
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    arguments = parser.parse_args()
    render_panels(
        panel=arguments.panel,
        analysis_id=arguments.analysis_id,
        analysis_root=arguments.analysis_root,
        reports_root=arguments.reports_root,
    )


if __name__ == "__main__":
    main()
