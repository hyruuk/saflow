"""Surface-based visualization for source and atlas-level statistical results.

Renders inflated brain maps (lateral + medial views, both hemispheres) using
nilearn, in the style of mario_fmri annotation panels. Supports:
- Atlas-level: ROI values painted onto fsaverage surface via MNE labels
- Source-level: vertex-level data rendered directly on fsaverage

Requires: nilearn, mne, matplotlib, numpy
"""

import logging
import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# View configuration matching mario_fmri style: 2x2 grid per hemisphere pair
VIEWS = [
    ("left", "lateral"),
    ("right", "lateral"),
    ("left", "medial"),
    ("right", "medial"),
]


def _get_fsaverage_surfaces() -> dict:
    """Fetch fsaverage inflated surfaces and sulcal depth maps from nilearn."""
    from nilearn import datasets

    return datasets.fetch_surf_fsaverage("fsaverage")


def _get_subjects_dir() -> Path:
    """Get the MNE fsaverage subjects directory."""
    import mne

    fsaverage_path = mne.datasets.fetch_fsaverage(verbose=False)
    return Path(fsaverage_path).parent


# Schaefer parcellation sizes available from the CBIG repo
_SCHAEFER_PARCELS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

_CBIG_BASE_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage/label"
)


def _ensure_schaefer_annot(atlas_name: str) -> None:
    """Download Schaefer annotation files if not present locally.

    Fetches .annot files from the CBIG GitHub repository into the
    MNE fsaverage label directory.
    """
    from code.source_reconstruction.apply_atlas import get_mne_atlas_name

    mne_name = get_mne_atlas_name(atlas_name)
    if not mne_name.startswith("Schaefer"):
        return

    subjects_dir = _get_subjects_dir()
    label_dir = subjects_dir / "fsaverage" / "label"

    # Check if already installed
    lh_annot = label_dir / f"lh.{mne_name}.annot"
    if lh_annot.exists():
        return

    import urllib.request

    logger.info(f"Downloading Schaefer annotation files for '{atlas_name}'...")
    for hemi in ["lh", "rh"]:
        fname = f"{hemi}.{mne_name}.annot"
        url = f"{_CBIG_BASE_URL}/{fname}"
        target = label_dir / fname
        try:
            urllib.request.urlretrieve(url, str(target))
            logger.info(f"  Downloaded {fname}")
        except Exception as e:
            logger.error(f"  Failed to download {fname}: {e}")
            raise


# In-process cache: (atlas_name, n_networks) -> (lh_labels, rh_labels). The
# overlay is reused across every brain panel in a figure; reading the annot
# + mapping vertices is the most expensive step.
_YEO_VERTEX_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}


def yeo_vertex_labels(
    atlas_name: str = "schaefer_400",
    n_networks: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-vertex Yeo network indices for fsaverage (lh, rh).

    Vertices not belonging to any cortical parcel (e.g. medial wall) get
    ``-1``. Indices match ``network_order(n_networks)`` ordering.
    """
    from code.source_reconstruction.apply_atlas import get_mne_atlas_name
    from code.utils.yeo_networks import (
        UNKNOWN_NETWORK,
        get_network_assignments,
        network_order,
    )
    import mne

    key = (atlas_name, n_networks)
    if key in _YEO_VERTEX_CACHE:
        return _YEO_VERTEX_CACHE[key]

    _ensure_schaefer_annot(atlas_name)
    subjects_dir = str(_get_subjects_dir())
    mne_atlas = get_mne_atlas_name(atlas_name)

    labels = mne.read_labels_from_annot(
        "fsaverage", parc=mne_atlas, subjects_dir=subjects_dir, verbose=False
    )
    label_names = [lab.name for lab in labels]
    assignments = get_network_assignments(label_names, n_networks=n_networks)
    nets = network_order(n_networks)
    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(nets)}

    n_vertices = 163842
    lh = np.full(n_vertices, -1, dtype=np.int16)
    rh = np.full(n_vertices, -1, dtype=np.int16)
    for lab, net in zip(labels, assignments):
        if net == UNKNOWN_NETWORK:
            continue
        idx = name_to_idx[net]
        if lab.hemi == "lh":
            lh[lab.vertices] = idx
        else:
            rh[lab.vertices] = idx

    _YEO_VERTEX_CACHE[key] = (lh, rh)
    return lh, rh


def roi_to_surface(
    roi_values: np.ndarray,
    roi_names: List[str],
    atlas_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map ROI values onto fsaverage surface vertices.

    Uses MNE labels from read_labels_from_annot to map each ROI value
    to the corresponding vertices on the fsaverage surface.

    Args:
        roi_values: Array of shape (n_rois,) with one value per ROI.
        roi_names: List of ROI names matching the atlas labels (with -lh/-rh suffix).
        atlas_name: Short atlas name (e.g., "aparc.a2009s", "schaefer_400").

    Returns:
        Tuple of (lh_data, rh_data) arrays, each of shape (n_vertices,),
        with NaN for vertices not in any ROI.
    """
    import mne

    from code.source_reconstruction.apply_atlas import get_mne_atlas_name

    _ensure_schaefer_annot(atlas_name)

    subjects_dir = str(_get_subjects_dir())
    mne_atlas = get_mne_atlas_name(atlas_name)

    labels = mne.read_labels_from_annot(
        "fsaverage", parc=mne_atlas, subjects_dir=subjects_dir, verbose=False
    )

    # Build lookup: label name -> label object
    label_map = {label.name: label for label in labels}

    # fsaverage surface vertex count
    n_vertices = 163842
    lh_data = np.full(n_vertices, np.nan)
    rh_data = np.full(n_vertices, np.nan)

    mapped_count = 0
    for roi_name, value in zip(roi_names, roi_values):
        if roi_name in label_map:
            label = label_map[roi_name]
            if label.hemi == "lh":
                lh_data[label.vertices] = value
            else:
                rh_data[label.vertices] = value
            mapped_count += 1
        else:
            logger.debug(f"ROI '{roi_name}' not found in atlas labels")

    logger.info(f"Mapped {mapped_count}/{len(roi_names)} ROIs to surface vertices")
    return lh_data, rh_data


# In-process cache: atlas_name -> (label_map, hemi-coords, hemi-midline-x).
_ROI_FACE_CACHE: Dict[str, tuple] = {}


def roi_view_faces(
    roi_names: List[str],
    atlas_name: str,
) -> List[Tuple[str, str]]:
    """Per-ROI ``(hemi, view)`` on the inflated fsaverage surface.

    ``hemi`` is ``"left"``/``"right"`` (from the ``-lh``/``-rh`` suffix);
    ``view`` is ``"medial"``/``"lateral"`` decided by the mean inflated-x of
    the ROI's vertices relative to its hemisphere's midline — "medial" means
    the vertices face the contralateral hemisphere. Used to auto-pick the
    single brain face that best shows a set of selected regions. Unknown ROI
    names fall back to ``("left", "lateral")``.
    """
    import mne
    import nilearn.surface as ns

    from code.source_reconstruction.apply_atlas import get_mne_atlas_name

    if atlas_name not in _ROI_FACE_CACHE:
        _ensure_schaefer_annot(atlas_name)
        subjects_dir = str(_get_subjects_dir())
        labels = mne.read_labels_from_annot(
            "fsaverage", parc=get_mne_atlas_name(atlas_name),
            subjects_dir=subjects_dir, verbose=False,
        )
        label_map = {label.name: label for label in labels}
        fsaverage = _get_fsaverage_surfaces()
        coords = {
            "left": np.asarray(ns.load_surf_mesh(fsaverage["infl_left"]).coordinates),
            "right": np.asarray(ns.load_surf_mesh(fsaverage["infl_right"]).coordinates),
        }
        midline = {h: float(c[:, 0].mean()) for h, c in coords.items()}
        _ROI_FACE_CACHE[atlas_name] = (label_map, coords, midline)

    label_map, coords, midline = _ROI_FACE_CACHE[atlas_name]
    faces: List[Tuple[str, str]] = []
    for name in roi_names:
        label = label_map.get(name)
        if label is None:
            faces.append(("left", "lateral"))
            continue
        hemi = "right" if label.hemi == "rh" else "left"
        mean_x = float(coords[hemi][label.vertices, 0].mean())
        # Left hemi: medial faces +x (toward midline); right hemi: -x.
        sign = 1.0 if hemi == "left" else -1.0
        view = "medial" if (mean_x - midline[hemi]) * sign > 0 else "lateral"
        faces.append((hemi, view))
    return faces


def render_inflated_view(
    lh_data: np.ndarray,
    rh_data: np.ndarray,
    hemi: str,
    view: str,
    fsaverage: dict,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    threshold: Optional[float] = None,
    yeo_overlay: Optional[int] = None,
    yeo_atlas_name: str = "schaefer_400",
    yeo_outline_color: str = "white",
    yeo_outline_lw: float = 0.6,
) -> np.ndarray:
    """Render a single inflated brain view and return as RGB array.

    Args:
        lh_data: Left hemisphere vertex data, shape (n_vertices,).
        rh_data: Right hemisphere vertex data, shape (n_vertices,).
        hemi: "left" or "right".
        view: "lateral" or "medial".
        fsaverage: nilearn fsaverage surface dict.
        cmap: Matplotlib colormap name.
        vmin, vmax: Color limits (symmetric around 0 for diverging cmaps).
        threshold: Values below this absolute threshold are not shown.
        yeo_overlay: If 7 or 17, draw Yeo network boundaries on top of the
            stat map. ``None`` (default) draws no overlay.
        yeo_atlas_name: Schaefer parcellation used to derive the Yeo
            assignments (e.g. ``schaefer_400``). The overlay is independent
            of the stat-map data; this only controls which Schaefer
            annotation file the network labels are read from.
        yeo_outline_color: Outline color (matplotlib spec).
        yeo_outline_lw: Outline line-width.

    Returns:
        RGB image array of shape (H, W, 3).
    """
    from nilearn import plotting

    surf_mesh = fsaverage[f"infl_{hemi}"]
    bg_map = fsaverage[f"sulc_{hemi}"]
    stat_map = lh_data if hemi == "left" else rh_data

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), subplot_kw={"projection": "3d"})
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Auto-close\(\)ing of figures upon backend switching.*",
            category=MatplotlibDeprecationWarning,
        )
        plotting.plot_surf_stat_map(
            surf_mesh,
            stat_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            colorbar=False,
            axes=ax,
            engine="matplotlib",
            darkness=None,
        )

        if yeo_overlay is not None:
            yeo_lh, yeo_rh = yeo_vertex_labels(
                atlas_name=yeo_atlas_name, n_networks=yeo_overlay,
            )
            roi_map = yeo_lh if hemi == "left" else yeo_rh
            # plot_surf_contours draws boundaries between distinct label
            # values; levels=positive label indices avoids the "Unknown"
            # sentinel (-1).
            levels = [int(v) for v in np.unique(roi_map) if v >= 0]
            try:
                plotting.plot_surf_contours(
                    surf_mesh,
                    roi_map=roi_map,
                    hemi=hemi,
                    view=view,
                    levels=levels,
                    colors=[yeo_outline_color] * len(levels),
                    linewidth=yeo_outline_lw,
                    axes=ax,
                    figure=fig,
                )
            except Exception as exc:
                logger.warning(f"Yeo overlay skipped on {hemi}/{view}: {exc}")

    # Render to array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    return _crop_whitespace(img)


def _crop_whitespace(img: np.ndarray, bg_color: int = 255) -> np.ndarray:
    """Crop whitespace borders from an RGB image array."""
    # Find non-white rows and columns
    non_white = np.any(img != bg_color, axis=2)
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)

    if not rows.any() or not cols.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin : rmax + 1, cmin : cmax + 1]


def plot_inflated_stat_map(
    lh_data: np.ndarray,
    rh_data: np.ndarray,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    threshold: Optional[float] = None,
    yeo_overlay: Optional[int] = None,
    yeo_atlas_name: str = "schaefer_400",
) -> Figure:
    """Create a 2x2 panel of inflated brain views for a single stat map.

    Layout:
        Left Lateral   | Right Lateral
        Left Medial    | Right Medial

    Args:
        lh_data: Left hemisphere data, shape (n_vertices,).
        rh_data: Right hemisphere data, shape (n_vertices,).
        title: Figure title.
        cmap: Colormap name.
        vmin, vmax: Symmetric color limits.
        threshold: Absolute threshold below which values are hidden.
        yeo_overlay: 7 / 17 / None — draw Yeo-N network boundaries on top.
        yeo_atlas_name: Schaefer parcellation that backs the Yeo labels.

    Returns:
        Matplotlib Figure with the 2x2 brain panel.
    """
    fsaverage = _get_fsaverage_surfaces()

    # Render each view
    images = {}
    for hemi, view in VIEWS:
        images[(hemi, view)] = render_inflated_view(
            lh_data, rh_data, hemi, view, fsaverage,
            cmap=cmap, vmin=vmin, vmax=vmax, threshold=threshold,
            yeo_overlay=yeo_overlay, yeo_atlas_name=yeo_atlas_name,
        )

    # Resize all views to same height
    target_height = max(img.shape[0] for img in images.values())
    resized = {}
    for key, img in images.items():
        if img.shape[0] != target_height:
            scale = target_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            from PIL import Image

            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((new_width, target_height), Image.LANCZOS)
            resized[key] = np.array(pil_img)
        else:
            resized[key] = img

    # Assemble 2x2: top = laterals, bottom = medials
    top = np.concatenate(
        [resized[("left", "lateral")], resized[("right", "lateral")]],
        axis=1,
    )
    bottom = np.concatenate(
        [resized[("left", "medial")], resized[("right", "medial")]],
        axis=1,
    )

    # Match widths
    max_width = max(top.shape[1], bottom.shape[1])
    for arr_name in ["top", "bottom"]:
        arr = locals()[arr_name]
        if arr.shape[1] < max_width:
            pad = np.full((arr.shape[0], max_width - arr.shape[1], 3), 255, dtype=np.uint8)
            if arr_name == "top":
                top = np.concatenate([arr, pad], axis=1)
            else:
                bottom = np.concatenate([arr, pad], axis=1)

    composite = np.concatenate([top, bottom], axis=0)

    # Create figure from composite
    dpi = 150
    fig_w = composite.shape[1] / dpi
    fig_h = composite.shape[0] / dpi + 0.4  # extra for title
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, composite.shape[0] / (composite.shape[0] + 0.4 * dpi)])
    ax.imshow(composite)
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    return fig


def plot_surface_stats_panel(
    results_files: List[Path],
    feature_names: List[str],
    space: str,
    alpha: float = 0.05,
    cmap: str = "RdBu_r",
    fig_title: str = "",
    cbar_label: str = "t-value",
    yeo_overlay: Optional[int] = None,
    yeo_atlas_name: Optional[str] = None,
) -> Figure:
    """Create a panel of inflated brain maps for multiple features.

    Each feature gets a 2x2 brain view panel (lat/med x L/R).
    A shared colorbar is added on the right.

    Args:
        results_files: List of .npz result files from run_group_statistics.
        feature_names: List of feature names (e.g., ["psd_theta", "psd_alpha"]).
        space: Analysis space (atlas name like "schaefer_400", or "source").
        alpha: Significance threshold for highlighting.
        cmap: Colormap name.
        fig_title: Overall figure title.

    Returns:
        Matplotlib Figure with the full panel.
    """
    import json

    fsaverage = _get_fsaverage_surfaces()
    n_features = len(results_files)

    # Load all results and determine global color limits
    all_results = []
    all_metadata = []
    all_tvals = []

    for results_file in results_files:
        results = np.load(results_file, allow_pickle=True)
        all_results.append(results)

        metadata_file = results_file.with_name(
            results_file.stem.replace("_results", "_metadata") + ".json"
        )
        if metadata_file.exists():
            with open(metadata_file) as f:
                all_metadata.append(json.load(f))
        else:
            all_metadata.append({})

        if "tvals" in results.files:
            all_tvals.append(results["tvals"].flatten())

    # Global symmetric color limits
    if all_tvals:
        all_vals = np.concatenate(all_tvals)
        vmax = float(np.nanpercentile(np.abs(all_vals), 98))
        vmin = -vmax
    else:
        vmin, vmax = -3.0, 3.0

    # Load ROI names from metadata to map results back to surface
    roi_names = _get_roi_names(space, all_results, all_metadata)

    # Render each feature
    feature_images = []
    feature_titles = []

    for feat_name, results in zip(feature_names, all_results):
        if "tvals" not in results.files:
            continue

        tvals = results["tvals"].flatten()

        # Get significance info. Canonical keys first; legacy keys retained
        # so old stats npz files still load.
        n_sig = 0
        for key in ("pvals_fdr_bh", "pvals_tmax", "pvals_bonferroni",
                    "pvals_corrected_fdr_bh", "pvals_corrected_fdr",
                    "pvals_corrected_tmax", "pvals_corrected_bonferroni"):
            if key in results.files:
                pvals = results[key].flatten()
                n_sig = int(np.sum(pvals < alpha))
                break

        # Map to surface
        lh_data, rh_data = _stat_values_to_surface(tvals, space, roi_names)

        # Render 4 views (with optional Yeo network outline overlay)
        overlay_atlas = yeo_atlas_name or space
        images = {}
        for hemi, view in VIEWS:
            images[(hemi, view)] = render_inflated_view(
                lh_data, rh_data, hemi, view, fsaverage,
                cmap=cmap, vmin=vmin, vmax=vmax,
                yeo_overlay=yeo_overlay,
                yeo_atlas_name=overlay_atlas,
            )

        # Resize to uniform height
        target_height = max(img.shape[0] for img in images.values())
        resized = {}
        for key, img in images.items():
            if img.shape[0] != target_height:
                scale = target_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(img)
                pil_img = pil_img.resize((new_width, target_height), PILImage.LANCZOS)
                resized[key] = np.array(pil_img)
            else:
                resized[key] = img

        # Assemble 2x2
        top = np.concatenate(
            [resized[("left", "lateral")], resized[("right", "lateral")]],
            axis=1,
        )
        bottom = np.concatenate(
            [resized[("left", "medial")], resized[("right", "medial")]],
            axis=1,
        )
        max_width = max(top.shape[1], bottom.shape[1])
        if top.shape[1] < max_width:
            top = np.concatenate(
                [top, np.full((top.shape[0], max_width - top.shape[1], 3), 255, dtype=np.uint8)],
                axis=1,
            )
        if bottom.shape[1] < max_width:
            bottom = np.concatenate(
                [bottom, np.full((bottom.shape[0], max_width - bottom.shape[1], 3), 255, dtype=np.uint8)],
                axis=1,
            )
        composite = np.concatenate([top, bottom], axis=0)
        feature_images.append(composite)

        short_name = feat_name.replace("psd_corrected_", "").replace("psd_", "").replace("fooof_", "")
        feature_titles.append(f"{short_name}\n(n={n_sig} sig)")

    if not feature_images:
        fig, ax = plt.subplots(1, 1)
        ax.text(0.5, 0.5, "No results to display", ha="center", va="center")
        return fig

    # Resize all feature composites to same dimensions
    max_h = max(img.shape[0] for img in feature_images)
    max_w = max(img.shape[1] for img in feature_images)

    padded = []
    for img in feature_images:
        pad_h = max_h - img.shape[0]
        pad_w = max_w - img.shape[1]
        if pad_h > 0 or pad_w > 0:
            img = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        padded.append(img)

    # Arrange in a row (one 2x2 brain per feature)
    panel = np.concatenate(padded, axis=1)

    # Build figure
    dpi = 150
    fig_w = panel.shape[1] / dpi + 1.0  # extra for colorbar
    fig_h = panel.shape[0] / dpi + 0.6  # extra for titles
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")

    # Main image
    ax_img = fig.add_axes([0, 0, panel.shape[1] / (fig_w * dpi), panel.shape[0] / (fig_h * dpi)])
    ax_img.imshow(panel)
    ax_img.axis("off")

    # Feature titles above each brain panel
    for i, title in enumerate(feature_titles):
        x_center = (i + 0.5) * max_w / panel.shape[1] * (panel.shape[1] / (fig_w * dpi))
        fig.text(x_center, 0.92, title, ha="center", va="bottom", fontsize=10)

    # Colorbar
    import matplotlib.colors as mcolors

    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels([f"{vmin:.1f}", "0", f"{vmax:.1f}"])

    if fig_title:
        fig.suptitle(fig_title, fontsize=12, y=0.99)

    return fig


def _get_roi_names(
    space: str,
    all_results: list,
    all_metadata: list,
) -> Optional[List[str]]:
    """Extract ROI names for atlas spaces.

    Checks metadata first, then falls back to loading from atlas definition.
    """
    if space == "source":
        return None

    # Try metadata (check both key names)
    for meta in all_metadata:
        roi_names = meta.get("spatial_names") or meta.get("roi_names")
        if not roi_names:
            data_meta = meta.get("data_metadata", {})
            roi_names = data_meta.get("spatial_names") or data_meta.get("roi_names")
        if roi_names:
            return roi_names

    # Try loading from a sample feature file
    # ROI names are stored in the atlas timeseries files
    logger.warning(
        f"ROI names not found in metadata for space '{space}'. "
        "Will attempt to infer from atlas definition."
    )
    return _load_roi_names_from_atlas(space)


def _load_roi_names_from_atlas(atlas_name: str) -> Optional[List[str]]:
    """Load ROI names by reading atlas labels from fsaverage."""
    import mne

    from code.source_reconstruction.apply_atlas import get_mne_atlas_name

    _ensure_schaefer_annot(atlas_name)

    subjects_dir = str(_get_subjects_dir())
    mne_atlas = get_mne_atlas_name(atlas_name)

    try:
        labels = mne.read_labels_from_annot(
            "fsaverage", parc=mne_atlas, subjects_dir=subjects_dir, verbose=False
        )
        # Return sorted names (matching the convention in apply_atlas)
        names = sorted(label.name for label in labels)
        logger.info(f"Loaded {len(names)} ROI names from atlas '{atlas_name}'")
        return names
    except Exception as e:
        logger.error(f"Failed to load atlas labels for '{atlas_name}': {e}")
        return None


def _stat_values_to_surface(
    values: np.ndarray,
    space: str,
    roi_names: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert stat values (per-ROI or per-vertex) to surface vertex arrays.

    Args:
        values: 1D array of values (n_rois for atlas, n_vertices for source).
        space: Analysis space name.
        roi_names: ROI names for atlas spaces.

    Returns:
        (lh_data, rh_data) arrays for fsaverage surface.
    """
    if space == "source":
        # Source-level: values are already per-vertex
        # Assume same vertex layout as fsaverage (lh + rh)
        n_per_hemi = 163842
        if len(values) == 2 * n_per_hemi:
            return values[:n_per_hemi], values[n_per_hemi:]
        # If different vertex count, pad/truncate
        logger.warning(
            f"Source data has {len(values)} vertices, expected {2 * n_per_hemi}. "
            "Mapping may be approximate."
        )
        lh_data = np.full(n_per_hemi, np.nan)
        rh_data = np.full(n_per_hemi, np.nan)
        n_half = len(values) // 2
        lh_data[:n_half] = values[:n_half]
        rh_data[:n_half] = values[n_half:]
        return lh_data, rh_data

    # Atlas-level: paint ROI values onto vertices
    if roi_names is None:
        logger.error("Cannot map atlas values to surface without ROI names")
        n_vertices = 163842
        return np.full(n_vertices, np.nan), np.full(n_vertices, np.nan)

    return roi_to_surface(values, roi_names, space)
