"""Visualize statistical results as topographic maps or inflated brain surfaces.

For sensor space: creates a panel of MNE topomaps (t-values).
For source/atlas spaces: creates inflated brain surface maps (t-values).

Usage:
    python -m code.visualization.run_viz_stats --feature-type psd --space sensor
    python -m code.visualization.run_viz_stats --feature-type fooof_exponent --space schaefer_400
    python -m code.visualization.run_viz_stats --feature-type psd --space aparc.a2009s --show
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Frequency band sort order
BAND_ORDER = ["delta", "theta", "alpha", "lobeta", "hibeta", "gamma1", "gamma2", "gamma3"]


def get_band_index(feat_name: str) -> int:
    for i, band in enumerate(BAND_ORDER):
        if feat_name.endswith(f"_{band}") or feat_name == band:
            return i
    return 999


def find_results(stats_dir: Path, feature_type: str, inout_str: str):
    """Find matching result files, supporting partial matching."""
    exact = stats_dir / f"feature-{feature_type}_inout-{inout_str}_test-paired_ttest_results.npz"
    if exact.exists():
        return [exact]
    pattern = f"feature-{feature_type}_*_inout-{inout_str}_test-paired_ttest_results.npz"
    return sorted(stats_dir.glob(pattern))


def extract_feature_names(results_files):
    """Extract feature names from result filenames."""
    names = []
    for f in results_files:
        name = f.stem.replace("_results", "")
        parts = name.split("_inout-")
        if parts:
            names.append(parts[0].replace("feature-", ""))
    return names


def viz_sensor(results_files, feature_names, alpha, cmap, config, data_root, cbar_label="t-value"):
    """Create sensor-level topomap panel."""
    import mne

    # Get sensor info
    logger.info("Loading sensor positions...")
    derivatives_dir = data_root / config["paths"]["derivatives"]
    sample_files = list(derivatives_dir.glob("preprocessed/sub-*/meg/*_proc-clean_meg.fif"))
    if not sample_files:
        sample_files = list(derivatives_dir.glob("**/sub-*/meg/*_meg.fif"))
    if not sample_files:
        logger.error("No preprocessed MEG files found to get sensor positions")
        return None

    raw = mne.io.read_raw_fif(sample_files[0], preload=False, verbose=False)
    raw.pick("mag")
    info = raw.info

    maps_to_plot = []
    masks_to_plot = []
    titles = []

    for feat_name, results_file in zip(feature_names, results_files):
        results = np.load(results_file, allow_pickle=True)
        if "tvals" in results.files:
            tvals = results["tvals"].flatten()
            maps_to_plot.append(tvals)
            mask = None
            n_sig = 0
            for key in results.files:
                if key.startswith("pvals_corrected_"):
                    pvals = results[key].flatten()
                    mask = pvals < alpha
                    n_sig = int(np.sum(mask))
                    break
            masks_to_plot.append(mask)
            short_name = feat_name.replace("psd_corrected_", "").replace("psd_", "").replace("fooof_", "")
            titles.append(f"{short_name}\n(n={n_sig} sig)")

    n_maps = len(maps_to_plot)
    logger.info(f"Creating panel with {n_maps} topomaps (t-values)...")

    fig, axes = plt.subplots(1, n_maps, figsize=(3 * n_maps + 1, 3.5), dpi=150)
    if n_maps == 1:
        axes = [axes]

    all_values = np.concatenate(maps_to_plot)
    vmax = float(np.nanpercentile(np.abs(all_values), 98))
    vmin = -vmax

    mask_params = dict(
        marker="o", markerfacecolor="w", markeredgecolor="k", linewidth=0, markersize=5,
    )

    im = None
    for ax, data, mask, title in zip(axes, maps_to_plot, masks_to_plot, titles):
        im = mne.viz.plot_topomap(
            data, info, axes=ax, show=False, cmap=cmap, mask=mask,
            mask_params=mask_params, vlim=(vmin, vmax),
            extrapolate="local", outlines="head", sphere=0.15, contours=0,
        )
        ax.set_title(title, fontsize=10)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im[0], cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

    fig.subplots_adjust(left=0.02, right=0.88, top=0.85, bottom=0.05, wspace=0.1)
    return fig


def viz_surface(results_files, feature_names, space, alpha, cmap, cbar_label="t-value"):
    """Create source/atlas-level inflated brain surface panel."""
    from code.visualization.plot_surface import plot_surface_stats_panel

    logger.info(f"Creating inflated brain surface panel for space '{space}'...")
    return plot_surface_stats_panel(
        results_files=results_files,
        feature_names=feature_names,
        space=space,
        alpha=alpha,
        cmap=cmap,
        cbar_label=cbar_label,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize statistical results as topographic or surface maps"
    )
    parser.add_argument(
        "--feature-type", required=True,
        help="Feature to visualize (e.g., fooof_exponent, psd_alpha, psd, fooof)",
    )
    parser.add_argument(
        "--space", default="sensor",
        help="Analysis space: 'sensor', 'source', or atlas name (e.g., 'schaefer_400')",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--cmap", default=None, help="Override colormap (default: from config)")
    parser.add_argument("--show", action="store_true", help="Display figure interactively")
    parser.add_argument("--no-save", action="store_true", help="Don't save figure to disk")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_root = Path(config["paths"]["data_root"])
    inout_bounds = config["analysis"]["inout_bounds"]
    inout_str = f"{inout_bounds[0]}{inout_bounds[1]}"
    stats_dir = data_root / "features" / f"statistics_{args.space}"

    print("=" * 80)
    print(f"Visualizing {args.feature_type} Statistics (space={args.space})")
    print("=" * 80)

    # Find results
    results_files = find_results(stats_dir, args.feature_type, inout_str)
    if not results_files:
        print(f"ERROR: No results found for feature type '{args.feature_type}'")
        available = list(stats_dir.glob("feature-*_results.npz"))
        if available:
            print(f"\nAvailable feature types in {stats_dir.name}/:")
            for f in sorted(available):
                name = f.stem.replace("_results", "")
                parts = name.split("_inout-")
                if parts:
                    feat = parts[0].replace("feature-", "")
                    print(f"  --feature-type={feat}")
        else:
            print(f"\nNo statistics found in {stats_dir}.")
            print(f"Run analysis.stats.* tasks first (e.g., invoke analysis.stats.psd --space={args.space})")
        return

    feature_names = extract_feature_names(results_files)

    # Sort by band order
    sorted_pairs = sorted(zip(feature_names, results_files), key=lambda x: get_band_index(x[0]))
    feature_names = [p[0] for p in sorted_pairs]
    results_files = [p[1] for p in sorted_pairs]

    print(f"Found {len(results_files)} feature(s): {', '.join(feature_names)}")

    # Resolve colormap from config (CLI --cmap overrides)
    from code.utils.visualization import resolve_colormap
    preset = resolve_colormap(args.feature_type, config, override=args.cmap)
    print(f"Colormap: {preset.cmap} ({preset.cbar_label})")

    # Dispatch
    if args.space == "sensor":
        fig = viz_sensor(
            results_files, feature_names, args.alpha, preset.cmap,
            config, data_root, cbar_label=preset.cbar_label,
        )
    else:
        fig = viz_surface(
            results_files, feature_names, args.space, args.alpha,
            preset.cmap, cbar_label=preset.cbar_label,
        )

    if fig is None:
        return

    # Title
    first_meta_file = results_files[0].with_name(
        results_files[0].stem.replace("_results", "_metadata") + ".json"
    )
    data_meta = {}
    if first_meta_file.exists():
        with open(first_meta_file) as f:
            meta = json.load(f)
            data_meta = meta.get("data_metadata", {})

    fig_title = (
        f"{args.feature_type} t-values ({args.space}) | IN/OUT: {inout_bounds} "
        f"| N={data_meta.get('n_subjects', '?')} subjects"
    )
    fig.suptitle(fig_title, fontsize=12, y=1.02)

    # Save
    if not args.no_save:
        fig_dir = Path("reports") / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"{args.feature_type}_{args.space}_stats.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to: {fig_path}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        plt.close(fig)

    # Summary
    print(f"\nResults summary:")
    print(f"  Features: {', '.join(feature_names)}")
    print(f"  Space: {args.space}")
    print(f"  Subjects: {data_meta.get('n_subjects', 'unknown')}")
    print(f"  IN trials: {data_meta.get('n_in', 'unknown')}")
    print(f"  OUT trials: {data_meta.get('n_out', 'unknown')}")


if __name__ == "__main__":
    main()
