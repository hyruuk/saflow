"""Reproduce Figure 3 C-F: the FOOOF spectral decomposition panel.

Picks the single most significant spatial unit from the FOOOF *exponent*
group-statistics map, then renders, at that sensor/region, the four-panel
FOOOF figure for IN vs OUT (in-the-zone vs out-of-the-zone):

    C  Raw spectrum (PSD)        - subject-averaged welch spectra
    D  Aperiodic component       - 1/f fit (offset - log10(f^exponent))
    E  Corrected spectrum (PSDc) - raw minus aperiodic
    F  Periodic components       - summed FOOOF Gaussian peaks

The selected sensor/region name is written into panel C, alongside an inset
topomap of the exponent t-values with that sensor marked (sensor space only).

By default the curves are averaged across all subjects (group mean +/- SEM);
pass --subject to reproduce the manuscript's single-example-subject panel.

Usage:
    python -m code.visualization.run_viz_spectra --space sensor
    python -m code.visualization.run_viz_spectra --space sensor --subject 07
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# IN = in-the-zone (low VTC), OUT = out-of-the-zone (high VTC). Manuscript
# colour convention: blue IN, orange OUT.
COLOR_IN = "#1f77b4"
COLOR_OUT = "#ff7f0e"


# ---------------------------------------------------------------------------
# Stats: find the exponent map and pick the most significant unit
# ---------------------------------------------------------------------------

def find_exponent_results(stats_dir: Path, stat_feature: str, inout_str: str) -> Path:
    """Locate the group-statistics result file for the selection feature."""
    candidates = sorted(stats_dir.glob(f"feature-{stat_feature}_*_results.npz"))
    candidates = [c for c in candidates if f"inout-{inout_str}" in c.name]
    if not candidates:
        raise FileNotFoundError(
            f"No '{stat_feature}' statistics found in {stats_dir} for "
            f"inout-{inout_str}. Run e.g.:\n"
            f"  invoke analysis.stats --features=fooof --space={stats_dir.name.split('_', 1)[-1]}"
        )
    # Prefer the subject-spectrum path (FOOOF fit on aggregated spectra).
    for c in candidates:
        if "path-subj-spectrum" in c.name:
            return c
    return candidates[0]


def select_unit(results_file: Path, select_by: str) -> Dict:
    """Return the most significant spatial unit from a stats result file.

    select_by: 'corrected' (default, uses pvals_corrected_*) or 'uncorrected'.
    """
    res = np.load(results_file, allow_pickle=True)
    tvals = res["tvals"].flatten()
    pvals = res["pvals"].flatten() if "pvals" in res.files else np.full_like(tvals, np.nan)

    corrected_key = next(
        (k for k in res.files if k.startswith("pvals_corrected_")), None
    )
    pvals_corr = (
        res[corrected_key].flatten() if corrected_key else np.full_like(tvals, np.nan)
    )

    use_corrected = select_by == "corrected" and corrected_key is not None
    ranking = pvals_corr if use_corrected else pvals
    # All-NaN ranking (e.g. corrected absent) -> fall back to |t|.
    if np.all(np.isnan(ranking)):
        ranking = -np.abs(tvals)
        rank_label = "max|t|"
    else:
        rank_label = corrected_key if use_corrected else "pvals (uncorrected)"

    idx = int(np.nanargmin(ranking))
    return {
        "index": idx,
        "tval": float(tvals[idx]),
        "pval": float(pvals[idx]),
        "pval_corrected": float(pvals_corr[idx]) if corrected_key else float("nan"),
        "rank_label": rank_label,
        "tvals": tvals,
        "n_units": tvals.size,
    }


# ---------------------------------------------------------------------------
# Spatial-unit names
# ---------------------------------------------------------------------------

def unit_names(space: str, n_units: int, config: Dict, data_root: Path) -> List[str]:
    """Channel names (sensor) or region labels (atlas) in result-map order."""
    if space == "sensor":
        from code.visualization.render import _get_sensor_info

        info = _get_sensor_info(config, data_root)
        if len(info["ch_names"]) != n_units:
            logger.warning(
                "Sensor info has %d channels but stats map has %d units; "
                "names may be misaligned.",
                len(info["ch_names"]), n_units,
            )
        return list(info["ch_names"])

    # Atlas / source: try a labels sidecar, else fall back to generic names.
    atlas_dir = data_root / config["paths"]["derivatives"] / f"atlas_timeseries_{space}"
    for labels_file in atlas_dir.glob("**/*labels*.json"):
        try:
            labels = json.loads(labels_file.read_text())
            if isinstance(labels, list) and len(labels) == n_units:
                return [str(x) for x in labels]
        except Exception:
            pass
    return [f"region-{i:03d}" for i in range(n_units)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _mean_sem(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Group mean and SEM across the subject axis (axis 0), NaN-safe."""
    mean = np.nanmean(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, sem


def _plot_curve(ax, freqs, arr_in, arr_out, single_subject: bool):
    """Draw IN/OUT group-mean (+/- SEM) curves on a log-frequency axis."""
    for arr, color, label in (
        (arr_in, COLOR_IN, "IN"),
        (arr_out, COLOR_OUT, "OUT"),
    ):
        mean, sem = _mean_sem(arr)
        ax.plot(freqs, mean, color=color, lw=1.8, label=label)
        if not single_subject:
            ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.25, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both", alpha=0.2)


def _panel_letter(ax, letter: str):
    ax.text(
        -0.16, 1.06, letter, transform=ax.transAxes,
        fontsize=15, fontweight="bold", va="top", ha="left",
    )


def _add_topomap_inset(ax_c, tvals, sel_index, config, data_root):
    """Inset topomap of exponent t-values with the selected sensor marked."""
    try:
        import mne
        from code.visualization.render import _get_sensor_info

        info = _get_sensor_info(config, data_root)
        valid = np.isfinite(tvals)
        if valid.sum() < 3:
            return
        info_plot = mne.pick_info(info, np.where(valid)[0])
        vals = tvals[valid]
        sel_in_plot = int(np.where(np.where(valid)[0] == sel_index)[0][0]) \
            if valid[sel_index] else None
        mask = np.zeros(vals.shape, dtype=bool)
        if sel_in_plot is not None:
            mask[sel_in_plot] = True
        vmax = float(np.nanpercentile(np.abs(vals), 98)) or 1.0

        inset = ax_c.inset_axes([0.60, 0.58, 0.40, 0.40])
        mne.viz.plot_topomap(
            vals, info_plot, axes=inset, show=False, cmap="RdBu_r",
            vlim=(-vmax, vmax), mask=mask,
            mask_params=dict(marker="o", markerfacecolor="lime",
                             markeredgecolor="k", markersize=7, linewidth=0),
            extrapolate="local", outlines="head", sphere=0.15, contours=0,
        )
        inset.set_title("exponent t-map", fontsize=7, pad=2)
    except Exception as exc:  # noqa: BLE001 - inset is decorative, never fatal
        logger.warning("Could not draw topomap inset: %s", exc)


def build_panel(
    spectra: Dict,
    unit_name: str,
    sel: Dict,
    space: str,
    fit_range: Tuple[float, float],
    config: Dict,
    data_root: Path,
    single_subject: bool,
):
    """Assemble the 2x2 C-D-E-F FOOOF panel figure."""
    freqs = np.asarray(spectra["freqs"], dtype=float)
    freqs_fit = np.asarray(spectra["freqs_fit"], dtype=float)
    lo, hi = fit_range
    full_mask = (freqs >= lo) & (freqs <= hi)
    f_full = freqs[full_mask]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)
    (ax_c, ax_d), (ax_e, ax_f) = axes

    # C - Raw spectrum
    _plot_curve(
        ax_c, f_full,
        spectra["IN"]["raw"][:, full_mask], spectra["OUT"]["raw"][:, full_mask],
        single_subject,
    )
    ax_c.set_title(f"Raw spectrum (PSD)  -  {unit_name}", fontsize=11)
    ax_c.set_ylabel("PSD (log$_{10}$ power)")
    ax_c.legend(loc="lower left", fontsize=9, frameon=False)
    _panel_letter(ax_c, "C")
    if space == "sensor":
        _add_topomap_inset(ax_c, sel["tvals"], sel["index"], config, data_root)

    # D - Aperiodic component
    _plot_curve(
        ax_d, f_full,
        spectra["IN"]["aperiodic"][:, full_mask],
        spectra["OUT"]["aperiodic"][:, full_mask],
        single_subject,
    )
    ax_d.set_title("Aperiodic component", fontsize=11)
    ax_d.set_ylabel("PSD (log$_{10}$ power)")
    ax_d.legend(loc="upper right", fontsize=9, frameon=False)
    _panel_letter(ax_d, "D")

    # E - Corrected spectrum
    _plot_curve(
        ax_e, f_full,
        spectra["IN"]["corrected"][:, full_mask],
        spectra["OUT"]["corrected"][:, full_mask],
        single_subject,
    )
    ax_e.set_title("Corrected spectrum (PSDc)", fontsize=11)
    ax_e.set_ylabel("PSDc (log$_{10}$ power)")
    ax_e.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax_e.legend(loc="upper right", fontsize=9, frameon=False)
    _panel_letter(ax_e, "E")

    # F - Periodic components (FOOOF Gaussian peaks)
    _plot_curve(
        ax_f, freqs_fit,
        spectra["IN"]["periodic"], spectra["OUT"]["periodic"],
        single_subject,
    )
    ax_f.set_title("Periodic components", fontsize=11)
    ax_f.set_ylabel("Periodic power (log$_{10}$)")
    ax_f.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax_f.legend(loc="upper right", fontsize=9, frameon=False)
    _panel_letter(ax_f, "F")

    # Exponent annotation on panel D (the parameter that drove the selection).
    exp_in = float(np.nanmean(spectra["IN"]["exponent"]))
    exp_out = float(np.nanmean(spectra["OUT"]["exponent"]))
    ax_d.text(
        0.03, 0.06,
        f"exponent  IN={exp_in:.2f}  OUT={exp_out:.2f}",
        transform=ax_d.transAxes, fontsize=8, color="0.3",
    )

    who = (
        f"subject {single_subject}" if isinstance(single_subject, str)
        else f"group mean +/- SEM (N={spectra['n_subjects']})"
    )
    p_txt = (
        f"p={sel['pval_corrected']:.4f} (corr)"
        if np.isfinite(sel["pval_corrected"]) else f"p={sel['pval']:.4f}"
    )
    fig.suptitle(
        f"FOOOF spectra at most significant exponent {('sensor' if space == 'sensor' else 'region')}: "
        f"{unit_name}  |  t={sel['tval']:.2f}, {p_txt}  |  {who}",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 3 C-F: FOOOF spectral decomposition panel."
    )
    parser.add_argument(
        "--space", default="sensor",
        help="'sensor' or atlas name (e.g. 'schaefer_400'). Default: sensor.",
    )
    parser.add_argument(
        "--stat-feature", default="fooof_exponent",
        help="Statistics map used to select the spatial unit. Default: fooof_exponent.",
    )
    parser.add_argument(
        "--select-by", default="corrected", choices=["corrected", "uncorrected"],
        help="Rank units by corrected or uncorrected p-values. Default: corrected.",
    )
    parser.add_argument(
        "--subject", default=None,
        help="Restrict the spectra to one subject (e.g. '07'). "
             "Default: group average across all subjects.",
    )
    parser.add_argument(
        "--n-events-window", type=int, default=8,
        help="Trials per welch window (selects the desc suffix). Default: 8.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure.")
    parser.add_argument("--no-save", action="store_true", help="Do not write a file.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    from code.statistics.subject_spectrum import load_channel_spectra

    config = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(config["paths"]["data_root"])
    inout_bounds = config["analysis"]["inout_bounds"]
    inout_str = f"{inout_bounds[0]}{inout_bounds[1]}"
    stats_dir = data_root / config["paths"]["results"] / f"statistics_{args.space}"

    print("=" * 80)
    print(f"FOOOF spectra panel (Fig. 3 C-F)  |  space={args.space}")
    print("=" * 80)

    # 1. Find the exponent stats map and pick the most significant unit.
    results_file = find_exponent_results(stats_dir, args.stat_feature, inout_str)
    print(f"Selection map : {results_file.name}")
    sel = select_unit(results_file, args.select_by)

    # 2. Resolve the unit name.
    names = unit_names(args.space, sel["n_units"], config, data_root)
    unit_name = names[sel["index"]] if sel["index"] < len(names) else f"unit-{sel['index']}"
    print(
        f"Selected unit : {unit_name} (index {sel['index']}) | "
        f"t={sel['tval']:.3f} | ranked by {sel['rank_label']}"
    )

    # 3. Match the statistics run's bad-trial handling so the spectra line up
    #    with the map the unit was selected from.
    meta_file = results_file.with_name(
        results_file.stem.replace("_results", "_metadata") + ".json"
    )
    bad_rule, interp_thr = "ar2", 0
    if meta_file.exists():
        dm = json.loads(meta_file.read_text()).get("data_metadata", {})
        bad_rule = str(dm.get("bad_trial_rule", bad_rule))
        interp_thr = int(dm.get("interp_reject_threshold", interp_thr) or 0)
    print(f"Bad-trial rule: {bad_rule} (interp_reject_threshold={interp_thr})")

    # 4. Load the per-subject spectra at the selected unit.
    subjects = [args.subject] if args.subject else None
    spectra = load_channel_spectra(
        channel_index=sel["index"],
        space=args.space,
        inout_bounds=tuple(inout_bounds),
        config=config,
        subjects=subjects,
        bad_trial_rule=bad_rule,
        interp_reject_threshold=interp_thr,
        n_events_window=args.n_events_window,
    )
    print(f"Loaded spectra: {spectra['n_subjects']} subject(s)")

    fooof_cfg = config.get("features", {}).get("fooof", [{}])
    fooof_params = fooof_cfg[0] if isinstance(fooof_cfg, list) else fooof_cfg
    fit_range = tuple(fooof_params.get("freq_range", [2.0, 120.0]))

    # 5. Build the panel.
    fig = build_panel(
        spectra=spectra, unit_name=unit_name, sel=sel, space=args.space,
        fit_range=fit_range, config=config, data_root=data_root,
        single_subject=(args.subject or False),
    )

    if not args.no_save:
        fig_dir = Path("reports") / "figures" / "statistics"
        fig_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9]+", "-", unit_name)
        subj_tag = f"_sub-{args.subject}" if args.subject else "_group"
        fig_path = fig_dir / f"spectra_panel_{args.space}_{safe_name}{subj_tag}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved figure  : {fig_path}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        plt.close(fig)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
