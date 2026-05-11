"""Unified visualization entry point.

Generates rows of topomaps (sensor) or inflated-brain panels (source/atlas) for
any metric registered in `code.visualization.metrics`. Picks the right loader
from the metric's file pattern; picks the right renderer from `--space`.

Examples:
    python -m code.visualization.run_viz --metric=tval --space=sensor --feature-set=psds
    python -m code.visualization.run_viz --metric=roc_auc --space=schaefer_400 --feature-set=psds
    python -m code.visualization.run_viz --metric=contrast --space=sensor --feature=fooof_exponent

When the underlying results aren't there, the script tells you exactly which
analysis task to run to produce them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from code.visualization.loaders import (
    BAND_ORDER,
    COMPLEXITY_ORDER,
    FOOOF_ORDER,
    MapResult,
    MissingResultsError,
    family_sort_key,
    feature_family,
    load_for_metric,
)
from code.visualization.metrics import METRICS
from code.visualization.render import render_row


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


FEATURE_SETS = {
    "psds": [f"psd_{b}" for b in BAND_ORDER],
    "psds_corrected": [f"psd_corrected_{b}" for b in BAND_ORDER],
    "fooof": [f"fooof_{p}" for p in ("exponent", "offset", "r_squared")],
    "complexity": [f"complexity_{m}" for m in COMPLEXITY_ORDER],
}
FEATURE_SETS["all"] = (
    FEATURE_SETS["psds"]
    + FEATURE_SETS["psds_corrected"]
    + FEATURE_SETS["fooof"]
    + FEATURE_SETS["complexity"]
)


def _features_to_glob(features: Optional[List[str]]) -> Optional[str]:
    """If a single feature is requested, pin the glob; otherwise glob all."""
    if not features:
        return None
    if len(features) == 1:
        return features[0]
    return None  # we'll filter results post-hoc


def _filter_loaded(maps: List[MapResult], features: Optional[List[str]]) -> List[MapResult]:
    if not features:
        return maps
    wanted = set(features)
    return [m for m in maps if m.feature in wanted]


def _group_by_family(maps: List[MapResult]) -> Dict[str, List[MapResult]]:
    out: Dict[str, List[MapResult]] = {}
    for m in maps:
        out.setdefault(m.family, []).append(m)
    return out


def _figure_filename(
    family: str, space: str, metric: str, correction: str,
    extra: Dict[str, str],
) -> str:
    pieces = [f"{metric}", f"{family}", f"space-{space}"]
    if "clf" in extra:
        pieces.append(f"clf-{extra['clf']}")
    pieces.append(f"correction-{correction}")
    return "_".join(pieces) + ".png"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified rows-of-maps viz (stats + classification, sensor + atlas)."
    )
    parser.add_argument(
        "--metric", required=True, choices=list(METRICS),
        help="What to plot. Examples: tval, contrast, roc_auc.",
    )
    parser.add_argument(
        "--space", default="sensor",
        help="'sensor', 'source', or atlas name (e.g. 'schaefer_400').",
    )
    parser.add_argument(
        "--feature", nargs="+", default=None,
        help="One or more feature names (e.g. 'psd_alpha psd_theta'). "
             "Default: all features found.",
    )
    parser.add_argument(
        "--feature-set", default=None, choices=list(FEATURE_SETS),
        help="Shortcut for a family (psds, psds_corrected, fooof, complexity, all).",
    )
    parser.add_argument(
        "--family", default=None,
        choices=["psd", "psd_corrected", "fooof", "complexity", "other"],
        help="If features span multiple families, render only this one.",
    )
    parser.add_argument(
        "--clf", default="lda",
        help="Classifier filter (used for classification metrics). Default: lda.",
    )
    parser.add_argument(
        "--cv", default="logo",
        help="CV strategy filter (classification only). Default: logo.",
    )
    parser.add_argument(
        "--mode", default="univariate",
        help="univariate or multivariate (classification only). Default: univariate.",
    )
    parser.add_argument(
        "--test", default="paired_ttest",
        help="Statistical test filter (statistics only). Default: paired_ttest.",
    )
    parser.add_argument(
        "--correction", default="auto",
        choices=["auto", "tmax", "fdr_bh", "bonferroni", "uncorrected"],
        help="Which p-value correction to use for the significance mask. "
             "'auto' = first available in the metric's preferred order.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cmap", default=None, help="Override colormap.")
    parser.add_argument(
        "--output-subdir", default="classification",
        help="Output goes into reports/figures/<output-subdir>/.",
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(config["paths"]["data_root"])
    results_subpath = config["paths"]["results"]
    inout_bounds = config["analysis"]["inout_bounds"]
    inout_str = f"{inout_bounds[0]}{inout_bounds[1]}"

    # Resolve feature list
    features: List[str] = []
    if args.feature_set:
        features.extend(FEATURE_SETS[args.feature_set])
    if args.feature:
        features.extend(args.feature)
    seen = set()
    features = [f for f in features if not (f in seen or seen.add(f))]
    feature_glob = _features_to_glob(features)

    # Per-metric extra placeholders
    metric = METRICS[args.metric]
    extra: Dict[str, str] = {}
    if "clf" in metric.results_glob:
        extra["clf"] = args.clf
    if "cv" in metric.results_glob:
        extra["cv"] = args.cv
    if "mode" in metric.results_glob:
        extra["mode"] = args.mode
    if "test" in metric.results_glob:
        extra["test"] = args.test
    if "space" in metric.results_glob:
        extra["space"] = args.space

    print("=" * 78)
    print(f"viz | metric={args.metric} | space={args.space}")
    if features:
        print(f"     | features={features}")
    if extra:
        print(f"     | filters={extra}")
    print(f"     | correction={args.correction} alpha={args.alpha}")
    print("=" * 78)

    try:
        maps = load_for_metric(
            metric_name=args.metric,
            space=args.space,
            data_root=data_root,
            results_subpath=results_subpath,
            feature=feature_glob,
            inout=inout_str,
            correction=args.correction,
            alpha=args.alpha,
            extra_filters=extra,
        )
    except MissingResultsError as exc:
        print(f"\n{exc}\n")
        return 2

    # Filter (when feature list has > 1, we couldn't pin the glob, so filter now)
    available_before = {m.feature for m in maps}
    maps = _filter_loaded(maps, features)
    if not maps:
        missing = [f for f in (features or []) if f not in available_before]
        print(
            f"\nFound {len(available_before)} '{metric.name}' result(s) for "
            f"space={args.space}, but none of the requested features were "
            f"present.\n"
        )
        if available_before:
            print("Already computed:")
            for f in sorted(available_before):
                print(f"  ✓ {f}")
            print()
        if missing:
            print("Missing — to produce them, run e.g.:")
            for f in missing:
                print(f"  {metric.missing_msg(space=args.space, feature=f, clf=args.clf)}")
        return 2

    # Sort each family in canonical order (bands by frequency, etc.)
    maps.sort(key=lambda m: family_sort_key(m.feature))

    # Group + render
    grouped = _group_by_family(maps)
    if args.family:
        grouped = {args.family: grouped.get(args.family, [])}
        if not grouped[args.family]:
            print(f"No results for family '{args.family}'.")
            return 2

    fig_dir = Path("reports") / "figures" / args.output_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(maps)} run(s):")
    for fam, fam_maps in grouped.items():
        print(f"  - {fam}: {len(fam_maps)}")

    for family, fam_maps in grouped.items():
        if not fam_maps:
            continue
        print(f"\nRendering {family}…")
        fig = render_row(
            fam_maps, args.space, config, data_root,
            cmap_override=args.cmap,
        )
        if fig is None:
            print(f"  (rendering returned no figure for {family})")
            continue

        # Title with provenance from the first map
        sample = fam_maps[0]
        md = sample.metadata.get("data_metadata", {})
        title_bits = [
            f"{family} {metric.cbar_label}",
            f"{args.space}",
            f"correction={args.correction} α={args.alpha}",
            f"N={md.get('n_subjects', '?')} subj"
            f" (IN={md.get('n_in', '?')}, OUT={md.get('n_out', '?')})",
        ]
        if "clf" in extra:
            title_bits.insert(2, f"clf={extra['clf']}")
        fig.suptitle(" | ".join(title_bits), fontsize=11, y=1.005)

        out = fig_dir / _figure_filename(
            family, args.space, args.metric, args.correction, extra,
        )
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  saved {out}  ({len(fam_maps)} feature(s))")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
