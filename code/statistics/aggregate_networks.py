"""Aggregate per-parcel paired-ttest stats to per-Yeo-network summaries.

Reads the per-feature stats npz files written by ``run_group_statistics.py``
for an atlas space (e.g. ``schaefer_400``) and produces one consolidated
network-level npz per (trial_type, correction) combination.

Output schema (one file per ``--trial-type`` × ``--correction``):
    stats-networks_yeo{7,17}_type-<trial>_correction-<corr>.npz
      features                  : (n_features,) feature names
      network_names             : (n_networks,)
      tvals_mean                : (n_features, n_networks)   mean per-parcel t
      d_mean                    : (n_features, n_networks)   mean per-parcel Cohen's d (paired)
      signed_count_sig          : (n_features, n_networks)   (#parcels with t>0 & p<α) − (#with t<0 & p<α)
      pooled_p_stouffer         : (n_features, n_networks)   Stouffer-combined two-sided p
      n_parcels_per_network     : (n_networks,)
      meta                      : pickled dict of provenance

The aggregation is post-hoc on per-parcel results; the original per-parcel
significance correction (FDR/tmax/etc.) is reused — we don't re-correct at
the network level. Stouffer's z is a pooled summary, not a strict test.

Usage:
    python -m code.statistics.aggregate_networks \\
        --space schaefer_400 --trial-type all \\
        --correction fdr --yeo 7
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from code.utils.config import load_config
from code.utils.paths import get_features_root, get_results_root
from code.utils.yeo_networks import (
    UNKNOWN_NETWORK,
    aggregate_to_networks,
    get_network_assignments,
    network_order,
    network_parcel_indices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Per-parcel stats files are written with this filename pattern. The
# leading ``feature-<name>`` token is the only varying part we glob on; we
# pin the rest via CLI filters so trial_type/inout don't bleed into one
# aggregated bundle by accident.
STATS_FNAME_RE = re.compile(
    r"^feature-(?P<feature>.+?)"
    r"_inout-(?P<inout>[^_]+)"
    r"(?:_sel-(?P<sel>[^_]+))?"
    r"_test-(?P<test>[^_]+(?:_[^_]+)?)"
    r"_path-(?P<path>[^_]+(?:-[^_]+)*)"
    r"_type-(?P<trial>[^_]+)"
    r"_results\.npz$"
)

# Per-correction → npz key in the per-parcel stats file. Mirrors PVAL_KEYS
# in stats_classif_panel.py so we accept both new and legacy keys.
PVAL_KEYS: Dict[str, Tuple[str, ...]] = {
    "fdr":         ("pvals_fdr_bh", "pvals_corrected_fdr_bh", "pvals_corrected_fdr"),
    "tmax":        ("pvals_tmax", "pvals_corrected_tmax"),
    "bonferroni":  ("pvals_bonferroni", "pvals_corrected_bonferroni"),
    "uncorrected": ("pvals_uncorrected", "pvals"),
}

# Cohen's d effect-size key candidates (paired preferred for paired-ttest).
D_KEYS: Tuple[str, ...] = (
    "effectsize_cohens_d_paired",
    "effectsize_hedges_g_paired",
    "effectsize_cohens_d",
)


def _resolve_pval(npz: np.lib.npyio.NpzFile, correction: str) -> np.ndarray:
    """Pull the right p-value array for ``correction`` from a stats npz."""
    if correction not in PVAL_KEYS:
        raise ValueError(f"Unknown correction {correction!r}, expected one of "
                         f"{list(PVAL_KEYS)}")
    for key in PVAL_KEYS[correction]:
        if key in npz.files:
            return np.asarray(npz[key]).flatten()
    raise KeyError(f"No p-value key for correction={correction} in stats npz "
                   f"(tried {PVAL_KEYS[correction]}, found {npz.files})")


def _resolve_d(npz: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    for key in D_KEYS:
        if key in npz.files:
            return np.asarray(npz[key]).flatten()
    return None


def _resolve_spatial_names(features_root: Path, space: str) -> List[str]:
    """Recover parcel names for ``space`` from any complexity feature npz.

    The complexity family is the canonical carrier of ``ch_names`` per
    [[feedback-ch-names-uniform]]; we pull from it because it's guaranteed
    to be populated for every subject/run on this space.
    """
    folder = features_root / f"complexity_{space}"
    if not folder.exists():
        raise FileNotFoundError(
            f"Cannot resolve spatial names: {folder} missing. Re-run "
            f"complexity feature extraction for space={space}, or backfill "
            f"ch_names via code.utils.backfill_ch_names."
        )
    for path in sorted(folder.rglob("*.npz")):
        with np.load(path, allow_pickle=True) as npz:
            if "ch_names" in npz.files:
                return [str(x) for x in np.asarray(npz["ch_names"])]
    raise FileNotFoundError(
        f"No npz with ch_names in {folder}. Run "
        f"code.utils.backfill_ch_names --space {space}"
    )


def _stouffer_two_sided(pvals: np.ndarray, tvals: np.ndarray) -> float:
    """Stouffer's signed-z combination of per-parcel two-sided p-values.

    Returns NaN if no finite parcels. Pools two-sided p-values into signed
    z-scores via the t-statistic sign; outputs a two-sided combined p.
    """
    pvals = np.asarray(pvals, dtype=float)
    tvals = np.asarray(tvals, dtype=float)
    mask = np.isfinite(pvals) & np.isfinite(tvals)
    if not mask.any():
        return float("nan")
    p = np.clip(pvals[mask], 1e-300, 1.0)
    z = sp_stats.norm.isf(p / 2.0) * np.sign(tvals[mask])
    z_comb = z.sum() / np.sqrt(z.size)
    return float(2.0 * sp_stats.norm.sf(abs(z_comb)))


def _scan_stats_files(
    stats_dir: Path,
    trial_type: str,
    inout_token: str = "2575",
    inout_selection: str = "strict",
) -> List[Tuple[Path, str]]:
    """Return ``[(path, feature_name), ...]`` for matching per-parcel stats files.

    ``inout_selection`` filters by the new ``_sel-<X>`` filename token (default
    ``strict`` matches files that don't carry the token at all, since that's
    how the default strategy writes them).
    """
    out: List[Tuple[Path, str]] = []
    expected_sel = None if inout_selection == "strict" else inout_selection
    for p in sorted(stats_dir.glob("*_results.npz")):
        m = STATS_FNAME_RE.match(p.name)
        if not m:
            continue
        if m.group("trial") != trial_type:
            continue
        if m.group("inout") != inout_token:
            continue
        if m.group("sel") != expected_sel:
            continue
        out.append((p, m.group("feature")))
    return out


def aggregate_one_combo(
    stats_dir: Path,
    ch_names: List[str],
    trial_type: str,
    correction: str,
    n_networks: int,
    alpha: float,
    inout_token: str = "2575",
    inout_selection: str = "strict",
) -> Dict[str, np.ndarray]:
    """Aggregate every feature in ``stats_dir`` for one (trial × correction)."""
    files = _scan_stats_files(
        stats_dir,
        trial_type=trial_type,
        inout_token=inout_token,
        inout_selection=inout_selection,
    )
    if not files:
        raise FileNotFoundError(
            f"No per-parcel stats for trial-type={trial_type} in {stats_dir}. "
            f"Run analysis.stats with --space matching first."
        )

    nets = network_order(n_networks)
    n_feats = len(files)
    n_nets = len(nets)

    feature_names: List[str] = []
    t_mean = np.full((n_feats, n_nets), np.nan)
    d_mean = np.full((n_feats, n_nets), np.nan)
    sc_sig = np.full((n_feats, n_nets), np.nan)
    p_pool = np.full((n_feats, n_nets), np.nan)

    # Drop the medial-wall / Unknown parcels once, up front.
    assignments = get_network_assignments(ch_names, n_networks=n_networks)
    keep = assignments != UNKNOWN_NETWORK
    ch_keep = [c for c, k in zip(ch_names, keep) if k]
    if not keep.all():
        logger.info(f"  dropping {(~keep).sum()} non-cortical (Unknown) parcels")

    parcel_idx = network_parcel_indices(ch_keep, n_networks=n_networks)

    for i, (npz_path, feature) in enumerate(files):
        feature_names.append(feature)
        with np.load(npz_path, allow_pickle=True) as npz:
            tvals = np.asarray(npz["tvals"]).flatten()
            pvals_corr = _resolve_pval(npz, correction)
            d_arr = _resolve_d(npz)

        if tvals.size != len(ch_names):
            logger.warning(
                f"  shape mismatch for {feature}: stats has {tvals.size} "
                f"parcels vs ch_names has {len(ch_names)}. Skipping."
            )
            continue

        # Drop Unknown parcels along the parcel axis to align with ch_keep.
        tvals = tvals[keep]
        pvals_corr = pvals_corr[keep]
        if d_arr is not None:
            d_arr = d_arr[keep]

        # Mean-of-parcels aggregation (using the helper)
        t_block, _ = aggregate_to_networks(tvals, ch_keep, n_networks=n_networks,
                                           agg="mean")
        t_mean[i, :] = t_block
        if d_arr is not None:
            d_block, _ = aggregate_to_networks(d_arr, ch_keep, n_networks=n_networks,
                                               agg="mean")
            d_mean[i, :] = d_block

        # Signed count of significant parcels: thresholded t-values then signed_count
        sig_t = np.where(pvals_corr < alpha, tvals, 0.0)
        sc_block, _ = aggregate_to_networks(sig_t, ch_keep, n_networks=n_networks,
                                            agg="signed_count")
        sc_sig[i, :] = sc_block

        # Stouffer-pooled p per network
        for j, net in enumerate(nets):
            idx = parcel_idx[net]
            if idx.size == 0:
                continue
            p_pool[i, j] = _stouffer_two_sided(pvals_corr[idx], tvals[idx])

    n_parcels_per_net = np.array(
        [parcel_idx[net].size for net in nets], dtype=int
    )

    return {
        "features": np.array(feature_names),
        "network_names": np.array(nets),
        "tvals_mean": t_mean,
        "d_mean": d_mean,
        "signed_count_sig": sc_sig,
        "pooled_p_stouffer": p_pool,
        "n_parcels_per_network": n_parcels_per_net,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-parcel stats to Yeo networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--space", required=True,
                   help="Atlas space (e.g. schaefer_400). Must match the "
                        "--space used by analysis.stats.")
    p.add_argument("--trial-type", default="all",
                   help="alltrials | correct | lapse | all (= run all three)")
    p.add_argument("--correction", default="fdr",
                   choices=list(PVAL_KEYS),
                   help="Per-parcel p-value correction to use for the "
                        "significance mask (drives signed_count_sig).")
    p.add_argument("--yeo", type=int, default=7, choices=[7, 17],
                   help="Yeo network granularity.")
    p.add_argument("--inout-token", default="2575",
                   help="inout bound token in the stats filename.")
    p.add_argument("--inout-selection", default=None,
                   choices=["strict", "lenient", "vtcfilt", "vtcraw"],
                   help="IN/OUT selection strategy whose stats to aggregate. "
                        "Defaults to the value in config.analysis.inout_selection "
                        "(or 'strict' if absent).")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--results-root", default=None,
                   help="Override results root (defaults to config).")
    p.add_argument("--features-root", default=None,
                   help="Override features root (defaults to config).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    results_root = Path(args.results_root) if args.results_root \
        else get_results_root(config)
    features_root = Path(args.features_root) if args.features_root \
        else get_features_root(config)

    stats_dir = results_root / f"statistics_{args.space}"
    out_dir = stats_dir / "group" / "networks"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Stats dir: {stats_dir}")
    logger.info(f"Output dir: {out_dir}")

    ch_names = _resolve_spatial_names(features_root, args.space)
    logger.info(f"Resolved {len(ch_names)} parcel names for space={args.space}")

    trial_types = (
        ["alltrials", "correct", "lapse"]
        if args.trial_type == "all" else [args.trial_type]
    )

    inout_selection = args.inout_selection or str(
        config.get("analysis", {}).get("inout_selection", "strict")
    )
    sel_tok = "" if inout_selection == "strict" else f"_sel-{inout_selection}"

    provenance = {
        "script": "code.statistics.aggregate_networks",
        "timestamp": datetime.utcnow().isoformat(),
        "space": args.space,
        "yeo": args.yeo,
        "correction": args.correction,
        "alpha": args.alpha,
        "inout_token": args.inout_token,
        "inout_selection": inout_selection,
    }

    for trial in trial_types:
        logger.info(f"--- trial-type={trial} ---")
        try:
            bundle = aggregate_one_combo(
                stats_dir=stats_dir,
                ch_names=ch_names,
                trial_type=trial,
                correction=args.correction,
                n_networks=args.yeo,
                alpha=args.alpha,
                inout_token=args.inout_token,
                inout_selection=inout_selection,
            )
        except FileNotFoundError as exc:
            logger.warning(f"  skipping {trial}: {exc}")
            continue

        out_name = (f"stats-networks_yeo{args.yeo}_"
                    f"type-{trial}_correction-{args.correction}{sel_tok}.npz")
        out_path = out_dir / out_name
        np.savez(out_path, **bundle, meta=np.asarray(json.dumps(provenance | {
            "trial_type": trial,
            "n_features": int(bundle["features"].size),
        })))
        logger.info(f"  wrote {out_path} "
                    f"({bundle['features'].size} features × "
                    f"{bundle['network_names'].size} networks)")


if __name__ == "__main__":
    main()
