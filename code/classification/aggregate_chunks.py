"""Aggregate per-chunk classification outputs into one final scores file.

Each chunked run of `code.classification.run_classification` writes a partial
`.npz` containing scores for a slice of the spatial dimension. Because every
chunk is run with the same RNG seed, the permutation-label sequences are
identical across chunks, so per-permutation max-across-spatial values can be
computed by concatenating the per-chunk `perm_scores` along the spatial axis.

Usage:
    python -m code.classification.aggregate_chunks \
        --feature psd_alpha --space sensor --mode univariate \
        --clf lda --cv logo

    # combined-feature run
    python -m code.classification.aggregate_chunks \
        --feature combined-10 --space schaefer_400 --mode univariate \
        --clf rf --cv logo --combined

Pass --delete-chunks to remove the per-chunk files after a successful merge.
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from code.classification.run_classification import (
    build_base_name,
    inout_bounds_to_string,
)
from code.statistics.corrections import (
    apply_bonferroni_correction,
    apply_fdr_correction,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_chunks(output_dir: Path, base: str) -> List[Tuple[int, int, Path]]:
    """Find chunk score files for a given base name. Returns [(idx, total, path)]."""
    pattern = re.compile(rf"^{re.escape(base)}_chunk-(\d+)of(\d+)_scores\.npz$")
    found: List[Tuple[int, int, Path]] = []
    for p in output_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(found)


def aggregate(
    output_dir: Path,
    feature_label: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    mode: str,
    combined: bool,
    delete_chunks: bool = False,
    trial_type: str = "alltrials",
) -> Path:
    base = build_base_name(
        feature_label, space, inout_bounds, clf_name, cv_name, mode, combined,
        trial_type=trial_type,
    )
    chunks = find_chunks(output_dir, base)
    if not chunks:
        raise FileNotFoundError(f"No chunk files matching '{base}_chunk-*' in {output_dir}")

    n_chunks_expected = chunks[0][1]
    if any(c[1] != n_chunks_expected for c in chunks):
        raise ValueError(
            f"Inconsistent total chunk counts in filenames: "
            f"{sorted({c[1] for c in chunks})}"
        )
    if {c[0] for c in chunks} != set(range(n_chunks_expected)):
        missing = set(range(n_chunks_expected)) - {c[0] for c in chunks}
        raise ValueError(f"Missing chunks: {sorted(missing)}")

    logger.info(f"Aggregating {n_chunks_expected} chunk(s) for '{base}'")

    observed_parts: List[np.ndarray] = []
    perm_parts: List[np.ndarray] = []
    importance_parts: List[np.ndarray] = []
    seeds: set = set()
    n_perms_set: set = set()
    n_spatial_total: Optional[int] = None
    chunk_meta_list: List[Dict] = []
    last_data_metadata: Optional[Dict] = None
    last_feature_list: Optional[List[str]] = None

    for chunk_idx, _, score_path in chunks:
        meta_path = score_path.with_name(score_path.name.replace("_scores.npz", "_metadata.json"))
        if meta_path.exists():
            chunk_meta = json.loads(meta_path.read_text())
            if "chunk" in chunk_meta:
                seeds.add(chunk_meta["chunk"]["seed"])
                if n_spatial_total is None:
                    n_spatial_total = chunk_meta["chunk"]["n_spatial_total"]
                elif n_spatial_total != chunk_meta["chunk"]["n_spatial_total"]:
                    raise ValueError(
                        f"n_spatial_total mismatch across chunks "
                        f"({n_spatial_total} vs {chunk_meta['chunk']['n_spatial_total']})"
                    )
            chunk_meta_list.append(chunk_meta)
            last_data_metadata = chunk_meta.get("data_metadata", last_data_metadata)
            last_feature_list = chunk_meta.get("feature_list", last_feature_list)
        with np.load(score_path) as npz:
            observed_parts.append(npz["observed"])
            perm_parts.append(npz["perm_scores"])
            n_perms_set.add(int(npz["perm_scores"].shape[0]))
            if "feature_importances" in npz.files:
                importance_parts.append(npz["feature_importances"])

    if len(seeds) > 1:
        raise ValueError(
            f"Chunks were produced with different seeds: {sorted(seeds)}. "
            f"Cannot aggregate — re-run with a shared --seed."
        )
    if len(n_perms_set) != 1:
        raise ValueError(f"Inconsistent n_permutations across chunks: {n_perms_set}")
    n_permutations = n_perms_set.pop()

    observed = np.concatenate(observed_parts, axis=0)
    perm_scores = np.concatenate(perm_parts, axis=1)
    n_spatial_actual = observed.shape[0]
    if n_spatial_total is not None and n_spatial_actual != n_spatial_total:
        raise ValueError(
            f"Concatenated n_spatial={n_spatial_actual} != reported total "
            f"{n_spatial_total}. Some chunks are corrupt or duplicated."
        )

    importances = None
    if importance_parts:
        if len(importance_parts) != n_chunks_expected:
            logger.warning(
                "Some chunks have feature_importances and others don't; skipping."
            )
        else:
            importances = np.concatenate(importance_parts, axis=0)

    pvals_unc = (np.sum(perm_scores >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    max_perm = np.nanmax(perm_scores, axis=1)
    pvals_tmax = (np.sum(max_perm[:, None] >= observed[None, :], axis=0) + 1) / (
        n_permutations + 1
    )
    pvals_fdr = apply_fdr_correction(pvals_unc, alpha=0.05, method="bh")
    pvals_bonf = apply_bonferroni_correction(pvals_unc, alpha=0.05)

    n_sig_tmax = int((pvals_tmax < 0.05).sum())
    n_sig_fdr = int((pvals_fdr < 0.05).sum())
    logger.info(
        f"Aggregated: n_spatial={n_spatial_actual}, n_permutations={n_permutations}, "
        f"max observed={observed.max():.3f}, "
        f"sig@.05 tmax={n_sig_tmax}/{n_spatial_actual}, FDR={n_sig_fdr}/{n_spatial_actual}"
    )

    npz_payload: Dict[str, np.ndarray] = dict(
        observed=observed,
        perm_scores=perm_scores,
        pvals_uncorrected=pvals_unc,
        pvals_tmax=pvals_tmax,
        pvals_fdr_bh=pvals_fdr,
        pvals_bonferroni=pvals_bonf,
    )
    if importances is not None:
        npz_payload["feature_importances"] = importances

    out_npz = output_dir / f"{base}_scores.npz"
    np.savez_compressed(out_npz, **npz_payload)

    aggregated_meta = {
        "feature": feature_label,
        "feature_list": last_feature_list,
        "combined": combined,
        "space": space,
        "inout_bounds": list(inout_bounds),
        "classifier": clf_name,
        "cv_strategy": cv_name,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "data_metadata": last_data_metadata,
        "summary": {
            "max_score": float(observed.max()),
            "mean_score": float(observed.mean()),
            "n_significant_tmax_a05": n_sig_tmax,
            "n_significant_fdr_bh_a05": n_sig_fdr,
            "n_permutations": int(n_permutations),
            "n_spatial": int(n_spatial_actual),
            "n_chunks_aggregated": int(n_chunks_expected),
        },
        "chunks": chunk_meta_list,
    }
    out_meta = output_dir / f"{base}_metadata.json"
    out_meta.write_text(json.dumps(aggregated_meta, indent=2, default=str))

    logger.info(f"Wrote aggregated scores -> {out_npz}")
    logger.info(f"Wrote aggregated metadata -> {out_meta}")

    if delete_chunks:
        for _, _, p in chunks:
            p.unlink(missing_ok=True)
            mp = p.with_name(p.name.replace("_scores.npz", "_metadata.json"))
            mp.unlink(missing_ok=True)
        logger.info(f"Deleted {len(chunks)} chunk file pair(s)")

    return out_npz


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-chunk classification outputs."
    )
    parser.add_argument("--feature", required=True,
                        help="Feature label as it appears in chunk filenames.")
    parser.add_argument("--space", required=True)
    parser.add_argument("--mode", default="univariate", choices=["univariate", "multivariate"])
    parser.add_argument("--clf", default="lda")
    parser.add_argument("--cv", default="logo")
    parser.add_argument("--combined", action="store_true",
                        help="Set if the original run used --combine-features.")
    parser.add_argument("--trial-type", default="alltrials",
                        choices=["alltrials", "correct", "rare", "lapse",
                                 "correct_commission"],
                        help="Trial-type filter the original run used "
                             "(must match the chunk filenames).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--delete-chunks", action="store_true",
                        help="Remove per-chunk score/metadata files after merging.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    inout_bounds = tuple(config["analysis"]["inout_bounds"])
    data_root = Path(config["paths"]["data_root"])
    output_dir = (
        data_root / config["paths"]["results"] / f"classification_{args.space}" / "group"
    )

    aggregate(
        output_dir=output_dir,
        feature_label=args.feature,
        space=args.space,
        inout_bounds=inout_bounds,
        clf_name=args.clf,
        cv_name=args.cv,
        mode=args.mode,
        combined=args.combined,
        delete_chunks=args.delete_chunks,
        trial_type=args.trial_type,
    )


if __name__ == "__main__":
    main()
