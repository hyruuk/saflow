"""Aggregate per-axis multi-feature classification outputs into one bundle.

A typical run of ``run_multifeature.py --axis all`` writes four files (one per
axis: per-spatial, per-feature, per-cell, joint). This aggregator reads them
back for a given (label, space, inout, clf, cv, trial_type) and produces a
single bundle file convenient for downstream visualization:

    feature-{label}_..._mf-bundle_scores.npz
    feature-{label}_..._mf-bundle_metadata.json

The npz contains arrays keyed by ``<axis>/<key>`` (e.g.
``per-spatial/observed``, ``per-feature/importances``, ``joint/observed``)
plus shared ``spatial_names`` and ``feature_names``.

Per-cell with chunked runs is merged on the fly (same shared-seed assumption
as ``aggregate_chunks.py``: concatenate ``observed`` along the spatial axis
and ``perm_scores`` along the spatial axis, then recompute p-values across
the full cell grid).

Usage:
    python -m code.classification.aggregate_multifeature \
        --label combined-19 --space sensor \
        --clf logistic --cv logo --importance permutation \
        --trial-type alltrials
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.classification.run_classification import (
    inout_bounds_to_string,
    load_config,
)
from code.classification.run_multifeature import (
    AXES,
    build_mf_base_name,
)
from code.statistics.corrections import (
    apply_bonferroni_correction,
    apply_fdr_correction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _find_per_cell_chunks(output_dir: Path, base: str) -> List[Tuple[int, int, Path]]:
    pattern = re.compile(rf"^{re.escape(base)}_chunk-(\d+)of(\d+)_scores\.npz$")
    found: List[Tuple[int, int, Path]] = []
    for p in output_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(found)


def _merge_per_cell_chunks(
    output_dir: Path, base: str, axis: str
) -> Optional[Tuple[Dict[str, np.ndarray], Dict]]:
    """Concatenate per-cell chunks (spatial axis 0 of observed; axis 1 of perm)."""
    chunks = _find_per_cell_chunks(output_dir, base)
    if not chunks:
        return None
    n_chunks_expected = chunks[0][1]
    if any(c[1] != n_chunks_expected for c in chunks):
        raise ValueError(
            f"Inconsistent chunk counts: {sorted({c[1] for c in chunks})}"
        )
    if {c[0] for c in chunks} != set(range(n_chunks_expected)):
        missing = set(range(n_chunks_expected)) - {c[0] for c in chunks}
        raise ValueError(f"Missing per-cell chunks: {sorted(missing)}")

    obs_parts: List[np.ndarray] = []
    perm_parts: List[np.ndarray] = []
    spatial_names = None
    feature_names = None
    last_meta: Optional[Dict] = None
    seeds: set = set()
    for chunk_idx, _, score_path in chunks:
        with np.load(score_path, allow_pickle=True) as npz:
            obs_parts.append(npz["observed"])
            perm_parts.append(npz["perm_scores"])
            if "spatial_names" in npz.files:
                spatial_names = list(npz["spatial_names"])
            if "feature_names" in npz.files:
                feature_names = list(npz["feature_names"])
        meta_path = score_path.with_name(
            score_path.name.replace("_scores.npz", "_metadata.json")
        )
        if meta_path.exists():
            chunk_meta = json.loads(meta_path.read_text())
            last_meta = chunk_meta
            if "chunk" in chunk_meta:
                seeds.add(chunk_meta["chunk"].get("seed"))

    if len(seeds) > 1:
        raise ValueError(
            f"per-cell chunks have inconsistent seeds {sorted(seeds)} — "
            f"cannot aggregate; re-run with shared --seed."
        )

    observed = np.concatenate(obs_parts, axis=0)  # (n_spatial, n_features)
    perm_scores = np.concatenate(
        perm_parts, axis=1
    )  # (n_perms, n_spatial, n_features)

    n_perms = perm_scores.shape[0]
    obs_flat = observed.ravel()
    perm_flat = perm_scores.reshape(n_perms, -1)
    pvals_unc = (np.sum(perm_flat >= obs_flat[None, :], axis=0) + 1) / (n_perms + 1)
    max_perm = np.nanmax(perm_flat, axis=1)
    pvals_tmax = (np.sum(max_perm[:, None] >= obs_flat[None, :], axis=0) + 1) / (
        n_perms + 1
    )
    nan_obs = np.isnan(obs_flat)
    if nan_obs.any():
        pvals_unc[nan_obs] = 1.0
        pvals_tmax[nan_obs] = 1.0
    pvals_fdr = apply_fdr_correction(pvals_unc, alpha=0.05, method="bh")
    pvals_bonf = apply_bonferroni_correction(pvals_unc, alpha=0.05)

    payload = {
        "observed": observed,
        "perm_scores": perm_scores,
        "pvals_uncorrected": pvals_unc.reshape(observed.shape),
        "pvals_tmax": pvals_tmax.reshape(observed.shape),
        "pvals_fdr_bh": pvals_fdr.reshape(observed.shape),
        "pvals_bonferroni": pvals_bonf.reshape(observed.shape),
    }
    if spatial_names is not None:
        payload["spatial_names"] = np.asarray(spatial_names)
    if feature_names is not None:
        payload["feature_names"] = np.asarray(feature_names)
    return payload, (last_meta or {})


def _load_axis(
    output_dir: Path,
    label: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    axis: str,
    importance: str,
    trial_type: str,
    analysis_level: Optional[str],
    inout_selection: str = "strict",
) -> Optional[Tuple[Dict[str, np.ndarray], Dict]]:
    base = build_mf_base_name(
        label, space, inout_bounds, clf_name, cv_name, axis,
        importance, trial_type, analysis_level=analysis_level,
        inout_selection=inout_selection,
    )
    score_path = output_dir / f"{base}_scores.npz"
    meta_path = output_dir / f"{base}_metadata.json"

    if score_path.exists():
        with np.load(score_path, allow_pickle=True) as npz:
            payload = {k: npz[k] for k in npz.files}
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return payload, meta

    # Maybe per-cell chunked
    if axis == "per-cell":
        merged = _merge_per_cell_chunks(output_dir, base, axis)
        if merged is not None:
            return merged
    return None


def aggregate(
    output_dir: Path,
    label: str,
    space: str,
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    importance: str,
    trial_type: str,
    analysis_level: Optional[str] = None,
    axes: Optional[List[str]] = None,
    inout_selection: str = "strict",
) -> Path:
    if axes is None:
        axes = list(AXES)

    bundle_payload: Dict[str, np.ndarray] = {}
    per_axis_meta: Dict[str, Dict] = {}
    spatial_names_seen: Optional[np.ndarray] = None
    feature_names_seen: Optional[np.ndarray] = None
    missing: List[str] = []

    for axis in axes:
        loaded = _load_axis(
            output_dir, label, space, inout_bounds, clf_name, cv_name,
            axis, importance, trial_type, analysis_level,
            inout_selection=inout_selection,
        )
        if loaded is None:
            missing.append(axis)
            continue
        payload, meta = loaded
        per_axis_meta[axis] = meta

        for k, v in payload.items():
            if k == "spatial_names":
                if spatial_names_seen is None:
                    spatial_names_seen = np.asarray(v)
                elif not np.array_equal(np.asarray(v), spatial_names_seen):
                    raise ValueError(
                        f"spatial_names disagree between axes for {label}/{space}"
                    )
                continue
            if k == "feature_names":
                if feature_names_seen is None:
                    feature_names_seen = np.asarray(v)
                elif not np.array_equal(np.asarray(v), feature_names_seen):
                    raise ValueError(
                        f"feature_names disagree between axes for {label}/{space}"
                    )
                continue
            bundle_payload[f"{axis}/{k}"] = v

    if not bundle_payload:
        raise FileNotFoundError(
            f"No axis files found for label={label} space={space} clf={clf_name} "
            f"cv={cv_name} importance={importance} type={trial_type} "
            f"level={analysis_level} in {output_dir}"
        )
    if missing:
        logger.warning(f"Missing axes (skipped): {missing}")

    if spatial_names_seen is not None:
        bundle_payload["spatial_names"] = spatial_names_seen
    if feature_names_seen is not None:
        bundle_payload["feature_names"] = feature_names_seen

    bundle_base = build_mf_base_name(
        label, space, inout_bounds, clf_name, cv_name,
        axis="bundle", importance=importance, trial_type=trial_type,
        analysis_level=analysis_level, inout_selection=inout_selection,
    )
    out_npz = output_dir / f"{bundle_base}_scores.npz"
    np.savez_compressed(out_npz, **bundle_payload)

    summary: Dict[str, object] = {"axes_included": list(per_axis_meta.keys())}
    for axis, m in per_axis_meta.items():
        summary[axis] = m.get("summary")
    bundle_meta = {
        "label": label,
        "space": space,
        "inout_bounds": list(inout_bounds),
        "classifier": clf_name,
        "cv_strategy": cv_name,
        "importance_method": importance,
        "trial_type": trial_type,
        "analysis_level": analysis_level,
        "timestamp": datetime.now().isoformat(),
        "per_axis_metadata": per_axis_meta,
        "summary": summary,
    }
    out_meta = output_dir / f"{bundle_base}_metadata.json"
    out_meta.write_text(json.dumps(bundle_meta, indent=2, default=str))

    logger.info(f"Bundle written -> {out_npz}")
    logger.info(f"Bundle meta    -> {out_meta}")
    return out_npz


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-feature classification outputs into one bundle."
    )
    parser.add_argument("--label", required=True,
                        help="Feature label used at run time (e.g. 'combined-19').")
    parser.add_argument("--space", required=True)
    parser.add_argument("--clf", default="logistic")
    parser.add_argument("--cv", default="logo")
    parser.add_argument("--importance", default="permutation",
                        choices=("permutation", "coef", "tree", "none"))
    parser.add_argument("--trial-type", default="alltrials",
                        choices=["alltrials", "correct", "rare", "lapse",
                                 "correct_commission"])
    parser.add_argument("--analysis-level", default="epoch")
    parser.add_argument("--axes", nargs="+", default=None,
                        help=f"Subset of axes to include. Default: all of {list(AXES)}.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inout-selection", default=None,
                        choices=["strict", "lenient", "vtcfilt", "vtcraw"],
                        help="IN/OUT selection strategy whose outputs to aggregate. "
                             "Defaults to config.analysis.inout_selection (or 'strict').")
    parser.add_argument("--output-dir", default=None,
                        help="Override input/output directory.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    inout_bounds = tuple(config["analysis"]["inout_bounds"])
    inout_selection = args.inout_selection or str(
        config.get("analysis", {}).get("inout_selection", "strict")
    )
    data_root = Path(config["paths"]["data_root"])
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            data_root / config["paths"]["results"]
            / f"classification_{args.space}" / "group_mf"
        )

    aggregate(
        output_dir=output_dir,
        label=args.label,
        space=args.space,
        inout_bounds=inout_bounds,
        clf_name=args.clf,
        cv_name=args.cv,
        importance=args.importance,
        trial_type=args.trial_type,
        analysis_level=args.analysis_level,
        axes=args.axes,
        inout_selection=inout_selection,
    )


if __name__ == "__main__":
    main()
