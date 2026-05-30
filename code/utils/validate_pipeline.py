"""Validate the saflow feature pipeline and print a copy-pasteable report.

Walks the feature tree (welch_psds, welch_psds_corrected, fooof, complexity)
across the requested spaces, plus a light preprocessed-stage presence check and
a group-statistics NaN check, and reports per (family, space):

  - coverage: files found vs the config subject x task-run grid (missing combos)
  - per-file health: shape, NaN %, fully-NaN trials/ROIs, all-zero metrics,
    value-range sanity, ch_names presence
  - the "contiguous valid-block" corruption signature (the fingerprint behind
    the 2026-05-29 regen: only ~N consecutive windows valid, rest NaN)

The output is plain ASCII designed to be copy-pasted back into a chat.

Usage:
    python -m code.utils.validate_pipeline
    python -m code.utils.validate_pipeline --space schaefer_400
    python -m code.utils.validate_pipeline --space schaefer_400 --families fooof,welch_psds
    python -m code.utils.validate_pipeline --subjects 17,18,26 --space schaefer_400
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Family specs: how to find files and what to validate in each.
#   desc      : the `desc-<x>` token (window size baked in; see --window)
#   psd       : True for 3D (n_trials, n_spatial, n_freqs) PSD arrays
#   positive  : True if the array should be strictly positive where valid
#               (raw welch power) — a zero/NaN trial counts as "dead"
#   metrics   : 2D arrays (n_trials, n_spatial) to check (fooof/complexity)
#   ranges    : optional {metric: (lo, hi)} physiological sanity bounds
# --------------------------------------------------------------------------
def _family_specs(window: int) -> Dict[str, dict]:
    w = f"w{window}" if window and window >= 2 else ""
    return {
        "welch_psds": dict(
            desc=f"welch{w}", psd=True, positive=True, metrics=["psds"],
        ),
        "welch_psds_corrected": dict(
            desc=f"welch-corr{w}", psd=True, positive=False, metrics=["psds"],
        ),
        "fooof": dict(
            desc=f"fooof{w}", psd=False, positive=False,
            metrics=["exponent", "offset", "r_squared"],
            ranges={"exponent": (-3.0, 6.0), "r_squared": (-0.01, 1.01),
                    "offset": (-15.0, 10.0)},
        ),
        "complexity": dict(
            desc=f"complexity{w}", psd=False, positive=False,
            metrics=["lzc_median", "entropy_permutation", "entropy_spectral",
                     "entropy_sample", "entropy_approximate", "entropy_svd",
                     "fractal_higuchi", "fractal_petrosian", "fractal_katz",
                     "fractal_dfa"],
        ),
    }


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# Per-file validation
# --------------------------------------------------------------------------
def _measure_file(path: Path, spec: dict) -> dict:
    """Load one feature npz and return raw per-metric measurements.

    No FAIL/WARN classification here — that happens in run() once the
    across-files context is known (so a metric that is dead in *every* file
    is reported as a systematic issue, not as per-subject corruption)."""
    out: dict = {"path": path, "readable": True, "file_issues": [],
                 "has_ch_names": False, "metrics": {}, "missing_metrics": []}
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as exc:                       # corrupt/truncated archive
        out["readable"] = False
        out["file_issues"].append(("FAIL", f"cannot read npz: {exc}"))
        return out

    keys = set(z.files)
    out["has_ch_names"] = "ch_names" in keys
    out["missing_metrics"] = [m for m in spec["metrics"] if m not in keys]

    n_trials = n_spatial = None
    for m in spec["metrics"]:
        if m not in keys:
            continue
        a = np.asarray(z[m])
        want_ndim = 3 if spec["psd"] else 2
        if a.ndim != want_ndim:
            out["file_issues"].append(
                ("FAIL", f"{m}: expected {want_ndim}D, got shape {a.shape}"))
            continue
        nt, ns = a.shape[0], a.shape[1]
        usable = np.isfinite(a)
        if spec["psd"] and spec["positive"]:
            usable &= a > 0
        reduce_axis = (1, 2) if spec["psd"] else 1
        valid_trial = usable.any(axis=reduce_axis)
        dead_roi_axis = (0, 2) if spec["psd"] else 0
        n_dead_roi = int((~usable.any(axis=dead_roi_axis)).sum())
        n_valid = int(valid_trial.sum())
        idx = np.flatnonzero(valid_trial)
        contiguous = bool(n_valid and (idx[-1] - idx[0] + 1) == n_valid)
        finite = a[np.isfinite(a)]
        rng = spec.get("ranges", {}).get(m)
        range_bad = None
        if finite.size and rng is not None:
            fmin, fmax = float(finite.min()), float(finite.max())
            if fmin < rng[0] or fmax > rng[1]:
                range_bad = f"[{fmin:.2f},{fmax:.2f}] outside [{rng[0]},{rng[1]}]"
        out["metrics"][m] = dict(
            n_trials=nt, n_spatial=ns, n_valid=n_valid,
            dead_frac=(1.0 - n_valid / nt) if nt else 1.0,
            contiguous=contiguous,
            block=(int(idx[0]), int(idx[-1])) if n_valid else None,
            zero_frac=float((a == 0).mean()),
            n_dead_roi=n_dead_roi,
            has_inf=bool(finite.size and not np.isfinite(finite).all()),
            range_bad=range_bad,
        )
        if n_trials is None:
            n_trials, n_spatial = nt, ns
        elif (nt, ns) != (n_trials, n_spatial):
            out["file_issues"].append(
                ("WARN", f"{m}: shape ({nt},{ns}) differs from siblings"))
    out["n_trials"], out["n_spatial"] = n_trials, n_spatial
    return out


# --------------------------------------------------------------------------
# Coverage discovery
# --------------------------------------------------------------------------
def _discover(features_root: Path, family: str, space: str, desc: str,
              subjects: List[str], runs: List[str]
              ) -> Tuple[Dict[Tuple[str, str], Path], List[Tuple[str, str]]]:
    """Return ({(sub,run): path}, missing[(sub,run)]) for the config grid."""
    fam_dir = features_root / f"{family}_{space}"
    found: Dict[Tuple[str, str], Path] = {}
    missing: List[Tuple[str, str]] = []
    for sub in subjects:
        for run in runs:
            hits = glob.glob(
                str(fam_dir / f"sub-{sub}" /
                    f"sub-{sub}_*run-{run}_*_desc-{desc}*.npz"))
            hits = [h for h in hits if not h.endswith("_params.json")]
            if hits:
                found[(sub, run)] = Path(sorted(hits)[0])
            else:
                missing.append((sub, run))
    return found, missing


def _fmt_combos(combos: List[Tuple[str, str]],
                all_runs: Optional[List[str]] = None) -> str:
    """Compress [(sub,run)...] into 'sub-XX:rRR,RR; ...'.

    A subject missing every run in ``all_runs`` collapses to 'sub-XX:all'."""
    by_sub: Dict[str, List[str]] = {}
    for sub, run in combos:
        by_sub.setdefault(sub, []).append(run)
    parts = []
    full = set(all_runs) if all_runs else None
    for s in sorted(by_sub):
        rs = sorted(by_sub[s])
        if full is not None and set(rs) == full:
            parts.append(f"sub-{s}:all")
        else:
            parts.append(f"sub-{s}:r{','.join(rs)}")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def run(config_path: str, spaces: List[str], families: List[str],
        subjects: List[str], runs: List[str], window: int) -> int:
    config = _load_config(config_path)
    data_root = Path(config["paths"]["data_root"])
    features_root = data_root / config["paths"]["features"]
    specs = _family_specs(window)

    lines: List[str] = []
    P = lines.append
    P("=" * 78)
    P("SAFLOW PIPELINE VALIDATION REPORT")
    P("=" * 78)
    P(f"data_root : {data_root}")
    P(f"window    : n_events_window={window} (desc token *{specs['fooof']['desc']}*)")
    P(f"grid      : {len(subjects)} subjects x {len(runs)} runs = "
      f"{len(subjects)*len(runs)} expected files per family/space")
    P(f"subjects  : {','.join(subjects)}")
    P(f"spaces    : {', '.join(spaces)}    families: {', '.join(families)}")
    P("")

    grand_fail = 0
    grand_warn = 0
    regen: Dict[Tuple[str, str], set] = {}   # (family,space) -> {(sub,run)}

    # ---- preprocessed presence (anchors expectations) --------------------
    prep = data_root / config["paths"]["derivatives"] / "preprocessed"
    if prep.exists():
        P("-" * 78)
        P("STAGE 0  preprocessed/  (proc-clean MEG)")
        P("-" * 78)
        pfound, pmiss = _discover_prep(prep, subjects, runs)
        P(f"  found {len(pfound)}/{len(subjects)*len(runs)} clean files")
        if pmiss:
            P(f"  absent: {_fmt_combos(pmiss, runs)}")
        P("")

    # ---- feature families x spaces ---------------------------------------
    for space in spaces:
        for family in families:
            spec = specs[family]
            fam_dir = features_root / f"{family}_{space}"
            P("-" * 78)
            P(f"{family}_{space}")
            P("-" * 78)
            if not fam_dir.exists():
                P("  (folder absent — not downloaded / not generated)")
                P("")
                continue
            found, missing = _discover(
                features_root, family, space, spec["desc"], subjects, runs)
            if not found:
                P(f"  coverage : 0/{len(subjects)*len(runs)} files "
                  "(none present — not downloaded / not generated)")
                P("")
                continue
            P(f"  coverage : {len(found)}/{len(subjects)*len(runs)} files"
              + (f"   MISSING: {_fmt_combos(missing, runs)}" if missing
                 else "   (complete)"))

            # ---- single pass: measure every file -------------------------
            measures: Dict[Tuple[str, str], dict] = {}
            for key, path in sorted(found.items()):
                measures[key] = _measure_file(path, spec)

            # ---- systematic context: which metrics are dead/degenerate ---
            # in (almost) EVERY file? Those are family-wide, not per-subject.
            present = [m for m in measures.values() if m["readable"]]
            n_present = len(present) or 1
            sys_dead, sys_zero = {}, {}
            for metric in spec["metrics"]:
                vals = [f["metrics"][metric] for f in present
                        if metric in f["metrics"]]
                if not vals:
                    continue
                n_dead = sum(1 for v in vals if v["dead_frac"] >= 0.5)
                n_zero = sum(1 for v in vals if v["zero_frac"] > 0.90)
                if n_dead >= 0.9 * len(vals):
                    sys_dead[metric] = (n_dead, len(vals),
                                        float(np.median([v["dead_frac"] for v in vals])))
                elif n_zero >= 0.9 * len(vals):
                    sys_zero[metric] = (n_zero, len(vals),
                                        float(np.median([v["zero_frac"] for v in vals])))
            for metric, (n, tot, med) in sorted(sys_dead.items()):
                P(f"  NOTE     : '{metric}' systematically empty — {med*100:.0f}% NaN "
                  f"in {n}/{tot} files (unpopulated metric, not per-subject corruption)")
                grand_warn += 1
            for metric, (n, tot, med) in sorted(sys_zero.items()):
                P(f"  NOTE     : '{metric}' ~{med*100:.0f}% exactly zero in {n}/{tot} "
                  f"files — degenerate, will NaN paired t-tests (zero variance)")
                grand_warn += 1

            # ---- per-file classification (excluding systematic metrics) --
            clean = 0
            for (sub, run), f in sorted(measures.items()):
                fails: List[str] = []
                warns: List[str] = [m for s, m in f["file_issues"] if s == "WARN"]
                fails += [m for s, m in f["file_issues"] if s == "FAIL"]
                if not f["readable"]:
                    fails.append("unreadable npz")
                if f["missing_metrics"]:
                    fails.append(f"missing keys {f['missing_metrics']}")
                if not f["has_ch_names"]:
                    warns.append("missing ch_names (numeric fallback bug)")

                # group metrics that share an identical corruption verdict so
                # exponent/offset/r_squared collapse into one line.
                buckets: Dict[str, List[str]] = {}
                for metric, mv in f["metrics"].items():
                    if metric in sys_dead or metric in sys_zero:
                        continue                       # reported family-wide
                    verdict = None
                    if mv["dead_frac"] >= 0.5:
                        if mv["n_valid"] == 0:
                            verdict = f"ALL {mv['n_trials']} trials dead"
                        elif mv["contiguous"]:
                            verdict = (f"CONTIGUOUS-BLOCK corruption — only "
                                       f"{mv['n_valid']}/{mv['n_trials']} valid windows "
                                       f"[{mv['block'][0]}-{mv['block'][1]}], "
                                       f"{mv['dead_frac']*100:.0f}% dead")
                        else:
                            verdict = (f"{mv['dead_frac']*100:.0f}% dead trials "
                                       f"({mv['n_valid']}/{mv['n_trials']} valid, scattered)")
                    if verdict:
                        buckets.setdefault(verdict, []).append(metric)
                    if mv["has_inf"]:
                        fails.append(f"{metric}: contains +/-inf")
                    if 0 < mv["dead_frac"] < 0.5:
                        warns.append(f"{metric}: {mv['dead_frac']*100:.0f}% dead trials")
                    if mv["range_bad"]:
                        warns.append(f"{metric}: values {mv['range_bad']}")
                    if 0 < mv["n_dead_roi"] < mv["n_spatial"]:
                        warns.append(f"{metric}: {mv['n_dead_roi']}/{mv['n_spatial']} ROIs never valid")
                for verdict, mets in buckets.items():
                    fails.append(f"{','.join(mets)}: {verdict}")

                if not fails and not warns:
                    clean += 1
                    continue
                if fails:
                    regen.setdefault((family, space), set()).add((sub, run))
                grand_fail += len(fails)
                grand_warn += len(warns)
                tag = "FAIL" if fails else "WARN"
                P(f"    [{tag}] sub-{sub} run-{run}")
                for m in fails:
                    P(f"        - FAIL: {m}")
                for m in warns:
                    P(f"        - warn: {m}")

            P(f"  health   : {clean}/{len(found)} files clean")
            P("")

    # ---- group statistics NaN check --------------------------------------
    results_root = data_root / config["paths"]["results"]
    for space in spaces:
        sdir = results_root / f"statistics_{space}"
        if not sdir.exists():
            continue
        P("-" * 78)
        P(f"GROUP STATS  statistics_{space}/  (all-NaN tvals = bad inputs)")
        P("-" * 78)
        nan_feats: Dict[str, int] = {}
        ok_files = 0
        n_nan = 0
        for f in sorted(sdir.glob("*_results.npz")):
            feat = f.name.split("_inout")[0].replace("feature-", "")
            try:
                with np.load(f, allow_pickle=True) as z:
                    t = np.asarray(z["tvals"]).ravel()
                if t.size and not np.isfinite(t).any():
                    nan_feats[feat] = nan_feats.get(feat, 0) + 1
                    n_nan += 1
                else:
                    ok_files += 1
            except Exception:
                nan_feats[feat + " (unreadable)"] = nan_feats.get(feat, 0) + 1
                n_nan += 1
        P(f"  {ok_files} ok, {n_nan} all-NaN results files "
          f"({len(nan_feats)} distinct features)")
        for feat, cnt in sorted(nan_feats.items()):
            P(f"    [FAIL] all-NaN tvals: {feat} ({cnt} file(s))")
            grand_fail += 1
        P("")

    # ---- regenerate summary ----------------------------------------------
    P("=" * 78)
    P("SUMMARY")
    P("=" * 78)
    P(f"  total FAIL findings: {grand_fail}    WARN findings: {grand_warn}")
    if regen:
        P("")
        P("  REGENERATE (force full recompute, e.g. --no-skip-existing):")
        for (family, space), combos in sorted(regen.items()):
            P(f"    {family}_{space}: {_fmt_combos(sorted(combos), runs)}")
        # suggest commands for the worst offenders
        P("")
        P("  suggested commands:")
        seen = set()
        for (family, space), combos in sorted(regen.items()):
            base = "fooof" if family == "fooof" else (
                "psd" if family.startswith("welch") else "complexity")
            subs = sorted({s for s, _ in combos})
            key = (base, space)
            if key in seen:
                continue
            seen.add(key)
            P(f"    invoke pipeline.features.{base} --space={space} "
              f"--subject={','.join(subs)}  # then re-derive downstream")
    else:
        P("  no FAIL-level corruption detected in the validated files.")
    P("=" * 78)

    verdict = "FAIL" if grand_fail else ("WARN" if grand_warn else "PASS")
    P(f"VERDICT: {verdict}")
    print("\n".join(lines))
    return 1 if grand_fail else 0


def _discover_prep(prep: Path, subjects: List[str], runs: List[str]
                   ) -> Tuple[Dict[Tuple[str, str], Path], List[Tuple[str, str]]]:
    found: Dict[Tuple[str, str], Path] = {}
    missing: List[Tuple[str, str]] = []
    for sub in subjects:
        for run in runs:
            hits = glob.glob(str(
                prep / f"sub-{sub}" / "meg" /
                f"sub-{sub}_*run-{run}_*proc-clean*meg.fif"))
            if hits:
                found[(sub, run)] = Path(hits[0])
            else:
                missing.append((sub, run))
    return found, missing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--space", default=None,
                   help="comma-separated spaces (default: sensor + every "
                        "atlas folder found on disk)")
    p.add_argument("--families", default=None,
                   help="comma-separated families (default: all four)")
    p.add_argument("--subjects", default=None,
                   help="comma-separated subjects (default: config grid)")
    p.add_argument("--window", type=int, default=8,
                   help="n_events_window baked into the desc token (default 8)")
    args = p.parse_args()

    config = _load_config(args.config)
    all_specs = _family_specs(args.window)

    subjects = (args.subjects.split(",") if args.subjects
                else list(config["bids"]["subjects"]))
    runs = list(config["bids"]["task_runs"])

    families = (args.families.split(",") if args.families
                else list(all_specs.keys()))

    if args.space:
        spaces = args.space.split(",")
    else:
        features_root = Path(config["paths"]["data_root"]) / config["paths"]["features"]
        found_spaces = set()
        for d in glob.glob(str(features_root / "*_*")):
            base = os.path.basename(d)
            for fam in all_specs:
                if base.startswith(fam + "_"):
                    found_spaces.add(base[len(fam) + 1:])
        spaces = sorted(found_spaces) or ["sensor"]

    return run(args.config, spaces, families, subjects, runs, args.window)


if __name__ == "__main__":
    raise SystemExit(main())
