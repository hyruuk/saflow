"""Panel 2 rebuilt around the cross-subject multifeature sweep.

Reads the output set of ``code.classification.run_multifeature_sweep`` and
tells the cross-subject generalization story in six subpanels:

    A  Sweep landscape      — mean LOSO AUC per (reduction x estimator).
                              Shows that spatial granularity, not classifier
                              family, is what moves the needle.
    B  Cross-subject AUC    — the winning cell's nested-CV AUC for each of the
                              held-out subjects, with its 95% CI and the
                              permutation null (inset).
    C  Selection honesty    — nested-select: which cell each outer fold picked
                              when selection was redone inside the fold, and
                              the resulting leak-free AUC.
    D  Feature families     — AUC per feature set at the winning reduction.
    E  Feature reliance     — held-out AUC drop when each feature block is
                              permuted, FDR-corrected across blocks.
    F  Pattern              — Haufe-transformed activation pattern
                              (spatial unit x feature) with a marginal bar of
                              per-network reliance.

An optional third row renders the personalized contrast (population vs
participant-specific decoding, and run stability) when a ``state_multifeature``
bundle is supplied — the sweep is purely cross-subject and cannot produce it.

Usage (synthetic, for prototyping):
    python -m code.visualization.synthetic_sweep_bundle --out /tmp/synth_sweep
    python -m code.visualization.multifeature_sweep_panel --bundle-dir /tmp/synth_sweep

Usage (real):
    python -m code.visualization.multifeature_sweep_panel --space schaefer_400
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from code.classification.run_multifeature_sweep import build_output_base
from code.classification.run_classification import get_git_hash
from code.visualization.loaders import family_sort_key, feature_family, short_label
from code.visualization.multifeature_story_panel import FAMILY_COLORS, FAMILY_DISPLAY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALPHA = 0.05
CHANCE = 0.5
ACCENT = "#D95319"
MUTED = "#8C8C8C"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _f(row: Dict[str, str], key: str) -> float:
    """Float from a CSV cell, NaN when absent or blank."""
    v = row.get(key)
    if v in (None, "", "nan"):
        return float("nan")
    return float(v)


def load_bundle(out_base: Path) -> Dict[str, object]:
    """Load whichever sweep stages have landed; missing stages come back None."""
    bundle: Dict[str, object] = {"out_base": out_base}

    csv_path = out_base.with_name(out_base.name + "_sweep.csv")
    if not csv_path.exists():
        raise SystemExit(
            f"No sweep CSV at {csv_path}. Run the sweep + merge stages first."
        )
    bundle["sweep"] = _read_csv(csv_path)
    logger.info(f"sweep: {len(bundle['sweep'])} cell(s) from {csv_path.name}")

    for key, suffix in (("confirm", "_confirm"), ("importance", "_importance"),
                        ("nested", "_nested-select")):
        npz_path = out_base.with_name(out_base.name + f"{suffix}.npz")
        json_path = out_base.with_name(out_base.name + f"{suffix}.json")
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as npz:
                bundle[key] = {k: npz[k] for k in npz.files}
            logger.info(f"{key}: loaded {npz_path.name}")
        else:
            bundle[key] = None
            logger.warning(f"{key}: {npz_path.name} not found — subpanel will be blank")
        bundle[f"{key}_meta"] = (json.loads(json_path.read_text())
                                 if json_path.exists() else None)
    return bundle


def _winner_row(sweep: List[Dict[str, str]], bundle: Dict) -> Dict[str, str]:
    """The cell the confirm stage used, else the top-ranked row."""
    meta = bundle.get("confirm_meta")
    if meta and meta.get("cell", {}).get("config_id"):
        cid = meta["cell"]["config_id"]
        for r in sweep:
            if r["config_id"] == cid:
                return r
    return max(sweep, key=lambda r: _f(r, "mean_auc"))


def _estimator_sort_key(spec: str) -> Tuple[int, float, str]:
    """Group estimators by family, then order numerically within a family.

    Plain sorting puts 'logistic:C=0.1' before 'logistic:C=1' but also
    interleaves families oddly; the landscape is much easier to read as
    linear-model block, then tree block.
    """
    family, _, rest = spec.partition(":")
    rank = {"logistic": 0, "linearsvc": 1, "lda": 2, "hgb": 3, "rf": 4}
    value = float("inf")
    for item in filter(None, rest.split(",")):
        _, _, v = item.partition("=")
        try:
            value = float(v)
            break
        except ValueError:
            continue
    return rank.get(family, 9), value, spec


def _stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < ALPHA else ""


# ---------------------------------------------------------------------------
# A — sweep landscape
# ---------------------------------------------------------------------------

def _panel_a(ax, sweep: List[Dict[str, str]], winner: Dict[str, str]) -> None:
    """reduction x estimator heatmap, best over feature set and normalization.

    Held at the winner's IN/OUT bounds so every cell scores the same windows.
    """
    rows = [r for r in sweep if r["inout"] == winner["inout"]]
    reductions = sorted({r["reduction"] for r in rows})
    estimators = sorted({r["estimator"] for r in rows}, key=_estimator_sort_key)

    grid = np.full((len(reductions), len(estimators)), np.nan)
    units: Dict[str, float] = {}
    for r in rows:
        i, j = reductions.index(r["reduction"]), estimators.index(r["estimator"])
        v = _f(r, "mean_auc")
        if np.isnan(grid[i, j]) or v > grid[i, j]:
            grid[i, j] = v
        units.setdefault(r["reduction"], _f(r, "n_reduced_spatial"))

    # Best reduction on top.
    order = np.argsort(-np.nanmax(grid, axis=1))
    grid, reductions = grid[order], [reductions[i] for i in order]


    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=6)
                continue
            lo, hi = np.nanmin(grid), np.nanmax(grid)
            rel = (grid[i, j] - lo) / (hi - lo) if hi > lo else 0.5
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=5.6,
                    color="white" if rel < 0.6 else "black")

    ax.set_xticks(range(len(estimators)))
    ax.set_xticklabels([e.replace("shrinkage=", "shrink=") for e in estimators],
                       rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(reductions)))
    ax.set_yticklabels(
        [f"{r}  ({int(units[r])} unit{'s' if units[r] != 1 else ''})"
         for r in reductions],
        fontsize=6.5,
    )
    ax.set_xlabel("Estimator", fontsize=8)
    ax.set_title(
        f"A  Sweep landscape — best of {len(rows)} cells at IN/OUT {winner['inout']}",
        fontsize=9, loc="left", fontweight="bold",
    )
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Mean held-out AUC", fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# B — cross-subject generalization
# ---------------------------------------------------------------------------

def _panel_b(ax, bundle: Dict, winner: Dict[str, str]) -> None:
    confirm = bundle.get("confirm")
    if confirm is None:
        _blank(ax, "B  Cross-subject AUC", "confirm stage not available")
        return

    auc = np.asarray(confirm["nested_auc_per_subject"], dtype=float)
    finite = auc[np.isfinite(auc)]
    order = np.argsort(finite)
    x = np.arange(len(finite))

    colors = [ACCENT if v > CHANCE else MUTED for v in finite[order]]
    ax.scatter(x, finite[order], s=22, c=colors, zorder=3, edgecolor="white", lw=0.4)
    ax.axhline(CHANCE, color="black", lw=0.8, ls="--", zorder=1)

    meta = bundle.get("confirm_meta") or {}
    nested = meta.get("nested", {})
    mean = nested.get("mean_auc", float(np.mean(finite)))
    lo, hi = nested.get("ci95_low"), nested.get("ci95_high")
    ax.axhline(mean, color=ACCENT, lw=1.6, zorder=2)
    if lo is not None and hi is not None:
        ax.axhspan(lo, hi, color=ACCENT, alpha=0.13, zorder=0)

    n_above = int((finite > CHANCE).sum())
    txt = [f"nested AUC = {mean:.3f}"]
    if lo is not None:
        txt.append(f"95% CI [{lo:.3f}, {hi:.3f}]")
    txt.append(f"{n_above}/{len(finite)} subjects > 0.5")
    if nested.get("wilcoxon_p") is not None:
        txt.append(f"Wilcoxon p = {nested['wilcoxon_p']:.1e}")
    if meta.get("permutation_pvalue") is not None:
        txt.append(f"permutation p = {meta['permutation_pvalue']:.3f}")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top",
            fontsize=6.5, linespacing=1.5)

    ax.set_xlabel("Held-out participant (sorted)", fontsize=8)
    ax.set_ylabel("Held-out AUC", fontsize=8)
    ax.set_title("B  Cross-subject generalization (nested CV)", fontsize=9,
                 loc="left", fontweight="bold")
    ax.tick_params(labelsize=6.5)

    perm = np.asarray(confirm.get("perm_mean_auc", []), dtype=float)
    if perm.size:
        # The null is over the mean AUC, so it belongs on its own axes rather
        # than behind per-subject dots that it is not comparable to.
        ins = inset_axes(ax, width="34%", height="26%", loc="lower right",
                         borderpad=0.9)
        ins.hist(perm, bins=25, color="#B8C5D1", edgecolor="white", lw=0.3)
        obs = meta.get("observed_fixed_hyperparams", mean)
        ins.axvline(obs, color=ACCENT, lw=1.4)
        ins.set_title("label-permutation null", fontsize=5.5, pad=2)
        ins.tick_params(labelsize=5, length=2)
        ins.set_yticks([])


# ---------------------------------------------------------------------------
# C — selection honesty
# ---------------------------------------------------------------------------

def _panel_c(ax, bundle: Dict, winner: Dict[str, str]) -> None:
    nested = bundle.get("nested")
    meta = bundle.get("nested_meta") or {}
    if nested is None:
        _blank(ax, "C  Selection honesty", "nested-select stage not available")
        return

    counts: Dict[str, int] = dict(meta.get("selection_counts", {}))
    if not counts:
        ids, n = np.unique(np.asarray(nested["chosen_ids"], dtype=str),
                           return_counts=True)
        counts = dict(zip(ids.tolist(), n.tolist()))
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    n_folds = int(sum(counts.values()))

    labels = []
    for cid, _ in top:
        parts = cid.split("|")
        labels.append(f"{parts[3]} · {parts[4]}\n{parts[2]} · {parts[1]}"
                      if len(parts) == 5 else cid)
    values = [c for _, c in top]
    y = np.arange(len(top))[::-1]
    ax.barh(y, values, color=ACCENT, alpha=0.85, height=0.62)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.03, yi, f"{v}/{n_folds}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.set_xlim(0, max(values) * 1.32)
    ax.set_xlabel("Outer folds selecting this cell", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.set_title("C  Selection remade inside every fold", fontsize=9, loc="left",
                 fontweight="bold")

    summ = meta.get("summary", {})
    lines = []
    if summ:
        lines.append(f"leak-free AUC = {summ.get('mean_auc', float('nan')):.3f}")
        if summ.get("ci95_low") is not None:
            lines.append(f"95% CI [{summ['ci95_low']:.3f}, {summ['ci95_high']:.3f}]")
        lines.append(f"{summ.get('n_above_chance')}/{summ.get('n_subjects')} > 0.5")
    sweep_auc = _f(winner, "mean_auc")
    if summ.get("mean_auc") is not None and np.isfinite(sweep_auc):
        lines.append(f"vs sweep winner {sweep_auc:.3f} "
                     f"(Δ {summ['mean_auc'] - sweep_auc:+.3f})")
    if lines:
        ax.text(0.98, 0.04, "\n".join(lines), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5, linespacing=1.5,
                bbox=dict(fc="white", ec=MUTED, lw=0.4, alpha=0.9, pad=3))


# ---------------------------------------------------------------------------
# D — feature families
# ---------------------------------------------------------------------------

def _panel_d(ax, sweep: List[Dict[str, str]], winner: Dict[str, str]) -> None:
    """Best AUC per feature set, holding reduction / normalization / bounds fixed."""
    rows = [r for r in sweep
            if r["inout"] == winner["inout"]
            and r["reduction"] == winner["reduction"]
            and r["normalization"] == winner["normalization"]]
    best: Dict[str, Dict[str, str]] = {}
    for r in rows:
        cur = best.get(r["feature_set"])
        if cur is None or _f(r, "mean_auc") > _f(cur, "mean_auc"):
            best[r["feature_set"]] = r
    if not best:
        _blank(ax, "D  Feature families", "no matching cells")
        return

    items = sorted(best.items(), key=lambda kv: -_f(kv[1], "mean_auc"))
    names = [k for k, _ in items]
    vals = np.array([_f(r, "mean_auc") for _, r in items])
    lo = np.array([_f(r, "ci95_low") for _, r in items])
    hi = np.array([_f(r, "ci95_high") for _, r in items])
    x = np.arange(len(names))

    palette = {"all": "#4C4C4C", "fooof": FAMILY_COLORS["fooof"],
               "psds": FAMILY_COLORS["psd"],
               "psds_corrected": FAMILY_COLORS["psd_corrected"],
               "complexity": FAMILY_COLORS["complexity"]}
    ax.bar(x, vals - CHANCE, bottom=CHANCE, width=0.62,
           color=[palette.get(n, MUTED) for n in names], alpha=0.9)
    ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none",
                ecolor="black", elinewidth=0.8, capsize=2.5)
    ax.axhline(CHANCE, color="black", lw=0.8, ls="--")
    for xi, (name, r) in zip(x, items):
        ax.text(xi, CHANCE - 0.004, f"{int(_f(r, 'n_features'))}f", ha="center",
                va="top", fontsize=5.8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6.5)
    ax.set_ylabel("Mean held-out AUC", fontsize=8)
    ax.set_ylim(min(CHANCE - 0.012, float(np.nanmin(lo)) - 0.005),
                float(np.nanmax(hi)) + 0.008)
    ax.tick_params(labelsize=6.5)
    ax.set_title(f"D  Feature families ({winner['reduction']}, "
                 f"{winner['normalization']})", fontsize=9, loc="left",
                 fontweight="bold")


# ---------------------------------------------------------------------------
# E / F — importance and pattern
# ---------------------------------------------------------------------------

def _ordered_features(names: List[str]) -> np.ndarray:
    return np.array(sorted(range(len(names)), key=lambda i: family_sort_key(names[i])))


def _panel_e(ax, bundle: Dict) -> None:
    imp = bundle.get("importance")
    if imp is None or "importance_by_feature" not in imp:
        _blank(ax, "E  Feature reliance", "importance stage not available")
        return

    names = [str(n) for n in imp["feature_names"]]
    vals = np.asarray(imp["importance_by_feature"], dtype=float)
    sem = np.asarray(imp.get("importance_by_feature_sem",
                             np.zeros_like(vals)), dtype=float)
    q = np.asarray(imp.get("importance_by_feature_pvalue_fdr",
                           np.full_like(vals, np.nan)), dtype=float)

    order = _ordered_features(names)
    vals, sem, q = vals[order], sem[order], q[order]
    labels = [short_label(names[i]) for i in order]
    fams = [feature_family(names[i]) for i in order]
    x = np.arange(len(order))

    ax.bar(x, vals, width=0.66, color=[FAMILY_COLORS.get(f, MUTED) for f in fams],
           alpha=0.9)
    ax.errorbar(x, vals, yerr=sem, fmt="none", ecolor="black", elinewidth=0.7,
                capsize=1.8)
    for xi, v, qi in zip(x, vals, q):
        s = _stars(qi)
        if s:
            ax.text(xi, v + sem[xi] + 0.0015, s, ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.8)
    ax.set_ylabel("Held-out AUC drop", fontsize=8)
    ax.tick_params(labelsize=6.5)
    n_sig = int(np.nansum(q < ALPHA))
    ax.set_title(f"E  Feature reliance — {n_sig}/{len(q)} at q<{ALPHA}",
                 fontsize=9, loc="left", fontweight="bold")
    present = [f for f in FAMILY_DISPLAY if f in set(fams)]
    if len(present) > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in present]
        ax.legend(handles, [FAMILY_DISPLAY[f] for f in present], fontsize=5.6,
                  frameon=False, ncol=len(present), loc="upper right")


def _panel_f(ax, bundle: Dict) -> None:
    imp = bundle.get("importance")
    if imp is None or "haufe_pattern" not in imp:
        _blank(ax, "F  Activation pattern", "importance stage not available")
        return

    names = [str(n) for n in imp["feature_names"]]
    spatial = [str(s) for s in imp["spatial_names"]]
    pattern = np.asarray(imp["haufe_pattern"], dtype=float)
    q = np.asarray(imp.get("haufe_pvalue_fdr",
                           np.full(pattern.shape, np.nan)), dtype=float)

    order = _ordered_features(names)
    pattern, q = pattern[:, order], q[:, order]
    labels = [short_label(names[i]) for i in order]

    vmax = float(np.nanmax(np.abs(pattern))) or 1.0
    im = ax.imshow(pattern, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    # Stipple the cells that survive correction rather than masking the rest,
    # so the reader still sees the full pattern.
    ys, xs = np.where(q < ALPHA)
    ax.scatter(xs, ys, s=3.2, c="black", marker="o", lw=0)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.8)
    ax.set_yticks(range(len(spatial)))
    ax.set_yticklabels(spatial, fontsize=6.5)
    ax.set_title("F  Haufe pattern (dots: q<0.05)", fontsize=9, loc="left",
                 fontweight="bold")
    # Dedicated cax to the right of the marginal reliance bars — a colorbar
    # attached to `ax` lands on top of them.
    cax = ax.inset_axes([1.30, 0.05, 0.035, 0.90])
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("Activation (per SD)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    if "importance_by_spatial" in imp:
        sp = np.asarray(imp["importance_by_spatial"], dtype=float)
        spq = np.asarray(imp.get("importance_by_spatial_pvalue_fdr",
                                 np.full_like(sp, np.nan)), dtype=float)
        side = ax.inset_axes([1.03, 0, 0.20, 1.0], sharey=ax)
        y = np.arange(len(sp))
        side.barh(y, sp, color=ACCENT, alpha=0.85, height=0.62)
        for yi, (v, qi) in enumerate(zip(sp, spq)):
            s = _stars(qi)
            if s:
                side.text(v, yi, " " + s, va="center", fontsize=5.5)
        side.axvline(0, color="black", lw=0.7)
        side.set_xlabel("AUC drop", fontsize=6)
        side.tick_params(labelsize=5.5, labelleft=False)
        side.set_title("reliance", fontsize=6)


# ---------------------------------------------------------------------------
# Optional row — personalized contrast (from state_multifeature)
# ---------------------------------------------------------------------------

def _panel_g(ax, state: Dict[str, np.ndarray]) -> None:
    pop = np.asarray(state["population_subject_auc"], dtype=float)
    ind = np.asarray(state["within_subject_auc"], dtype=float)
    n = min(len(pop), len(ind))
    pop, ind = pop[:n], ind[:n]
    for a, b in zip(pop, ind):
        ax.plot((0, 1), (a, b), color="#A8A8A8", lw=0.6, alpha=0.65)
    ax.scatter(np.zeros(n), pop, s=14, color=ACCENT, zorder=3)
    ax.scatter(np.ones(n), ind, s=14, color="#2878B5", zorder=3)
    ax.plot((0, 1), (np.nanmean(pop), np.nanmean(ind)), color="black", lw=2.0)
    ax.axhline(CHANCE, color="black", lw=0.8, ls="--")
    ax.set_xticks((0, 1))
    ax.set_xticklabels(("Population\n(LOSO)", "Personalized\n(leave-one-run-out)"),
                       fontsize=6.5)
    ax.set_ylabel("Held-out AUC", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.text(0.5, -0.16, f"personalized higher in {int(np.sum(ind > pop))}/{n}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
    ax.set_title("G  Population vs personalized", fontsize=9, loc="left",
                 fontweight="bold")


def _panel_h(ax, state: Dict[str, np.ndarray]) -> None:
    values = np.asarray(state["within_run_auc"], dtype=float)
    order = np.argsort(np.nanmean(values, axis=1))
    im = ax.imshow(values[order], cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels([f"Run {i + 1}" for i in range(values.shape[1])],
                       rotation=45, ha="right", fontsize=6)
    ax.set_yticks((0, len(order) - 1))
    ax.set_yticklabels(("lower mean", "higher mean"), fontsize=6)
    ax.set_ylabel("Participants (ordered)", fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Held-out AUC", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("H  Run stability (personalized)", fontsize=9, loc="left",
                 fontweight="bold")


def _blank(ax, title: str, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8, color=MUTED,
            transform=ax.transAxes)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def render_panel(bundle: Dict, title: Optional[str] = None,
                 state: Optional[Dict[str, np.ndarray]] = None) -> plt.Figure:
    sweep = bundle["sweep"]
    winner = _winner_row(sweep, bundle)

    n_rows = 3 if state else 2
    fig = plt.figure(figsize=(16.5, 4.3 * n_rows + 0.7))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.42, wspace=0.42,
                  left=0.055, right=0.93, top=0.89, bottom=0.08)

    _panel_a(fig.add_subplot(gs[0, 0]), sweep, winner)
    _panel_b(fig.add_subplot(gs[0, 1]), bundle, winner)
    _panel_c(fig.add_subplot(gs[0, 2]), bundle, winner)
    _panel_d(fig.add_subplot(gs[1, 0]), sweep, winner)
    _panel_e(fig.add_subplot(gs[1, 1]), bundle)
    _panel_f(fig.add_subplot(gs[1, 2]), bundle)
    if state:
        _panel_g(fig.add_subplot(gs[2, 0]), state)
        _panel_h(fig.add_subplot(gs[2, 1]), state)

    if title is None:
        title = (
            "Cross-subject decoding of attentional state — winning cell: "
            f"{winner['reduction']} · {winner['feature_set']} · "
            f"{winner['estimator']} · {winner['normalization']} · "
            f"IN/OUT {winner['inout']}"
        )
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.965)
    if (bundle.get("confirm_meta") or {}).get("synthetic"):
        fig.text(0.5, 0.935, "SYNTHETIC DATA — prototyping only", ha="center",
                 fontsize=9, color=ACCENT, fontweight="bold")
    return fig


def export_table(bundle: Dict, out_csv: Path) -> None:
    """Write the numbers behind E and F so the manuscript can cite them."""
    imp = bundle.get("importance")
    if imp is None or "importance_by_feature" not in imp:
        return
    names = [str(n) for n in imp["feature_names"]]
    spatial = [str(s) for s in imp["spatial_names"]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block_type", "block", "auc_drop", "sem", "p", "q_fdr"])
        for kind, labels, key in (("feature", names, "importance_by_feature"),
                                  ("spatial", spatial, "importance_by_spatial")):
            if key not in imp:
                continue
            vals = np.asarray(imp[key], dtype=float)
            sem = np.asarray(imp.get(f"{key}_sem", np.full_like(vals, np.nan)))
            p = np.asarray(imp.get(f"{key}_pvalue", np.full_like(vals, np.nan)))
            q = np.asarray(imp.get(f"{key}_pvalue_fdr", np.full_like(vals, np.nan)))
            for i, lab in enumerate(labels):
                w.writerow([kind, lab, f"{vals[i]:.6f}", f"{sem[i]:.6f}",
                            f"{p[i]:.6g}", f"{q[i]:.6g}"])
        w.writerow([])
        w.writerow(["haufe_pattern", "spatial\\feature"] + names)
        pattern = np.asarray(imp["haufe_pattern"], dtype=float)
        for i, s in enumerate(spatial):
            w.writerow(["haufe_pattern", s] + [f"{v:.6f}" for v in pattern[i]])
    logger.info(f"Wrote table -> {out_csv}")




CAPTION = (
    "Figure X. Cross-subject multifeature decoding of IN versus OUT attentional "
    "state. Models were fit on 8-trial analysis windows and evaluated with "
    "leave-one-subject-out cross-validation. (A) Mean held-out AUC for every "
    "combination of spatial reduction and estimator in the sweep, taking the "
    "best feature set and per-subject normalisation in each cell. (B) Nested "
    "cross-validated AUC for each held-out participant in the winning cell, "
    "with the mean, its 95% confidence interval, and the within-subject "
    "label-permutation null (inset). (C) Cells selected by each outer fold when "
    "the entire model choice was remade inside the fold, and the resulting "
    "selection-free AUC. (D) Mean held-out AUC per feature family at the "
    "winning spatial reduction. (E) Held-out AUC decrease when each feature is "
    "permuted across all spatial units. (F) Haufe-transformed activation "
    "pattern over spatial units and features, with per-network reliance at "
    "right. Stars and dots denote Benjamini-Hochberg q < 0.05 across blocks. "
    "Decoding included 32 participants."
)


def write_sidecars(output: Path, bundle: Dict, dpi: int, synthetic: bool) -> None:
    """Write the .json / .txt provenance pair next to a manuscript figure.

    Mirrors the key set the analysis pipeline's figure schema requires
    (``panel``, ``path``, ``dpi``, ``data_mode``, ``render_parameters`` plus the
    provenance block). ``data_mode`` is "synthetic" for a prototype render —
    ``analysis.audit`` requires every sidecar to read "real", so a prototype
    left in place announces itself instead of passing silently.
    """
    winner = _winner_row(bundle["sweep"], bundle)
    meta = {
        "panel": "panel2",
        "component": "composite",
        "path": str(output),
        "dpi": dpi,
        "data_mode": "synthetic" if synthetic else "real",
        "render_parameters": {
            "dpi": dpi,
            "synthetic_watermark": bool(synthetic),
            "white_background": True,
        },
        "analysis_id": None if synthetic else str(bundle["out_base"].parent),
        "config_hash": None,
        "git": {"commit": get_git_hash()},
        "inputs": sorted(
            str(p) for p in bundle["out_base"].parent.glob(
                bundle["out_base"].name + "_*")
        ),
        "software": {"renderer": "code.visualization.multifeature_sweep_panel"},
        "winning_cell": {k: winner.get(k) for k in
                         ("config_id", "inout", "normalization", "feature_set",
                          "reduction", "estimator")},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.2.0",
    }
    if synthetic:
        meta["note"] = (
            "SYNTHETIC prototype render — numbers are generated, not measured. "
            "Regenerate from real sweep output before use."
        )
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    caption = CAPTION
    if synthetic:
        caption = "[SYNTHETIC PROTOTYPE — numbers are generated, not measured] " + caption
    output.with_suffix(".txt").write_text(caption + "\n")
    logger.info(f"Wrote sidecars -> {output.with_suffix('.json').name}, "
                f"{output.with_suffix('.txt').name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle-dir", default=None,
                   help="Directory holding the sweep output set (overrides discovery).")
    p.add_argument("--space", default="schaefer_400")
    p.add_argument("--trial-type", default="alltrials")
    p.add_argument("--n-events-window", type=int, default=8)
    p.add_argument("--inout-selection", default="strict")
    p.add_argument("--state-bundle", default=None,
                   help="Optional npz with population_subject_auc / "
                        "within_subject_auc / within_run_auc for the third row.")
    p.add_argument("--output", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--provenance", action="store_true",
                   help="Also write the .json/.txt sidecar pair next to the PNG.")
    p.add_argument("--title", default=None)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    if args.bundle_dir:
        out_dir = Path(args.bundle_dir)
    else:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        out_dir = (Path(cfg["paths"]["data_root"]) / cfg["paths"]["results"]
                   / f"classification_{args.space}" / "group_sweep")
    out_base = build_output_base(out_dir, args.space, args.trial_type,
                                 args.n_events_window, args.inout_selection)
    bundle = load_bundle(out_base)

    state = None
    if args.state_bundle:
        with np.load(args.state_bundle, allow_pickle=True) as npz:
            state = {k: npz[k] for k in npz.files}
        missing = {"population_subject_auc", "within_subject_auc",
                   "within_run_auc"} - set(state)
        if missing:
            logger.warning(f"--state-bundle lacks {sorted(missing)}; skipping row 3")
            state = None

    fig = render_panel(bundle, title=args.title, state=state)

    output = Path(args.output) if args.output else (
        Path("reports") / "figures"
        / f"panel2_multifeature_sweep_space-{args.space}_type-{args.trial_type}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote panel -> {output}")

    if args.provenance:
        write_sidecars(output, bundle, args.dpi,
                       synthetic=bool((bundle.get("confirm_meta") or {}).get("synthetic")))

    table = Path(args.table) if args.table else (
        Path("reports") / "tables"
        / f"panel2_multifeature_sweep_space-{args.space}_type-{args.trial_type}.csv"
    )
    export_table(bundle, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
