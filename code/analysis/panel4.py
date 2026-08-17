"""Build Panel 4 network attribution from validated Panel 2/3 bundles.

Inputs are immutable participant-level four-cell network means and held-out
feature-by-network reliance estimates. Outputs contain paired IN/OUT network
effects and synchronized maximum-|t| family-wise inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from code.analysis.contracts import CORRECTED_FEATURES
from code.analysis.networks import synchronized_sign_flip_test
from code.analysis.provenance import active_analysis_id, resolve_analysis_directory
from code.classification.multifeature_provenance import git_state


def compute_state_means(cell_means: np.ndarray, cell_counts: np.ndarray) -> np.ndarray:
    """Return exact count-weighted IN and OUT means from four outcome cells."""
    means = np.asarray(cell_means, dtype=float)
    counts = np.asarray(cell_counts, dtype=float)
    return np.stack(
        [
            _weighted_cell_mean(means[:, indices], counts[:, indices])
            for indices in ((0, 1), (2, 3))
        ],
        axis=1,
    )


def _weighted_cell_mean(means: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Pool cell means using their retained-window counts."""
    valid = np.isfinite(means) & (counts[..., None, None] > 0)
    weights = np.where(valid, counts[..., None, None], 0.0)
    numerator = np.nansum(means * weights, axis=1)
    denominator = np.sum(weights, axis=1)
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )


def compute_modulation_inference(
    differences: np.ndarray, *, permutations: int, seed: int
) -> dict[str, np.ndarray]:
    """Test OUT-minus-IN effects in separate FOOOF and corrected-PSD families."""
    results: dict[str, np.ndarray] = {}
    for name, feature_slice, family_seed in (
        ("fooof", slice(0, 2), seed),
        ("corrected_psd", slice(2, 9), seed + 1),
    ):
        values = differences[..., feature_slice]
        flat = values.reshape(values.shape[0], -1)
        test = synchronized_sign_flip_test({"OUT_minus_IN": flat}, permutations, family_seed)
        shape = values.shape[1:]
        results[f"{name}_t_values"] = np.asarray(test["t_values"])[0].reshape(shape)
        results[f"{name}_p_fwer"] = np.asarray(test["p_values_fwer"])[0].reshape(shape)
    return results


def build_panel4_arrays(
    analysis_dir: Path, *, permutations: int = 10_000, seed: int = 42
) -> dict[str, np.ndarray]:
    """Assemble real Panel 4 arrays from existing immutable primary bundles."""
    means, counts, subjects = _load_network_partials(analysis_dir)
    state_means = compute_state_means(means, counts)
    differences = state_means[:, 1] - state_means[:, 0]
    arrays = {
        "subject_order": subjects,
        "state_network_means": state_means,
        "network_modulation": differences,
        **compute_modulation_inference(differences, permutations=permutations, seed=seed),
    }
    arrays.update(_load_reliance(analysis_dir))
    return arrays


def _load_network_partials(analysis_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and align the nine feature-wise network summaries."""
    means, reference_counts, reference_subjects = [], None, None
    for feature in CORRECTED_FEATURES:
        path = (
            analysis_dir / "network_dynamics" / "partials" / "modulation" / feature / "observed.npz"
        )
        with np.load(path, allow_pickle=False) as bundle:
            feature_means = np.asarray(bundle["result_network_cell_means"])[..., 0]
            counts = np.asarray(bundle["result_cell_counts"])
            subjects = np.asarray(bundle["result_subject_order"])
        if reference_counts is not None and (
            not np.array_equal(counts, reference_counts)
            or not np.array_equal(subjects, reference_subjects)
        ):
            raise ValueError("Panel 4 network partials are not aligned")
        means.append(feature_means)
        reference_counts, reference_subjects = counts, subjects
    return np.stack(means, axis=-1), reference_counts, reference_subjects


def _load_reliance(analysis_dir: Path) -> dict[str, np.ndarray]:
    """Load network-major held-out reliance matrices from Panel 2."""
    path = analysis_dir / "multifeature_decoding" / "observed.npz"
    output: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as bundle:
        for regime in ("population", "within_subject"):
            values = np.asarray(bundle[f"{regime}_cell_reliance"])
            p_values = np.asarray(bundle[f"{regime}_cell_reliance_p_fwer"])
            output[f"{regime}_cell_reliance"] = values.reshape(len(values), 7, 9)
            output[f"{regime}_cell_reliance_p_fwer"] = p_values.reshape(7, 9)
    return output


def write_panel4_bundle(
    analysis_dir: Path,
    arrays: dict[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> Path:
    """Write an immutable, provenance-linked Panel 4 compact bundle."""
    target = analysis_dir / "network_state_attribution"
    target.mkdir(parents=True, exist_ok=True)
    archive, metadata = target / "observed.npz", target / "observed.json"
    if archive.exists() or metadata.exists():
        raise FileExistsError(f"immutable Panel 4 bundle exists: {target}")
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(archive)
    provenance = json.loads((analysis_dir / "provenance.json").read_text())
    source = Path(__file__)
    payload: dict[str, Any] = {
        "provenance": {
            "analysis_id": provenance["analysis_id"],
            "data_mode": "real",
            "git": provenance["git"],
            "config_hash": provenance["config_hash"],
            "software": provenance.get("software", {}),
            "inputs": [
                str(analysis_dir / "network_dynamics" / "partials" / "modulation"),
                str(analysis_dir / "multifeature_decoding" / "observed.npz"),
            ],
        },
        "summary": {
            "subject_n": int(arrays["network_modulation"].shape[0]),
            "weighting": "equal-window pooled within participant and state",
            "permutations": permutations,
            "correction": "synchronized sign-flip max-|t|; FOOOF and corrected-PSD families separate",
            "reliance_source": "held-out grouped shuffling from Panel 2",
        },
        "producer": {
            "script": str(source.relative_to(Path.cwd())),
            "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "git": git_state(Path.cwd()),
            "parameters": {"permutations": permutations, "random_seed": seed},
        },
    }
    metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def main() -> None:
    """Build the real Panel 4 compact bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--analysis-id")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analysis_id = args.analysis_id or active_analysis_id(args.analysis_root)
    directory = resolve_analysis_directory(args.analysis_root, analysis_id)
    arrays = build_panel4_arrays(directory, permutations=args.permutations, seed=args.seed)
    print(write_panel4_bundle(directory, arrays, permutations=args.permutations, seed=args.seed))


if __name__ == "__main__":
    main()
