"""Small schema-compatible synthetic run through all Phase C workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from code.paper_panels.chunks import build_chunk_specs, write_chunk
from code.paper_panels.contracts import PANEL1_FEATURES, PANEL23_FEATURES, SCHEMA_VERSION
from code.paper_panels.decoding import DecodingConfig
from code.paper_panels.networks import CELL_ORDER, YEO7_ORDER
from code.paper_panels.workers import (
    compute_panel1_statistics,
    compute_panel2_models,
    compute_panel3_modulation,
    compute_panel3_coupling,
)


def run_synthetic_phase_c(output_dir: Path, seed: int = 42) -> Path:
    """Execute all observed workers and write deterministic resumable artifacts."""
    generator = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "analysis_id": "synthetic-phase-c",
        "data_mode": "synthetic",
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
    }
    _run_panel1(output_dir, generator, provenance)
    _run_panel2(output_dir, generator, provenance)
    _run_panel3(output_dir, generator, provenance)
    _write_chunks(output_dir, provenance)
    manifest = output_dir / "phase_c_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                **provenance,
                "panels": ["panel1", "panel2", "panel3"],
                "status": "complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest


def _run_panel1(
    root: Path, generator: np.random.Generator, provenance: dict[str, Any]
) -> None:
    inside = generator.normal(size=(8, len(PANEL1_FEATURES), 20))
    outside = inside + generator.normal(0.2, 0.3, size=inside.shape)
    result = compute_panel1_statistics(
        inside, outside, feature_order=PANEL1_FEATURES
    )
    _write_result(root / "panel1", result, provenance)


def _run_panel2(
    root: Path, generator: np.random.Generator, provenance: dict[str, Any]
) -> None:
    subjects = np.repeat(np.arange(4), 24)
    states = np.tile(np.repeat(["IN", "OUT"], 12), 4)
    outcomes = np.tile(
        ["correct_omission"] * 6 + ["commission_error"] * 6, 8
    )
    tensor = generator.normal(size=(len(subjects), 5, len(PANEL23_FEATURES)))
    tensor[:, 0, 0] += (states == "OUT") * 1.5
    tensor[:, 1, 1] += (np.asarray(outcomes) == "commission_error") * 1.5
    result = compute_panel2_models(
        tensor,
        states,
        np.asarray(outcomes),
        subjects,
        feature_order=PANEL23_FEATURES,
        parcel_order=tuple(f"parcel-{index:03d}" for index in range(5)),
        config=DecodingConfig(c_grid=(0.1, 1.0), inner_splits=3, seed=7),
    )
    _write_result(root / "panel2", result, provenance)


def _run_panel3(
    root: Path, generator: np.random.Generator, provenance: dict[str, Any]
) -> None:
    subjects = np.repeat(np.arange(4), len(CELL_ORDER) * 5)
    cells = np.tile(np.repeat(CELL_ORDER, 5), 4)
    assignments = np.asarray(YEO7_ORDER)
    values = generator.normal(
        size=(len(subjects), len(assignments), len(PANEL23_FEATURES))
    )
    values[cells == "OUT_commission_error"] += 0.4
    runs = np.tile(np.repeat(["02", "03"], 10), 4)
    result = {
        "modulation": compute_panel3_modulation(
            values,
            cells,
            subjects,
            assignments,
            minimum_windows=5,
            n_permutations=19,
            seed=11,
        ),
        "coupling": compute_panel3_coupling(
            values,
            cells,
            subjects,
            runs,
            assignments,
            minimum_windows=5,
            n_permutations=19,
            seed=12,
        ),
    }
    _write_result(root / "panel3", result, provenance)


def _write_result(
    directory: Path, result: dict[str, Any], provenance: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    summary = _extract_json(result, arrays)
    np.savez_compressed(directory / "observed.npz", **arrays)
    (directory / "observed.json").write_text(
        json.dumps(
            {"provenance": provenance, "summary": summary},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _extract_json(value: Any, arrays: dict[str, np.ndarray], prefix: str = "") -> Any:
    """Move arrays to NPZ and return a JSON-safe structural summary."""
    if isinstance(value, np.ndarray):
        name = prefix or f"array_{len(arrays)}"
        arrays[name] = value
        return {"array": name, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {
            str(key): _extract_json(item, arrays, f"{prefix}_{key}".strip("_"))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _extract_json(item, arrays, f"{prefix}_{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_chunks(root: Path, provenance: dict[str, Any]) -> None:
    specs = build_chunk_specs(
        analysis_id=provenance["analysis_id"],
        endpoint="panel2",
        family="joint_models",
        n_permutations=19,
        chunk_size=7,
        config_hash="synthetic",
        git_commit="synthetic",
        feature_order=PANEL23_FEATURES,
    )
    for spec in specs:
        values = np.random.default_rng(spec.seed).random(
            (spec.stop - spec.start, 3)
        )
        write_chunk(
            root / "panel2" / "chunks" / f"chunk-{spec.chunk_index:04d}.npz",
            spec,
            values,
        )


def main() -> None:
    """Run the synthetic Phase C workflow from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    run_synthetic_phase_c(arguments.output_dir, arguments.seed)


if __name__ == "__main__":
    main()
