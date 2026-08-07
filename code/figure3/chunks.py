"""Immutable deterministic permutation chunks and strict aggregation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ChunkSpec:
    """Describe one half-open permutation interval."""

    analysis_id: str
    endpoint: str
    family: str
    chunk_index: int
    start: int
    stop: int
    seed: int
    config_hash: str
    git_commit: str
    feature_order: tuple[str, ...]


def derive_chunk_seed(
    analysis_id: str, endpoint: str, family: str, chunk_index: int
) -> int:
    """Derive a stable NumPy-compatible seed from immutable identifiers."""
    key = f"{analysis_id}|{endpoint}|{family}|{chunk_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**32)


def build_chunk_specs(
    *,
    analysis_id: str,
    endpoint: str,
    family: str,
    n_permutations: int,
    chunk_size: int,
    config_hash: str,
    git_commit: str,
    feature_order: Sequence[str],
) -> list[ChunkSpec]:
    """Build a gap-free deterministic chunk manifest."""
    if n_permutations < 1 or chunk_size < 1:
        raise ValueError("permutation and chunk counts must be positive")
    specs = []
    for index, start in enumerate(range(0, n_permutations, chunk_size)):
        specs.append(
            ChunkSpec(
                analysis_id=analysis_id,
                endpoint=endpoint,
                family=family,
                chunk_index=index,
                start=start,
                stop=min(start + chunk_size, n_permutations),
                seed=derive_chunk_seed(analysis_id, endpoint, family, index),
                config_hash=config_hash,
                git_commit=git_commit,
                feature_order=tuple(feature_order),
            )
        )
    return specs


def write_chunk(path: Path, spec: ChunkSpec, values: np.ndarray) -> None:
    """Atomically write one immutable chunk and its JSON sidecar."""
    array = np.asarray(values)
    if array.shape[0] != spec.stop - spec.start:
        raise ValueError("chunk row count does not match its permutation interval")
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"immutable chunk already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, values=array)
    temporary.replace(path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(asdict(spec), indent=2, sort_keys=True) + "\n")


def read_chunk(path: Path) -> tuple[ChunkSpec, np.ndarray]:
    """Load one chunk and validate its local interval and seed."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["feature_order"] = tuple(metadata["feature_order"])
    spec = ChunkSpec(**metadata)
    expected_seed = derive_chunk_seed(
        spec.analysis_id, spec.endpoint, spec.family, spec.chunk_index
    )
    if spec.seed != expected_seed:
        raise ValueError(f"incompatible seed in {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["values"])
    if values.shape[0] != spec.stop - spec.start:
        raise ValueError(f"wrong permutation row count in {path.name}")
    return spec, values


def aggregate_chunks(
    paths: Sequence[Path], expected: Sequence[ChunkSpec]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate chunks only when every expected interval is compatible."""
    if len(paths) != len(expected):
        raise ValueError("missing or duplicate chunk files")
    loaded = [read_chunk(path) for path in paths]
    loaded.sort(key=lambda item: item[0].start)
    ordered_expected = sorted(expected, key=lambda spec: spec.start)
    for (observed, _), wanted in zip(loaded, ordered_expected):
        if observed != wanted:
            raise ValueError(
                f"incompatible chunk {observed.chunk_index}; expected {wanted}"
            )
    intervals = [(spec.start, spec.stop) for spec, _ in loaded]
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] != current[0]:
            raise ValueError("chunk intervals contain a gap or overlap")
    values = np.concatenate([array for _, array in loaded], axis=0)
    manifest = {
        "analysis_id": ordered_expected[0].analysis_id,
        "endpoint": ordered_expected[0].endpoint,
        "family": ordered_expected[0].family,
        "permutation_interval": [intervals[0][0], intervals[-1][1]],
        "chunks": [asdict(spec) for spec, _ in loaded],
    }
    return values, manifest
