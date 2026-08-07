"""Atomic JSON/NPZ serialization for Panel analysis scientific results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def write_result_bundle(
    directory: Path,
    result: dict[str, Any],
    provenance: dict[str, Any],
    *,
    stem: str = "observed",
) -> tuple[Path, Path]:
    """Write numeric arrays and JSON metadata as one immutable result pair."""
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / f"{stem}.npz"
    metadata_path = directory / f"{stem}.json"
    if archive_path.exists() or metadata_path.exists():
        raise FileExistsError(f"immutable result bundle already exists: {directory / stem}")
    arrays: dict[str, np.ndarray] = {}
    summary = _extract_json(result, arrays)
    temporary = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(archive_path)
    metadata_path.write_text(
        json.dumps(
            {"provenance": provenance, "summary": summary},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return archive_path, metadata_path


def read_result_bundle(directory: Path, *, stem: str = "observed") -> dict[str, Any]:
    """Reconstruct one JSON/NPZ result pair and validate array references."""
    metadata_path = directory / f"{stem}.json"
    archive_path = directory / f"{stem}.npz"
    if not metadata_path.exists() or not archive_path.exists():
        raise FileNotFoundError(f"incomplete result bundle: {directory / stem}")
    metadata = json.loads(metadata_path.read_text())
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return {
        "provenance": metadata["provenance"],
        "result": _restore_json(metadata["summary"], arrays),
    }


def _extract_json(
    value: Any, arrays: dict[str, np.ndarray], prefix: str = "result"
) -> Any:
    """Move arrays into an NPZ payload and retain references in JSON."""
    if isinstance(value, np.ndarray):
        key = _unique_key(prefix, arrays)
        arrays[key] = value
        return {"array": key, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {
            str(key): _extract_json(nested, arrays, f"{prefix}_{key}")
            for key, nested in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _extract_json(nested, arrays, f"{prefix}_{index}")
            for index, nested in enumerate(value)
        ]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"result contains unsupported value type: {type(value).__name__}")


def _unique_key(prefix: str, arrays: dict[str, np.ndarray]) -> str:
    """Return a collision-free normalized NPZ field name."""
    base = prefix.replace(" ", "_").replace("/", "_")
    candidate = base
    index = 2
    while candidate in arrays:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _restore_json(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Restore NPZ array references embedded in result JSON."""
    if isinstance(value, dict) and set(value) == {"array", "shape", "dtype"}:
        key = value["array"]
        if key not in arrays:
            raise ValueError(f"result metadata references missing array: {key}")
        array = arrays[key]
        if list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
            raise ValueError(f"result array contract mismatch: {key}")
        return array
    if isinstance(value, dict):
        return {key: _restore_json(nested, arrays) for key, nested in value.items()}
    if isinstance(value, list):
        return [_restore_json(nested, arrays) for nested in value]
    return value
