"""Analysis identities, immutable directories, validation, and compact export."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ANALYSIS_ID_PATTERN = re.compile(
    r"^mf-(\d{8}T\d{6}Z)-g([0-9a-f]+|unknown)-c([0-9a-f]{12})$"
)


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable 12-character SHA-256 hash of resolved configuration."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:12]


def git_state(project_root: Path) -> dict[str, Any]:
    """Return commit identity and dirty state without mutating the repository."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": "unknown", "short_commit": "unknown", "dirty": None}
    return {"commit": commit, "short_commit": commit[:8], "dirty": dirty}


def create_analysis_id(
    config: Mapping[str, Any], project_root: Path, timestamp: datetime | None = None
) -> str:
    """Create an immutable analysis ID from UTC time, Git, and configuration."""
    current = timestamp or datetime.now(timezone.utc)
    utc_token = current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"mf-{utc_token}-g{git_state(project_root)['short_commit']}"
        f"-c{canonical_config_hash(config)}"
    )


def validate_analysis_id(analysis_id: str) -> None:
    """Reject identifiers that could escape or ambiguously name a directory."""
    if not ANALYSIS_ID_PATTERN.fullmatch(analysis_id):
        raise ValueError(f"Invalid multifeature analysis ID: {analysis_id!r}")


def initialize_analysis_directory(
    root: Path,
    analysis_id: str,
    config: Mapping[str, Any],
    cli_arguments: Mapping[str, Any],
    project_root: Path,
) -> Path:
    """Create one new derivative directory and its reproducibility metadata."""
    validate_analysis_id(analysis_id)
    analysis_dir = root / analysis_id
    if analysis_dir.exists():
        raise FileExistsError(f"Analysis directory already exists: {analysis_dir}")
    analysis_dir.mkdir(parents=True)
    state = git_state(project_root)
    generated_by = {
        "Name": "Saflow corrected multifeature analysis",
        "Version": state["commit"],
    }
    _write_json(
        analysis_dir / "dataset_description.json",
        {
            "Name": f"Saflow multifeature derivative {analysis_id}",
            "BIDSVersion": "1.10.0",
            "DatasetType": "derivative",
            "GeneratedBy": [generated_by],
        },
    )
    (analysis_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=True)
    )
    _write_json(analysis_dir / "cli_arguments.json", dict(cli_arguments))
    _write_json(
        analysis_dir / "provenance.json",
        {
            "analysis_id": analysis_id,
            "config_hash": canonical_config_hash(config),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git": state,
        },
    )
    _write_json(analysis_dir / "environment.json", environment_snapshot())
    return analysis_dir


def environment_snapshot() -> dict[str, Any]:
    """Capture Python, platform, and installed package versions."""
    packages = sorted(
        (item.metadata["Name"], item.version)
        for item in distributions()
        if item.metadata["Name"]
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(packages),
    }


def validate_result(
    scores_path: Path,
    metadata_path: Path,
    *,
    analysis_id: str,
    config_hash: str,
    seed: int,
    n_permutations: int,
    required_arrays: Mapping[str, Sequence[int] | None],
) -> tuple[bool, str]:
    """Validate arrays, shapes, axes, and immutable run metadata."""
    try:
        metadata = json.loads(metadata_path.read_text())
        expected = {
            "analysis_id": analysis_id,
            "config_hash": config_hash,
            "seed": seed,
            "n_permutations": n_permutations,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False, f"metadata {key}={metadata.get(key)!r}, expected {value!r}"
        with np.load(scores_path, allow_pickle=False) as result:
            for key, shape in required_arrays.items():
                if key not in result.files:
                    return False, f"missing array {key}"
                if shape is not None and tuple(result[key].shape) != tuple(shape):
                    return False, f"array {key} has shape {result[key].shape}, expected {shape}"
            for axis_name in ("feature_names", "spatial_names"):
                if axis_name not in result.files or result[axis_name].size == 0:
                    return False, f"missing or empty {axis_name}"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return False, str(error)
    return True, "valid"


def export_analysis(analysis_dir: Path, destination: Path) -> Path:
    """Create a compact export containing results and metadata, never features."""
    required = ("dataset_description.json", "resolved_config.yaml", "provenance.json")
    missing = [name for name in required if not (analysis_dir / name).exists()]
    if missing:
        raise ValueError(f"Analysis is incomplete; missing {missing}")
    if destination.exists():
        raise FileExistsError(f"Export destination already exists: {destination}")
    destination.mkdir(parents=True)
    allowed_suffixes = {".json", ".yaml", ".yml", ".npz", ".csv", ".tsv", ".png", ".svg"}
    excluded_parts = {"features", "subject_features", "chunks"}
    copied: list[str] = []
    for source in analysis_dir.rglob("*"):
        relative = source.relative_to(analysis_dir)
        if not source.is_file() or excluded_parts.intersection(relative.parts):
            continue
        if source.suffix.lower() not in allowed_suffixes:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(relative))
    _write_json(destination / "export_manifest.json", {"files": sorted(copied)})
    return destination


def inventory_legacy_results(source: Path, manifest_path: Path) -> Path:
    """Write a dry-run archive inventory without moving or deleting results."""
    entries = []
    if source.exists():
        for path in sorted(source.rglob("*combined-24*")):
            if path.is_file():
                entries.append({
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_hash(path),
                })
    _write_json(
        manifest_path,
        {
            "mode": "dry-run-inventory",
            "source": str(source),
            "files": entries,
            "moved": False,
            "deleted": False,
        },
    )
    return manifest_path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
