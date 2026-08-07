"""Immutable Saflow analysis analysis identities and artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from code.classification.multifeature_provenance import environment_snapshot, git_state

ID_PATTERN = re.compile(r"^analysis-(\d{8}T\d{6}Z)-g([0-9a-f]+|unknown)-c([0-9a-f]{12})$")


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash a resolved configuration deterministically."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def create_analysis_id(config: Mapping[str, Any], project_root: Path,
                       timestamp: datetime | None = None) -> str:
    """Create a Saflow analysis ID from UTC time, Git commit, and configuration."""
    moment = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return (f"analysis-{moment.strftime('%Y%m%dT%H%M%SZ')}-"
            f"g{git_state(project_root)['short_commit']}-c{config_hash(config)}")


def validate_analysis_id(analysis_id: str) -> None:
    """Reject unsafe or noncanonical analysis identifiers."""
    if not ID_PATTERN.fullmatch(analysis_id):
        raise ValueError(f"invalid Saflow analysis analysis ID: {analysis_id!r}")


def initialize(root: Path, analysis_id: str, config: Mapping[str, Any],
               arguments: Mapping[str, Any], project_root: Path) -> Path:
    """Create a new analysis directory; existing IDs are immutable."""
    validate_analysis_id(analysis_id)
    destination = root / analysis_id
    if destination.exists():
        raise FileExistsError(f"immutable analysis already exists: {destination}")
    destination.mkdir(parents=True)
    state = git_state(project_root)
    software = environment_snapshot()
    input_roots = [
        {
            "kind": key,
            "path": str(config.get("paths", {}).get(key, "")),
            "exists_at_initialization": Path(
                str(config.get("paths", {}).get(key, ""))
            ).exists(),
        }
        for key in ("raw", "bids", "features")
        if config.get("paths", {}).get(key)
    ]
    (destination / "resolved_config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=True))
    artifacts = {
        "dataset_description.json": {"Name": f"Saflow corrected Saflow analysis {analysis_id}",
            "BIDSVersion": "1.10.0", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "saflow Saflow analysis", "Version": state["commit"]}]},
        "provenance.json": {"analysis_id": analysis_id, "config_hash": config_hash(config),
            "created_utc": datetime.now(timezone.utc).isoformat(), "git": state,
            "inputs": input_roots, "software": software},
        "environment.json": software,
        "cli_arguments.json": dict(arguments),
    }
    for name, payload in artifacts.items():
        (destination / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    for folder in ("manifests", "qc", "statistics", "classification", "sensitivity", "tables", "figures"):
        (destination / folder).mkdir()
    return destination
