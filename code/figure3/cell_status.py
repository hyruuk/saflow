"""Execute one immutable DAG cell and atomically record its status."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def execute_cell(spec_path: Path) -> Path:
    """Execute a JSON command specification and write complete/failed status."""
    spec = json.loads(spec_path.read_text())
    command = spec.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(value, str) for value in command
    ):
        raise ValueError("cell command must be a non-empty string list")
    status_path = Path(spec["status_path"])
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, check=False)
    payload: dict[str, Any] = {
        field: spec[field]
        for field in (
            "analysis_id",
            "node",
            "cell_index",
            "config_hash",
            "git_commit",
        )
    }
    payload.update(
        {
            "status": "complete" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "command": command,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
    )
    _write_atomic_json(status_path, payload)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return status_path


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a status record without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    """Execute a scheduler-generated cell specification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    execute_cell(parser.parse_args().spec)


if __name__ == "__main__":
    main()
