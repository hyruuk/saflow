"""Execute one immutable execution plan cell and atomically record its status."""

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
    commands = _validated_commands(spec)
    status_path = Path(spec["status_path"])
    started = datetime.now(timezone.utc)
    steps = []
    return_code = 0
    failed_command: list[str] | None = None
    for step_index, command in enumerate(commands):
        step_started = datetime.now(timezone.utc)
        result = subprocess.run(command, check=False)
        steps.append(
            {
                "step_index": step_index,
                "command": command,
                "return_code": result.returncode,
                "started_utc": step_started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        if result.returncode:
            return_code = result.returncode
            failed_command = command
            break
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
    if "execution_git_commit" in spec:
        payload["execution_git_commit"] = spec["execution_git_commit"]
    payload.update(
        {
            "status": "complete" if return_code == 0 else "failed",
            "return_code": return_code,
            "commands": commands,
            "steps": steps,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
    )
    _write_atomic_json(status_path, payload)
    if return_code:
        raise subprocess.CalledProcessError(return_code, failed_command)
    return status_path


def _validated_commands(spec: dict[str, Any]) -> list[list[str]]:
    """Return validated sequential commands, accepting legacy one-command specs."""
    commands = spec.get("commands")
    if commands is None and "command" in spec:
        commands = [spec["command"]]
    if (
        not isinstance(commands, list)
        or not commands
        or any(
            not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) for value in command)
            for command in commands
        )
    ):
        raise ValueError("cell commands must be a non-empty list of string lists")
    return commands


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
