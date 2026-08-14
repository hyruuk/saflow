"""Submit the complete fast Schaefer-400 state workflow to SLURM."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code.analysis.provenance import resolve_analysis_directory
from code.utils.config import load_config

LOGGER = logging.getLogger(__name__)


def _resource(config: dict[str, Any], node: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Resolve one fast-state node resource request."""
    configured = config.get("analysis_workflow", {}).get("node_resources", {}).get(
        node, {}
    )
    return {**fallback, **configured}


def _script(
    *,
    name: str,
    command: list[str],
    project_root: Path,
    venv: Path,
    logs: Path,
    account: str,
    resource: dict[str, Any],
    array: str | None = None,
) -> str:
    """Render one self-contained SLURM script."""
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --cpus-per-task={int(resource['cpus'])}",
        f"#SBATCH --mem={int(resource['memory_gb'])}G",
        f"#SBATCH --time={resource['time']}",
        f"#SBATCH --output={logs / (name + '_%A_%a.out' if array else name + '_%j.out')}",
        f"#SBATCH --error={logs / (name + '_%A_%a.err' if array else name + '_%j.err')}",
    ]
    if array:
        directives.append(f"#SBATCH --array={array}")
    body = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(project_root))}",
        f"source {shlex.quote(str(venv / 'bin' / 'activate'))}",
        shlex.join(command),
    ]
    return "\n".join([*directives, "", *body, ""])


def build_scripts(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, str]:
    """Build the four dependency-linked workflow scripts without submitting."""
    if args.n_permutations < 1 or args.permutations_per_job < 1:
        raise ValueError("permutation counts must be positive")
    if args.array_throttle < 1:
        raise ValueError("array throttle must be positive")
    analysis_root = Path(args.analysis_root)
    resolve_analysis_directory(analysis_root)
    project_root = Path.cwd().resolve()
    venv = Path(config["paths"]["venv"]).resolve()
    logs = Path(config["paths"]["logs"]).resolve() / "analysis_workflow"
    account = str(args.account or config["computing"]["slurm"]["account"])
    if not account:
        raise ValueError("computing.slurm.account is required")
    python = str(venv / "bin" / "python")
    common = ["--analysis-root", str(analysis_root), "--config", args.config]
    prepare_command = [
        python,
        "-m",
        "code.analysis.fast_state_workflow",
        "prepare",
        *common,
        "--n-permutations",
        str(args.n_permutations),
        "--alpha",
        str(args.alpha),
        "--tolerance",
        str(args.tolerance),
    ]
    observed_command = [
        python,
        "-m",
        "code.analysis.fast_state_workflow",
        "run",
        *common,
        "--observed",
        "--stage-local",
        "--skip-valid",
    ]
    permutation_command = [
        python,
        "-m",
        "code.analysis.fast_state_workflow",
        "run",
        *common,
        "--permutations-per-job",
        str(args.permutations_per_job),
        "--stage-local",
        "--skip-valid",
    ]
    aggregate_command = [
        python,
        "-m",
        "code.analysis.fast_state_workflow",
        "aggregate",
        *common,
    ]
    prepare_resource = _resource(
        config, "fast_state_prepare", {"time": "24:00:00", "memory_gb": 64, "cpus": 4}
    )
    run_resource = _resource(
        config, "fast_state_permutations", {"time": "06:00:00", "memory_gb": 64, "cpus": 4}
    )
    aggregate_resource = {"time": "01:00:00", "memory_gb": 8, "cpus": 1}
    batch_count = math.ceil(args.n_permutations / args.permutations_per_job)
    array = f"0-{batch_count - 1}%{args.array_throttle}"
    common_script = {
        "project_root": project_root,
        "venv": venv,
        "logs": logs,
        "account": account,
    }
    return {
        "prepare": _script(name="saflow_state_prepare", command=prepare_command, resource=prepare_resource, **common_script),
        "observed": _script(name="saflow_state_observed", command=observed_command, resource=run_resource, **common_script),
        "permutations": _script(name="saflow_state_permutations", command=permutation_command, resource=run_resource, array=array, **common_script),
        "aggregate": _script(name="saflow_state_aggregate", command=aggregate_command, resource=aggregate_resource, **common_script),
    }


def _submit(path: Path, dependency: str | None = None) -> str:
    """Submit one script and return its parsable SLURM job ID."""
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(path))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().split(";", maxsplit=1)[0]


def submit(args: argparse.Namespace) -> dict[str, Any]:
    """Write scripts, submit their dependency graph, and journal job IDs."""
    config = load_config(args.config)
    analysis_directory = resolve_analysis_directory(Path(args.analysis_root))
    (Path(config["paths"]["logs"]).resolve() / "analysis_workflow").mkdir(
        parents=True, exist_ok=True
    )
    scripts = build_scripts(args, config)
    script_directory = analysis_directory / "slurm" / "scripts" / "fast_state"
    script_directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, content in scripts.items():
        path = script_directory / f"{name}.sh"
        path.write_text(content)
        paths[name] = path
    prepare_job = _submit(paths["prepare"])
    observed_job = _submit(paths["observed"], prepare_job)
    permutation_job = _submit(paths["permutations"], prepare_job)
    aggregate_job = _submit(
        paths["aggregate"], f"{observed_job}:{permutation_job}"
    )
    journal = {
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_directory": str(analysis_directory),
        "n_permutations": args.n_permutations,
        "permutations_per_job": args.permutations_per_job,
        "array_throttle": args.array_throttle,
        "jobs": {
            "prepare": prepare_job,
            "observed": observed_job,
            "permutations": permutation_job,
            "aggregate": aggregate_job,
        },
        "scripts": {name: str(path) for name, path in paths.items()},
    }
    journal_path = analysis_directory / "slurm" / "fast_state_submission.json"
    journal_path.parent.mkdir(exist_ok=True)
    journal_path.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Submitted fast-state workflow: %s", journal["jobs"])
    return journal


def main() -> None:
    """Submit the complete workflow from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--permutations-per-job", type=int, default=10)
    parser.add_argument("--array-throttle", type=int, default=25)
    parser.add_argument("--account")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    submit(parser.parse_args())


if __name__ == "__main__":
    main()
