"""Submit the complete state-multifeature dependency graph to SLURM."""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from code.analysis.fast_state_submit import _analysis_root, _resource, _script, _submit
from code.analysis.provenance import resolve_analysis_directory
from code.utils.config import load_config

LOGGER = logging.getLogger(__name__)


def _command(python: str, stage: str, common: list[str], *arguments: str) -> list[str]:
    return [python, "-m", "code.analysis.state_multifeature_workflow", stage, *common, *arguments]


def build_scripts(args: argparse.Namespace, config: dict) -> dict[str, str]:
    """Build all state-multifeature scripts without submission."""
    root = _analysis_root(config, args.analysis_root)
    resolve_analysis_directory(root)
    project = Path.cwd().resolve()
    venv = Path(config["paths"]["venv"]).resolve()
    logs = Path(config["paths"]["logs"]).resolve() / "analysis_workflow"
    account = str(args.account or config["computing"]["slurm"]["account"])
    python = str(venv / "bin" / "python")
    common = ["--analysis-root", str(root), "--config", args.config]
    subjects = config["bids"]["subjects"]
    population_batches = math.ceil(args.n_permutations / args.permutations_per_job)
    within_batches = math.ceil(
        args.n_permutations / args.within_permutations_per_job
    )
    within_cells = len(subjects) * within_batches
    prepare = _command(
        python, "prepare", common,
        "--n-permutations", str(args.n_permutations),
        "--alpha", str(args.alpha),
        "--tolerance", str(args.tolerance),
        "--reliance-repeats", str(args.reliance_repeats),
    )
    commands = {
        "prepare": prepare,
        "population_observed": _command(python, "population", common, "--observed", "--stage-local", "--skip-valid"),
        "population_permutations": _command(python, "population", common, "--permutations-per-job", str(args.permutations_per_job), "--stage-local", "--skip-valid"),
        "population_reliance": _command(python, "reliance", common, "--regime", "population", "--stage-local", "--skip-valid"),
        "within_observed": _command(python, "within", common, "--observed", "--stage-local", "--skip-valid"),
        "within_permutations": _command(python, "within", common, "--permutations-per-job", str(args.within_permutations_per_job), "--stage-local", "--skip-valid"),
        "within_reliance": _command(python, "reliance", common, "--regime", "within_subject", "--stage-local", "--skip-valid"),
        "aggregate": _command(python, "aggregate", common, "--sign-flip-permutations", str(args.sign_flip_permutations)),
    }
    prepare_resource = _resource(config, "state_multifeature_prepare", {"time": "24:00:00", "memory_gb": 64, "cpus": 4})
    model_resource = _resource(config, "state_multifeature_models", {"time": "06:00:00", "memory_gb": 64, "cpus": 4})
    reliance_resource = _resource(config, "state_multifeature_reliance", {"time": "06:00:00", "memory_gb": 64, "cpus": 4})
    aggregate_resource = _resource(config, "state_multifeature_results", {"time": "01:00:00", "memory_gb": 16, "cpus": 1})
    shared = {"project_root": project, "venv": venv, "logs": logs, "account": account}
    return {
        "prepare": _script(name="saflow_state_mf_prepare", command=commands["prepare"], resource=prepare_resource, **shared),
        "population_observed": _script(name="saflow_state_mf_population", command=commands["population_observed"], resource=model_resource, **shared),
        "population_permutations": _script(name="saflow_state_mf_population_null", command=commands["population_permutations"], resource=model_resource, array=f"0-{population_batches - 1}%{args.array_throttle}", **shared),
        "population_reliance": _script(name="saflow_state_mf_population_reliance", command=commands["population_reliance"], resource=reliance_resource, array=f"0-{len(subjects) - 1}%{args.subject_throttle}", **shared),
        "within_observed": _script(name="saflow_state_mf_within", command=commands["within_observed"], resource=model_resource, array=f"0-{len(subjects) - 1}%{args.subject_throttle}", **shared),
        "within_permutations": _script(name="saflow_state_mf_within_null", command=commands["within_permutations"], resource=model_resource, array=f"0-{within_cells - 1}%{args.array_throttle}", **shared),
        "within_reliance": _script(name="saflow_state_mf_within_reliance", command=commands["within_reliance"], resource=reliance_resource, array=f"0-{len(subjects) - 1}%{args.subject_throttle}", **shared),
        "aggregate": _script(name="saflow_state_mf_results", command=commands["aggregate"], resource=aggregate_resource, **shared),
    }


def submit(args: argparse.Namespace) -> dict:
    """Submit preparation, parallel branches, and dependent aggregation."""
    config = load_config(args.config)
    root = _analysis_root(config, args.analysis_root)
    analysis = resolve_analysis_directory(root)
    logs = Path(config["paths"]["logs"]).resolve() / "analysis_workflow"
    logs.mkdir(parents=True, exist_ok=True)
    scripts = build_scripts(args, config)
    script_directory = analysis / "slurm" / "scripts" / "state_multifeature"
    script_directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, content in scripts.items():
        path = script_directory / f"{name}.sh"
        path.write_text(content)
        paths[name] = path
    jobs = {"prepare": _submit(paths["prepare"])}
    branches = [name for name in scripts if name not in {"prepare", "aggregate"}]
    for name in branches:
        jobs[name] = _submit(paths[name], jobs["prepare"])
    jobs["aggregate"] = _submit(
        paths["aggregate"], ":".join(jobs[name] for name in branches)
    )
    journal = {
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_directory": str(analysis),
        "jobs": jobs,
        "scripts": {name: str(path) for name, path in paths.items()},
        "parameters": vars(args),
    }
    journal_path = analysis / "slurm" / "state_multifeature_submission.json"
    journal_path.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Submitted state-multifeature workflow: %s", jobs)
    return journal


def build_parser() -> argparse.ArgumentParser:
    """Build the shared submission parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--permutations-per-job", type=int, default=10)
    parser.add_argument("--within-permutations-per-job", type=int, default=100)
    parser.add_argument("--reliance-repeats", type=int, default=20)
    parser.add_argument("--sign-flip-permutations", type=int, default=10000)
    parser.add_argument("--array-throttle", type=int, default=25)
    parser.add_argument("--subject-throttle", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--account")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    submit(build_parser().parse_args())


if __name__ == "__main__":
    main()
