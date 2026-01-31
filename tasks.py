"""Invoke tasks for saflow development.

This module provides common development tasks using the invoke library.
Run `invoke --list` to see all available tasks.

Usage:
    invoke test              # Run tests
    invoke lint              # Run linting
    invoke format            # Format code
    invoke clean             # Clean generated files
    invoke setup --mode=dev  # Setup environment
"""

from pathlib import Path
from invoke import task


# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent
SAFLOW_PKG = PROJECT_ROOT / "saflow"
CODE_DIR = PROJECT_ROOT / "code"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Directories to clean
CLEAN_PATTERNS = [
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.egg-info",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".coverage",
    "coverage.xml",
    "build",
    "dist",
]


# ==============================================================================
# Testing
# ==============================================================================

@task
def test(c, verbose=False, coverage=True, markers=None):
    """Run tests with pytest.

    Args:
        verbose: Verbose output (-v)
        coverage: Generate coverage report (default: True)
        markers: Only run tests matching given marker expression

    Examples:
        invoke test
        invoke test --verbose
        invoke test --markers="not slow"
        invoke test --no-coverage
    """
    cmd = ["pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=saflow", "--cov=code", "--cov-report=term-missing"])

    if markers:
        cmd.append(f"-m '{markers}'")

    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), pty=True)


@task
def test_fast(c):
    """Run only fast tests (exclude slow markers)."""
    test(c, markers="not slow")


@task
def test_unit(c):
    """Run only unit tests."""
    test(c, markers="unit")


@task
def test_integration(c):
    """Run only integration tests."""
    test(c, markers="integration")


# ==============================================================================
# Code Quality
# ==============================================================================

@task
def lint(c, fix=False):
    """Run ruff linting.

    Args:
        fix: Automatically fix issues where possible

    Examples:
        invoke lint
        invoke lint --fix
    """
    cmd = ["ruff", "check", str(SAFLOW_PKG), str(CODE_DIR)]

    if fix:
        cmd.append("--fix")

    print(f"Running: {' '.join(cmd)}")
    result = c.run(" ".join(cmd), warn=True, pty=True)

    if result.exited != 0:
        print("\n⚠️  Linting issues found. Run 'invoke lint --fix' to auto-fix.")
        return result

    print("✓ No linting issues found!")


@task
def format(c, check=False):
    """Format code with ruff.

    Args:
        check: Only check formatting, don't modify files

    Examples:
        invoke format
        invoke format --check
    """
    cmd = ["ruff", "format", str(SAFLOW_PKG), str(CODE_DIR)]

    if check:
        cmd.append("--check")

    print(f"Running: {' '.join(cmd)}")
    result = c.run(" ".join(cmd), warn=True, pty=True)

    if result.exited != 0 and check:
        print("\n⚠️  Code formatting issues found. Run 'invoke format' to fix.")
        return result

    if not check:
        print("✓ Code formatted successfully!")


@task
def typecheck(c):
    """Run mypy type checking."""
    cmd = ["mypy", str(SAFLOW_PKG), str(CODE_DIR)]

    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), warn=True, pty=True)


@task
def check(c):
    """Run all code quality checks (lint, format check, type check).

    This is useful for pre-commit or CI checks.
    """
    print("=" * 80)
    print("Running all code quality checks...")
    print("=" * 80)

    print("\n1. Linting...")
    lint_result = lint(c)

    print("\n2. Format checking...")
    format_result = format(c, check=True)

    print("\n3. Type checking...")
    typecheck(c)

    print("\n" + "=" * 80)
    if lint_result and lint_result.exited != 0:
        print("❌ Linting failed")
    elif format_result and format_result.exited != 0:
        print("❌ Format checking failed")
    else:
        print("✓ All code quality checks passed!")
    print("=" * 80)


# ==============================================================================
# Cleaning
# ==============================================================================

@task
def clean(c, bytecode=True, cache=True, coverage=True, build=True, logs=False, reports=False):
    """Clean generated files.

    Args:
        bytecode: Remove Python bytecode files (default: True)
        cache: Remove tool cache directories (default: True)
        coverage: Remove coverage reports (default: True)
        build: Remove build artifacts (default: True)
        logs: Remove log files (default: False)
        reports: Remove report files (default: False)

    Examples:
        invoke clean
        invoke clean --logs --reports  # Clean everything including logs/reports
        invoke clean --no-bytecode     # Clean everything except bytecode
    """
    patterns = []

    if bytecode:
        patterns.extend(["**/__pycache__", "**/*.pyc", "**/*.pyo"])

    if cache:
        patterns.extend([".pytest_cache", ".mypy_cache", ".ruff_cache"])

    if coverage:
        patterns.extend(["htmlcov", ".coverage", "coverage.xml"])

    if build:
        patterns.extend(["**/*.egg-info", "build", "dist"])

    if logs:
        patterns.append("logs")

    if reports:
        patterns.append("reports")

    print("Cleaning generated files...")
    removed_count = 0

    for pattern in patterns:
        if "**" in pattern:
            # Recursive glob pattern
            for path in PROJECT_ROOT.rglob(pattern.replace("**/", "")):
                if path.exists():
                    if path.is_file():
                        path.unlink()
                        removed_count += 1
                    elif path.is_dir():
                        c.run(f"rm -rf {path}", hide=True)
                        removed_count += 1
        else:
            # Direct path
            path = PROJECT_ROOT / pattern
            if path.exists():
                if path.is_file():
                    path.unlink()
                    print(f"  Removed: {path.relative_to(PROJECT_ROOT)}")
                    removed_count += 1
                elif path.is_dir():
                    c.run(f"rm -rf {path}", hide=True)
                    print(f"  Removed: {path.relative_to(PROJECT_ROOT)}/")
                    removed_count += 1

    print(f"✓ Cleaned {removed_count} items")


@task
def clean_all(c):
    """Clean everything including logs and reports."""
    clean(c, logs=True, reports=True)


# ==============================================================================
# Environment Setup
# ==============================================================================

@task
def setup(c, mode="basic", python="python3.9", force=False):
    """Run the setup script to create development environment.

    Args:
        mode: Installation mode (basic, dev, hpc, all)
        python: Python executable to use
        force: Force reinstall if venv exists

    Examples:
        invoke setup
        invoke setup --mode=dev
        invoke setup --mode=all --python=python3.10
        invoke setup --force
    """
    cmd = ["./setup.sh"]

    if mode != "basic":
        cmd.append(f"--{mode}")

    if python != "python3.9":
        cmd.extend(["--python", python])

    if force:
        cmd.append("--force")

    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), pty=True)


# ==============================================================================
# Documentation
# ==============================================================================

@task
def docs(c, serve=False, port=8000):
    """Build documentation with Sphinx.

    Args:
        serve: Serve documentation locally after building
        port: Port for local server (default: 8000)

    Examples:
        invoke docs
        invoke docs --serve
        invoke docs --serve --port=8080
    """
    docs_build = DOCS_DIR / "_build"

    print("Building documentation...")
    c.run(f"sphinx-build -b html {DOCS_DIR} {docs_build}/html", pty=True)

    print(f"✓ Documentation built in {docs_build}/html")

    if serve:
        print(f"\nServing documentation at http://localhost:{port}")
        print("Press Ctrl+C to stop server")
        c.run(f"python -m http.server {port} -d {docs_build}/html", pty=True)


# ==============================================================================
# Utility Tasks
# ==============================================================================

@task
def info(c):
    """Display project information."""
    print("=" * 80)
    print("Saflow Project Information")
    print("=" * 80)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Saflow package:   {SAFLOW_PKG}")
    print(f"Code directory:   {CODE_DIR}")
    print(f"Tests directory:  {TESTS_DIR}")
    print(f"Docs directory:   {DOCS_DIR}")
    print()

    # Check if config exists
    config_file = PROJECT_ROOT / "config.yaml"
    config_template = PROJECT_ROOT / "config.yaml.template"

    print("Configuration:")
    print(f"  config.yaml:          {'✓ exists' if config_file.exists() else '✗ missing'}")
    print(f"  config.yaml.template: {'✓ exists' if config_template.exists() else '✗ missing'}")
    print()

    # Check if venv exists
    venv_dir = PROJECT_ROOT / "env"
    print(f"Virtual environment:  {'✓ exists' if venv_dir.exists() else '✗ missing'}")

    if venv_dir.exists():
        # Try to get Python version
        python_exe = venv_dir / "bin" / "python"
        if python_exe.exists():
            result = c.run(f"{python_exe} --version", hide=True)
            print(f"  Python version:     {result.stdout.strip()}")

    print("=" * 80)


@task
def validate_config(c):
    """Validate configuration file."""
    print("Validating configuration...")

    cmd = [
        "python", "-c",
        "'from code.utils.config import load_config; "
        "config = load_config(); "
        "print(\"✓ Configuration is valid\")'"
    ]

    c.run(" ".join(cmd), pty=True)


# ==============================================================================
# Development Workflow
# ==============================================================================

@task(pre=[format, lint, test])
def precommit(c):
    """Run pre-commit checks (format, lint, test).

    This is useful to run before committing changes.
    """
    print("\n" + "=" * 80)
    print("✓ All pre-commit checks passed!")
    print("=" * 80)


@task(pre=[clean])
def rebuild(c, mode="dev"):
    """Clean and rebuild environment.

    Args:
        mode: Installation mode (basic, dev, hpc, all)
    """
    setup(c, mode=mode, force=True)


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_python_executable():
    """Get the Python executable, preferring venv if it exists."""
    venv_python = PROJECT_ROOT / "env" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python"


# ==============================================================================
# Pipeline Tasks
# ==============================================================================

@task
def validate_inputs(c, data_root=None, verbose=False):
    """Validate that raw input data is present and complete.

    Checks for:
    - Raw MEG data directory exists
    - Behavioral logfiles directory exists
    - Required .ds files present
    - Behavioral logfiles present for expected subjects

    Args:
        data_root: Override data root from config
        verbose: Show detailed file listing

    Examples:
        invoke validate-inputs
        invoke validate-inputs --verbose
        invoke validate-inputs --data-root=/path/to/data
    """
    print("=" * 80)
    print("Validating Input Data")
    print("=" * 80)

    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command - run as module with PYTHONPATH set
    cmd = [python_exe, "-m", "code.utils.validation", "--check-inputs"]

    if data_root:
        cmd.extend(["--data-root", str(data_root)])

    if verbose:
        cmd.append("--verbose")

    # Set PYTHONPATH to project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


@task
def bids(
    c,
    input_dir=None,
    output_dir=None,
    subjects=None,
    log_level="INFO",
    dry_run=False,
):
    """Run BIDS conversion (Stage 0).

    Converts raw MEG data (.ds files) to BIDS format and enriches
    gradCPT events with behavioral data (VTC, RT, performance, IN/OUT zones).

    Args:
        input_dir: Override raw data directory from config
        output_dir: Override BIDS output directory from config
        subjects: Process specific subjects only (space-separated)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        dry_run: Validate inputs without processing

    Examples:
        invoke bids
        invoke bids --subjects "04 05 06"
        invoke bids --log-level=DEBUG
        invoke bids --input-dir=/path/to/raw --output-dir=/path/to/bids
        invoke bids --dry-run
    """
    print("=" * 80)
    print("BIDS Conversion - Stage 0")
    print("=" * 80)

    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command - run as module with PYTHONPATH set
    cmd = [python_exe, "-m", "code.bids.generate_bids"]

    if input_dir:
        cmd.extend(["--input", str(input_dir)])

    if output_dir:
        cmd.extend(["--output", str(output_dir)])

    if subjects:
        cmd.extend(["--subjects"] + subjects.split())

    cmd.extend(["--log-level", log_level])

    if dry_run:
        cmd.append("--dry-run")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Set PYTHONPATH to include project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


@task
def preprocess(
    c,
    subject=None,
    runs=None,
    bids_root=None,
    log_level="INFO",
    skip_existing=True,
    slurm=False,
    dry_run=False,
):
    """Run MEG preprocessing (Stage 1).

    Preprocesses BIDS MEG data with filtering, AutoReject, and ICA artifact removal.
    Saves preprocessed continuous data, epochs, logs, and HTML reports.

    Args:
        subject: Subject ID to process (default: all subjects from config)
        runs: Run numbers to process (space-separated, default: all task runs from config)
        bids_root: Override BIDS root directory from config
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        skip_existing: Skip runs already preprocessed (default: True)
        slurm: Submit jobs to SLURM instead of running locally (default: False)
        dry_run: Generate SLURM scripts without submitting (requires --slurm)

    Examples:
        # Local execution (single subject)
        invoke preprocess --subject=04
        invoke preprocess --subject=04 --runs="02 03"
        invoke preprocess --subject=04 --log-level=DEBUG

        # SLURM execution (distributed, one job per run)
        invoke preprocess --subject=04 --slurm
        invoke preprocess --slurm  # Process all subjects
        invoke preprocess --slurm --dry-run  # Generate scripts without submitting
    """
    print("=" * 80)
    print("MEG Preprocessing - Stage 1")
    print("=" * 80)

    if slurm:
        # SLURM execution: submit jobs
        _preprocess_slurm(
            c,
            subject=subject,
            runs=runs,
            bids_root=bids_root,
            log_level=log_level,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
    else:
        # Local execution: run directly
        if not subject:
            print("ERROR: --subject is required for local execution")
            print("Use --slurm to process all subjects in parallel on HPC")
            return

        _preprocess_local(
            c,
            subject=subject,
            runs=runs,
            bids_root=bids_root,
            log_level=log_level,
            skip_existing=skip_existing,
        )


def _preprocess_local(
    c,
    subject: str,
    runs=None,
    bids_root=None,
    log_level="INFO",
    skip_existing=True,
):
    """Run preprocessing locally (helper function)."""
    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command - run as module with PYTHONPATH set
    cmd = [python_exe, "-m", "code.preprocessing.run_preprocessing"]

    cmd.extend(["--subject", subject])

    if runs:
        cmd.extend(["--runs"] + runs.split())

    if bids_root:
        cmd.extend(["--bids-root", str(bids_root)])

    cmd.extend(["--log-level", log_level])

    if skip_existing:
        cmd.append("--skip-existing")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Set PYTHONPATH to include project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


def _preprocess_slurm(
    c,
    subject=None,
    runs=None,
    bids_root=None,
    log_level="INFO",
    skip_existing=True,
    dry_run=False,
):
    """Submit preprocessing jobs to SLURM (helper function)."""
    from datetime import datetime

    # Import after checking --slurm to avoid import errors on local machines
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, submit_slurm_job, save_job_manifest

    print("\n[SLURM Mode] Submitting preprocessing jobs to cluster\n")

    # Load config
    config = load_config()

    # Determine subjects to process
    if subject:
        subjects = [subject]
        print(f"Processing subject: {subject}")
    else:
        subjects = config["bids"]["subjects"]
        print(f"Processing all subjects: {len(subjects)} subjects")

    # Determine runs to process
    if runs:
        run_list = runs.split()
        print(f"Processing runs: {', '.join(run_list)}")
    else:
        run_list = config["bids"]["task_runs"]
        print(f"Processing all task runs: {', '.join(run_list)}")

    # Determine BIDS root
    if not bids_root:
        data_root = Path(config["paths"]["data_root"])
        bids_root = data_root / config["paths"]["derivatives"] / "bids"
    else:
        bids_root = Path(bids_root)

    print(f"BIDS root: {bids_root}")
    print(f"Total jobs to submit: {len(subjects)} subjects × {len(run_list)} runs = {len(subjects) * len(run_list)} jobs")
    print()

    # Get SLURM configuration
    slurm_config = config["computing"]["slurm"]
    preproc_resources = slurm_config["preprocessing"]

    # Create output directories
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "preprocessing"
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "preprocessing"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    # Get venv path
    venv_path = PROJECT_ROOT / config["paths"]["venv"]

    # Generate and submit jobs
    job_ids = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for subj in subjects:
        for run in run_list:
            # Job name
            job_name = f"preproc_sub-{subj}_run-{run}"

            # Template context
            context = {
                "job_name": job_name,
                "account": slurm_config["account"],
                "partition": slurm_config.get("partition", "standard"),
                "time": preproc_resources["time"],
                "cpus": preproc_resources["cpus"],
                "mem": preproc_resources["mem"],
                "email": slurm_config.get("email"),
                "modules": slurm_config.get("modules", []),
                "venv_path": str(venv_path),
                "project_root": str(PROJECT_ROOT),
                "log_dir": str(log_dir),
                "timestamp": timestamp,
                # Preprocessing-specific
                "subject": subj,
                "run": run,
                "bids_root": str(bids_root),
                "log_level": log_level,
                "skip_existing": skip_existing,
            }

            # Render SLURM script
            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script(
                "preprocessing.sh.j2",
                context,
                output_path=script_path,
            )

            # Submit job
            job_id = submit_slurm_job(
                script_path,
                dry_run=dry_run,
            )

            if job_id:
                job_ids.append(job_id)

    # Save job manifest
    if job_ids:
        manifest_path = log_dir / f"preprocessing_manifest_{timestamp}.json"
        save_job_manifest(
            job_ids,
            manifest_path,
            metadata={
                "stage": "preprocessing",
                "timestamp": timestamp,
                "subjects": subjects,
                "runs": run_list,
                "num_subjects": len(subjects),
                "num_runs": len(run_list),
                "num_jobs": len(job_ids),
            },
        )

        print(f"\n{'=' * 80}")
        print(f"✓ Submitted {len(job_ids)} preprocessing jobs")
        print(f"  Manifest: {manifest_path}")
        print(f"  Scripts:  {script_dir}")
        print(f"  Logs:     {log_dir}")
        print(f"{'=' * 80}")
    else:
        print("\n✗ No jobs were submitted")

    if dry_run:
        print("\n[DRY RUN] Scripts generated but not submitted")
# ==============================================================================
# Source Reconstruction
# ==============================================================================

@task
def source_recon(
    c,
    subject=None,
    runs=None,
    bids_root=None,
    log_level="INFO",
    skip_existing=True,
    slurm=False,
    dry_run=False,
):
    """Run source reconstruction (Stage 2).

    Computes inverse solution and morphs source estimates to fsaverage template.
    Requires preprocessed data from Stage 1.

    Args:
        subject: Subject ID to process (default: all subjects from config)
        runs: Run numbers to process (space-separated, default: all task runs from config)
        bids_root: Override BIDS root directory from config
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        skip_existing: Skip runs already processed (default: True)
        slurm: Submit jobs to SLURM instead of running locally (default: False)
        dry_run: Generate SLURM scripts without submitting (requires --slurm)

    Examples:
        # Local execution (single subject)
        invoke source-recon --subject=04 --runs="02 03"
        invoke source-recon --subject=04 --log-level=DEBUG

        # SLURM execution (distributed, one job per run)
        invoke source-recon --subject=04 --slurm
        invoke source-recon --slurm  # Process all subjects
        invoke source-recon --slurm --dry-run  # Generate scripts without submitting
    """
    print("=" * 80)
    print("Source Reconstruction - Stage 2")
    print("=" * 80)

    if slurm:
        # SLURM execution: submit jobs
        _source_recon_slurm(
            c,
            subject=subject,
            runs=runs,
            bids_root=bids_root,
            log_level=log_level,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
    else:
        # Local execution: run directly
        if not subject:
            print("ERROR: --subject is required for local execution")
            print("Use --slurm to process all subjects in parallel on HPC")
            return

        _source_recon_local(
            c,
            subject=subject,
            runs=runs,
            bids_root=bids_root,
            log_level=log_level,
            skip_existing=skip_existing,
        )


def _source_recon_local(
    c,
    subject: str,
    runs=None,
    bids_root=None,
    log_level="INFO",
    skip_existing=True,
):
    """Run source reconstruction locally (helper function)."""
    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command - run as module with PYTHONPATH set
    cmd = [python_exe, "-m", "code.source_reconstruction.run_inverse_solution"]

    cmd.extend(["--subject", subject])

    if runs:
        cmd.extend(["--runs", runs])

    if bids_root:
        cmd.extend(["--bids-root", str(bids_root)])

    cmd.extend(["--log-level", log_level])

    if skip_existing:
        cmd.append("--skip-existing")
    else:
        cmd.append("--no-skip-existing")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Set PYTHONPATH to include project root
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    # Run command
    c.run(" ".join(cmd), env=env, pty=True)


def _source_recon_slurm(
    c,
    subject=None,
    runs=None,
    bids_root=None,
    log_level="INFO",
    skip_existing=True,
    dry_run=False,
):
    """Submit source reconstruction jobs to SLURM (helper function)."""
    from datetime import datetime

    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    # Load configuration
    config = load_config()

    # Determine subjects and runs
    if subject:
        subjects = [subject]
    else:
        subjects = config["bids"]["subjects"]

    if runs:
        run_list = runs.split()
    else:
        run_list = config["bids"]["task_runs"]

    # Get SLURM settings
    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        print("Set computing.slurm.enabled: true")
        return

    src_recon_resources = slurm_config.get("source_reconstruction", {})
    if not src_recon_resources:
        print("ERROR: No source_reconstruction resources in config.yaml")
        print("Add computing.slurm.source_reconstruction section")
        return

    # Get paths
    data_root = Path(config["paths"]["data_root"])
    if bids_root:
        bids_root = Path(bids_root)
    else:
        bids_root = data_root / config["paths"]["bids"]

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    # Prepare output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "source_reconstruction"
    script_dir.mkdir(parents=True, exist_ok=True)

    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "source_reconstruction"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"SLURM Job Submission - Source Reconstruction")
    print(f"{'=' * 80}")
    print(f"  Subjects:   {len(subjects)} ({', '.join(subjects[:3])}{', ...' if len(subjects) > 3 else ''})")
    print(f"  Runs:       {len(run_list)} ({', '.join(run_list)})")
    print(f"  Total jobs: {len(subjects) * len(run_list)}")
    print(f"  Resources:  {src_recon_resources['cpus']} CPUs, {src_recon_resources['mem']} RAM, {src_recon_resources['time']} time")
    print(f"  Dry run:    {dry_run}")
    print(f"{'=' * 80}\n")

    if dry_run:
        print("[DRY RUN MODE] Generating scripts without submitting...\n")

    # Submit jobs (one per subject-run combination)
    job_ids = []
    for subj in subjects:
        for run in run_list:
            job_name = f"srcrecon_sub-{subj}_run-{run}"

            # Prepare template context
            context = {
                "job_name": job_name,
                "account": slurm_config["account"],
                "partition": slurm_config.get("partition", "standard"),
                "email": slurm_config.get("email", ""),
                "cpus": src_recon_resources["cpus"],
                "mem": src_recon_resources["mem"],
                "time": src_recon_resources["time"],
                "log_dir": str(log_dir),
                "venv_path": str(venv_path),
                "project_root": str(PROJECT_ROOT),
                "subject": subj,
                "run": run,
                "bids_root": str(bids_root),
                "log_level": log_level,
                "skip_existing": skip_existing,
            }

            # Generate script
            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script(
                "source_reconstruction.sh.j2",
                context,
                output_path=script_path,
            )

            print(f"Generated script: {script_path.name}")

            # Submit job (unless dry run)
            if not dry_run:
                try:
                    job_id = submit_slurm_job(
                        script_path,
                        job_name=job_name,
                        dry_run=False,
                    )
                    job_ids.append(job_id)
                    print(f"  → Submitted job {job_id}")
                except Exception as e:
                    print(f"  ✗ Failed to submit: {e}")

    # Save manifest
    if job_ids:
        manifest_path = log_dir / f"source_reconstruction_manifest_{timestamp}.json"
        save_job_manifest(
            job_ids,
            manifest_path,
            metadata={
                "stage": "source_reconstruction",
                "timestamp": timestamp,
                "subjects": subjects,
                "runs": run_list,
                "num_subjects": len(subjects),
                "num_runs": len(run_list),
                "num_jobs": len(job_ids),
            },
        )

        print(f"\n{'=' * 80}")
        print(f"✓ Submitted {len(job_ids)} source reconstruction jobs")
        print(f"  Manifest: {manifest_path}")
        print(f"  Scripts:  {script_dir}")
        print(f"  Logs:     {log_dir}")
        print(f"{'=' * 80}")
    else:
        print("\n✗ No jobs were submitted")

    if dry_run:
        print("\n[DRY RUN] Scripts generated but not submitted")


# ==============================================================================
# Statistics and Classification Tasks
# ==============================================================================

@task
def statistics(
    c,
    feature_type,
    space="sensor",
    test="paired_ttest",
    corrections="fdr bonferroni",
    alpha=0.05,
    n_permutations=10000,
    visualize=False,
    slurm=False,
    dry_run=False,
):
    """Run group-level statistical analysis (IN vs OUT).

    Computes group-level statistics comparing IN and OUT attentional states,
    applies multiple comparison corrections, and computes effect sizes.

    Args:
        feature_type: Feature to analyze (e.g., 'fooof_exponent', 'psd_alpha', 'lzc')
        space: Analysis space ('sensor', 'source', 'atlas')
        test: Statistical test ('paired_ttest', 'independent_ttest', 'permutation')
        corrections: Correction methods (space-separated, e.g., 'fdr bonferroni')
        alpha: Significance threshold (default: 0.05)
        n_permutations: Number of permutations for permutation test
        visualize: Generate visualization plots
        slurm: Submit job to SLURM instead of running locally
        dry_run: Generate SLURM script without submitting (requires --slurm)

    Examples:
        # Local execution
        invoke statistics --feature-type=fooof_exponent
        invoke statistics --feature-type=psd_alpha --space=sensor --test=paired_ttest
        invoke statistics --feature-type=lzc --corrections="fdr bonferroni tmax"

        # With visualization
        invoke statistics --feature-type=fooof_exponent --visualize

        # SLURM execution
        invoke statistics --feature-type=psd_alpha --slurm
        invoke statistics --feature-type=fooof_exponent --slurm --dry-run
    """
    print("=" * 80)
    print("Group-Level Statistical Analysis")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented for statistics")
        print("Run locally for now")
        return

    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command
    cmd = [python_exe, "-m", "code.statistics.run_group_statistics"]
    cmd.extend(["--feature-type", feature_type])
    cmd.extend(["--space", space])
    cmd.extend(["--test", test])
    cmd.extend(["--correction"] + corrections.split())
    cmd.extend(["--alpha", str(alpha)])
    cmd.extend(["--n-permutations", str(n_permutations)])

    if visualize:
        cmd.append("--visualize")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Set PYTHONPATH to include project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


@task
def classify(
    c,
    features,
    clf="lda",
    cv="logo",
    space="sensor",
    n_permutations=1000,
    no_balance=False,
    visualize=False,
    slurm=False,
    dry_run=False,
):
    """Run classification analysis (IN vs OUT).

    Trains classifiers to decode IN vs OUT attentional states from neural
    features using cross-validation and permutation testing.

    Args:
        features: Feature type(s) (space-separated for multivariate, e.g., 'fooof_exponent psd_alpha')
        clf: Classifier ('lda', 'svm', 'rf', 'logistic')
        cv: Cross-validation strategy ('logo', 'stratified', 'group')
        space: Analysis space ('sensor', 'source', 'atlas')
        n_permutations: Number of permutations for significance testing
        no_balance: Disable class balancing
        visualize: Generate visualization plots
        slurm: Submit job to SLURM instead of running locally
        dry_run: Generate SLURM script without submitting (requires --slurm)

    Examples:
        # Single feature classification
        invoke classify --features=fooof_exponent
        invoke classify --features=psd_alpha --clf=svm --cv=logo

        # Multivariate classification
        invoke classify --features="fooof_exponent psd_alpha psd_theta"
        invoke classify --features="psd_alpha psd_beta" --clf=rf

        # With visualization
        invoke classify --features=fooof_exponent --visualize

        # SLURM execution
        invoke classify --features=psd_alpha --clf=svm --slurm
        invoke classify --features=fooof_exponent --slurm --dry-run
    """
    print("=" * 80)
    print("Classification Analysis")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented for classification")
        print("Run locally for now")
        return

    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command
    cmd = [python_exe, "-m", "code.classification.run_classification"]
    cmd.extend(["--features"] + features.split())
    cmd.extend(["--clf", clf])
    cmd.extend(["--cv", cv])
    cmd.extend(["--space", space])
    cmd.extend(["--n-permutations", str(n_permutations)])

    if no_balance:
        cmd.append("--no-balance")

    if visualize:
        cmd.append("--visualize")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Set PYTHONPATH to include project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


# ==============================================================================
# Visualization Tasks
# ==============================================================================

@task
def plot_behavior(
    c,
    subject="07",
    run="4",
    inout_bounds="25 75",
    output=None,
    verbose=False,
):
    """Generate behavioral analysis figure.

    Creates a publication-quality figure with:
    - Panel A: VTC time course for example subject
    - Panel B: Lapse rate comparison (IN vs OUT)
    - Panel C: Omission error rate comparison (IN vs OUT)
    - Panel D: Pre-event RT comparison by condition

    Args:
        subject: Example subject ID for VTC plot (default: 07)
        run: Example run number for VTC plot (default: 4)
        inout_bounds: Percentile bounds for IN/OUT classification (default: "25 75")
        output: Custom output path (default: reports/figures/behavior/)
        verbose: Enable verbose logging

    Examples:
        invoke plot-behavior
        invoke plot-behavior --subject=04 --run=3
        invoke plot-behavior --inout-bounds="50 50"
        invoke plot-behavior --verbose
        invoke plot-behavior --output=my_figure.png
    """
    print("=" * 80)
    print("Behavioral Analysis Visualization")
    print("=" * 80)

    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command
    cmd = [python_exe, "-m", "code.visualization.plot_behavior"]

    cmd.extend(["--subject", subject])
    cmd.extend(["--run", run])

    # Parse inout_bounds
    bounds = inout_bounds.split()
    if len(bounds) == 2:
        cmd.extend(["--inout-bounds", bounds[0], bounds[1]])

    if output:
        cmd.extend(["--output", output])

    if verbose:
        cmd.append("--verbose")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Set PYTHONPATH to include project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


# ==============================================================================
# List Tasks
# ==============================================================================

@task(default=True)
def help(c):
    """Show available tasks (default task)."""
    c.run("invoke --list")
