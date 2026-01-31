"""Invoke tasks for saflow MEG pipeline.

This module provides organized tasks using invoke's Collection feature.
Tasks are grouped into namespaces for clean organization.

Usage:
    invoke --list                          # List all tasks
    invoke dev.check.dataset               # Check dataset completeness
    invoke dev.check.qc --subject=04       # Run data quality checks
    invoke dev.check.code                  # Run linting/formatting checks
    invoke pipeline.preprocess --subject=04
    invoke features.fooof --subject=04
    invoke analysis.statistics --feature-type=fooof_exponent

Namespaces:
    dev.check   - Data validation (dataset, qc, code)
    dev         - Development tasks (test, clean)
    env         - Environment tasks (setup, info)
    pipeline    - Data processing pipeline (bids, preprocess, source-recon)
    features    - Feature extraction (fooof, psd, complexity)
    analysis    - Statistical analysis (statistics, classify)
    viz         - Visualization (behavior)
"""

import os
from pathlib import Path

from invoke import Collection, task

# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent
SAFLOW_PKG = PROJECT_ROOT / "saflow"
CODE_DIR = PROJECT_ROOT / "code"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"


def get_python_executable():
    """Get the Python executable, preferring venv if it exists."""
    venv_python = PROJECT_ROOT / "env" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python"


def get_env_with_pythonpath():
    """Get environment dict with PYTHONPATH set to project root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


# ==============================================================================
# dev.check.* Tasks - Validation & Quality Checks
# ==============================================================================

@task
def check_dataset(c, verbose=False):
    """Check dataset completeness - which files exist for each subject.

    Scans all data directories (sourcedata, BIDS, derivatives, processed)
    and reports which files are present/missing for each subject.

    Outputs a summary table showing:
    - MEG raw files (sourcedata)
    - Behavioral logfiles (sourcedata)
    - BIDS converted files
    - Preprocessed files
    - Feature files

    Examples:
        invoke dev.check.dataset
        invoke dev.check.dataset --verbose
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.qc.check_dataset"]

    if verbose:
        cmd.append("--verbose")

    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def check_qc(c, subject=None, runs=None, output_dir="reports/qc", verbose=False):
    """Run data quality checks on BIDS MEG data.

    By default, checks all subjects. Use --subject to check a single subject.

    Reports on:
    - ISI (Inter-Stimulus Interval) consistency and outliers
    - Response rates and reaction times
    - Channel quality (flat, noisy, saturated channels)
    - Event/trigger counts and timing
    - Data integrity (sampling rate, duration)
    - Head motion (if HPI available)

    Examples:
        invoke dev.check.qc                    # All subjects (default)
        invoke dev.check.qc --subject=04       # Single subject
        invoke dev.check.qc --subject=04 --runs="02 03 04"
        invoke dev.check.qc --verbose
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.qc.check_qc"]

    if subject:
        cmd.extend(["--subject", subject])
    else:
        cmd.append("--all")

    if runs:
        cmd.extend(["--runs"] + runs.split())

    cmd.extend(["--output-dir", output_dir])

    if verbose:
        cmd.append("--verbose")

    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def check_code(c, fix=False):
    """Run code quality checks (lint, format, typecheck).

    Checks:
    - Ruff linting (style, errors, complexity)
    - Ruff formatting (code style consistency)
    - Mypy type checking (type annotations)

    Examples:
        invoke dev.check.code
        invoke dev.check.code --fix
    """
    print("=" * 80)
    print("Code Quality Checks")
    print("=" * 80)

    all_passed = True

    # Linting
    print("\n[1/3] Linting with ruff...")
    lint_cmd = ["ruff", "check", str(SAFLOW_PKG), str(CODE_DIR)]
    if fix:
        lint_cmd.append("--fix")
    result = c.run(" ".join(lint_cmd), warn=True, pty=True)
    if result.exited != 0:
        all_passed = False
        if not fix:
            print("  → Run with --fix to auto-fix issues")

    # Formatting
    print("\n[2/3] Checking format with ruff...")
    format_cmd = ["ruff", "format", "--check", str(SAFLOW_PKG), str(CODE_DIR)]
    if fix:
        format_cmd.remove("--check")
    result = c.run(" ".join(format_cmd), warn=True, pty=True)
    if result.exited != 0:
        all_passed = False
        if not fix:
            print("  → Run with --fix to auto-format")

    # Type checking
    print("\n[3/3] Type checking with mypy...")
    type_cmd = ["mypy", str(SAFLOW_PKG), str(CODE_DIR)]
    result = c.run(" ".join(type_cmd), warn=True, pty=True)
    if result.exited != 0:
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All code quality checks passed!")
    else:
        print("✗ Some checks failed")
    print("=" * 80)


# ==============================================================================
# dev.* Tasks - Development & Testing
# ==============================================================================

@task
def test(c, verbose=False, coverage=True, markers=None):
    """Run tests with pytest.

    Examples:
        invoke dev.test
        invoke dev.test --verbose
        invoke dev.test --markers="not slow"
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
def clean(c, bytecode=True, cache=True, coverage=True, build=True, logs=False):
    """Clean generated files.

    Examples:
        invoke dev.clean
        invoke dev.clean --logs
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

    print("Cleaning generated files...")
    removed_count = 0

    for pattern in patterns:
        if "**" in pattern:
            for path in PROJECT_ROOT.rglob(pattern.replace("**/", "")):
                if path.exists():
                    if path.is_file():
                        path.unlink()
                        removed_count += 1
                    elif path.is_dir():
                        c.run(f"rm -rf {path}", hide=True)
                        removed_count += 1
        else:
            path = PROJECT_ROOT / pattern
            if path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    c.run(f"rm -rf {path}", hide=True)
                removed_count += 1

    print(f"✓ Cleaned {removed_count} items")


@task(pre=[check_code, test])
def precommit(c):
    """Run pre-commit checks (code checks + tests)."""
    print("\n" + "=" * 80)
    print("✓ All pre-commit checks passed!")
    print("=" * 80)


# ==============================================================================
# env.* Tasks - Environment Management
# ==============================================================================

@task
def setup(c, mode="basic", python="python3.9", force=False):
    """Run the setup script to create development environment.

    Examples:
        invoke env.setup
        invoke env.setup --mode=dev
        invoke env.setup --mode=all --force
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


@task
def info(c):
    """Display project information."""
    print("=" * 80)
    print("Saflow Project Information")
    print("=" * 80)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Saflow package:   {SAFLOW_PKG}")
    print(f"Code directory:   {CODE_DIR}")
    print()

    config_file = PROJECT_ROOT / "config.yaml"
    print("Configuration:")
    print(f"  config.yaml: {'✓ exists' if config_file.exists() else '✗ missing'}")

    venv_dir = PROJECT_ROOT / "env"
    print(f"\nVirtual environment: {'✓ exists' if venv_dir.exists() else '✗ missing'}")

    if venv_dir.exists():
        python_exe = venv_dir / "bin" / "python"
        if python_exe.exists():
            result = c.run(f"{python_exe} --version", hide=True)
            print(f"  Python version: {result.stdout.strip()}")
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


@task(pre=[clean])
def rebuild(c, mode="dev"):
    """Clean and rebuild environment."""
    setup(c, mode=mode, force=True)


# ==============================================================================
# pipeline.* Tasks - Data Processing Pipeline
# ==============================================================================

@task
def validate_inputs(c, data_root=None, verbose=False):
    """Validate that raw input data is present and complete.

    Examples:
        invoke pipeline.validate-inputs
        invoke pipeline.validate-inputs --verbose
    """
    print("=" * 80)
    print("Validating Input Data")
    print("=" * 80)

    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.utils.validation", "--check-inputs"]

    if data_root:
        cmd.extend(["--data-root", str(data_root)])
    if verbose:
        cmd.append("--verbose")

    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def bids(c, input_dir=None, output_dir=None, subjects=None, log_level="INFO", dry_run=False):
    """Run BIDS conversion (Stage 0).

    Examples:
        invoke pipeline.bids
        invoke pipeline.bids --subjects "04 05 06"
        invoke pipeline.bids --dry-run
    """
    print("=" * 80)
    print("BIDS Conversion - Stage 0")
    print("=" * 80)

    python_exe = get_python_executable()
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
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def preprocess(c, subject=None, runs=None, bids_root=None, log_level="INFO",
               skip_existing=True, crop=None, with_autoreject=False, slurm=False, dry_run=False):
    """Run MEG preprocessing (Stage 1).

    Examples:
        invoke pipeline.preprocess --subject=04
        invoke pipeline.preprocess --subject=04 --runs="02 03"
        invoke pipeline.preprocess --subject=04 --crop=50  # Quick test with 50s
        invoke pipeline.preprocess --subject=04 --with-autoreject  # Enable 2nd AR pass
        invoke pipeline.preprocess --slurm
    """
    print("=" * 80)
    print("MEG Preprocessing - Stage 1")
    if crop:
        print(f"[TEST MODE] Cropping to first {crop} seconds")
    if with_autoreject:
        print("[WITH AUTOREJECT] Second AutoReject pass enabled")
    print("=" * 80)

    if slurm:
        _preprocess_slurm(c, subject, runs, bids_root, log_level, skip_existing, dry_run)
    else:
        if not subject:
            print("ERROR: --subject is required for local execution")
            print("Use --slurm to process all subjects in parallel on HPC")
            return
        _preprocess_local(c, subject, runs, bids_root, log_level, skip_existing, crop, with_autoreject)


@task
def source_recon(c, subject=None, runs=None, bids_root=None, log_level="INFO",
                 skip_existing=True, slurm=False, dry_run=False):
    """Run source reconstruction (Stage 2).

    Examples:
        invoke pipeline.source-recon --subject=04
        invoke pipeline.source-recon --slurm
    """
    print("=" * 80)
    print("Source Reconstruction - Stage 2")
    print("=" * 80)

    if slurm:
        _source_recon_slurm(c, subject, runs, bids_root, log_level, skip_existing, dry_run)
    else:
        if not subject:
            print("ERROR: --subject is required for local execution")
            print("Use --slurm to process all subjects in parallel on HPC")
            return
        _source_recon_local(c, subject, runs, bids_root, log_level, skip_existing)


# ==============================================================================
# features.* Tasks - Feature Extraction
# ==============================================================================

@task
def fooof(c, subject=None, runs=None, space="sensor", skip_existing=True, slurm=False, dry_run=False):
    """Extract FOOOF aperiodic parameters and corrected PSDs.

    Computes:
    - Aperiodic parameters (exponent, offset, knee)
    - Goodness of fit (r_squared, error)
    - Corrected PSDs (aperiodic component removed)

    FOOOF parameters (freq_range, aperiodic_mode) come from config.yaml.

    Examples:
        invoke features.fooof --subject=04
        invoke features.fooof --subject=04 --runs="02 03"
    """
    from code.utils.config import load_config

    print("=" * 80)
    print("Feature Extraction - FOOOF")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented")
        return

    if not subject:
        print("ERROR: --subject is required for local execution")
        return

    config = load_config()
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    python_exe = get_python_executable()

    for run in run_list:
        print(f"\n[Processing run {run}]")
        cmd = [python_exe, "-m", "code.features.compute_fooof"]
        cmd.extend(["--subject", subject])
        cmd.extend(["--run", run])
        cmd.extend(["--space", space])

        if skip_existing:
            cmd.append("--skip-existing")
        else:
            cmd.append("--no-skip-existing")

        print(f"Running: {' '.join(cmd)}\n")
        result = c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)
        if result.exited != 0:
            print(f"WARNING: Run {run} failed")

    print(f"\n✓ FOOOF extraction complete for sub-{subject}")


@task
def psd(c, subject=None, runs=None, space="sensor", skip_existing=True, slurm=False, dry_run=False):
    """Extract power spectral density features.

    Computes:
    - Welch PSD estimates per trial
    - Saves with IN/OUT classification metadata

    Examples:
        invoke features.psd --subject=04
        invoke features.psd --subject=04 --runs="02 03"
        invoke features.psd --subject=04 --space=sensor
    """
    from code.utils.config import load_config

    print("=" * 80)
    print("Feature Extraction - PSD")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented")
        return

    if not subject:
        print("ERROR: --subject is required for local execution")
        return

    config = load_config()
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    python_exe = get_python_executable()

    for run in run_list:
        print(f"\n[Processing run {run}]")
        cmd = [python_exe, "-m", "code.features.compute_welch_psd"]
        cmd.extend(["--subject", subject])
        cmd.extend(["--run", run])
        cmd.extend(["--space", space])

        if skip_existing:
            cmd.append("--skip-existing")
        else:
            cmd.append("--no-skip-existing")

        print(f"Running: {' '.join(cmd)}\n")
        result = c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)
        if result.exited != 0:
            print(f"WARNING: Run {run} failed")

    print(f"\n✓ PSD extraction complete for sub-{subject}")


@task
def complexity(c, subject=None, runs=None, complexity_type="lzc entropy fractal",
               overwrite=False, slurm=False, dry_run=False):
    """Extract complexity and entropy measures.

    Computes:
    - Lempel-Ziv Complexity (LZC)
    - Entropy measures (permutation, spectral, sample, approximate, SVD)
    - Fractal dimensions (Higuchi, Petrosian, Katz, DFA)

    Examples:
        invoke features.complexity --subject=04
        invoke features.complexity --subject=04 --runs="02 03"
        invoke features.complexity --complexity-type="lzc entropy"
    """
    from code.utils.config import load_config

    print("=" * 80)
    print("Feature Extraction - Complexity")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented")
        return

    if not subject:
        print("ERROR: --subject is required for local execution")
        return

    config = load_config()
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    python_exe = get_python_executable()

    for run in run_list:
        print(f"\n[Processing run {run}]")
        cmd = [python_exe, "-m", "code.features.compute_complexity"]
        cmd.extend(["--subject", subject])
        cmd.extend(["--run", run])
        cmd.extend(["--complexity-type"] + complexity_type.split())

        if overwrite:
            cmd.append("--overwrite")

        print(f"Running: {' '.join(cmd)}\n")
        result = c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)
        if result.exited != 0:
            print(f"WARNING: Run {run} failed")

    print(f"\n✓ Complexity extraction complete for sub-{subject}")


@task
def extract_all(c, subject=None, overwrite=False, slurm=False, dry_run=False):
    """Extract all feature types (PSD, FOOOF, complexity).

    Order: PSD -> FOOOF -> Complexity (FOOOF depends on PSD)

    Examples:
        invoke features.all --subject=04
        invoke features.all --subject=04 --overwrite
    """
    print("=" * 80)
    print("Feature Extraction - All Features")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented")
        return

    if not subject:
        print("ERROR: --subject is required for local execution")
        return

    # Convert overwrite to skip_existing for fooof/psd tasks
    skip_existing = not overwrite

    print("\n[1/3] Extracting PSD features...")
    psd(c, subject=subject, skip_existing=skip_existing)

    print("\n[2/3] Extracting FOOOF features...")
    fooof(c, subject=subject, skip_existing=skip_existing)

    print("\n[3/3] Extracting Complexity features...")
    complexity(c, subject=subject, overwrite=overwrite)

    print("\n" + "=" * 80)
    print("✓ All feature extraction complete!")
    print("=" * 80)


# ==============================================================================
# analysis.* Tasks - Statistical Analysis
# ==============================================================================

@task
def statistics(c, feature_type, space="sensor", test="paired_ttest",
               corrections="fdr bonferroni", alpha=0.05, n_permutations=10000,
               visualize=False, slurm=False, dry_run=False):
    """Run group-level statistical analysis (IN vs OUT).

    Feature types:
    - FOOOF: fooof_exponent, fooof_offset, fooof_knee, fooof_r_squared
    - PSD: psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma
    - Complexity: lzc, perm_entropy, spectral_entropy, higuchi_fd

    Examples:
        invoke analysis.statistics --feature-type=fooof_exponent
        invoke analysis.statistics --feature-type=psd_alpha --visualize
    """
    print("=" * 80)
    print("Group-Level Statistical Analysis")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented")
        return

    python_exe = get_python_executable()
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
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def classify(c, features, clf="lda", cv="logo", space="sensor",
             n_permutations=1000, no_balance=False, visualize=False,
             slurm=False, dry_run=False):
    """Run classification analysis (IN vs OUT).

    Examples:
        invoke analysis.classify --features=fooof_exponent
        invoke analysis.classify --features="fooof_exponent psd_alpha"
        invoke analysis.classify --features=psd_alpha --clf=svm
    """
    print("=" * 80)
    print("Classification Analysis")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented")
        return

    python_exe = get_python_executable()
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
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


# ==============================================================================
# viz.* Tasks - Visualization
# ==============================================================================

@task
def behavior(c, subject="07", run="4", inout_bounds="25 75", output=None, verbose=False):
    """Generate behavioral analysis figure.

    Examples:
        invoke viz.behavior
        invoke viz.behavior --subject=04 --run=3
    """
    print("=" * 80)
    print("Behavioral Analysis Visualization")
    print("=" * 80)

    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.plot_behavior"]
    cmd.extend(["--subject", subject])
    cmd.extend(["--run", run])

    bounds = inout_bounds.split()
    if len(bounds) == 2:
        cmd.extend(["--inout-bounds", bounds[0], bounds[1]])

    if output:
        cmd.extend(["--output", output])
    if verbose:
        cmd.append("--verbose")

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


# ==============================================================================
# Helper Functions (Private)
# ==============================================================================

def _preprocess_local(c, subject, runs=None, bids_root=None, log_level="INFO", skip_existing=True, crop=None, with_autoreject=False):
    """Run preprocessing locally."""
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.preprocessing.run_preprocessing"]
    cmd.extend(["--subject", subject])

    if runs:
        cmd.extend(["--runs"] + runs.split())
    if bids_root:
        cmd.extend(["--bids-root", str(bids_root)])
    cmd.extend(["--log-level", log_level])
    if skip_existing:
        cmd.append("--skip-existing")
    if crop:
        cmd.extend(["--crop", str(crop)])
    if with_autoreject:
        cmd.append("--with-autoreject")

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


def _preprocess_slurm(c, subject=None, runs=None, bids_root=None,
                      log_level="INFO", skip_existing=True, dry_run=False):
    """Submit preprocessing jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, submit_slurm_job, save_job_manifest

    print("\n[SLURM Mode] Submitting preprocessing jobs to cluster\n")

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    if not bids_root:
        data_root = Path(config["paths"]["data_root"])
        bids_root = data_root / "bids"
    else:
        bids_root = Path(bids_root)

    slurm_config = config["computing"]["slurm"]
    preproc_resources = slurm_config["preprocessing"]

    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "preprocessing"
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "preprocessing"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    venv_path = PROJECT_ROOT / config["paths"]["venv"]
    job_ids = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Subjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total jobs: {len(subjects) * len(run_list)}")

    for subj in subjects:
        for run in run_list:
            job_name = f"preproc_sub-{subj}_run-{run}"
            context = {
                "job_name": job_name,
                "account": slurm_config["account"],
                "partition": slurm_config.get("partition", "standard"),
                "time": preproc_resources["time"],
                "cpus": preproc_resources["cpus"],
                "mem": preproc_resources["mem"],
                "venv_path": str(venv_path),
                "project_root": str(PROJECT_ROOT),
                "log_dir": str(log_dir),
                "subject": subj,
                "run": run,
                "bids_root": str(bids_root),
                "log_level": log_level,
                "skip_existing": skip_existing,
            }

            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("preprocessing.sh.j2", context, output_path=script_path)
            job_id = submit_slurm_job(script_path, dry_run=dry_run)
            if job_id:
                job_ids.append(job_id)

    if job_ids:
        manifest_path = log_dir / f"preprocessing_manifest_{timestamp}.json"
        save_job_manifest(job_ids, manifest_path, metadata={
            "stage": "preprocessing",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted {len(job_ids)} preprocessing jobs")


def _source_recon_local(c, subject, runs=None, bids_root=None, log_level="INFO", skip_existing=True):
    """Run source reconstruction locally."""
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.source_reconstruction.run_inverse_solution"]
    cmd.extend(["--subject", subject])

    if runs:
        cmd.extend(["--runs", runs])
    if bids_root:
        cmd.extend(["--bids-root", str(bids_root)])
    cmd.extend(["--log-level", log_level])
    cmd.append("--skip-existing" if skip_existing else "--no-skip-existing")

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), env=get_env_with_pythonpath(), pty=True)


def _source_recon_slurm(c, subject=None, runs=None, bids_root=None,
                        log_level="INFO", skip_existing=True, dry_run=False):
    """Submit source reconstruction jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    src_recon_resources = slurm_config.get("source_reconstruction", {})
    if not src_recon_resources:
        print("ERROR: No source_reconstruction resources in config.yaml")
        return

    data_root = Path(config["paths"]["data_root"])
    if bids_root:
        bids_root = Path(bids_root)
    else:
        bids_root = data_root / config["paths"]["bids"]

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "source_reconstruction"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "source_reconstruction"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSubjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total jobs: {len(subjects) * len(run_list)}")

    job_ids = []
    for subj in subjects:
        for run in run_list:
            job_name = f"srcrecon_sub-{subj}_run-{run}"
            context = {
                "job_name": job_name,
                "account": slurm_config["account"],
                "partition": slurm_config.get("partition", "standard"),
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

            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("source_reconstruction.sh.j2", context, output_path=script_path)

            if not dry_run:
                try:
                    job_id = submit_slurm_job(script_path, job_name=job_name, dry_run=False)
                    job_ids.append(job_id)
                except Exception as e:
                    print(f"  ✗ Failed to submit: {e}")

    if job_ids:
        manifest_path = log_dir / f"source_reconstruction_manifest_{timestamp}.json"
        save_job_manifest(job_ids, manifest_path, metadata={
            "stage": "source_reconstruction",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted {len(job_ids)} source reconstruction jobs")


# ==============================================================================
# Build Namespace Collections
# ==============================================================================

# dev.check.* subcollection
check = Collection("check")
check.add_task(check_dataset, name="dataset")
check.add_task(check_qc, name="qc")
check.add_task(check_code, name="code")

# Development tasks
dev = Collection("dev")
dev.add_task(test)
dev.add_task(test_fast, name="test-fast")
dev.add_task(clean)
dev.add_task(precommit)
dev.add_collection(check)  # Nested: dev.check.*

# Environment tasks
env = Collection("env")
env.add_task(setup)
env.add_task(info)
env.add_task(validate_config, name="validate-config")
env.add_task(rebuild)

# Pipeline tasks
pipeline = Collection("pipeline")
pipeline.add_task(validate_inputs, name="validate-inputs")
pipeline.add_task(bids)
pipeline.add_task(preprocess)
pipeline.add_task(source_recon, name="source-recon")

# Feature extraction tasks
features = Collection("features")
features.add_task(fooof)
features.add_task(psd)
features.add_task(complexity)
features.add_task(extract_all, name="all")

# Analysis tasks
analysis = Collection("analysis")
analysis.add_task(statistics)
analysis.add_task(classify)

# Visualization tasks
viz = Collection("viz")
viz.add_task(behavior)

# Build main namespace
namespace = Collection()
namespace.add_collection(dev)
namespace.add_collection(env)
namespace.add_collection(pipeline)
namespace.add_collection(features)
namespace.add_collection(analysis)
namespace.add_collection(viz)


# Default task
@task(default=True)
def help_task(c):
    """Show available tasks (default task)."""
    c.run("invoke --list")


namespace.add_task(help_task, name="help")
