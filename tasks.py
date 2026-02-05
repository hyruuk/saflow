"""Invoke tasks for saflow MEG pipeline.

This module provides organized tasks using invoke's Collection feature.
Tasks are grouped into namespaces for clean organization.

Usage:
    invoke --list                          # List all tasks
    invoke dev.check.dataset               # Check dataset completeness
    invoke dev.check.qc --subject=04       # Run data quality checks
    invoke dev.check.code                  # Run linting/formatting checks
    invoke pipeline.preprocess --subject=04
    invoke pipeline.features.fooof --subject=04
    invoke pipeline.features.all --slurm   # All features on HPC
    invoke analysis.statistics --feature-type=fooof_exponent

Namespaces:
    dev.check           - Data validation (dataset, qc, code)
    dev                 - Development tasks (test, clean)
    env                 - Environment tasks (setup, info)
    get                 - Data downloads (atlases)
    pipeline            - Data processing pipeline (bids, preprocess, source-recon, atlas)
    pipeline.features   - Feature extraction (psd, fooof, complexity, all)
    analysis            - Statistical analysis (statistics, classify)
    viz                 - Visualization (behavior)

SLURM Support:
    Most pipeline tasks support --slurm for HPC execution:
    - invoke pipeline.preprocess --slurm
    - invoke pipeline.source-recon --slurm
    - invoke pipeline.features.psd --slurm
    - invoke pipeline.features.all --slurm --space=aparc.a2009s
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
# get.* Tasks - Data Downloads
# ==============================================================================

@task
def get_atlases(c):
    """Download FreeSurfer atlas files (Schaefer, Destrieux).

    Downloads atlas annotation files required for source-level parcellation:
    - Destrieux (aparc.a2009s): 148 ROIs
    - Schaefer 100, 200, 400 parcels (7 Networks)

    Files are downloaded to the fsaverage/label directory specified in config.yaml.

    Examples:
        invoke get.atlases
    """
    print("=" * 80)
    print("Downloading FreeSurfer Atlases")
    print("=" * 80)

    script_path = PROJECT_ROOT / "scripts" / "download_atlases.sh"

    if not script_path.exists():
        print(f"ERROR: Download script not found: {script_path}")
        return

    c.run(str(script_path), pty=True)


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
               skip_existing=True, crop=None, skip_second_ar=False, slurm=False, dry_run=False):
    """Run MEG preprocessing (Stage 1).

    By default, runs second AutoReject pass (fit only) to detect bad epochs post-ICA.
    Use --skip-second-ar to skip this step.

    Examples:
        invoke pipeline.preprocess --subject=04
        invoke pipeline.preprocess --subject=04 --runs="02 03"
        invoke pipeline.preprocess --subject=04 --crop=50  # Quick test with 50s
        invoke pipeline.preprocess --subject=04 --skip-second-ar  # Skip 2nd AR pass
        invoke pipeline.preprocess --slurm
    """
    print("=" * 80)
    print("MEG Preprocessing - Stage 1")
    if crop:
        print(f"[TEST MODE] Cropping to first {crop} seconds")
    if skip_second_ar:
        print("[SKIP SECOND AR] Second AutoReject pass will be skipped")
    else:
        print("[DEFAULT] Second AutoReject pass enabled (fit only, for bad epoch detection)")
    print("=" * 80)

    if slurm:
        _preprocess_slurm(c, subject, runs, bids_root, log_level, skip_existing, dry_run)
    else:
        if not subject:
            print("ERROR: --subject is required for local execution")
            print("Use --slurm to process all subjects in parallel on HPC")
            return
        _preprocess_local(c, subject, runs, bids_root, log_level, skip_existing, crop, skip_second_ar)


@task
def source_recon(c, subject=None, runs=None, bids_root=None, log_level="INFO",
                 skip_existing=True, slurm=False, dry_run=False):
    """Run source reconstruction (Stage 2).

    By default processes all subjects from config. Use --subject for a single subject.

    Examples:
        invoke pipeline.source-recon                # All subjects (default)
        invoke pipeline.source-recon --subject=04  # Single subject
        invoke pipeline.source-recon --slurm       # All subjects on HPC
    """
    from code.utils.config import load_config

    print("=" * 80)
    print("Source Reconstruction - Stage 2")
    print("=" * 80)

    if slurm:
        _source_recon_slurm(c, subject, runs, bids_root, log_level, skip_existing, dry_run)
    else:
        config = load_config()
        subjects = [subject] if subject else config["bids"]["subjects"]
        print(f"Processing {len(subjects)} subject(s): {', '.join(subjects)}\n")

        for subj in subjects:
            print(f"\n{'='*40}")
            print(f"Subject: {subj}")
            print(f"{'='*40}")
            _source_recon_local(c, subj, runs, bids_root, log_level, skip_existing)


@task
def atlas(c, subject=None, runs=None, atlases=None, skip_existing=True, slurm=False, dry_run=False):
    """Apply atlas parcellation to morphed source estimates (Stage 2b).

    Extracts ROI-level time series from vertex-level source estimates using
    cortical parcellations. By default applies:
    - aparc.a2009s (Destrieux, 148 ROIs)
    - schaefer_100, schaefer_200, schaefer_400 (Schaefer parcellations)

    Output: derivatives/atlas_timeseries_{atlas}/sub-{subject}/*.npz

    By default processes all subjects from config. Use --subject for a single subject.

    Examples:
        invoke pipeline.atlas                      # All subjects (default)
        invoke pipeline.atlas --subject=04         # Single subject
        invoke pipeline.atlas --subject=04 --atlases="aparc.a2009s"
        invoke pipeline.atlas --slurm              # All subjects on cluster
    """
    from code.utils.config import load_config

    print("=" * 80)
    print("Atlas Application - Stage 2b")
    print("=" * 80)

    if slurm:
        _atlas_slurm(c, subject, runs, atlases, skip_existing, dry_run)
        return

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    print(f"Processing {len(subjects)} subject(s): {', '.join(subjects)}\n")

    python_exe = get_python_executable()

    for subj in subjects:
        print(f"\n{'='*40}")
        print(f"Subject: {subj}")
        print(f"{'='*40}")

        cmd = [python_exe, "-m", "code.source_reconstruction.apply_atlas"]
        cmd.extend(["--subject", subj])

        if runs:
            cmd.extend(["--runs", runs])

        if atlases:
            cmd.extend(["--atlases", atlases])

        if skip_existing:
            cmd.append("--skip-existing")
        else:
            cmd.append("--no-skip-existing")

        print(f"Running: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)

    print(f"\n✓ Atlas application complete for {len(subjects)} subject(s)")


# ==============================================================================
# pipeline.features.* Tasks - Feature Extraction
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
        invoke pipeline.features.fooof --subject=04
        invoke pipeline.features.fooof --subject=04 --space=aparc.a2009s
        invoke pipeline.features.fooof --slurm  # All subjects on cluster
    """
    print("=" * 80)
    print("Feature Extraction - FOOOF")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "fooof", subject, runs, space, skip_existing=skip_existing, dry_run=dry_run)
        return

    if not subject:
        print("ERROR: --subject is required for local execution")
        print("Use --slurm to process all subjects in parallel on HPC")
        return

    _features_local(c, "fooof", subject, runs, space, skip_existing=skip_existing)
    print(f"\n✓ FOOOF extraction complete for sub-{subject}")


@task
def psd(c, subject=None, runs=None, space="sensor", skip_existing=True, slurm=False, dry_run=False):
    """Extract power spectral density features.

    Computes:
    - Welch PSD estimates per trial
    - Saves with IN/OUT classification metadata

    Examples:
        invoke pipeline.features.psd --subject=04
        invoke pipeline.features.psd --subject=04 --space=aparc.a2009s
        invoke pipeline.features.psd --slurm  # All subjects on cluster
    """
    print("=" * 80)
    print("Feature Extraction - PSD")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "psd", subject, runs, space, skip_existing=skip_existing, dry_run=dry_run)
        return

    if not subject:
        print("ERROR: --subject is required for local execution")
        print("Use --slurm to process all subjects in parallel on HPC")
        return

    _features_local(c, "psd", subject, runs, space, skip_existing=skip_existing)
    print(f"\n✓ PSD extraction complete for sub-{subject}")


@task
def complexity(c, subject=None, runs=None, space="sensor", complexity_type="lzc entropy fractal",
               overwrite=False, slurm=False, dry_run=False):
    """Extract complexity and entropy measures.

    Computes:
    - Lempel-Ziv Complexity (LZC)
    - Entropy measures (permutation, spectral, sample, approximate, SVD)
    - Fractal dimensions (Higuchi, Petrosian, Katz, DFA)

    Examples:
        invoke pipeline.features.complexity --subject=04
        invoke pipeline.features.complexity --subject=04 --space=aparc.a2009s
        invoke pipeline.features.complexity --slurm  # All subjects on cluster
    """
    print("=" * 80)
    print("Feature Extraction - Complexity")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "complexity", subject, runs, space, skip_existing=not overwrite,
                        complexity_types=complexity_type, dry_run=dry_run)
        return

    if not subject:
        print("ERROR: --subject is required for local execution")
        print("Use --slurm to process all subjects in parallel on HPC")
        return

    _features_local(c, "complexity", subject, runs, space, skip_existing=not overwrite,
                    complexity_types=complexity_type)
    print(f"\n✓ Complexity extraction complete for sub-{subject}")


@task
def extract_all(c, subject=None, runs=None, space="sensor", overwrite=False, slurm=False, dry_run=False):
    """Extract all feature types (PSD, FOOOF, complexity).

    Order: PSD -> FOOOF -> Complexity (FOOOF depends on PSD)

    Examples:
        invoke pipeline.features.all --subject=04
        invoke pipeline.features.all --subject=04 --space=aparc.a2009s
        invoke pipeline.features.all --slurm  # All subjects on cluster
    """
    print("=" * 80)
    print(f"Feature Extraction - All Features (space={space})")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "all", subject, runs, space, skip_existing=not overwrite, dry_run=dry_run)
        return

    if not subject:
        print("ERROR: --subject is required for local execution")
        print("Use --slurm to process all subjects in parallel on HPC")
        return

    # Convert overwrite to skip_existing for fooof/psd tasks
    skip_existing = not overwrite

    print("\n[1/3] Extracting PSD features...")
    _features_local(c, "psd", subject, runs, space, skip_existing=skip_existing)

    print("\n[2/3] Extracting FOOOF features...")
    _features_local(c, "fooof", subject, runs, space, skip_existing=skip_existing)

    print("\n[3/3] Extracting Complexity features...")
    _features_local(c, "complexity", subject, runs, space, skip_existing=skip_existing)

    print("\n" + "=" * 80)
    print("✓ All feature extraction complete!")
    print("=" * 80)


# ==============================================================================
# analysis.* Tasks - Statistical Analysis
# ==============================================================================

@task
def statistics(c, feature_type, space="sensor", test="paired_ttest",
               corrections="fdr bonferroni", alpha=0.05, n_permutations=10000,
               visualize=False, average_trials=False, slurm=False, dry_run=False):
    """Run group-level statistical analysis (IN vs OUT).

    By default, runs trial-level independent t-test (matching cc_saflow).
    Use --average-trials for subject-level paired t-test.

    Feature types:
    - FOOOF: fooof_exponent, fooof_offset, fooof_knee, fooof_r_squared
    - PSD: psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma
    - Complexity: complexity (uses dedicated script)

    Examples:
        invoke analysis.statistics --feature-type=fooof_exponent
        invoke analysis.statistics --feature-type=psd_alpha --average-trials
        invoke analysis.statistics --feature-type=complexity
    """
    print("=" * 80)
    print("Group-Level Statistical Analysis")
    print("=" * 80)

    if slurm:
        print("ERROR: SLURM execution not yet implemented")
        return

    python_exe = get_python_executable()

    # Use dedicated script for complexity features
    if feature_type == "complexity" or feature_type.startswith("complexity_"):
        print("Using dedicated complexity statistics script...")
        cmd = [python_exe, "-m", "code.statistics.run_complexity_stats"]
        cmd.extend(["--space", space])
        # Map correction names
        corr = corrections.split()[0] if corrections else "fdr"
        cmd.extend(["--correction", corr])
        cmd.extend(["--alpha", str(alpha)])
        if "permutation" in corrections:
            cmd.extend(["--n-permutations", str(n_permutations)])
    else:
        cmd = [python_exe, "-m", "code.statistics.run_group_statistics"]
        cmd.extend(["--feature-type", feature_type])
        cmd.extend(["--space", space])
        cmd.extend(["--test", test])
        cmd.extend(["--correction"] + corrections.split())
        cmd.extend(["--alpha", str(alpha)])
        cmd.extend(["--n-permutations", str(n_permutations)])

        if average_trials:
            cmd.append("--average-trials")
        if visualize:
            cmd.append("--visualize")

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def stats_complexity(c, space="sensor", correction="fdr", alpha=0.05, n_permutations=1000):
    """Run t-tests on complexity measures (IN vs OUT).

    Runs paired t-tests comparing IN vs OUT attentional states for all
    complexity metrics:
    - LZC (Lempel-Ziv Complexity)
    - Entropy measures (permutation, spectral, sample, approximate, SVD)
    - Fractal dimensions (Higuchi, Petrosian, Katz, DFA)

    Outputs:
    - Topographic figure: reports/figures/complexity_ttest_{correction}.png
    - Numerical results: {data_root}/features/statistics_{space}/complexity_ttest_results.npz

    Examples:
        invoke analysis.stats.complexity
        invoke analysis.stats.complexity --correction=bonferroni
        invoke analysis.stats.complexity --correction=permutation --n-permutations=5000
        invoke analysis.stats.complexity --correction=none --alpha=0.01
    """
    print("=" * 80)
    print("Complexity Statistics: IN vs OUT (Paired T-tests)")
    print(f"Space: {space}, Correction: {correction}, Alpha: {alpha}")
    print("=" * 80)

    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.statistics.run_complexity_stats"]
    cmd.extend(["--space", space])
    cmd.extend(["--correction", correction])
    cmd.extend(["--alpha", str(alpha)])
    cmd.extend(["--n-permutations", str(n_permutations)])

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def stats_fooof(c, space="sensor", correction="fdr", alpha=0.05, n_permutations=1000,
                n_jobs=1, average_trials=False):
    """Run t-tests on FOOOF parameters (IN vs OUT).

    By default, runs trial-level independent t-test.
    Use --average-trials for subject-level paired t-test.

    Runs t-tests comparing IN vs OUT attentional states for:
    - Aperiodic exponent (1/f slope)
    - Aperiodic offset
    - Model fit (r_squared)

    Examples:
        invoke analysis.stats.fooof
        invoke analysis.stats.fooof --average-trials
        invoke analysis.stats.fooof --correction=tmax --n-permutations=10000
        invoke analysis.stats.fooof --n-jobs=4
    """
    test_label = "Paired T-tests (subject-level)" if average_trials else "Independent T-tests (trial-level)"
    print("=" * 80)
    print(f"FOOOF Statistics: IN vs OUT ({test_label})")
    print(f"Space: {space}, Correction: {correction}, Alpha: {alpha}, n_jobs: {n_jobs}")
    print("=" * 80)

    python_exe = get_python_executable()

    # Run for each FOOOF parameter
    fooof_params = ["fooof_exponent", "fooof_offset", "fooof_r_squared"]

    for param in fooof_params:
        print(f"\n[{param}]")
        cmd = [python_exe, "-m", "code.statistics.run_group_statistics"]
        cmd.extend(["--feature-type", param])
        cmd.extend(["--space", space])
        cmd.extend(["--test", "paired_ttest"])
        cmd.extend(["--correction", correction])
        cmd.extend(["--alpha", str(alpha)])
        cmd.extend(["--n-permutations", str(n_permutations)])
        cmd.extend(["--n-jobs", str(n_jobs)])
        if average_trials:
            cmd.append("--average-trials")

        print(f"Running: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)


@task
def stats_psd(c, space="sensor", correction="fdr", alpha=0.05, n_permutations=1000,
              bands="theta alpha lobeta hibeta gamma1 gamma2 gamma3", n_jobs=1,
              average_trials=False):
    """Run t-tests on raw PSD band power (IN vs OUT).

    By default, runs trial-level independent t-test.
    Use --average-trials for subject-level paired t-test.

    Default bands: delta, theta, alpha, lobeta, hibeta, gamma1

    Examples:
        invoke analysis.stats.psd
        invoke analysis.stats.psd --average-trials
        invoke analysis.stats.psd --correction=tmax --n-permutations=10000
        invoke analysis.stats.psd --bands="theta alpha"
        invoke analysis.stats.psd --n-jobs=4
    """
    test_label = "Paired T-tests (subject-level)" if average_trials else "Independent T-tests (trial-level)"
    print("=" * 80)
    print(f"PSD Statistics: IN vs OUT ({test_label})")
    print(f"Space: {space}, Correction: {correction}, Alpha: {alpha}, n_jobs: {n_jobs}")
    print("=" * 80)

    python_exe = get_python_executable()
    band_list = bands.split()

    for band in band_list:
        print(f"\n[psd_{band}]")
        cmd = [python_exe, "-m", "code.statistics.run_group_statistics"]
        cmd.extend(["--feature-type", f"psd_{band}"])
        cmd.extend(["--space", space])
        cmd.extend(["--test", "paired_ttest"])
        cmd.extend(["--correction", correction])
        cmd.extend(["--alpha", str(alpha)])
        cmd.extend(["--n-permutations", str(n_permutations)])
        cmd.extend(["--n-jobs", str(n_jobs)])
        if average_trials:
            cmd.append("--average-trials")

        print(f"Running: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)


@task
def stats_psd_corrected(c, space="sensor", correction="fdr", alpha=0.05, n_permutations=1000,
                        bands="theta alpha lobeta hibeta gamma1 gamma2 gamma3", n_jobs=1,
                        average_trials=False):
    """Run t-tests on aperiodic-corrected PSD band power (IN vs OUT).

    By default, runs trial-level independent t-test.
    Use --average-trials for subject-level paired t-test.

    Default bands: delta, theta, alpha, lobeta, hibeta, gamma1

    Examples:
        invoke analysis.stats.psd-corrected
        invoke analysis.stats.psd-corrected --average-trials
        invoke analysis.stats.psd-corrected --correction=tmax --n-permutations=10000
        invoke analysis.stats.psd-corrected --bands="theta alpha"
        invoke analysis.stats.psd-corrected --n-jobs=4
    """
    test_label = "Paired T-tests (subject-level)" if average_trials else "Independent T-tests (trial-level)"
    print("=" * 80)
    print(f"PSD Corrected Statistics: IN vs OUT ({test_label})")
    print(f"Space: {space}, Correction: {correction}, Alpha: {alpha}, n_jobs: {n_jobs}")
    print("=" * 80)

    python_exe = get_python_executable()
    band_list = bands.split()

    for band in band_list:
        print(f"\n[psd_corrected_{band}]")
        cmd = [python_exe, "-m", "code.statistics.run_group_statistics"]
        cmd.extend(["--feature-type", f"psd_corrected_{band}"])
        cmd.extend(["--space", space])
        cmd.extend(["--test", "paired_ttest"])
        cmd.extend(["--correction", correction])
        cmd.extend(["--alpha", str(alpha)])
        cmd.extend(["--n-permutations", str(n_permutations)])
        cmd.extend(["--n-jobs", str(n_jobs)])
        if average_trials:
            cmd.append("--average-trials")

        print(f"Running: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)


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
def viz_stats(c, feature_type="fooof_exponent", space="sensor", alpha=0.05,
              cmap="RdBu", show=False, save=True):
    """Visualize saved statistical results as topographic maps.

    Creates a panel of topomaps showing contrast, t-values, and effect sizes.
    All maps share the same colormap and a single colorbar.

    Args:
        feature_type: Feature to visualize (e.g., fooof_exponent, psd_alpha)
        space: Analysis space (sensor, source, atlas)
        alpha: Significance threshold for marking
        cmap: Colormap for all maps (default: RdBu_r)
        show: Display the figure interactively
        save: Save the figure to reports/figures/

    Examples:
        invoke viz.stats --feature-type=fooof_exponent --show
        invoke viz.stats --feature-type=psd_alpha --cmap=viridis
        invoke viz.stats --feature-type=psd_corrected_theta --save
    """
    from pathlib import Path
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import mne

    print("=" * 80)
    print(f"Visualizing {feature_type} Statistics")
    print("=" * 80)

    # Load config and build filename
    from code.utils.config import load_config
    config = load_config()
    data_root = Path(config["paths"]["data_root"])
    inout_bounds = config["analysis"]["inout_bounds"]
    inout_str = f"{inout_bounds[0]}{inout_bounds[1]}"

    stats_dir = data_root / "features" / f"statistics_{space}"

    # Find matching result files (support partial matching like "psd" for all psd_* features)
    exact_match = stats_dir / f"feature-{feature_type}_inout-{inout_str}_test-paired_ttest_results.npz"
    if exact_match.exists():
        results_files = [exact_match]
    else:
        # Try partial match (e.g., "psd" matches "psd_alpha", "psd_theta", etc.)
        pattern = f"feature-{feature_type}_*_inout-{inout_str}_test-paired_ttest_results.npz"
        results_files = sorted(stats_dir.glob(pattern))

    if not results_files:
        print(f"ERROR: No results found for feature type '{feature_type}'")
        # List available features
        available_files = list(stats_dir.glob("feature-*_results.npz"))
        if available_files:
            print(f"\nAvailable feature types in {stats_dir.name}/:")
            for f in sorted(available_files):
                # Extract feature type from filename: feature-{type}_inout-...
                name = f.stem.replace("_results", "")
                parts = name.split("_inout-")
                if parts:
                    feat = parts[0].replace("feature-", "")
                    print(f"  --feature-type={feat}")
        else:
            print(f"\nNo statistics found. Run analysis.stats.* tasks first:")
            print(f"  invoke analysis.stats.fooof")
            print(f"  invoke analysis.stats.psd")
            print(f"  invoke analysis.stats.psd-corrected")
        return

    # Extract feature names from files
    feature_names = []
    for f in results_files:
        name = f.stem.replace("_results", "")
        parts = name.split("_inout-")
        if parts:
            feature_names.append(parts[0].replace("feature-", ""))

    # Sort by frequency band order (low to high frequency)
    band_order = ["delta", "theta", "alpha", "lobeta", "hibeta", "gamma1", "gamma2", "gamma3"]

    def get_band_index(feat_name):
        """Get sort index based on frequency band."""
        for i, band in enumerate(band_order):
            if feat_name.endswith(f"_{band}") or feat_name == band:
                return i
        return 999  # Unknown bands go last

    # Sort features and files together
    sorted_pairs = sorted(zip(feature_names, results_files), key=lambda x: get_band_index(x[0]))
    feature_names = [p[0] for p in sorted_pairs]
    results_files = [p[1] for p in sorted_pairs]

    print(f"Found {len(results_files)} feature(s): {', '.join(feature_names)}")

    # Get sensor info from a sample preprocessed file
    print("Loading sensor positions...")
    derivatives_dir = data_root / config["paths"]["derivatives"]
    sample_files = list(derivatives_dir.glob("preprocessed/sub-*/meg/*_proc-clean_meg.fif"))
    if not sample_files:
        sample_files = list(derivatives_dir.glob("**/sub-*/meg/*_meg.fif"))
    if not sample_files:
        print("ERROR: No preprocessed MEG files found to get sensor positions")
        return

    raw = mne.io.read_raw_fif(sample_files[0], preload=False, verbose=False)
    raw.pick("mag")  # Pick only magnetometers (not ref_meg)
    info = raw.info

    # Load all results
    all_results = []
    all_metadata = []
    for results_file in results_files:
        results = np.load(results_file, allow_pickle=True)
        all_results.append(results)
        metadata_file = results_file.with_name(results_file.stem.replace("_results", "_metadata") + ".json")
        if metadata_file.exists():
            with open(metadata_file) as f:
                all_metadata.append(json.load(f))
        else:
            all_metadata.append({})

    # Collect maps to plot - one t-value map per feature
    maps_to_plot = []
    masks_to_plot = []
    titles = []

    for feat_name, results in zip(feature_names, all_results):
        # Get t-values for this feature
        if "tvals" in results.files:
            tvals = results["tvals"].flatten()
            maps_to_plot.append(tvals)
            # Get significance mask from corrected p-values
            mask = None
            n_sig = 0
            for key in results.files:
                if key.startswith("pvals_corrected_"):
                    pvals = results[key].flatten()
                    mask = pvals < alpha
                    n_sig = np.sum(mask)
                    break
            masks_to_plot.append(mask)
            # Clean up feature name for title
            short_name = feat_name.replace("psd_corrected_", "").replace("psd_", "").replace("fooof_", "")
            titles.append(f"{short_name}\n(n={n_sig} sig)")

    n_maps = len(maps_to_plot)
    print(f"Creating panel with {n_maps} topomaps (t-values)...")

    # Create figure with subplots + colorbar space
    fig, axes = plt.subplots(1, n_maps, figsize=(3 * n_maps + 1, 3.5), dpi=150)
    if n_maps == 1:
        axes = [axes]

    # Determine global color limits (symmetric around 0 for diverging colormaps)
    all_values = np.concatenate([m for m in maps_to_plot])
    vmax = np.nanpercentile(np.abs(all_values), 98)
    vmin = -vmax

    # Mask params for significant sensors (white circles with black borders)
    mask_params = dict(
        marker="o",
        markerfacecolor="w",
        markeredgecolor="k",
        linewidth=0,
        markersize=5,
    )

    # Plot each topomap
    im = None
    for ax, data, mask, title in zip(axes, maps_to_plot, masks_to_plot, titles):
        im = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            cmap=cmap,
            mask=mask,
            mask_params=mask_params,
            vlim=(vmin, vmax),
            extrapolate="local",
            outlines="head",
            sphere=0.15,
            contours=0,
        )
        ax.set_title(title, fontsize=10)

    # Add single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im[0], cax=cbar_ax)
    cbar.set_label("Value", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    # Add min/max ticks
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

    # Add figure title
    first_meta = all_metadata[0] if all_metadata else {}
    data_meta = first_meta.get("data_metadata", {})
    fig_title = f"{feature_type} t-values | IN/OUT: {inout_bounds} | N={data_meta.get('n_subjects', '?')} subjects"
    fig.suptitle(fig_title, fontsize=12, y=1.02)

    # Adjust layout manually (avoid tight_layout warning with colorbar)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.85, bottom=0.05, wspace=0.1)

    # Save figure
    if save:
        fig_dir = Path("reports") / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"{feature_type}_stats.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to: {fig_path}")

    # Show or close
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Print summary
    print(f"\nResults summary:")
    print(f"  Features: {', '.join(feature_names)}")
    print(f"  Subjects: {data_meta.get('n_subjects', 'unknown')}")
    print(f"  IN trials: {data_meta.get('n_in', 'unknown')}")
    print(f"  OUT trials: {data_meta.get('n_out', 'unknown')}")


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

def _preprocess_local(c, subject, runs=None, bids_root=None, log_level="INFO", skip_existing=True, crop=None, skip_second_ar=False):
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
    if skip_second_ar:
        cmd.append("--skip-second-ar")

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
                "partition": slurm_config.get("partition", ""),
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
        bids_root = data_root / "bids"

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
                "partition": slurm_config.get("partition", ""),
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


def _atlas_slurm(c, subject=None, runs=None, atlases=None,
                 skip_existing=True, dry_run=False):
    """Submit atlas application jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    print("\n[SLURM Mode] Submitting atlas application jobs to cluster\n")

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    # Get atlas-specific resources from config
    atlas_resources = slurm_config.get("atlas", {})
    if not atlas_resources:
        print("ERROR: No atlas resources in config.yaml (computing.slurm.atlas)")
        return

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "atlas"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "atlas"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total jobs: {len(subjects)}")  # One job per subject (all runs)

    job_ids = []
    for subj in subjects:
        job_name = f"atlas_sub-{subj}"
        runs_str = " ".join(run_list)
        context = {
            "job_name": job_name,
            "account": slurm_config["account"],
            "partition": slurm_config.get("partition", ""),
            "cpus": atlas_resources["cpus"],
            "mem": atlas_resources["mem"],
            "time": atlas_resources["time"],
            "log_dir": str(log_dir),
            "venv_path": str(venv_path),
            "project_root": str(PROJECT_ROOT),
            "subject": subj,
            "runs": runs_str,
            "atlases": atlases,
            "skip_existing": skip_existing,
            "timestamp": timestamp,
        }

        script_path = script_dir / f"{job_name}_{timestamp}.sh"
        render_slurm_script("atlas.sh.j2", context, output_path=script_path)

        if not dry_run:
            try:
                job_id = submit_slurm_job(script_path, job_name=job_name, dry_run=False)
                if job_id:
                    job_ids.append(job_id)
            except Exception as e:
                print(f"  ✗ Failed to submit: {e}")
        else:
            print(f"[DRY RUN] Would submit: {script_path.name}")

    if job_ids:
        manifest_path = log_dir / f"atlas_manifest_{timestamp}.json"
        save_job_manifest(job_ids, manifest_path, metadata={
            "stage": "atlas",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted {len(job_ids)} atlas application jobs")
    elif dry_run:
        print(f"\n[DRY RUN] Would have submitted {len(subjects)} jobs")


def _features_local(c, feature_type, subject, runs=None, space="sensor",
                    skip_existing=True, complexity_types=None, log_level="INFO"):
    """Run feature extraction locally."""
    from code.utils.config import load_config

    config = load_config()
    run_list = runs.split() if runs else config["bids"]["task_runs"]
    python_exe = get_python_executable()

    for run in run_list:
        print(f"\n[Processing run {run}]")

        if feature_type == "psd":
            cmd = [python_exe, "-m", "code.features.compute_welch_psd"]
            cmd.extend(["--subject", subject, "--run", run, "--space", space])
            cmd.extend(["--log-level", log_level])
            cmd.append("--skip-existing" if skip_existing else "--no-skip-existing")

        elif feature_type == "fooof":
            cmd = [python_exe, "-m", "code.features.compute_fooof"]
            cmd.extend(["--subject", subject, "--run", run, "--space", space])
            cmd.extend(["--log-level", log_level])
            cmd.append("--skip-existing" if skip_existing else "--no-skip-existing")

        elif feature_type == "complexity":
            cmd = [python_exe, "-m", "code.features.compute_complexity"]
            cmd.extend(["--subject", subject, "--run", run, "--space", space])
            if complexity_types:
                cmd.extend(["--complexity-type"] + complexity_types.split())
            if not skip_existing:
                cmd.append("--overwrite")

        else:
            print(f"ERROR: Unknown feature type '{feature_type}'")
            return

        print(f"Running: {' '.join(cmd)}\n")
        result = c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)
        if result.exited != 0:
            print(f"WARNING: Run {run} failed")


def _features_slurm(c, feature_type, subject=None, runs=None, space="sensor",
                    skip_existing=True, complexity_types=None, log_level="INFO", dry_run=False):
    """Submit feature extraction jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    print(f"\n[SLURM Mode] Submitting {feature_type} feature extraction jobs to cluster\n")

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    features_resources = slurm_config.get("features", {})
    if not features_resources:
        print("ERROR: No features resources in config.yaml")
        return

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "features"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "features"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Feature type: {feature_type}")
    print(f"Space: {space}")
    print(f"Subjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total jobs: {len(subjects) * len(run_list)}")

    job_ids = []
    for subj in subjects:
        for run in run_list:
            job_name = f"{feature_type}_sub-{subj}_run-{run}_{space}"
            context = {
                "job_name": job_name,
                "account": slurm_config["account"],
                "partition": slurm_config.get("partition", ""),
                "cpus": features_resources["cpus"],
                "mem": features_resources["mem"],
                "time": features_resources["time"],
                "log_dir": str(log_dir),
                "venv_path": str(venv_path),
                "project_root": str(PROJECT_ROOT),
                "feature_type": feature_type,
                "subject": subj,
                "run": run,
                "space": space,
                "log_level": log_level,
                "skip_existing": skip_existing,
                "complexity_types": complexity_types,
                "timestamp": timestamp,
            }

            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("features.sh.j2", context, output_path=script_path)

            if not dry_run:
                try:
                    job_id = submit_slurm_job(script_path, job_name=job_name, dry_run=False)
                    if job_id:
                        job_ids.append(job_id)
                except Exception as e:
                    print(f"  ✗ Failed to submit: {e}")
            else:
                print(f"[DRY RUN] Would submit: {script_path.name}")

    if job_ids:
        manifest_path = log_dir / f"{feature_type}_manifest_{timestamp}.json"
        save_job_manifest(job_ids, manifest_path, metadata={
            "stage": f"features_{feature_type}",
            "feature_type": feature_type,
            "space": space,
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted {len(job_ids)} {feature_type} feature extraction jobs")
    elif dry_run:
        print(f"\n[DRY RUN] Would have submitted {len(subjects) * len(run_list)} jobs")


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

# Data download tasks
get = Collection("get")
get.add_task(get_atlases, name="atlases")

# Feature extraction tasks (nested under pipeline)
features = Collection("features")
features.add_task(fooof)
features.add_task(psd)
features.add_task(complexity)
features.add_task(extract_all, name="all")

# Pipeline tasks (includes features as subcollection)
pipeline = Collection("pipeline")
pipeline.add_task(validate_inputs, name="validate-inputs")
pipeline.add_task(bids)
pipeline.add_task(preprocess)
pipeline.add_task(source_recon, name="source-recon")
pipeline.add_task(atlas)
pipeline.add_collection(features)  # Nested: pipeline.features.*

# Statistics subcollection (under analysis)
stats = Collection("stats")
stats.add_task(stats_complexity, name="complexity")
stats.add_task(stats_fooof, name="fooof")
stats.add_task(stats_psd, name="psd")
stats.add_task(stats_psd_corrected, name="psd-corrected")

# Analysis tasks
analysis = Collection("analysis")
analysis.add_task(statistics)
analysis.add_task(classify)
analysis.add_collection(stats)  # Nested: analysis.stats.*

# Visualization tasks
viz = Collection("viz")
viz.add_task(viz_stats, name="stats")
viz.add_task(behavior)

# Build main namespace
namespace = Collection()
namespace.add_collection(dev)
namespace.add_collection(env)
namespace.add_collection(get)
namespace.add_collection(pipeline)
namespace.add_collection(analysis)
namespace.add_collection(viz)


# Default task
@task(default=True)
def help_task(c):
    """Show available tasks (default task)."""
    c.run("invoke --list")


namespace.add_task(help_task, name="help")
