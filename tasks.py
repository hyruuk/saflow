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
    invoke analysis.stats --features=all
    invoke analysis.classify --features=all

Namespaces:
    dev.check           - Data validation (dataset, qc, code)
    dev                 - Development tasks (test, clean)
    env                 - Environment tasks (setup, info)
    get                 - Data downloads (atlases)
    pipeline            - Data processing pipeline (bids, preprocess, source-recon, atlas)
    pipeline.features   - Feature extraction (psd, fooof, complexity, all)
    analysis            - Statistical analysis (stats, classify)
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
SAFLOW_PKG = PROJECT_ROOT / "code"
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


# Feature-set shortcuts shared by analysis.stats and analysis.classify so the
# two CLIs accept identical --features values.
_FEATURE_SHORTCUTS = {"psds", "psds_corrected", "fooof", "complexity", "all"}


def resolve_features(features):
    """Resolve a `--features` value into a concrete list of feature names.

    Accepts:
      - A single feature name: "fooof_exponent"
      - A space-separated list: "fooof_exponent psd_alpha"
      - A shortcut: "psds", "psds_corrected", "fooof", "complexity", "all"
      - A mix: "fooof psd_alpha"  (shortcut + extra feature)

    Shortcuts expand via classification.run_classification.expand_feature_set,
    which reads bands from config.yaml so PSD families track config.
    """
    from code.classification.run_classification import expand_feature_set
    from code.utils.config import load_config

    if not features:
        raise ValueError("--features is required (single name, list, or shortcut: "
                         "psds, psds_corrected, fooof, complexity, all)")

    tokens = features.split() if isinstance(features, str) else list(features)
    config = None
    resolved = []
    for tok in tokens:
        if tok in _FEATURE_SHORTCUTS:
            if config is None:
                config = load_config()
            resolved.extend(expand_feature_set(tok, config))
        else:
            resolved.append(tok)

    seen = set()
    return [f for f in resolved if not (f in seen or seen.add(f))]


# Trial-type shortcut shared by analysis.stats and analysis.classify. "all"
# expands to the three variants we report on: every window, baseline-only
# (no errors) and lapse windows (>=1 commission error).
_TRIAL_TYPE_SETS = {"all": ["alltrials", "correct", "lapse"]}


def resolve_trial_types(trial_type):
    """Resolve a `--trial-type` value into a concrete list.

    Accepts a single type ('alltrials', 'correct', 'rare', 'lapse',
    'correct_commission') or the shortcut 'all' → alltrials + correct
    (baseline) + lapse.
    """
    if not trial_type:
        return list(_TRIAL_TYPE_SETS["all"])
    if trial_type in _TRIAL_TYPE_SETS:
        return list(_TRIAL_TYPE_SETS[trial_type])
    return [trial_type]


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
        cmd.extend(["--cov=code", "--cov-report=term-missing"])
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
               skip_existing=True, crop=None, slurm=False, dry_run=False):
    """Run MEG preprocessing (Stage 1).

    Pipeline: Filter -> Epoch (Freq+Rare only) -> AR1 -> ICA -> AR2 (fit+transform)
    Outputs: continuous Raw + BAD annotations, ICA epochs, AR2-interpolated epochs.

    Examples:
        invoke pipeline.preprocess --subject=04
        invoke pipeline.preprocess --subject=04 --runs="02 03"
        invoke pipeline.preprocess --subject=04 --crop=50  # Quick test with 50s
        invoke pipeline.preprocess --slurm
    """
    print("=" * 80)
    print("MEG Preprocessing - Stage 1")
    if crop:
        print(f"[TEST MODE] Cropping to first {crop} seconds")
    print("=" * 80)

    if slurm:
        _preprocess_slurm(c, subject, runs, bids_root, log_level, skip_existing, dry_run)
    else:
        from code.utils.config import load_config
        config = load_config()
        subjects = [subject] if subject else config["bids"]["subjects"]
        print(f"Processing {len(subjects)} subject(s): {', '.join(subjects)}\n")
        for subj in subjects:
            print(f"\n{'='*40}")
            print(f"Subject: {subj}")
            print(f"{'='*40}")
            _preprocess_local(c, subj, runs, bids_root, log_level, skip_existing, crop)


@task
def preprocess_report(c, subject=None, dataset=False):
    """Generate preprocessing aggregate reports.

    Examples:
        invoke pipeline.preprocess-report --subject=04
        invoke pipeline.preprocess-report --dataset
        invoke pipeline.preprocess-report --subject=04 --dataset
    """
    from code.utils.config import load_config
    config = load_config()

    # Default: all subjects + dataset report
    if not subject and not dataset:
        subjects = config["bids"]["subjects"]
        dataset = True
    else:
        subjects = [subject] if subject else []

    python_exe = get_python_executable()
    for subj in subjects:
        cmd = [python_exe, "-m", "code.preprocessing.aggregate_reports", "-s", subj]
        print(f"\nRunning: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)

    if dataset:
        cmd = [python_exe, "-m", "code.preprocessing.aggregate_reports", "--dataset"]
        print(f"\nRunning: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


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
def fooof(c, subject=None, runs=None, space="sensor", skip_existing=True, slurm=False, dry_run=False,
          n_events_window=8):
    """Extract FOOOF aperiodic parameters and corrected PSDs.

    Computes:
    - Aperiodic parameters (exponent, offset)
    - Goodness of fit (r_squared, error)
    - Corrected PSDs (aperiodic component removed)

    FOOOF parameters (freq_range, fixed aperiodic_mode) come from config.yaml.

    Args:
        n_events_window: Window size used by upstream Welch (1 = single-trial,
            8 = cc_saflow's sliding window, default). Determines which Welch
            file to load and which desc suffix to write.

    Examples:
        invoke pipeline.features.fooof --subject=04
        invoke pipeline.features.fooof --subject=04 --space=aparc.a2009s
        invoke pipeline.features.fooof --slurm  # All subjects on cluster
        invoke pipeline.features.fooof --n-events-window=1  # single-trial mode
    """
    print("=" * 80)
    print("Feature Extraction - FOOOF")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "fooof", subject, runs, space, skip_existing=skip_existing,
                        dry_run=dry_run, n_events_window=n_events_window)
        return

    from code.utils.config import load_config
    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    print(f"Processing {len(subjects)} subject(s)\n")
    for subj in subjects:
        _features_local(c, "fooof", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)
    print(f"\n✓ FOOOF extraction complete")


@task
def psd(c, subject=None, runs=None, space="sensor", skip_existing=True, slurm=False, dry_run=False,
        n_events_window=8):
    """Extract power spectral density features.

    Computes:
    - Welch PSD estimates per epoch
    - Saves with IN/OUT classification metadata

    Args:
        n_events_window: Number of consecutive stim trials per epoch.
            1 = single-trial, 8 = cc_saflow's sliding window (default).

    Examples:
        invoke pipeline.features.psd --subject=04
        invoke pipeline.features.psd --subject=04 --space=aparc.a2009s
        invoke pipeline.features.psd --slurm  # All subjects on cluster
        invoke pipeline.features.psd --n-events-window=1  # single-trial mode
    """
    print("=" * 80)
    print("Feature Extraction - PSD")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "psd", subject, runs, space, skip_existing=skip_existing,
                        dry_run=dry_run, n_events_window=n_events_window)
        return

    from code.utils.config import load_config
    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    print(f"Processing {len(subjects)} subject(s)\n")
    for subj in subjects:
        _features_local(c, "psd", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)
    print(f"\n✓ PSD extraction complete")


@task
def complexity(c, subject=None, runs=None, space="sensor", complexity_type="lzc entropy fractal",
               overwrite=False, slurm=False, dry_run=False, n_events_window=8):
    """Extract complexity and entropy measures.

    Computes:
    - Lempel-Ziv Complexity (LZC)
    - Entropy measures (permutation, spectral, sample, approximate, SVD)
    - Fractal dimensions (Higuchi, Petrosian, Katz, DFA)

    Args:
        n_events_window: Number of consecutive stim trials per epoch.
            1 = single-trial, 8 = cc_saflow's sliding window (default).

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
                        complexity_types=complexity_type, dry_run=dry_run,
                        n_events_window=n_events_window)
        return

    from code.utils.config import load_config
    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    print(f"Processing {len(subjects)} subject(s)\n")
    for subj in subjects:
        _features_local(c, "complexity", subj, runs, space, skip_existing=not overwrite,
                        complexity_types=complexity_type, n_events_window=n_events_window)
    print(f"\n✓ Complexity extraction complete")


@task
def extract_all(c, subject=None, runs=None, space="sensor", overwrite=False, slurm=False, dry_run=False,
                n_events_window=8):
    """Extract all feature types (PSD, FOOOF, complexity).

    Order: PSD -> FOOOF -> Complexity (FOOOF depends on PSD)

    Args:
        n_events_window: Number of consecutive stim trials per epoch.
            1 = single-trial, 8 = cc_saflow's sliding window (default).

    Examples:
        invoke pipeline.features.all --subject=04
        invoke pipeline.features.all --subject=04 --space=aparc.a2009s
        invoke pipeline.features.all --slurm  # All subjects on cluster
        invoke pipeline.features.all --n-events-window=1  # single-trial mode
    """
    print("=" * 80)
    print(f"Feature Extraction - All Features (space={space}, window={n_events_window})")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "all", subject, runs, space, skip_existing=not overwrite,
                        dry_run=dry_run, n_events_window=n_events_window)
        return

    from code.utils.config import load_config
    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    skip_existing = not overwrite
    print(f"Processing {len(subjects)} subject(s)\n")

    for subj in subjects:
        print(f"\n{'='*40}\nSubject: {subj}\n{'='*40}")
        print("\n[1/3] Extracting PSD features...")
        _features_local(c, "psd", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)
        print("\n[2/3] Extracting FOOOF features...")
        _features_local(c, "fooof", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)
        print("\n[3/3] Extracting Complexity features...")
        _features_local(c, "complexity", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)

    print("\n" + "=" * 80)
    print("✓ All feature extraction complete!")
    print("=" * 80)


# ==============================================================================
# analysis.* Tasks - Statistical Analysis
# ==============================================================================

@task
def stats(c, features="all", space="sensor", test="paired_ttest",
          correction="fdr", alpha=0.05, n_permutations=10000,
          n_jobs=1, analysis_level="both", aggregate="median", visualize=False,
          continue_on_error=True,
          trial_type="all", zoning="per-run", n_events_window=8,
          slurm_time=None, slurm_mem=None, slurm_cpus=None,
          slurm=False, dry_run=False):
    """Run group-level statistical analysis (IN vs OUT) on one or more features.

    With no arguments, runs every feature through every trial-type variant at
    every analysis level (`--features=all --trial-type=all
    --analysis-level=both`).

    `--features` accepts (default: 'all'):
      - A single feature name: --features=fooof_exponent
      - A space-separated list: --features="fooof_exponent psd_alpha"
      - A shortcut: --features=psds | psds_corrected | fooof | complexity | all

    `--trial-type` accepts (default: 'all'):
      - A single type: alltrials | correct | rare | lapse | correct_commission
      - The shortcut 'all' → alltrials + correct (baseline) + lapse, each run
        separately into its own `_type-<...>` result files.

    `--analysis-level` accepts (default: 'both'):
      - epoch:   per-epoch independent t-test (level-epoch file).
      - average: 1 IN + 1 OUT per subject, paired t-test (level-average file).
                  psd_/fooof_ → subject-spectrum (pool → FOOOF once);
                  complexity_ → per-subject median/mean of per-epoch features.
      - both:    run both, save one results file per level.

    --aggregate ('median' default | 'mean') controls the per-subject reducer
    used by level=average on complexity_* features. Ignored elsewhere.

    Examples:
        invoke analysis.stats --features=fooof_exponent
        invoke analysis.stats --features=psds --space=schaefer_400
        invoke analysis.stats --features=all --n-jobs=4
        invoke analysis.stats --features=complexity --correction=fdr
        invoke analysis.stats --features=complexity --aggregate=mean
        invoke analysis.stats --features="psd_alpha psd_theta" --analysis-level=epoch
        invoke analysis.stats --features=psds --slurm
    """
    feature_list = resolve_features(features)
    trial_types = resolve_trial_types(trial_type)
    print("=" * 80)
    print(f"analysis.stats | space={space}  correction={correction}  α={alpha}")
    print(f"               | n_perm={n_permutations}  analysis-level={analysis_level}")
    print(f"               | features ({len(feature_list)}): {' '.join(feature_list)}")
    print(f"               | trial-types ({len(trial_types)}): {' '.join(trial_types)}")
    print("=" * 80)

    if slurm:
        for tt in trial_types:
            print(f"\n{'#' * 80}\n# SLURM submit | trial-type: {tt}\n{'#' * 80}")
            _stats_slurm(
                c,
                feature_list=feature_list,
                space=space,
                test=test,
                correction=correction,
                alpha=alpha,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                analysis_level=analysis_level,
                aggregate=aggregate,
                visualize=visualize,
                trial_type=tt,
                zoning=zoning,
                n_events_window=n_events_window,
                slurm_time=slurm_time,
                slurm_mem=slurm_mem,
                slurm_cpus=slurm_cpus,
                dry_run=dry_run,
            )
        return

    python_exe = get_python_executable()
    failures = []

    for tt in trial_types:
        print(f"\n{'#' * 80}\n# trial-type: {tt}\n{'#' * 80}")
        cmd = [python_exe, "-m", "code.statistics.run_group_statistics",
               "--feature-type", *feature_list,
               "--space", space, "--test", test,
               "--correction", correction, "--alpha", str(alpha),
               "--n-permutations", str(n_permutations), "--n-jobs", str(n_jobs),
               "--analysis-level", analysis_level,
               "--aggregate", aggregate,
               "--trial-type", tt,
               "--zoning", zoning,
               "--n-events-window", str(n_events_window)]
        if visualize:
            cmd.append("--visualize")
        print(f"\n>>> run_group_statistics on {len(feature_list)} feature(s) "
              f"(level={analysis_level})")
        print(f"Running: {' '.join(cmd)}\n")
        result = c.run(" ".join(cmd), pty=True,
                       env=get_env_with_pythonpath(), warn=continue_on_error)
        if result is not None and getattr(result, "exited", 0) != 0:
            failures.append((f"run_group_statistics[{tt}]", f"exit {result.exited}"))

    print("\n" + "=" * 80)
    if failures:
        print(f"Done with {len(failures)} failure(s):")
        for fam, msg in failures:
            print(f"  - {fam}: {msg}")
    else:
        print("All features completed.")
    print("=" * 80)


@task
def classify(c, features="all", clf="logistic", cv="auto",
             space="sensor", mode="univariate", n_permutations=1000,
             no_balance=False, n_jobs=-1, continue_on_error=False,
             combine_features=False, importances=False, label=None,
             n_chunks=1, seed=42, aggregate=True, delete_chunks=False,
             trial_type="all", zoning="per-run", n_events_window=8,
             analysis_level="both", standardize="auto",
             slurm_time=None, slurm_mem=None, slurm_cpus=None,
             slurm=False, dry_run=False):
    """Run classification analysis (IN vs OUT) on one or more features.

    With no arguments, runs every feature through every trial-type variant
    (`--features=all --trial-type=all`).

    `--features` accepts (default: 'all'):
      - A single feature name: --features=fooof_exponent
      - A space-separated list: --features="fooof_exponent psd_alpha"
      - A shortcut: --features=psds | psds_corrected | fooof | complexity | all

    `--trial-type` accepts (default: 'all'):
      - A single type: alltrials | correct | rare | lapse | correct_commission
      - The shortcut 'all' → alltrials + correct (baseline) + lapse, each run
        separately into its own `_type-<...>` score files.

    Spatial mode:
      - univariate (default): per-channel/ROI classifier + shared-permutation t-max
      - multivariate: pool spatial dim, single permutation_test_score

    Analysis level & CV:
      - --analysis-level=epoch: per-epoch classification (LOSO CV).
      - --analysis-level=average: 1 IN + 1 OUT row per subject — subject-
        spectrum (pool → FOOOF) for psd_/fooof_, mean of per-epoch features
        for complexity_. GroupKFold(k=6).
      - --analysis-level=both (default): run both levels, save one scores
        file per level.
      - --cv=auto resolves from the level (logo for epoch, group for
        average). --cv (logo|stratified|group) forces a specific splitter.

    Feature handling:
      - Default loop: each feature is classified independently (single-feature framework).
      - --combine-features: stack all selected features into a single classification
        (great for RF feature importance with --importances).

    Examples:
        # Single feature, per-channel LDA + tmax (recommended for single features)
        invoke analysis.classify --features=fooof_exponent --clf=lda

        # All 8 PSD bands, each as its own classification
        invoke analysis.classify --features=psds --space=sensor

        # Every feature on disk, each as its own classification
        invoke analysis.classify --features=all --space=schaefer_400

        # Combine all complexity metrics, RF per ROI, save importances
        invoke analysis.classify --features=complexity --space=schaefer_400 \\
            --clf=rf --combine-features --importances

        # Combine + multivariate: one big RF over (n_features × n_spatial)
        invoke analysis.classify --features=all --combine-features \\
            --mode=multivariate --clf=rf --importances

        # Submit each feature as a separate SLURM job
        invoke analysis.classify --features=psds --slurm

        # Per-epoch classification only (LOSO CV)
        invoke analysis.classify --features=fooof_exponent --analysis-level=epoch
    """
    print("=" * 80)
    print("Classification Analysis")
    print("=" * 80)

    feature_list = resolve_features(features)
    trial_types = resolve_trial_types(trial_type)
    print(f"Features ({len(feature_list)}): {' '.join(feature_list)}")
    print(f"Trial-types ({len(trial_types)}): {' '.join(trial_types)}")

    if slurm:
        for tt in trial_types:
            print(f"\n{'#' * 80}\n# SLURM submit | trial-type: {tt}\n{'#' * 80}")
            _classify_slurm(
                c,
                feature_list=feature_list,
                clf=clf,
                cv=cv,
                space=space,
                mode=mode,
                n_permutations=n_permutations,
                no_balance=no_balance,
                n_jobs=n_jobs,
                continue_on_error=continue_on_error,
                combine_features=combine_features,
                importances=importances,
                label=label,
                n_chunks=n_chunks,
                seed=seed,
                aggregate=aggregate,
                delete_chunks=delete_chunks,
                trial_type=tt,
                zoning=zoning,
                n_events_window=n_events_window,
                analysis_level=analysis_level,
                standardize=standardize,
                slurm_time=slurm_time,
                slurm_mem=slurm_mem,
                slurm_cpus=slurm_cpus,
                dry_run=dry_run,
            )
        return

    if n_chunks > 1:
        print(
            f"NOTE: --n-chunks={n_chunks} requires running each chunk separately. "
            f"Use --slurm to fan out automatically, or pass --chunk-idx N to the "
            f"underlying script directly."
        )
        return

    python_exe = get_python_executable()
    for tt in trial_types:
        print(f"\n{'#' * 80}\n# trial-type: {tt}\n{'#' * 80}")
        cmd = [python_exe, "-m", "code.classification.run_classification",
               "--feature", *feature_list]
        cmd.extend(["--clf", clf])
        cmd.extend(["--cv", cv])
        cmd.extend(["--space", space])
        cmd.extend(["--mode", mode])
        cmd.extend(["--n-permutations", str(n_permutations)])
        cmd.extend(["--n-jobs", str(n_jobs)])
        cmd.extend(["--trial-type", tt])
        cmd.extend(["--zoning", zoning])
        cmd.extend(["--n-events-window", str(n_events_window)])
        cmd.extend(["--analysis-level", analysis_level])
        cmd.extend(["--standardize", standardize])
        cmd.extend(["--seed", str(seed)])
        if no_balance:
            cmd.append("--no-balance")
        if continue_on_error:
            cmd.append("--continue-on-error")
        if combine_features:
            cmd.append("--combine-features")
        if importances:
            cmd.append("--importances")
        if label:
            cmd.extend(["--label", label])

        print(f"\nRunning: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


# ==============================================================================
# viz.* Tasks - Visualization
# ==============================================================================

@task
def viz_stats(c, feature_type="fooof_exponent", space="sensor", alpha=0.05,
              trial_type="alltrials", cmap=None, show=False, save=True):
    """Visualize saved statistical results as topographic or surface maps.

    For sensor space: creates a panel of topomaps (t-values).
    For source/atlas spaces: creates inflated brain surface maps (t-values).

    Colormap is resolved from config.yaml visualization section based on
    feature type. Use --cmap to override.

    Convention (for contrast colormaps):
        Red  = OUT > IN (positive t-values)
        Blue = IN > OUT (negative t-values)

    Args:
        feature_type: Feature to visualize (e.g., fooof_exponent, psd_alpha)
        space: Analysis space ('sensor', 'source', or atlas name like 'schaefer_400')
        alpha: Significance threshold for marking
        cmap: Override colormap (default: from config.yaml)
        show: Display the figure interactively
        save: Save the figure to reports/figures/

    Examples:
        invoke viz.stats --feature-type=fooof_exponent --show
        invoke viz.stats --feature-type=psd_alpha --space=schaefer_400
        invoke viz.stats --feature-type=psd --space=aparc.a2009s --save
        invoke viz.stats --feature-type=psd --cmap=coolwarm
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.run_viz_stats"]
    cmd.extend(["--feature-type", feature_type])
    cmd.extend(["--space", space])
    cmd.extend(["--alpha", str(alpha)])
    cmd.extend(["--trial-type", trial_type])
    if cmap:
        cmd.extend(["--cmap", cmap])
    if show:
        cmd.append("--show")
    if not save:
        cmd.append("--no-save")

    print(f"Running: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def viz_maps(c, metric, space="sensor", feature=None, feature_set=None,
             family=None, clf="lda", cv="logo", mode="univariate",
             test="paired_ttest", trial_type="alltrials", correction="auto",
             alpha=0.05, cmap=None, output_subdir=None, config="config.yaml"):
    """Unified rows-of-maps viz (stats + classification, sensor + atlas).

    One row of topomaps (sensor) or inflated-brain panels (source/atlas) per
    feature family. Auto-discovers result files for the chosen metric and
    prints exactly what to run when nothing is found.

    Args:
        metric: 'tval' | 'contrast' | 'balanced_accuracy' (see
            code.visualization.metrics).
        space: 'sensor', 'source', or atlas name (e.g., 'schaefer_400').
        feature: space-separated feature names (e.g., 'psd_alpha psd_theta').
        feature_set: shortcut family ('psds', 'psds_corrected', 'fooof',
            'complexity', 'all').
        family: filter rendering to one family when features span multiple.
        clf, cv, mode: classification result filters.
        test: statistics test filter.
        trial_type: trial-type variant to plot ('alltrials', 'correct',
            'lapse', ...).
        correction: which p-value correction for the significance mask
            ('auto', 'tmax', 'fdr_bh', 'bonferroni', 'uncorrected').
        alpha: significance threshold for the mask.
        cmap: override colormap (default: from metric).
        output_subdir: subfolder under reports/figures/ (default: 'classification').

    Examples:
        invoke viz.maps --metric=balanced_accuracy --space=sensor --feature-set=psds
        invoke viz.maps --metric=balanced_accuracy --space=schaefer_400 --feature-set=all
        invoke viz.maps --metric=tval --space=sensor --feature=fooof_exponent
        invoke viz.maps --metric=contrast --space=sensor --feature-set=psds --trial-type=lapse
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.run_viz",
           "--metric", metric, "--space", space,
           "--clf", clf, "--cv", cv, "--mode", mode,
           "--test", test, "--trial-type", trial_type,
           "--correction", correction,
           "--alpha", str(alpha), "--config", config]
    if feature:
        cmd.extend(["--feature"] + feature.split())
    if feature_set:
        cmd.extend(["--feature-set", feature_set])
    if family:
        cmd.extend(["--family", family])
    if cmap:
        cmd.extend(["--cmap", cmap])
    if output_subdir:
        cmd.extend(["--output-subdir", output_subdir])

    print(f"Running: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def spectra(c, space="sensor", stat_feature="fooof_exponent", select_by="corrected",
            subject=None, n_events_window=8, show=False, save=True,
            config="config.yaml"):
    """Reproduce Figure 3 C-F: the FOOOF spectral decomposition panel.

    Selects the most significant sensor/region from the FOOOF exponent
    group-statistics map, then renders, at that unit, the four-panel FOOOF
    figure for IN vs OUT:

        C  Raw spectrum (PSD)
        D  Aperiodic component
        E  Corrected spectrum (PSDc)
        F  Periodic components

    The selected unit name is written into panel C (plus an exponent t-map
    inset for sensor space). Curves are group means +/- SEM unless --subject
    pins a single subject (the manuscript's example-subject case).

    Args:
        space: 'sensor' or atlas name (e.g. 'schaefer_400').
        stat_feature: statistics map used to pick the unit (default fooof_exponent).
        select_by: rank by 'corrected' or 'uncorrected' p-values.
        subject: restrict spectra to one subject (default: group average).
        n_events_window: trials per welch window (welch desc suffix).
        show: display the figure interactively.
        save: write the figure to reports/figures/statistics/.

    Examples:
        invoke viz.spectra
        invoke viz.spectra --space=sensor --select-by=uncorrected
        invoke viz.spectra --subject=07
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.run_viz_spectra",
           "--space", space, "--stat-feature", stat_feature,
           "--select-by", select_by, "--n-events-window", str(n_events_window),
           "--config", config]
    if subject:
        cmd.extend(["--subject", str(subject)])
    if show:
        cmd.append("--show")
    if not save:
        cmd.append("--no-save")

    print(f"Running: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def stats_classif_panel(c, space="sensor", trial_type="alltrials",
                        correction="fdr", clf="logistic", cv="group",
                        alpha=0.05, n_events_window=8,
                        output=None, config="config.yaml"):
    """Render the paper-ready stats + classification multi-panel (Fig. 3).

    A single composite figure (panels A-J): per-band t-values and AUC for
    raw and corrected PSD, FOOOF parameter t-values and AUC, plus the
    IN-vs-OUT FOOOF spectral decomposition lines (raw, aperiodic,
    corrected, periodic). Significance is computed within each topomap
    (per feature), never pooled across bands/metrics.

    Args:
        space: 'sensor' (topomaps) or an atlas name like 'schaefer_400' /
            'aparc.a2009s' (inflated-brain panels). Vertex-level 'source'
            is not wired.
        trial_type: alltrials | correct | lapse.
        correction: significance mask -- fdr (default) | tmax | bonferroni |
            uncorrected. Always applied per-feature.
        clf, cv: classifier file selectors (default logistic/group). The
            panel is necessarily univariate.
        alpha: significance threshold for the mask (default 0.05).
        n_events_window: trials per welch window (welch desc suffix).
        output: override output path. Default writes to
            reports/figures/stats_classif_panel_space-<space>_type-<trial>_correction-<corr>.png.

    Examples:
        invoke viz.stats-classif-panel
        invoke viz.stats-classif-panel --trial-type=correct
        invoke viz.stats-classif-panel --trial-type=lapse --correction=tmax
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.stats_classif_panel",
           "--config", config,
           "--space", space,
           "--trial-type", trial_type,
           "--correction", correction,
           "--clf", clf, "--cv", cv,
           "--alpha", str(alpha),
           "--n-events-window", str(n_events_window)]
    if output:
        cmd.extend(["--output", output])

    print(f"Running: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


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


@task
def viz_auto(c, results_dir=None, metric=None, space=None, trial_type=None,
             clf=None, cv=None, mode=None, test=None, feature_set="all",
             correction="auto", alpha=0.05, cmap=None, dry_run=False,
             continue_on_error=True, config="config.yaml"):
    """Scan the results folder and render every available visualization.

    Walks <data_root>/<paths.results>/ for `statistics_<space>` and
    `classification_<space>` result folders, discovers which metrics, spaces,
    trial-types and (for classification) clf/cv/mode combinations actually have
    result files on disk, and runs `viz.maps` once per discovered combination.

    Every filter argument defaults to None = "discover and render all". Pass a
    value to restrict to it (e.g. --space=sensor renders only sensor maps);
    every other dimension is still auto-discovered.

    Args:
        results_dir: override the results root (default: from config.yaml).
        metric: restrict to one metric (tval | contrast | balanced_accuracy).
        space: restrict to one space (sensor | schaefer_400 | ...).
        trial_type: restrict to one trial-type (alltrials | correct | lapse | ...).
        clf, cv, mode: restrict classification combinations.
        test: restrict statistics test (default: discover all).
        feature_set: feature family set forwarded to viz.maps (default: all).
        correction, alpha, cmap: forwarded to viz.maps.
        dry_run: print the planned viz.maps commands without running them.
        continue_on_error: keep going if one render fails (default: True).

    Examples:
        invoke viz.auto                  # everything found on disk
        invoke viz.auto --dry-run        # preview the plan
        invoke viz.auto --space=sensor   # only sensor maps
        invoke viz.auto --metric=tval    # only t-value statistics maps
    """
    import re

    from code.utils.config import load_config
    from code.visualization.metrics import METRICS

    if results_dir:
        results_root = Path(results_dir).expanduser()
    else:
        results_root = Path(load_config(config)["paths"]["results"])

    if not results_root.is_dir():
        print(f"Results folder not found: {results_root}")
        return

    metric_names = [metric] if metric else list(METRICS)
    for mn in metric_names:
        if mn not in METRICS:
            raise ValueError(f"Unknown metric '{mn}'. Choices: {', '.join(METRICS)}")

    print("=" * 80)
    print(f"viz.auto | scanning {results_root}")
    print("=" * 80)

    # Each plan entry is a fully-resolved set of viz.maps arguments.
    plan = []
    for mn in metric_names:
        m = METRICS[mn]
        family = m.results_subdir.split("_{space}")[0]   # statistics | classification
        is_classif = "{clf}" in m.results_glob
        value_suffix = "_" + m.results_glob.rsplit("_", 1)[-1]   # _results.npz | _scores.npz

        for space_dir in sorted(results_root.glob(f"{family}_*")):
            if not space_dir.is_dir():
                continue
            disc_space = space_dir.name[len(family) + 1:]
            if space and disc_space != space:
                continue
            file_dir = results_root / m.results_subdir.format(space=disc_space)
            if not file_dir.is_dir():
                continue

            combos = set()
            for f in sorted(file_dir.glob("*" + value_suffix)):
                name = "_" + f.name
                tt = re.search(r"_type-([a-z_]+)" + re.escape(value_suffix), name)
                if not tt:
                    continue
                tt = tt.group(1)
                if trial_type and tt != trial_type:
                    continue
                if is_classif:
                    c_ = re.search(r"_clf-([^_]+)_cv-", name)
                    v_ = re.search(r"_cv-([^_]+)_mode-", name)
                    md_ = re.search(r"_mode-([^_]+)_type-", name)
                    fclf = c_.group(1) if c_ else "lda"
                    fcv = v_.group(1) if v_ else "logo"
                    fmode = md_.group(1) if md_ else "univariate"
                    if (clf and fclf != clf) or (cv and fcv != cv) or (mode and fmode != mode):
                        continue
                    combos.add((tt, fclf, fcv, fmode, None))
                else:
                    t_ = re.search(r"_test-(.+?)_path-", name)
                    ftest = t_.group(1) if t_ else "paired_ttest"
                    if test and ftest != test:
                        continue
                    combos.add((tt, None, None, None, ftest))

            for tt, fclf, fcv, fmode, ftest in sorted(combos):
                plan.append({
                    "metric": mn, "space": disc_space, "trial_type": tt,
                    "clf": fclf, "cv": fcv, "mode": fmode, "test": ftest,
                    "output_subdir": family,
                })

    if not plan:
        print("\nNo result files discovered — nothing to render.")
        print("Run analysis.stats / analysis.classify first, or check --results-dir.")
        return

    print(f"\nDiscovered {len(plan)} visualization(s) to render:")
    for p in plan:
        extra = (f"clf={p['clf']} cv={p['cv']} mode={p['mode']}"
                 if p["test"] is None else f"test={p['test']}")
        print(f"  - {p['metric']:18s} space={p['space']:14s} "
              f"type={p['trial_type']:12s} {extra}")

    python_exe = get_python_executable()
    failures = []
    for i, p in enumerate(plan, 1):
        cmd = [python_exe, "-m", "code.visualization.run_viz",
               "--metric", p["metric"], "--space", p["space"],
               "--feature-set", feature_set,
               "--trial-type", p["trial_type"],
               "--correction", correction, "--alpha", str(alpha),
               "--output-subdir", p["output_subdir"],
               "--config", config]
        if p["clf"]:
            cmd.extend(["--clf", p["clf"]])
        if p["cv"]:
            cmd.extend(["--cv", p["cv"]])
        if p["mode"]:
            cmd.extend(["--mode", p["mode"]])
        if p["test"]:
            cmd.extend(["--test", p["test"]])
        if cmap:
            cmd.extend(["--cmap", cmap])

        print(f"\n{'#' * 80}\n# [{i}/{len(plan)}] {p['metric']} | "
              f"space={p['space']} | type={p['trial_type']}\n{'#' * 80}")
        print(f"Running: {' '.join(cmd)}\n")
        if dry_run:
            continue
        result = c.run(" ".join(cmd), pty=True,
                       env=get_env_with_pythonpath(), warn=continue_on_error)
        if result is not None and getattr(result, "exited", 0) != 0:
            failures.append((p, f"exit {result.exited}"))

    print("\n" + "=" * 80)
    if dry_run:
        print(f"Dry run — {len(plan)} visualization(s) planned, none rendered.")
    elif failures:
        print(f"Done with {len(failures)}/{len(plan)} failure(s):")
        for p, msg in failures:
            print(f"  - {p['metric']} space={p['space']} type={p['trial_type']}: {msg}")
    else:
        print(f"All {len(plan)} visualization(s) rendered.")
    print("=" * 80)


# ==============================================================================
# Helper Functions (Private)
# ==============================================================================

def _preprocess_local(c, subject, runs=None, bids_root=None, log_level="INFO", skip_existing=True, crop=None):
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

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


def _preprocess_slurm(c, subject=None, runs=None, bids_root=None,
                      log_level="INFO", skip_existing=True, dry_run=False):
    """Submit preprocessing jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import (
        render_slurm_script, submit_slurm_job, submit_job_array, save_job_manifest,
    )

    print("\n[SLURM Mode] Submitting preprocessing job array to cluster\n")

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    if not bids_root:
        bids_root = Path(config["paths"]["bids"])
    else:
        bids_root = Path(bids_root)

    slurm_config = config["computing"]["slurm"]
    preproc_resources = slurm_config["preprocessing"]

    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "preprocessing"
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "preprocessing"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    venv_path = PROJECT_ROOT / config["paths"]["venv"]
    base_resources = dict(
        account=slurm_config["account"],
        partition=slurm_config.get("partition", ""),
        cpus=preproc_resources["cpus"],
        mem=preproc_resources["mem"],
        time=preproc_resources["time"],
        log_dir=str(log_dir),
        venv_path=str(venv_path),
        project_root=str(PROJECT_ROOT),
    )
    max_concurrent = slurm_config.get("array_throttle", 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Subjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total array tasks: {len(subjects) * len(run_list)}")

    # Render one per-task script per (subject, run); a single job array
    # dispatches them all instead of hundreds of separate sbatch submissions.
    task_scripts = []
    for subj in subjects:
        for run in run_list:
            job_name = f"preproc_sub-{subj}_run-{run}"
            context = {
                **base_resources,
                "job_name": job_name,
                "timestamp": timestamp,
                "subject": subj,
                "run": run,
                "bids_root": str(bids_root),
                "log_level": log_level,
                "skip_existing": skip_existing,
            }
            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("preprocessing.sh.j2", context, output_path=script_path)
            task_scripts.append(script_path)

    array_job_id = submit_job_array(
        task_scripts, "preproc_array", base_resources, script_dir, timestamp,
        max_concurrent=max_concurrent, dry_run=dry_run,
    )

    # One report job after the whole preprocessing array finishes.
    # aggregate_reports --dataset regenerates every subject report AND the
    # dataset report, so a single afterany-dependent job replaces the old
    # per-subject report fan-out (and avoids concurrent dataset-report writes).
    report_resources = slurm_config.get("report", {"time": "00:15:00", "cpus": 2, "mem": "4G"})
    report_ctx = {
        **base_resources,
        "cpus": report_resources["cpus"],
        "mem": report_resources["mem"],
        "time": report_resources["time"],
        "job_name": "preproc_report",
        "timestamp": timestamp,
    }
    report_script = script_dir / f"preproc_report_{timestamp}.sh"
    render_slurm_script("preprocess_report.sh.j2", report_ctx, output_path=report_script)
    report_job_id = submit_slurm_job(
        report_script,
        dependencies=[array_job_id] if array_job_id else None,
        dep_type="afterany",  # run even if some preprocessing tasks failed
        dry_run=dry_run,
    )

    all_job_ids = [j for j in (array_job_id, report_job_id) if j]
    if all_job_ids:
        manifest_path = log_dir / f"preprocessing_manifest_{timestamp}.json"
        save_job_manifest(all_job_ids, manifest_path, metadata={
            "stage": "preprocessing",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
            "preprocessing_array_job_id": array_job_id,
            "report_job_id": report_job_id,
        })
        print(f"\n✓ Submitted preprocessing array ({len(task_scripts)} tasks)"
              f" + 1 dependent report job")
    elif dry_run:
        print(f"\n[DRY RUN] Would submit a {len(task_scripts)}-task preprocessing "
              f"array + 1 report job")


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
    from code.utils.slurm import (
        render_slurm_script, save_job_manifest, submit_slurm_job, submit_job_array,
    )

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

    if bids_root:
        bids_root = Path(bids_root)
    else:
        bids_root = Path(config["paths"]["bids"])

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "source_reconstruction"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "source_reconstruction"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSubjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total array tasks: {len(subjects) * len(run_list)}")

    base_resources = dict(
        account=slurm_config["account"],
        partition=slurm_config.get("partition", ""),
        cpus=src_recon_resources["cpus"],
        mem=src_recon_resources["mem"],
        time=src_recon_resources["time"],
        log_dir=str(log_dir),
        venv_path=str(venv_path),
        project_root=str(PROJECT_ROOT),
    )
    max_concurrent = slurm_config.get("array_throttle", 0)

    task_scripts = []
    for subj in subjects:
        for run in run_list:
            job_name = f"srcrecon_sub-{subj}_run-{run}"
            context = {
                **base_resources,
                "job_name": job_name,
                "timestamp": timestamp,
                "subject": subj,
                "run": run,
                "bids_root": str(bids_root),
                "log_level": log_level,
                "skip_existing": skip_existing,
            }
            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("source_reconstruction.sh.j2", context, output_path=script_path)
            task_scripts.append(script_path)

    array_job_id = submit_job_array(
        task_scripts, "srcrecon_array", base_resources, script_dir, timestamp,
        max_concurrent=max_concurrent, dry_run=dry_run,
    )

    if array_job_id:
        manifest_path = log_dir / f"source_reconstruction_manifest_{timestamp}.json"
        save_job_manifest([array_job_id], manifest_path, metadata={
            "stage": "source_reconstruction",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted source reconstruction array ({len(task_scripts)} tasks)")
    elif dry_run:
        print(f"\n[DRY RUN] Would submit a {len(task_scripts)}-task array")


def _atlas_slurm(c, subject=None, runs=None, atlases=None,
                 skip_existing=True, dry_run=False):
    """Submit atlas application jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import (
        render_slurm_script, save_job_manifest, submit_slurm_job, submit_job_array,
    )

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
    print(f"Total array tasks: {len(subjects)}")  # One task per subject (all runs)

    base_resources = dict(
        account=slurm_config["account"],
        partition=slurm_config.get("partition", ""),
        cpus=atlas_resources["cpus"],
        mem=atlas_resources["mem"],
        time=atlas_resources["time"],
        log_dir=str(log_dir),
        venv_path=str(venv_path),
        project_root=str(PROJECT_ROOT),
    )
    max_concurrent = slurm_config.get("array_throttle", 0)

    task_scripts = []
    for subj in subjects:
        job_name = f"atlas_sub-{subj}"
        context = {
            **base_resources,
            "job_name": job_name,
            "timestamp": timestamp,
            "subject": subj,
            "runs": " ".join(run_list),
            "atlases": atlases,
            "skip_existing": skip_existing,
        }
        script_path = script_dir / f"{job_name}_{timestamp}.sh"
        render_slurm_script("atlas.sh.j2", context, output_path=script_path)
        task_scripts.append(script_path)

    array_job_id = submit_job_array(
        task_scripts, "atlas_array", base_resources, script_dir, timestamp,
        max_concurrent=max_concurrent, dry_run=dry_run,
    )

    if array_job_id:
        manifest_path = log_dir / f"atlas_manifest_{timestamp}.json"
        save_job_manifest([array_job_id], manifest_path, metadata={
            "stage": "atlas",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted atlas application array ({len(task_scripts)} tasks)")
    elif dry_run:
        print(f"\n[DRY RUN] Would submit a {len(task_scripts)}-task array")


def _features_local(c, feature_type, subject, runs=None, space="sensor",
                    skip_existing=True, complexity_types=None, log_level="INFO",
                    n_events_window=8):
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
            cmd.extend(["--n-events-window", str(n_events_window)])
            cmd.extend(["--log-level", log_level])
            cmd.append("--skip-existing" if skip_existing else "--no-skip-existing")

        elif feature_type == "fooof":
            cmd = [python_exe, "-m", "code.features.compute_fooof"]
            cmd.extend(["--subject", subject, "--run", run, "--space", space])
            cmd.extend(["--n-events-window", str(n_events_window)])
            cmd.extend(["--log-level", log_level])
            cmd.append("--skip-existing" if skip_existing else "--no-skip-existing")

        elif feature_type == "complexity":
            cmd = [python_exe, "-m", "code.features.compute_complexity"]
            cmd.extend(["--subject", subject, "--run", run, "--space", space])
            cmd.extend(["--n-events-window", str(n_events_window)])
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
                    skip_existing=True, complexity_types=None, log_level="INFO", dry_run=False,
                    n_events_window=8):
    """Submit feature extraction jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import (
        render_slurm_script, save_job_manifest, submit_slurm_job, submit_job_array,
    )

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
    print(f"Total array tasks: {len(subjects) * len(run_list)}")

    base_resources = dict(
        account=slurm_config["account"],
        partition=slurm_config.get("partition", ""),
        cpus=features_resources["cpus"],
        mem=features_resources["mem"],
        time=features_resources["time"],
        log_dir=str(log_dir),
        venv_path=str(venv_path),
        project_root=str(PROJECT_ROOT),
    )
    max_concurrent = slurm_config.get("array_throttle", 0)

    task_scripts = []
    for subj in subjects:
        for run in run_list:
            job_name = f"{feature_type}_sub-{subj}_run-{run}_{space}"
            context = {
                **base_resources,
                "job_name": job_name,
                "timestamp": timestamp,
                "feature_type": feature_type,
                "subject": subj,
                "run": run,
                "space": space,
                "log_level": log_level,
                "skip_existing": skip_existing,
                "complexity_types": complexity_types,
                "n_events_window": n_events_window,
            }
            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("features.sh.j2", context, output_path=script_path)
            task_scripts.append(script_path)

    array_job_id = submit_job_array(
        task_scripts, f"{feature_type}_array", base_resources, script_dir, timestamp,
        max_concurrent=max_concurrent, dry_run=dry_run,
    )

    if array_job_id:
        manifest_path = log_dir / f"{feature_type}_manifest_{timestamp}.json"
        save_job_manifest([array_job_id], manifest_path, metadata={
            "stage": f"features_{feature_type}",
            "feature_type": feature_type,
            "space": space,
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted {feature_type} feature extraction array "
              f"({len(task_scripts)} tasks)")
    elif dry_run:
        print(f"\n[DRY RUN] Would submit a {len(task_scripts)}-task array")


def _stats_slurm(c, feature_list, space="sensor", test="paired_ttest",
                 correction="fdr", alpha=0.05, n_permutations=10000,
                 n_jobs=1, analysis_level="both", aggregate="median",
                 visualize=False,
                 trial_type="alltrials", zoning="per-run",
                 n_events_window=8,
                 slurm_time=None, slurm_mem=None, slurm_cpus=None,
                 dry_run=False):
    """Submit one statistics job per feature to SLURM (each runs all
    requested levels in-process)."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import (
        render_slurm_script, save_job_manifest, submit_slurm_job, submit_job_array,
    )

    print(f"\n[SLURM Mode] Submitting statistics jobs to cluster\n")

    config = load_config()
    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    stats_resources = slurm_config.get("statistics", {})
    if not stats_resources:
        print("ERROR: No statistics resources in config.yaml (computing.slurm.statistics)")
        return

    features = list(feature_list)
    if not features:
        print("ERROR: no features to submit")
        return

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "statistics"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "statistics"
    log_dir.mkdir(parents=True, exist_ok=True)

    levels = ["epoch", "average"] if analysis_level == "both" else [analysis_level]

    print(f"Space: {space}  Test: {test}  Correction: {correction}")
    print(f"n_permutations: {n_permutations}  alpha: {alpha}  "
          f"analysis_level: {analysis_level} ({'+'.join(levels)})")
    print(f"Jobs: {len(features)} feature(s) × {len(levels)} level(s) "
          f"= {len(features) * len(levels)} (one job array per level)")

    cpus_resolved = slurm_cpus if slurm_cpus is not None else stats_resources["cpus"]
    mem_resolved = slurm_mem if slurm_mem is not None else stats_resources["mem"]
    time_resolved = slurm_time if slurm_time is not None else stats_resources["time"]
    print(f"Resources: cpus={cpus_resolved}  mem={mem_resolved}  time={time_resolved}")

    base_resources = dict(
        account=slurm_config["account"],
        partition=slurm_config.get("partition", ""),
        cpus=cpus_resolved,
        mem=mem_resolved,
        time=time_resolved,
        log_dir=str(log_dir),
        venv_path=str(venv_path),
        project_root=str(PROJECT_ROOT),
    )

    max_concurrent = slurm_config.get("array_throttle", 0)
    all_ids: List[str] = []
    all_scripts: List[Path] = []
    for level in levels:
        level_scripts = []
        for feat in features:
            job_name = f"stats_{feat}_{space}_{test}_{trial_type}_level-{level}"
            context = {
                **base_resources,
                "job_name": job_name,
                "feature_label": feat,
                "space": space,
                "test": test,
                "correction": correction,
                "alpha": alpha,
                "n_permutations": n_permutations,
                "n_jobs": n_jobs,
                "analysis_level": level,
                "aggregate": aggregate,
                "visualize": visualize,
                "trial_type": trial_type,
                "zoning": zoning,
                "n_events_window": n_events_window,
                "timestamp": timestamp,
            }
            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("statistics.sh.j2", context, output_path=script_path)
            level_scripts.append(script_path)

        array_job_id = submit_job_array(
            level_scripts,
            f"stats_array_{trial_type}_level-{level}", base_resources,
            script_dir, timestamp, max_concurrent=max_concurrent,
            dry_run=dry_run,
        )
        if array_job_id:
            all_ids.append(array_job_id)
        all_scripts.extend(level_scripts)

    if all_ids:
        manifest_path = log_dir / f"statistics_manifest_{trial_type}_{timestamp}.json"
        save_job_manifest(all_ids, manifest_path, metadata={
            "stage": "statistics",
            "space": space,
            "test": test,
            "correction": correction,
            "alpha": alpha,
            "n_permutations": n_permutations,
            "analysis_levels": levels,
            "features": features,
            "trial_type": trial_type,
            "timestamp": timestamp,
        })
        print(f"\n✓ Submitted {len(all_ids)} statistics array(s) "
              f"({len(all_scripts)} tasks total); manifest: {manifest_path}")
    elif dry_run:
        print(
            f"\n[DRY RUN] Would submit {len(levels)} statistics array(s) "
            f"({len(all_scripts)} task(s))"
        )


def _classify_slurm(c, feature_list, clf="logistic", cv="auto",
                    space="sensor", mode="univariate", n_permutations=1000,
                    no_balance=False, n_jobs=-1, continue_on_error=False,
                    combine_features=False, importances=False, label=None,
                    n_chunks=1, seed=42, aggregate=True, delete_chunks=False,
                    trial_type="alltrials", zoning="per-run",
                    n_events_window=8, analysis_level="both",
                    standardize="auto",
                    slurm_time=None, slurm_mem=None, slurm_cpus=None,
                    dry_run=False):
    """Submit classification jobs to SLURM.

    Job fan-out:
      - One job per (feature, level). When --analysis-level=both, each
        feature yields two jobs (one per level).
      - When n_chunks > 1 (univariate only), each classification is split into
        n_chunks parallel jobs over the spatial dimension. All chunks share the
        same seed so the permutation y-shuffle sequence is identical, and an
        aggregation job (afterok dependency) merges them into the final output.
    """
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import (
        render_slurm_script, save_job_manifest, submit_slurm_job, submit_job_array,
    )
    from code.classification.run_classification import _cv_for_level

    print(f"\n[SLURM Mode] Submitting classification jobs to cluster\n")

    levels = ["epoch", "average"] if analysis_level == "both" else [analysis_level]

    def _cv_for_lvl(level: str) -> str:
        return cv if cv != "auto" else _cv_for_level(level)

    config = load_config()
    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    classification_resources = slurm_config.get("classification", {})
    if not classification_resources:
        print("ERROR: No classification resources in config.yaml (computing.slurm.classification)")
        return

    if n_chunks > 1 and mode != "univariate":
        print("ERROR: --n-chunks requires --mode=univariate (multivariate has no spatial dim to split)")
        return

    features = list(feature_list)
    if not features:
        print("ERROR: no features to submit")
        return

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "classification"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "classification"
    log_dir.mkdir(parents=True, exist_ok=True)

    # One classification per feature, OR one combined classification.
    # Each entry: (label_for_filename, feature_args_for_cli, feature_list)
    if combine_features:
        classifications = [(
            label or f"combined-{len(features)}",
            "--feature " + " ".join(features),
            features,
        )]
    else:
        classifications = [(feat, f"--feature {feat}", [feat]) for feat in features]

    print(f"Space: {space}  Mode: {mode}  Classifier: {clf}  CV: {cv}")
    print(f"n_permutations: {n_permutations}  combine_features: {combine_features}  "
          f"analysis_level: {analysis_level} ({'+'.join(levels)})")
    print(f"Classifications: {len(classifications)} × {len(levels)} level(s)  "
          f"chunks per classification: {n_chunks}")

    # Resolve resources: per-invocation overrides win over config defaults
    cpus_resolved = slurm_cpus if slurm_cpus is not None else classification_resources["cpus"]
    mem_resolved = slurm_mem if slurm_mem is not None else classification_resources["mem"]
    time_resolved = slurm_time if slurm_time is not None else classification_resources["time"]
    print(f"Resources: cpus={cpus_resolved}  mem={mem_resolved}  time={time_resolved}")

    base_resources = dict(
        account=slurm_config["account"],
        partition=slurm_config.get("partition", ""),
        cpus=cpus_resolved,
        mem=mem_resolved,
        time=time_resolved,
        log_dir=str(log_dir),
        venv_path=str(venv_path),
        project_root=str(PROJECT_ROOT),
    )

    max_concurrent = slurm_config.get("array_throttle", 0)

    # One job array per analysis level (so the user can monitor / re-submit
    # epoch and average independently). Each array contains (feature × chunk)
    # tasks for that level, plus an optional aggregation array when chunked.
    all_classify_ids: List[str] = []
    all_agg_ids: List[str] = []
    all_classify_scripts: List[Path] = []
    all_agg_jobs: List[Tuple[str, str]] = []

    for level in levels:
        level_scripts = []
        level_agg_jobs: List[Tuple[str, str]] = []
        for feat_label, feature_args, feat_list in classifications:
            for chunk_idx in range(n_chunks):
                chunk_suffix = f"_chunk-{chunk_idx}of{n_chunks}" if n_chunks > 1 else ""
                job_name = (
                    f"classify_{feat_label}_{space}_{mode}_{clf}_"
                    f"{trial_type}_level-{level}{chunk_suffix}"
                )
                context = {
                    **base_resources,
                    "job_name": job_name,
                    "feature_label": feat_label,
                    "feature_args": feature_args,
                    "space": space,
                    "mode": mode,
                    "clf": clf,
                    "cv": cv,
                    "analysis_level": level,
                    "n_permutations": n_permutations,
                    "n_jobs": n_jobs,
                    "no_balance": no_balance,
                    "combine_features": combine_features,
                    "importances": importances,
                    "continue_on_error": continue_on_error,
                    "label": label,
                    "n_chunks": n_chunks,
                    "chunk_idx": chunk_idx,
                    "seed": seed,
                    "trial_type": trial_type,
                    "zoning": zoning,
                    "n_events_window": n_events_window,
                    "standardize": standardize,
                    "timestamp": timestamp,
                }
                script_path = script_dir / f"{job_name}_{timestamp}.sh"
                render_slurm_script("classification.sh.j2", context, output_path=script_path)
                level_scripts.append(script_path)
            if n_chunks > 1 and aggregate:
                level_agg_jobs.append((feat_label, level))

        classify_array_id = submit_job_array(
            level_scripts,
            f"classify_array_{trial_type}_level-{level}", base_resources,
            script_dir, timestamp, max_concurrent=max_concurrent,
            dry_run=dry_run,
        )

        # Chunk aggregation (only when n_chunks > 1): one task per
        # (feature, level), in a second array depending on this level's
        # classification array.
        agg_array_id = None
        if level_agg_jobs:
            agg_resources = dict(base_resources, cpus=1, mem="8G", time="0:30:00")
            agg_scripts = []
            for feat_label, _lvl in level_agg_jobs:
                cv_resolved = _cv_for_lvl(level)
                agg_job_name = (
                    f"aggregate_{feat_label}_{space}_{mode}_{clf}_"
                    f"{trial_type}_level-{level}"
                )
                agg_context = {
                    **agg_resources,
                    "job_name": agg_job_name,
                    "timestamp": timestamp,
                    "feature_label": feat_label,
                    "space": space,
                    "mode": mode,
                    "clf": clf,
                    "cv": cv_resolved,
                    "analysis_level": level,
                    "combined": combine_features,
                    "delete_chunks": delete_chunks,
                    "trial_type": trial_type,
                }
                agg_script = script_dir / f"{agg_job_name}_{timestamp}.sh"
                render_slurm_script(
                    "classification_aggregate.sh.j2", agg_context,
                    output_path=agg_script,
                )
                agg_scripts.append(agg_script)
            agg_array_id = submit_job_array(
                agg_scripts,
                f"aggregate_array_{trial_type}_level-{level}", agg_resources,
                script_dir, timestamp, max_concurrent=max_concurrent,
                dependencies=[classify_array_id] if classify_array_id else None,
                dep_type="afterok", dry_run=dry_run,
            )

        if classify_array_id:
            all_classify_ids.append(classify_array_id)
        if agg_array_id:
            all_agg_ids.append(agg_array_id)
        all_classify_scripts.extend(level_scripts)
        all_agg_jobs.extend(level_agg_jobs)

    all_job_ids = all_classify_ids + all_agg_ids
    if all_job_ids:
        manifest_path = log_dir / f"classification_manifest_{trial_type}_{timestamp}.json"
        save_job_manifest(all_job_ids, manifest_path, metadata={
            "stage": "classification",
            "space": space,
            "mode": mode,
            "clf": clf,
            "cv": cv,
            "n_permutations": n_permutations,
            "combine_features": combine_features,
            "n_chunks": n_chunks,
            "analysis_levels": levels,
            "classify_array_job_ids": all_classify_ids,
            "aggregate_array_job_ids": all_agg_ids,
            "features": features,
            "trial_type": trial_type,
            "timestamp": timestamp,
        })
        msg = (
            f"\n✓ Submitted {len(all_classify_ids)} classification array(s) "
            f"({len(all_classify_scripts)} tasks total)"
        )
        if all_agg_jobs:
            msg += (
                f" + {len(all_agg_ids)} aggregation array(s) "
                f"({len(all_agg_jobs)} tasks total)"
            )
        print(msg + f"; manifest: {manifest_path}")
    elif dry_run:
        n_total = len(classifications) * len(levels) * n_chunks
        print(
            f"\n[DRY RUN] Would submit {len(levels)} classification array(s) "
            f"({n_total} task(s)) + optional per-level aggregation arrays"
        )


@task
def classify_multifeature(c, features="all", label=None,
                          clf="logistic", cv="logo", space="sensor",
                          axis="all", importance="permutation",
                          n_permutations=1000, n_jobs=-1,
                          per_feature_scale=True, no_balance=False,
                          seed=42, trial_type="alltrials",
                          zoning="per-run", n_events_window=8,
                          standardize="per-subject", analysis_level="epoch",
                          n_chunks=1, chunk_idx=0,
                          importance_n_repeats=5,
                          keep_bad_trials=False, output_dir=None,
                          aggregate=True, config="config.yaml"):
    """Run multi-feature classification (IN vs OUT) along one or all axes.

    Loads all selected features as one stacked tensor (n_trials, n_spatial,
    n_features) and runs IN-vs-OUT decoding along one or more axes:

      - per-spatial : per-sensor/ROI classifier using all features as input.
                      Output: scores(n_spatial,), importance(n_spatial, n_feat).
      - per-feature : per-feature classifier using all spatial units as input.
                      Output: scores(n_features,), importance(n_features, n_sp).
      - per-cell    : one classifier per (sensor, feature) pair.
                      Output: scores(n_spatial, n_features).
      - joint       : one classifier on flattened (n_spatial * n_features).
                      Output: scalar score, importance(n_spatial, n_feat).
      - all (default): run all four axes, write one file per axis.

    `--features` follows the same conventions as `analysis.classify` (single
    name, space-separated list, or shortcut: psds, psds_corrected, fooof,
    complexity, all).

    Feature-importance backend (`--importance`):
      - permutation (default): model-agnostic, averaged over CV test folds.
      - coef: signed coefficients (linear models only: lda, logistic, svm).
      - tree: feature_importances_ (rf only).
      - none: skip importance.

    Examples:
        # All four axes, all features, sensor space, logreg + permutation importance
        invoke analysis.classify-multifeature

        # Per-feature axis only on schaefer_400, RF + tree importance
        invoke analysis.classify-multifeature --axis=per-feature \\
            --space=schaefer_400 --clf=rf --importance=tree

        # Joint axis, all features, with bundled aggregation after
        invoke analysis.classify-multifeature --axis=joint --features=all
    """
    print("=" * 80)
    print("Multi-Feature Classification Analysis")
    print("=" * 80)

    feature_list = resolve_features(features)
    print(f"Features ({len(feature_list)}): {' '.join(feature_list)}")
    print(f"Axis: {axis}  clf: {clf}  importance: {importance}  space: {space}")

    if n_chunks > 1 and axis not in ("per-cell",):
        print(
            f"NOTE: --n-chunks={n_chunks} only takes effect for --axis=per-cell."
        )

    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.classification.run_multifeature",
           "--feature", *feature_list]
    cmd.extend(["--clf", clf])
    cmd.extend(["--cv", cv])
    cmd.extend(["--space", space])
    cmd.extend(["--axis", axis])
    cmd.extend(["--importance", importance])
    cmd.extend(["--importance-n-repeats", str(importance_n_repeats)])
    cmd.extend(["--n-permutations", str(n_permutations)])
    cmd.extend(["--n-jobs", str(n_jobs)])
    cmd.extend(["--trial-type", trial_type])
    cmd.extend(["--zoning", zoning])
    cmd.extend(["--n-events-window", str(n_events_window)])
    cmd.extend(["--standardize", standardize])
    cmd.extend(["--analysis-level", analysis_level])
    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--n-chunks", str(n_chunks)])
    cmd.extend(["--chunk-idx", str(chunk_idx)])
    cmd.extend(["--config", config])
    if not per_feature_scale:
        cmd.append("--no-per-feature-scale")
    if no_balance:
        cmd.append("--no-balance")
    if keep_bad_trials:
        cmd.append("--keep-bad-trials")
    if label:
        cmd.extend(["--label", label])
    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())

    if aggregate and axis == "all":
        actual_label = label or f"combined-{len(feature_list)}"
        agg_cmd = [
            python_exe, "-m", "code.classification.aggregate_multifeature",
            "--label", actual_label,
            "--space", space,
            "--clf", clf,
            "--cv", cv,
            "--importance", importance,
            "--trial-type", trial_type,
            "--analysis-level", analysis_level,
            "--config", config,
        ]
        if output_dir:
            agg_cmd.extend(["--output-dir", output_dir])
        print(f"\nAggregating bundle: {' '.join(agg_cmd)}\n")
        c.run(" ".join(agg_cmd), pty=True, env=get_env_with_pythonpath())


@task
def classify_multifeature_aggregate(c, label, space, clf="logistic", cv="logo",
                                    importance="permutation",
                                    trial_type="alltrials",
                                    analysis_level="epoch",
                                    axes=None,
                                    output_dir=None, config="config.yaml"):
    """Bundle per-axis multi-feature classification outputs into one file.

    Examples:
        invoke analysis.classify-multifeature-aggregate --label=combined-19 --space=sensor
    """
    python_exe = get_python_executable()
    cmd = [
        python_exe, "-m", "code.classification.aggregate_multifeature",
        "--label", label,
        "--space", space,
        "--clf", clf,
        "--cv", cv,
        "--importance", importance,
        "--trial-type", trial_type,
        "--analysis-level", analysis_level,
        "--config", config,
    ]
    if axes:
        cmd.extend(["--axes", *axes.split()])
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def classify_aggregate(c, feature, space, clf="lda", cv="logo",
                       mode="univariate", combined=False,
                       delete_chunks=False, trial_type="alltrials",
                       config="config.yaml"):
    """Manually aggregate per-chunk classification outputs.

    Use this if --aggregate=False was used at submission, or if the afterok
    aggregator job failed and chunk files are still on disk.

    Examples:
        invoke analysis.classify-aggregate --feature=psd_alpha --space=sensor
        invoke analysis.classify-aggregate --feature=combined-10 --space=schaefer_400 --combined
    """
    python_exe = get_python_executable()
    cmd = [
        python_exe, "-m", "code.classification.aggregate_chunks",
        "--feature", feature,
        "--space", space,
        "--mode", mode,
        "--clf", clf,
        "--cv", cv,
        "--trial-type", trial_type,
        "--config", config,
    ]
    if combined:
        cmd.append("--combined")
    if delete_chunks:
        cmd.append("--delete-chunks")
    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


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
pipeline.add_task(preprocess_report, name="preprocess-report")
pipeline.add_task(source_recon, name="source-recon")
pipeline.add_task(atlas)
pipeline.add_collection(features)  # Nested: pipeline.features.*


@task
def slurm_jobs(c, pattern=None, state=None, user=None):
    """List your running/queued SLURM jobs, optionally filtered by name glob.

    Args:
        pattern: glob-style name filter (e.g. 'classify_*', '*chunk-0*'). Default: all.
        state: filter by SLURM state code (R, PD, CG, ...). Default: all.
        user: SLURM username to query (default: current user).

    Examples:
        invoke slurm.jobs
        invoke slurm.jobs --pattern='classify_*'
        invoke slurm.jobs --pattern='aggregate_*' --state=PD
    """
    import fnmatch
    from code.utils.slurm import get_user_jobs

    jobs = get_user_jobs(user=user)
    if pattern:
        jobs = [j for j in jobs if fnmatch.fnmatch(j["JobName"], pattern)]
    if state:
        jobs = [j for j in jobs if j["State"] == state]

    if not jobs:
        print("No matching jobs.")
        return

    print(f"{'JobID':>10}  {'State':>5}  {'Time':>10}  {'CPUs':>4}  {'Mem':>6}  Name")
    print("-" * 80)
    for j in jobs:
        print(
            f"{j['JobID']:>10}  {j['State']:>5}  {j['Time']:>10}  "
            f"{j['CPUs']:>4}  {j['Memory']:>6}  {j['JobName']}"
        )
    print(f"\n{len(jobs)} job(s)")


@task
def slurm_cancel(c, pattern=None, job_ids=None, state=None, user=None,
                 dry_run=False, yes=False):
    """Cancel SLURM jobs matching a name glob (or explicit IDs).

    Safety: by default, prints what would be cancelled and asks for confirmation.
    Pass --yes to skip the prompt, or --dry-run to print without cancelling.

    Args:
        pattern: glob-style name filter (e.g. 'classify_psd_*', '*chunk-*').
        job_ids: comma- or space-separated explicit job IDs (overrides pattern matching).
        state: optional SLURM state code filter (R, PD, CG, ...). Useful with pattern.
        user: SLURM username to query (default: current user).
        dry_run: just print the matches; don't cancel.
        yes: skip the confirmation prompt and cancel immediately.

    Examples:
        invoke slurm.cancel --pattern='classify_psd_*' --dry-run
        invoke slurm.cancel --pattern='aggregate_*' --state=PD --yes
        invoke slurm.cancel --pattern='*chunk-0of*'
        invoke slurm.cancel --job-ids='123,124,125'
    """
    import fnmatch
    from code.utils.slurm import cancel_job, get_user_jobs

    if not pattern and not job_ids:
        print("ERROR: pass --pattern or --job-ids")
        return

    targets: list = []
    if job_ids:
        ids = [s for s in job_ids.replace(",", " ").split() if s]
        # If state/pattern also given, look up names from squeue and apply filters
        if state or pattern:
            all_jobs = {j["JobID"]: j for j in get_user_jobs(user=user)}
            for jid in ids:
                if jid not in all_jobs:
                    print(f"  ! {jid}: not in queue, skipping")
                    continue
                j = all_jobs[jid]
                if state and j["State"] != state:
                    continue
                if pattern and not fnmatch.fnmatch(j["JobName"], pattern):
                    continue
                targets.append(j)
        else:
            targets = [{"JobID": jid, "JobName": "(unknown)", "State": "?"} for jid in ids]
    else:
        jobs = get_user_jobs(user=user)
        targets = [j for j in jobs if fnmatch.fnmatch(j["JobName"], pattern)]
        if state:
            targets = [j for j in targets if j["State"] == state]

    if not targets:
        print("No matching jobs.")
        return

    print(f"Targets ({len(targets)}):")
    for j in targets:
        print(f"  {j['JobID']:>10}  [{j.get('State', '?'):>3}]  {j['JobName']}")

    if dry_run:
        print(f"\n[DRY RUN] Would cancel {len(targets)} job(s)")
        return

    if not yes:
        try:
            ans = input(f"\nCancel {len(targets)} job(s)? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    cancelled = 0
    for j in targets:
        if cancel_job(j["JobID"]):
            cancelled += 1
    print(f"\n✓ Cancelled {cancelled}/{len(targets)} job(s)")


# Analysis tasks
analysis = Collection("analysis")
analysis.add_task(stats)
analysis.add_task(classify)
analysis.add_task(classify_aggregate, name="classify-aggregate")
analysis.add_task(classify_multifeature, name="classify-multifeature")
analysis.add_task(classify_multifeature_aggregate,
                  name="classify-multifeature-aggregate")

# Visualization tasks
viz = Collection("viz")
viz.add_task(viz_stats, name="stats")
viz.add_task(viz_maps, name="maps")
viz.add_task(viz_auto, name="auto")
viz.add_task(spectra)
viz.add_task(stats_classif_panel, name="stats-classif-panel")
viz.add_task(behavior)

# SLURM job-management tasks
slurm = Collection("slurm")
slurm.add_task(slurm_jobs, name="jobs")
slurm.add_task(slurm_cancel, name="cancel")

# Build main namespace
namespace = Collection()
namespace.add_collection(dev)
namespace.add_collection(env)
namespace.add_collection(get)
namespace.add_collection(pipeline)
namespace.add_collection(analysis)
namespace.add_collection(viz)
namespace.add_collection(slurm)


# Default task
@task(default=True)
def help_task(c):
    """Show available tasks (default task)."""
    c.run("invoke --list")


namespace.add_task(help_task, name="help")
