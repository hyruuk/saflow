# Saflow Refactoring Progress Tracker

**Last Updated**: 2026-01-31
**Current Phase**: Phase 4 - Feature Extraction (IN PROGRESS)
**Analysis Focus**: Unified sensor/source/atlas analysis
**Status**: ✅ Stages 0-2 Complete | ✅ Core Feature Extraction Complete | 📋 Statistics & Classification Pending

---

## Key Context for Resuming

### What We're Doing
Complete refactoring of cc_saflow → saflow with:
- Clean configuration system (no hardcoded paths)
- Proper logging (no print statements)
- Modern Python packaging (pyproject.toml)
- Sensor/source analysis space architecture
- BIDS-compliant organization

### Critical Paths
- **Old codebase**: `/home/hyruuk/GitHub/cocolab/cc_saflow`
- **New codebase**: `/home/hyruuk/GitHub/cocolab/saflow`
- **Data location**: `/media/hyruuk/YH_storage/DATA/saflow`

### Important Notes
- Log directories should be created by scripts as needed (not pre-created)
- Phase 1 focuses on SENSOR-LEVEL analysis only
- Source-level analysis is Phase 2 (post-refactoring)
- Architecture supports both spaces from the start

---

## Completed Tasks

### ✅ Phase 1.1: Create project structure
**Status**: COMPLETE
**What was done**:
- Created all directory structure:
  - `saflow/` package (with proper `__init__.py` file, not folder!)
  - `code/` with all subdirectories including sensor/ and source/ splits
  - `slurm/templates/`
  - `tests/`, `docs/`, `reports/` with sensor/source subdirs
  - `logs/` (empty, to be populated by scripts)
- Created `__init__.py` files for all Python modules
- Copied LICENSE from cc_saflow

**Files created**:
- Directory structure
- Multiple `__init__.py` files
- `LICENSE` (copied)

### ✅ Phase 1.2: Implement configuration system
**Status**: COMPLETE
**What was done**:
- ✅ Created `config.yaml.template` with comprehensive schema
  - Includes analysis.space configuration (sensor/source)
  - All paths with placeholders
  - BIDS configuration with exact subject list from cc_saflow
  - Preprocessing, source reconstruction, features settings
  - SLURM resource allocations
- ✅ Created `code/utils/config.py` with:
  - Configuration loading and validation
  - Placeholder detection
  - Path expansion (relative → absolute)
  - Helper functions (get_subjects, get_analysis_space, etc.)
- ✅ Created `code/utils/paths.py` with:
  - BIDS path construction helpers
  - Path getters for all pipeline stages
  - Analysis space-aware path routing
- ✅ Created `code/utils/space.py` with:
  - Analysis space validation
  - Sensor/source routing utilities
  - Source-level prerequisite checks
  - Phase 1 sensor-only enforcement
- ✅ Created working `config.yaml` for testing
- ✅ Validated YAML syntax and structure

### ✅ Phase 1.3: Implement logging system
**Status**: COMPLETE
**What was done**:
- ✅ Created `code/utils/logging_config.py` with:
  - `setup_logging()` function for configuring loggers
  - Console and file output support
  - Color-coded console output (ANSI colors)
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Automatic timestamp addition to log files
  - Support for multiple file handlers
  - `add_file_handler()` for adding additional log files
  - `log_provenance()` for recording git hash, config, timestamp
- ✅ Features:
  - Log format and level configurable via config.yaml
  - Automatic log directory creation
  - Support for nested log directories (e.g., logs/preprocessing/subject_01/)
  - Colored console output for better readability
  - File output without ANSI codes
  - Integration with configuration system
- ✅ Tested all logging functionality:
  - Basic logging (console + file)
  - Console-only logging
  - Multiple file handlers
  - Provenance logging
  - Subdirectory logging
  - All tests passed ✓

**Files created**:
- `code/utils/logging_config.py`
- `code/__init__.py` (created to fix package imports)

### ✅ Phase 1.4: Setup dependency management
**Status**: COMPLETE
**What was done**:
- ✅ Created `pyproject.toml` with modern Python packaging (PEP 621)
  - Build system configuration (setuptools)
  - Project metadata (name, version, description, authors, keywords, classifiers)
  - Python version constraint: >=3.9,<3.13
  - Core dependencies with version constraints extracted from cc_saflow/requirements.txt:
    - MEG/EEG: mne, mne-bids, autoreject
    - Scientific computing: numpy, scipy, pandas
    - Machine learning: scikit-learn, xgboost, mlneurotools
    - Signal processing: fooof, antropy
    - Statistics: statsmodels, statannotations
    - Visualization: matplotlib, seaborn
    - Utilities: joblib, tqdm, rich, h5io, str2bool
    - Configuration: pyyaml, python-dotenv, click
  - Optional dependency groups:
    - `[dev]`: ruff, mypy, pre-commit, ipython, jupyter
    - `[test]`: pytest, pytest-cov, pytest-xdist, coverage
    - `[docs]`: sphinx, sphinx-rtd-theme, myst-parser
    - `[hpc]`: jinja2 (for SLURM templating)
    - `[all]`: All optional dependencies combined
- ✅ Comprehensive tool configurations:
  - **Ruff**: Linting and formatting with ~20 rule sets enabled
    - Line length: 100
    - Target Python 3.9+
    - Excludes: .git, .venv, __pycache__, build, logs, reports, data
    - isort configuration with first-party packages
  - **Pytest**: Testing configuration
    - Test discovery patterns
    - Coverage reporting (term, html, xml)
    - Custom markers for test organization (slow, integration, unit, etc.)
    - Test paths and exclusions
  - **Mypy**: Type checking configuration
    - Python 3.9 target
    - Strict equality, no implicit optional
    - Ignore missing imports for third-party packages (mne, etc.)
  - **Coverage**: Code coverage configuration
    - Source paths, omit patterns
    - Exclude lines (pragma, abstract methods, etc.)

**Key features**:
- All dependencies from cc_saflow captured with reasonable version constraints
- Separate optional dependency groups for different use cases
- Comprehensive linting/testing/type-checking configurations
- Ready for `pip install -e .[all]` for development setup
- Supports Python 3.9-3.12

**Files created**:
- `pyproject.toml`

### ✅ Phase 1.5: Create setup script
**Status**: COMPLETE
**What was done**:
- ✅ Created `setup.sh` bash script with comprehensive environment setup
  - Virtual environment creation with configurable Python version
  - Package installation with multiple modes (basic, dev, hpc, all)
  - Automatic config.yaml creation from template
  - Project directory creation (logs, reports, etc.)
  - Development tools setup (pre-commit hooks if available)
  - Installation verification
  - Color-coded output for better readability
- ✅ Features:
  - Command-line options:
    - `--dev`: Install development dependencies
    - `--hpc`: Install HPC dependencies
    - `--all`: Install all optional dependencies
    - `--python`: Specify Python executable
    - `--force`: Force reinstall if venv exists
    - `--help`: Show help message
  - Preflight checks:
    - Python version validation (requires 3.9+)
    - venv module availability check
    - Existing environment handling
  - Package installation:
    - Upgrades pip, setuptools, wheel
    - Installs saflow in editable mode
    - Installs optional dependencies based on mode
  - Configuration setup:
    - Creates config.yaml from template
    - Warns about placeholder replacement
  - Verification:
    - Tests package imports
    - Checks configuration loading
    - Provides next steps summary
  - Made executable with proper permissions
- ✅ Error handling:
  - Exits on any error (set -e)
  - Clear error messages for missing dependencies
  - User prompts for existing environments

**Key features**:
- One-command environment setup
- Flexible installation modes for different use cases
- Comprehensive verification and helpful next steps
- Production-ready with proper error handling

**Files created**:
- `setup.sh` (executable)

### ✅ Phase 1.6: Create .gitignore
**Status**: COMPLETE
**What was done**:
- ✅ Created comprehensive `.gitignore` file covering all necessary exclusions
  - Configuration files: config.yaml, .env (keeps config.yaml.template)
  - Python artifacts: __pycache__/, *.pyc, *.egg-info/, build/, dist/
  - Virtual environments: env/, venv/, ENV/
  - Testing artifacts: .pytest_cache/, .coverage, htmlcov/, .tox/
  - Tool caches: .mypy_cache/, .ruff_cache/, .pylint.d/
  - IDE files: .vscode/, .idea/, *.swp, .ipynb_checkpoints
  - Project-specific:
    - logs/ (generated by pipeline)
    - reports/ (generated outputs)
    - data/, sourcedata/, derivatives/, processed/ (external data)
    - tmp/, temp/, *.tmp
  - MEG/EEG specific:
    - .mne/ cache
    - fs_subjects/ (FreeSurfer outputs)
    - *.fif, *.ds, *.sqd (large binary files)
  - Scientific computing: *.h5, *.hdf5, *.npy, *.npz, *.pkl, *.mat
  - SLURM: slurm-*.out, *.sbatch
  - Documentation: docs/_build/
  - OS-specific: .DS_Store, Thumbs.db, .directory
  - Backup files: *.bak, *.backup, *.old, *.orig
  - Compressed archives: *.zip, *.tar, *.gz, *.rar
- ✅ Exception rules to keep important files:
  - !config.yaml.template
  - !.gitkeep
  - !README.md, !LICENSE, !CHANGELOG.md, !AGENTS.md, !PROGRESS.md, !TASKS.md
  - !tests/data/, !tests/fixtures/

**Key features**:
- Comprehensive coverage for Python/MEG project
- Organized by category with clear section headers
- Protects user-specific and generated files
- Keeps templates and documentation
- Prevents accidental commits of large data files
- Cross-platform (macOS, Windows, Linux)

**Files created**:
- `.gitignore`

### ✅ Phase 1.7: Create task runner
**Status**: COMPLETE
**What was done**:
- ✅ Created `tasks.py` with comprehensive invoke tasks for development workflow
  - **Testing tasks**:
    - `invoke test`: Run pytest with coverage
    - `invoke test-fast`: Run only fast tests (exclude slow markers)
    - `invoke test-unit`: Run only unit tests
    - `invoke test-integration`: Run only integration tests
  - **Code quality tasks**:
    - `invoke lint [--fix]`: Run ruff linting with optional auto-fix
    - `invoke format [--check]`: Format code with ruff (or check only)
    - `invoke typecheck`: Run mypy type checking
    - `invoke check`: Run all quality checks (lint + format + typecheck)
  - **Cleaning tasks**:
    - `invoke clean`: Clean generated files (bytecode, cache, coverage, build)
    - `invoke clean-all`: Clean everything including logs and reports
  - **Environment tasks**:
    - `invoke setup [--mode=MODE]`: Run setup script with specified mode
    - `invoke rebuild [--mode=MODE]`: Clean and rebuild environment
  - **Documentation tasks**:
    - `invoke docs [--serve] [--port=PORT]`: Build and optionally serve docs
  - **Utility tasks**:
    - `invoke info`: Display project information
    - `invoke validate-config`: Validate configuration file
  - **Workflow tasks**:
    - `invoke precommit`: Run format + lint + test before committing
    - `invoke help`: Show available tasks (default)
- ✅ Added `invoke>=2.2.0` to pyproject.toml [dev] dependencies
- ✅ Features:
  - All tasks have comprehensive docstrings with examples
  - Tasks support command-line arguments with sensible defaults
  - Organized into logical sections (testing, quality, cleaning, etc.)
  - Color-coded output where appropriate (✓, ❌, ⚠️)
  - Pre-task dependencies (e.g., precommit runs format, lint, test)
  - Path-agnostic (uses Path objects, works cross-platform)

**Key features**:
- Complete development workflow automation
- Reduces cognitive load for common tasks
- Consistent command interface across team members
- Easy to extend with new tasks
- Self-documenting with `invoke --list`

**Usage examples**:
```bash
invoke --list              # Show all tasks
invoke test                # Run tests with coverage
invoke lint --fix          # Lint and auto-fix issues
invoke format              # Format all code
invoke check               # Run all quality checks
invoke precommit           # Pre-commit workflow
invoke clean               # Clean generated files
invoke setup --mode=dev    # Setup dev environment
invoke info                # Show project info
```

**Files created**:
- `tasks.py`

**Files modified**:
- `pyproject.toml` (added invoke to dev dependencies)

---

### ✅ Phase 2: Core Utilities (REVISED)
**Status**: COMPLETE
**What was done**:
- ✅ **ARCHITECTURE CHANGE**: Eliminated `saflow/` package (project-specific, not library)
- ✅ All utilities moved to `code/utils/` following flat structure
- ✅ Removed `saflow/__init__.py` and hardcoded FREQS constants
- ✅ Migrated all utility functions from cc_saflow/saflow/ → saflow/
- ✅ Created saflow/__init__.py with package constants (FREQS, FREQS_NAMES)
  - NO hardcoded paths (removed DATA_ROOT, BIDS_PATH, LOGS_DIR, etc.)
  - Kept frequency band definitions and constants
- ✅ Created saflow/core.py - BIDS utilities and I/O helpers
  - create_fnames() - BIDS path construction (removed hardcoded bids_root)
  - get_meg_picks_and_info() - MEG channel information
  - segment_sourcelevel() - Source-level data segmentation
  - create_pval_mask() - Statistical mask creation
- ✅ Created saflow/data.py - Data loading and feature extraction
  - load_features() - Load PSD, LZC, FOOOF features
  - balance_dataset() - Balance class distributions
  - get_VTC_bounds(), get_inout() - VTC classification
  - select_epoch(), select_trial() - Epoch quality control
  - get_trial_data() - Trial-level feature extraction
- ✅ Created saflow/neuro.py - MEG/EEG preprocessing utilities
  - average_bands() - Frequency band power computation
  - get_present_events() - Event detection
  - trim_events(), trim_INOUT_idx() - Epoch trimming
  - compute_PSD_hilbert(), compute_envelopes_hilbert() - Hilbert transforms
- ✅ Created saflow/behav.py - Behavioral analysis utilities
  - interpolate_RT(), compute_VTC() - VTC computation
  - get_VTC_from_file() - Load and process behavioral data
  - clean_comerr() - Performance classification
  - SDT() - Signal detection theory measures
- ✅ Created saflow/stats.py - Statistical utilities
  - subject_average() - Subject-level averaging
  - simple_contrast(), subject_contrast() - Statistical contrasts
  - singlefeat_classif() - Single-feature classification
  - apply_tmax(), compute_pval() - Multiple comparison correction
- ✅ Created saflow/viz.py - Visualization utilities
  - grid_topoplot() - Grid of topographic maps
  - plot_psd_topomap() - PSD visualization
  - plot_contrast_topomap() - Statistical contrast visualization

**Key changes applied to all functions**:
- Removed ALL hardcoded paths (now use config or function parameters)
- Added type hints to all functions
- Added Google-style docstrings with examples
- Replaced print() statements with logging
- Added error handling with try/except blocks
- Used pathlib.Path for path handling

**Final file structure** (`code/utils/`):
- `config.py` (existing) - Configuration loading
- `logging_config.py` (existing) - Logging setup
- `paths.py` (existing) - Path construction
- `space.py` (existing) - Analysis space routing
- `bids_utils.py` (NEW) - BIDS operations (segment_sourcelevel, get_meg_picks_and_info, create_pval_mask)
- `data_loading.py` (NEW) - Data loading (load_features, balance_dataset, trial selection)
- `signal_processing.py` (NEW) - Signal processing (average_bands, Hilbert transforms, event detection)
- `behavioral.py` (NEW) - Behavioral analysis (VTC, RT, SDT) - removed unused threshold_VTC()
- `statistics.py` (NEW) - Statistical utilities (subject_average, contrasts, classification)
- `visualization.py` (NEW) - Visualization (grid_topoplot, convenience wrappers)

**Key architectural decisions**:
- Project is codebase-specific, not a reusable library
- All utilities in `code/utils/` (flat structure for easy navigation)
- Frequency bands: DEFAULT_FREQS in signal_processing.py as fallback, but scripts should load from config
- Removed invented/unused functions (threshold_VTC)
- Fixed all imports (removed saflow references)
- Updated pyproject.toml to remove saflow package

**Files deleted**:
- Entire `saflow/` package directory

---

### ✅ Phase 3.1: BIDS Generation (Stage 0)
**Status**: COMPLETE
**What was done**:
- ✅ Created `code/bids/` module with three files:
  - `__init__.py` - Package initialization with exports
  - `utils.py` - BIDS conversion helper functions
  - `generate_bids.py` - Main BIDS conversion script
- ✅ Migrated from cc_saflow/saflow/data/raw2bids.py with major improvements:
  - Uses config.yaml for all paths (no hardcoded ACQ_PATH, BIDS_PATH, SUBJ_LIST)
  - Added comprehensive logging (replaced all print statements)
  - Added progress bar with rich library
  - Added provenance tracking (git hash, timestamp, subjects processed)
  - Added error handling for all major operations
  - Added CLI arguments for path overrides and subject filtering
  - Type hints and Google-style docstrings throughout
  - Short, focused functions (~10 lines where possible)

**Helper functions in utils.py**:
- `parse_info_from_name()` - Extract subject/run/task from filename
- `load_meg_recording()` - Load CTF data and create BIDSPath
- `detect_events()` - Find events from trigger channel
- `add_trial_indices()` - Add trial index column to events
- `find_trial_type()` - Look up trial type from performance dict
- `add_behavioral_info()` - Enrich events with VTC, RT, task performance
- `add_inout_zones()` - Add IN/OUT classifications for multiple bounds (50/50, 25/75, 10/90)

**Main script features** (generate_bids.py):
- Argparse CLI with --input, --output, --subjects, --log-level
- Config-driven paths (overridable via CLI)
- Process noise files (empty-room recordings)
- Process subject recordings (MEG data)
- Enrich gradCPT events with behavioral data from logfiles
- Progress tracking with rich Progress
- Provenance saved to bids_root/code/provenance_bids.json
- Comprehensive logging to logs/bids/

**Key improvements over original**:
1. No hardcoded paths - all from config or CLI
2. Fixed bug: line 109 in original used undefined `ds_file` variable
3. Better error handling (try/except for each file)
4. Provenance tracking (git hash, timestamp)
5. Progress reporting (console + logs)
6. Short functions (most ~5-15 lines)
7. Comprehensive docstrings
8. Type hints on all functions
9. Uses code.utils.behavioral.get_VTC_from_file (with logs_dir parameter)

**Files created**:
- `code/bids/__init__.py`
- `code/bids/utils.py`
- `code/bids/generate_bids.py`
- `code/utils/validation.py` - Input data validation utility
- `TASKS.md` - Comprehensive task documentation

**Invoke tasks added**:
- `invoke validate-inputs` - Validate raw data is present and complete
- `invoke bids` - Run BIDS conversion (Stage 0)

**Usage via invoke** (recommended):
```bash
# Validate inputs first
invoke validate-inputs --verbose

# Run BIDS conversion (dry run)
invoke bids --dry-run

# Process all subjects
invoke bids

# Process specific subjects
invoke bids --subjects "04 05 06"

# Debug mode
invoke bids --log-level=DEBUG
```

**Direct usage** (also works):
```bash
# Use paths from config
python code/bids/generate_bids.py

# Override paths
python code/bids/generate_bids.py -i /path/to/raw -o /path/to/bids

# Process specific subjects
python code/bids/generate_bids.py --subjects 04 05 06
```

---

### ✅ Phase 3.2: Preprocessing Pipeline (Stage 1)
**Status**: COMPLETE
**What was done**:
- ✅ Created `code/preprocessing/` module with four files:
  - `__init__.py` - Package initialization
  - `utils.py` - Preprocessing utilities (paths, filtering, epoching, noise covariance)
  - `ica_pipeline.py` - ICA artifact removal pipeline
  - `autoreject_pipeline.py` - AutoReject epoch rejection
  - `run_preprocessing.py` - Main orchestration script
- ✅ Migrated from cc_saflow/saflow/data/preprocessing.py with major improvements:
  - **Two-pass AutoReject approach** (IMPROVEMENT over original):
    1. First pass (fit only) on aggressively filtered data → identify bad epochs for ICA
    2. Second pass (fit_transform) on ICA-cleaned data → interpolate bad channels
  - **Saves BOTH versions**:
    - ICA-only epochs (processing="ica") for comparison
    - ICA+AR epochs (processing="icaar") as final cleaned data
  - **Comprehensive comparison report** showing data quality and rejection statistics for both versions
  - Uses config.yaml for all parameters
  - Comprehensive logging throughout
  - Progress tracking with rich library
  - Provenance metadata saved with all outputs

**Two-pass AutoReject details**:
- **First pass**: Fit on aggressively filtered epochs (1 Hz highpass), get bad epochs mask for ICA fitting
- **Second pass**: Fit_transform on ICA-cleaned epochs, interpolates bad channels and rejects remaining bad epochs
- Report compares both versions with summary table showing:
  - Bad epochs from each pass
  - Channels interpolated in second pass
  - Final data retention percentage
  - Evoked responses for both versions

**Helper functions in utils.py**:
- `create_preprocessing_paths()` - Create BIDSPath objects for inputs/outputs
- `compute_or_load_noise_cov()` - Compute or cache noise covariance from empty-room
- `apply_filtering()` - Apply bandpass + notch filters, create two versions (0.1 Hz and 1 Hz highpass)
- `create_epochs()` - Create epochs from continuous data

**ICA pipeline** (ica_pipeline.py):
- `fit_ica()` - Fit ICA on good epochs
- `find_ecg_components()` - Detect cardiac artifacts
- `find_eog_components()` - Detect ocular artifacts
- `apply_ica()` - Remove artifact components
- `run_ica_pipeline()` - Complete ICA workflow

**AutoReject pipeline** (autoreject_pipeline.py):
- `run_autoreject()` - First pass (fit only, for bad epoch identification)
- `run_autoreject_transform()` - Second pass (fit_transform, with interpolation)
- `get_good_epochs_mask()` - Extract good epochs mask

**Main script features** (run_preprocessing.py):
- Argparse CLI with --subject, --runs, --bids-root, --log-level, --skip-existing
- Config-driven preprocessing parameters
- Two filtered versions: 0.1-200 Hz (final) and 1-200 Hz (for AR/ICA)
- Gradient compensation (grade 3) for source reconstruction
- Two-pass AutoReject with comparison
- ICA artifact removal (ECG + EOG)
- HTML report with diagnostic plots comparing both versions
- Saves preprocessed continuous data, both epoch versions, both AR logs, report, and metadata
- Memory cleanup after processing

**Output structure** (three data versions saved):
1. **Continuous (ICA-cleaned)**: `derivatives/preprocessed/sub-{subject}/meg/*_proc-clean_meg.fif`
2. **Epochs (ICA only)**: `derivatives/epochs/sub-{subject}/meg/*_proc-ica_meg.fif`
3. **Epochs (ICA+AR)**: `derivatives/epochs/sub-{subject}/meg/*_proc-icaar_meg.fif`

**Metadata and reports**:
- AR logs (both passes): `derivatives/preprocessed/sub-{subject}/meg/*_desc-ARlog{1,2}_meg.pkl`
- HTML report: `derivatives/preprocessed/sub-{subject}/meg/*_desc-report_meg.html`
- **Text summary**: `derivatives/preprocessed/sub-{subject}/meg/*_desc-report_meg_summary.txt` (NEW!)
- Processing metadata: `derivatives/preprocessed/sub-{subject}/meg/*_params.json`
- Execution logs: `logs/preprocessing/preprocessing_sub-{subject}_YYYYMMDD_HHMMSS.log`

**Key improvements over original**:
1. **Two-pass AutoReject** (original had second pass commented out) - better data quality
2. **Saves THREE versions** for comparison: continuous (ICA), epochs (ICA), epochs (ICA+AR)
3. **Comprehensive comparison report** with structured HTML tables and text summary
4. **Text-based summary file** (*_summary.txt) for easy parsing and review
5. No hardcoded paths - all from config or CLI
6. Proper logging throughout with clear progress indicators (✓ checkmarks)
7. Modular design (separate files for ICA, AR, utilities)
8. Provenance metadata saved with detailed metrics
9. Progress reporting with rich library
10. Type hints and docstrings throughout
11. Comprehensive error handling

**Files created**:
- `code/preprocessing/__init__.py`
- `code/preprocessing/utils.py`
- `code/preprocessing/ica_pipeline.py`
- `code/preprocessing/autoreject_pipeline.py`
- `code/preprocessing/run_preprocessing.py`

**Invoke task added**:
- `invoke preprocess --subject=04` - Run preprocessing (Stage 1)

**Usage via invoke** (recommended):
```bash
# Process all task runs for subject 04
invoke preprocess --subject=04

# Process specific runs only
invoke preprocess --subject=04 --runs="02 03"

# Reprocess existing files
invoke preprocess --subject=04 --no-skip-existing

# Debug mode
invoke preprocess --subject=04 --log-level=DEBUG
```

**Direct usage** (alternative):
```bash
# Process all task runs
python -m code.preprocessing.run_preprocessing -s 04

# Process specific runs
python -m code.preprocessing.run_preprocessing -s 04 -r 02 03
```

---

### ✅ Phase 3.3: Source Reconstruction (Stage 2)
**Status**: COMPLETE
**What was done**:
- ✅ Created `code/source_reconstruction/` module with three files:
  - `__init__.py` - Package initialization
  - `utils.py` - Source reconstruction utilities (700+ lines)
  - `run_inverse_solution.py` - Main orchestration script (320+ lines)
  - `apply_atlas.py` - Atlas/parcellation application (optional Stage 2b)
- ✅ Migrated from cc_saflow/saflow/source_reconstruction/inverse_solution.py with major improvements:
  - **Modular design**: Split monolithic script into reusable utility functions
  - **7-step pipeline** with clear progress indicators:
    1. Coregistration (MEG ↔ MRI coordinate systems)
    2. Source space setup (cortical surface model)
    3. BEM model creation (boundary element model)
    4. Forward solution computation (leadfield matrix)
    5. Noise covariance estimation (from empty-room recording)
    6. Inverse solution application (dSPM/MNE/sLORETA)
    7. Morphing to fsaverage template
  - **Automatic MRI detection**: Uses individual MRI if available, falls back to fsaverage
  - **Caching strategy**: Saves coregistration, forward solution, and noise covariance for reuse
  - Uses config.yaml for all parameters (method, SNR, atlas)
  - Comprehensive logging with timing and memory information
  - Provenance metadata saved with all outputs

**Utility functions** (code/source_reconstruction/utils.py):
- `create_output_paths()` - Create BIDSPath objects for all inputs/outputs
- `compute_coregistration()` - Automatic coregistration with ICP fitting
- `setup_source_space()` - Create source space (oct6 spacing)
- `create_bem_model()` - Create BEM solution (single-layer for MEG)
- `compute_forward_solution()` - Compute forward solution
- `compute_noise_covariance()` - Compute noise covariance from empty-room
- `apply_inverse_continuous()` - Apply inverse operator to continuous data
- `apply_inverse_to_epochs()` - Apply inverse operator to epochs
- `morph_to_fsaverage()` - Morph source estimates to template
- `save_source_estimate()` - Save source estimate to HDF5
- `check_mri_availability()` - Check if individual MRI exists

**Main script features** (run_inverse_solution.py):
- Argparse CLI with --subject, --runs, --bids-root, --log-level, --skip-existing
- Supports both local and SLURM execution (via tasks.py)
- Config-driven source reconstruction parameters
- Automatic caching of intermediate results (trans, fwd, noise_cov)
- Progress tracking with 7-step workflow
- Comprehensive error handling with informative messages
- Saves morphed source estimates and metadata
- Memory cleanup after processing

**Atlas application** (apply_atlas.py - optional):
- Applies cortical parcellation to morphed source estimates
- Averages time series within each ROI/label
- Supports any FreeSurfer atlas (e.g., aparc.a2009s, aparc, HCPMMP1)
- Outputs ROI-averaged time series as pickle files
- Useful for ROI-based analyses in source space

**Output structure**:
1. **Coregistration transform**: `derivatives/trans/sub-{subject}/meg/*_proc-trans_meg.fif`
2. **Forward solution**: `derivatives/fwd/sub-{subject}/meg/*_proc-forward_meg.fif`
3. **Noise covariance**: `derivatives/noise_cov/sub-emptyroom/meg/*.fif`
4. **Source estimates**: `derivatives/minimum-norm-estimate/sub-{subject}/meg/*_desc-sources_meg-stc.h5`
5. **Morphed sources**: `derivatives/morphed_sources/sub-{subject}/meg/*_desc-morphed_meg-stc.h5`
6. **Atlased sources** (optional): `derivatives/atlased_sources_{atlas}/sub-{subject}/meg/*-avg.pkl`

**Metadata and logs**:
- Processing metadata: `derivatives/morphed_sources/sub-{subject}/meg/*_params.json`
- Execution logs: `logs/source_reconstruction/source_recon_sub-{subject}_YYYYMMDD_HHMMSS.log`

**Key improvements over original**:
1. **Modular utility functions** (original was monolithic 325-line script)
2. **Comprehensive logging** with clear progress indicators (1/7, 2/7, etc.)
3. **Robust caching** - reuses trans, fwd, noise_cov across runs
4. **Better error handling** with try/except and informative messages
5. No hardcoded paths - all from config or CLI
6. Type hints and Google-style docstrings throughout
7. Separate atlas application script for optional ROI-based analysis
8. Supports multi-run processing with --runs argument
9. Provenance metadata with MRI availability, n_sources, sfreq
10. Ready for SLURM execution (invoke task created)

**SLURM integration**:
- Created SLURM template: `slurm/templates/source_reconstruction.sh.j2`
- Added invoke task: `invoke source-recon` with --slurm support
- Per-run distribution (one job per subject-run combination)
- Configuration in config.yaml under `computing.slurm.source_reconstruction`
- Resource requirements: 1 CPU, 256G memory, 2-hour time limit
- Job manifests saved to `logs/slurm/source_reconstruction/`

**Files created**:
- `code/source_reconstruction/__init__.py`
- `code/source_reconstruction/utils.py` (700+ lines)
- `code/source_reconstruction/run_inverse_solution.py` (320+ lines)
- `code/source_reconstruction/apply_atlas.py` (optional, 280+ lines)
- `slurm/templates/source_reconstruction.sh.j2`

**Files modified**:
- `tasks.py`: Added `invoke source-recon` task with local and SLURM execution
- `TASKS.md`: Added comprehensive source reconstruction documentation

**Invoke task added**:
- `invoke source-recon --subject=04 --runs="02 03"` - Run source reconstruction (Stage 2)
- `invoke source-recon --slurm` - Submit to SLURM cluster

**Usage via invoke** (recommended):
```bash
# Local execution - single subject
invoke source-recon --subject=04 --runs="02 03"

# Local execution - debug mode
invoke source-recon --subject=04 --log-level=DEBUG

# SLURM execution - single subject (6 jobs)
invoke source-recon --subject=04 --slurm

# SLURM execution - all subjects (192 jobs)
invoke source-recon --slurm

# Dry run - generate scripts without submitting
invoke source-recon --slurm --dry-run
```

**Direct usage** (alternative):
```bash
# Process specific runs
python -m code.source_reconstruction.run_inverse_solution --subject 04 --runs "02 03"

# Apply atlas (optional)
python -m code.source_reconstruction.apply_atlas --subject 04 --runs "02 03"
```

**Configuration used**:
```yaml
source_reconstruction:
  method: dSPM              # Inverse method
  snr: 3.0                  # Signal-to-noise ratio
  atlas: aparc.a2009s       # Atlas for parcellation

paths:
  freesurfer_subjects_dir: fs_subjects/  # FreeSurfer directory

computing:
  n_jobs: -1                # Parallel jobs for forward solution
  slurm:
    source_reconstruction:
      cpus: 1
      mem: 256G             # Large memory for morphing
      time: "2:00:00"
```

---

### ✅ Phase 4: Feature Extraction & Architectural Improvements
**Status**: COMPLETE (2026-01-31)
**What was done**:

#### Task 4.1: Fix Critical Import Errors
- ✅ Added missing validation functions to `code/utils/validation.py`:
  - `validate_subject(subject, config) -> bool`
  - `validate_run(run, config) -> bool`
  - `validate_subject_run(subject, run, config) -> bool`
- ✅ Fixed source reconstruction import errors (run_inverse_solution.py, apply_atlas.py)
- ✅ Fixed compute_complexity.py import paths (was using `utils.*`, now `code.utils.*`)

#### Task 4.2: Unified Data Loaders
- ✅ Created `code/features/loaders.py` with unified multi-space architecture:
  - Single `load_data()` function works for sensor/source/atlas spaces
  - `SpatialData` namedtuple: consistent return type (data, sfreq, spatial_names)
  - Supports continuous and epoched data
  - Space-specific internal loaders: `_load_sensor_data()`, `_load_source_data()`, `_load_atlas_data()`
- ✅ **Architectural decision**: No separate sensor/source folders - single codebase with `space` parameter
  - Removed empty `code/features/sensor/` and `code/features/source/` directories
  - All feature scripts use unified loaders

#### Task 4.3: VTC Framework Organization
- ✅ Moved `classify_trials_from_vtc()` from features to behavioral module:
  - Source: `code/utils/behavioral.py` (with other VTC functions)
  - Updated imports in: `compute_fooof.py`, `plot_behavior.py`, `features/__init__.py`
- ✅ **Design principle**: Behavioral metrics separate from neural features
  - VTC computation: `get_VTC_from_file()` in behavioral module
  - Trial classification: `classify_trials_from_vtc()` in behavioral module
  - Feature extraction: Uses behavioral module for trial classification

#### Task 4.4: Complete FOOOF Implementation
- ✅ Implemented full FOOOF pipeline (`code/features/compute_fooof.py`, 474 lines):
  - `load_welch_psd()`: Load pre-computed Welch PSDs from derivatives
  - `fit_fooof_group()`: Fit FOOOFGroup to multi-channel/ROI PSD
  - `extract_fooof_params()`: Extract aperiodic (exponent, offset) and periodic (peaks) parameters
  - `average_psd_by_zone()`: Average trials within IN/OUT zones for state comparison
  - `process_subject_run()`: Main pipeline with per-trial + averaged fitting
- ✅ Features:
  - Per-trial FOOOF fitting for temporal dynamics
  - IN/OUT zone averaging for state contrasts
  - Comprehensive parameter extraction (exponent, offset, r², peak CF/PW/BW)
  - Multi-space support (sensor/source/atlas)
  - Provenance tracking (saves .pkl + .json metadata)
  - Config-driven (freq_range, peak_width_limits, aperiodic_mode, etc.)

#### Task 4.5: Documentation
- ✅ Created comprehensive README.md:
  - Pipeline workflow (expected task sequence)
  - VTC framework explanation (theory, computation, usage)
  - Dataset specificities (gradCPT task, data composition)
  - Design choices (unified architecture, two-pass AR, config-driven)
  - Usage examples (single subject, multi-space, HPC)
  - Configuration guide
- ✅ Created CHANGELOG.md:
  - Version 0.2.0 with all feature additions
  - Version 0.1.0 with initial pipeline
  - Migration notes from cc_saflow
- ✅ Updated PROGRESS.md:
  - Phase 4 completion status
  - Task list updates
  - Current status summary

**Files created**:
- `code/features/loaders.py` (300+ lines)
- `code/features/compute_fooof.py` (474 lines)
- `README.md` (600+ lines)
- `CHANGELOG.md` (350+ lines)

**Files modified**:
- `code/utils/validation.py` (added 3 functions, ~90 lines)
- `code/utils/behavioral.py` (added classify_trials_from_vtc, ~90 lines)
- `code/features/compute_complexity.py` (fixed imports)
- `code/features/__init__.py` (updated imports)
- `code/visualization/plot_behavior.py` (updated imports)
- `PROGRESS.md` (updated status)

**Files deleted**:
- `code/features/sensor/` (empty directory)
- `code/features/source/` (empty directory)

**Testing results**:
- ✅ All validation functions work
- ✅ Source reconstruction imports successfully
- ✅ Welch PSD script recognizes loaders
- ✅ FOOOF imports and CLI functional
- ✅ Unified architecture verified

**Key improvements**:
1. **Unified architecture**: Single codebase for all analysis spaces (no code duplication)
2. **VTC in behavioral module**: Clear separation of behavioral vs. neural features
3. **Complete FOOOF**: Per-trial + averaged fitting with comprehensive parameter extraction
4. **Import consistency**: All scripts use `from code.utils.*` (no relative imports)
5. **Documentation**: Comprehensive README explaining workflow, VTC, design choices

---

### ✅ Phase 6.2-6.3: SLURM Integration (HPC)
**Status**: COMPLETE (implemented early due to user request)
**What was done**:
- ✅ Created SLURM utilities module (`code/utils/slurm.py`)
  - Job submission with `sbatch` integration
  - Job status checking with `sacct`
  - Job cancellation with `scancel`
  - SLURM script rendering with Jinja2 templates
  - Job manifest tracking (JSON files with job IDs and metadata)
- ✅ Created Jinja2 template system (`slurm/templates/`)
  - `base.sh.j2`: Base SLURM template with common directives
  - `preprocessing.sh.j2`: Preprocessing-specific template
  - Support for config-driven resource allocation
  - Environment setup (venv activation, PYTHONPATH, OMP threads)
  - Comprehensive job logging (stdout/stderr)
- ✅ Enhanced `invoke preprocess` task with `--slurm` support
  - Local execution: `invoke preprocess --subject=04` (runs directly)
  - SLURM execution: `invoke preprocess --slurm` (submits jobs)
  - Distributed processing: one SLURM job per (subject, run) combination
  - Supports all subjects: `invoke preprocess --slurm` submits 192 jobs (32 subjects × 6 runs)
  - Supports single subject: `invoke preprocess --subject=04 --slurm` submits 6 jobs
  - Dry run support: `invoke preprocess --slurm --dry-run` generates scripts without submitting
- ✅ Job tracking and management
  - Job manifests saved to `logs/slurm/preprocessing/preprocessing_manifest_YYYYMMDD_HHMMSS.json`
  - Contains: job IDs, subjects, runs, timestamps, metadata
  - SLURM scripts saved to `slurm/scripts/preprocessing/`
  - Job logs saved to `logs/slurm/preprocessing/*_{jobid}.out/.err`
- ✅ Configuration integration
  - SLURM settings in `config.yaml`: account, partition, resources per stage
  - Per-stage resource allocation (CPUs, memory, time)
  - Email notifications (optional)
  - Module loading support (for HPC environments)
- ✅ Documentation
  - Updated TASKS.md with SLURM usage examples
  - Added SLURM Integration section with job management commands
  - Documented job manifests and log locations
  - Clear examples for local vs. SLURM execution

**Key design decisions**:
- **Per-run distribution**: One SLURM job per run (not per subject) to maximize parallelization
  - Allows processing 192 runs in parallel across cluster
  - Better resource utilization than batching by subject
- **Template-based**: Jinja2 templates for maintainability and extensibility
  - Base template inherited by stage-specific templates
  - Easy to add new pipeline stages (features, classification, etc.)
- **Config-driven**: All SLURM parameters from config.yaml
  - No hardcoded accounts, partitions, or resource allocations
  - Easy to adapt for different HPC environments
- **Job tracking**: Manifest files for batch job management
  - Track all jobs from a single submission
  - Useful for implementing dependencies in future stages

**Files created**:
- `code/utils/slurm.py` (310 lines)
- `slurm/templates/base.sh.j2`
- `slurm/templates/preprocessing.sh.j2`

**Files modified**:
- `tasks.py`: Enhanced preprocess task with --slurm support
- `TASKS.md`: Added SLURM Integration section

**Impact on plan**:
- Phase 6.2-6.3 implemented early (originally scheduled for Days 19-21)
- Required for preprocessing at scale on HPC
- Architecture ready for future stages (features, statistics, classification)
- All future pipeline tasks will use same template system

---

## Next Steps

### ✅ PHASE 1 COMPLETE ✅ (Infrastructure)
### ✅ PHASE 2 COMPLETE ✅ (Core Utilities)
### ✅ PHASE 3 COMPLETE ✅ (Data Processing Pipeline)
### ✅ PHASE 4 COMPLETE ✅ (Feature Extraction & Architectural Improvements)
### ✅ PHASE 6.2-6.3 COMPLETE ✅ (SLURM Integration)

**Completed Phases:**
1. ~~Phase 1: Project Infrastructure~~ ✅
   - Project structure, config system, logging, dependencies, setup script, gitignore, task runner
2. ~~Phase 2: Core Utilities~~ ✅
   - Behavioral, BIDS, data loading, signal processing, statistics, visualization utilities
3. ~~Phase 3: Data Processing Pipeline~~ ✅
   - Stage 0: BIDS Generation
   - Stage 1: Preprocessing (ICA, AutoReject)
   - Stage 2: Source Reconstruction (MNE inverse solutions)
4. ~~Phase 4: Feature Extraction~~ ✅
   - Validation functions (fix import errors)
   - Unified data loaders (sensor/source/atlas)
   - VTC framework organization (behavioral module)
   - Complete FOOOF implementation
   - Comprehensive documentation (README, CHANGELOG)
5. ~~Phase 6.2-6.3: SLURM Integration~~ ✅ (implemented early)
   - Job submission, templates, task integration

**Remaining Phases:**
- **Phase 5**: Statistics & Classification (IN PROGRESS)
- **Phase 6**: Enhanced Visualization
- **Phase 7**: Testing & Quality Assurance
- **Phase 8**: Validation & Publication Readiness

**Current Status:**
✅ BIDS generation (Stage 0) complete and tested
✅ Preprocessing pipeline (Stage 1) complete with two-pass AutoReject
✅ Source reconstruction (Stage 2) complete with morphing to fsaverage
✅ SLURM integration complete for preprocessing AND source reconstruction
✅ Feature extraction (Stage 3) - Welch PSD, FOOOF, Complexity complete
✅ Unified multi-space architecture implemented (no separate sensor/source code)
✅ VTC framework in behavioral module
⏳ Ready to test full pipeline (Stages 0-3) with real data
📋 Next: Phase 5 - Statistics & Classification modules

---

## Task List

### ✅ Completed Tasks
- [x] #1: Phase 1.1 - Create project structure
- [x] #2: Phase 1.2 - Implement configuration system
- [x] #3: Phase 1.3 - Implement logging system
- [x] #4: Phase 1.4 - Setup dependency management
- [x] #5: Phase 1.5 - Create setup script
- [x] #6: Phase 1.6 - Create comprehensive gitignore
- [x] #7: Phase 1.7 - Create task runner
- [x] #8: Phase 2.1 - Migrate utilities to code/utils/
- [x] #9: **Phase 3.1** - BIDS generation (Stage 0) ✅
- [x] #10: **Phase 3.2** - Preprocessing (Stage 1) ✅
- [x] #11: **Phase 3.3** - Source reconstruction (Stage 2) ✅
- [x] #12: **Phase 4.1** - Fix validation imports ✅
- [x] #13: **Phase 4.2** - Unified data loaders ✅
- [x] #14: **Phase 4.3** - VTC framework organization ✅
- [x] #15: **Phase 4.4** - Complete FOOOF implementation ✅
- [x] #16: **Phase 4.5** - Comprehensive documentation ✅
- [x] #17: **Phase 6.2-6.3** - SLURM integration ✅ (implemented early)

### 🔄 Current/Next Tasks
- [ ] #18: Phase 5.1 - Statistics module (group contrasts, IN/OUT comparisons) ← **NEXT**

### 📋 Upcoming Tasks (Phase 5+)
- [ ] #18: Phase 5.1 - Statistics module (group-level stats)
- [ ] #19: Phase 5.2 - Classification module (ML decoding)
- [ ] #20: Phase 6.1 - Enhanced visualization (topoplots, TFR)
- [ ] #21: Phase 7.1 - Create test suite (unit + integration)
- [ ] #22: Phase 7.2 - Code quality automation (pre-commit hooks)
- [ ] #23: Phase 8.1 - End-to-end pipeline test with real data
- [ ] #24: Phase 8.2 - HPC testing at scale
- [ ] #25: Phase 8.3 - Multi-subject validation
- [ ] #26: Phase 8.4 - Publication-ready outputs

### Workflow Notes
- **Incremental approach**: Implement one stage → test with real data → iterate → next stage
- **User validation required** between each major stage
- **Trial filtering redesign**: Deferred until we have real data to test (during feature extraction phase)
- **Architecture principle**: Project-specific code, not a reusable library

---

## Files Created This Session

1. `/home/hyruuk/GitHub/cocolab/saflow/config.yaml.template`
2. `/home/hyruuk/GitHub/cocolab/saflow/config.yaml`
3. `/home/hyruuk/GitHub/cololab/saflow/code/utils/config.py`
4. `/home/hyruuk/GitHub/cocolab/saflow/code/utils/paths.py`
5. `/home/hyruuk/GitHub/cocolab/saflow/code/utils/space.py`
6. `/home/hyruuk/GitHub/cocolab/saflow/code/utils/logging_config.py`
7. `/home/hyruuk/GitHub/cocolab/saflow/LICENSE` (copied)
8. `/home/hyruuk/GitHub/cocolab/saflow/code/__init__.py`
9. `/home/hyruuk/GitHub/cocolab/saflow/pyproject.toml`
10. `/home/hyruuk/GitHub/cocolab/saflow/setup.sh`
11. `/home/hyruuk/GitHub/cocolab/saflow/.gitignore`
12. `/home/hyruuk/GitHub/cocolab/saflow/tasks.py`
13. Multiple other `__init__.py` files

## Path Structure Updates

Updated path organization in both config files:
- `raw:` changed from `raw/` to `sourcedata/` (relative to data_root)
- `reports:`, `logs:`, `venv:` now use relative paths (`./reports`, `./logs`, `./env`) pointing to saflow project directory
- `slurm_output:` now `./logs/slurm` (within logs directory)
- `freesurfer_subjects_dir:` changed to `fs_subjects/` (relative to data_root, not absolute path)

---

## Issues Fixed

1. **Issue**: Created `saflow/__init__.py` as folder instead of file
   - **Fixed**: Removed directory, created proper file
2. **Issue**: Pre-created log subdirectories
   - **Fixed**: Removed them (scripts will create as needed)
3. **Issue**: setup.sh Python version detection failing with parsing errors
   - **Problem**: `python --version` output varied across systems, causing arithmetic errors at line 125
   - **Fixed**: Changed to use `sys.version_info` directly, added validation for integer version numbers
4. **Issue**: setup.sh venv check failing despite venv being installed
   - **Problem**: Using `--help` flag was unreliable for checking venv availability
   - **Fixed**: Changed to direct import test: `python -c "import venv"`

---

## Dependencies to Extract from cc_saflow

When creating pyproject.toml, pin these from `cc_saflow/requirements.txt`:
- mne
- numpy
- scipy
- pandas
- matplotlib
- seaborn
- scikit-learn
- mne-bids
- autoreject
- fooof
- pyyaml
- python-dotenv
- etc.

---

## Verification Checklist (End of Phase 1)

- [x] No hardcoded paths in any code
- [x] Configuration system loads and validates correctly
- [x] Logging system works (console + file)
- [x] pyproject.toml has all pinned dependencies
- [x] setup.sh creates working environment
- [x] .gitignore comprehensive
- [x] Task runner (invoke) functional

**✅ ALL PHASE 1 VERIFICATION ITEMS COMPLETE**
