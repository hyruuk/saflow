# Saflow Refactoring Progress Tracker

**Last Updated**: 2026-01-30
**Current Phase**: Phase 3.2 COMPLETE - Preprocessing Pipeline ✅
**Analysis Focus**: Sensor-level analysis (Phase 1), Source-level planned for Phase 2
**Status**: ✅ Stage 0-1 Implemented - Testing preprocessing on sub-04 to sub-08

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
    - Signal processing: fooof, neurokit2
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

## Next Steps

### ✅ PHASE 1 COMPLETE ✅
### ✅ PHASE 2 COMPLETE ✅
### ✅ PHASE 3.1 COMPLETE ✅
### ✅ PHASE 3.2 COMPLETE ✅

**All Phase 1 sub-phases complete:**
1. ~~Phase 1.1: Create project structure~~ ✅
2. ~~Phase 1.2: Implement configuration system~~ ✅
3. ~~Phase 1.3: Implement logging system~~ ✅
4. ~~Phase 1.4: Setup dependency management~~ ✅
5. ~~Phase 1.5: Create setup script~~ ✅
6. ~~Phase 1.6: Create .gitignore~~ ✅
7. ~~Phase 1.7: Create task runner~~ ✅

**Phase 2 Progress:**
1. ~~Phase 2: Migrate utilities to code/utils/~~ ✅

**Phase 3 Progress:**
1. ~~Phase 3.1: Stage 0 - BIDS Generation~~ ✅
2. ~~Phase 3.2: Stage 1 - Preprocessing~~ ✅
3. Phase 3.3: Stage 2 - Source Reconstruction ← **NEXT**

**Current Status:**
✅ BIDS generation (Stage 0) complete and tested
✅ Preprocessing pipeline (Stage 1) complete with two-pass AutoReject
⏳ Ready to test preprocessing on subject 04 (BIDS data available)

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
- [x] #8: Phase 2.1 - Migrate saflow package utilities (REVISED: moved to code/utils/)
- [x] #9: **Phase 3.1** - Migrate BIDS generation (Stage 0) ✅

### 🔄 Current/Next Tasks
- [ ] #10: **Phase 3.2** - Migrate preprocessing (Stage 1) ← **NEXT**
- [ ] #11: Phase 3.3 - Migrate source reconstruction (Stage 2)

### 📋 Upcoming Tasks (Phase 4+)
- [ ] #12: Phase 4.1 - Migrate sensor-level feature extraction
- [ ] #13: Phase 4.2 - De-duplicate feature utilities
- [ ] #14: Phase 5.1 - Migrate sensor-level statistics
- [ ] #15: Phase 5.2 - Migrate sensor-level classification
- [ ] #16: Phase 6.1 - Migrate sensor-level visualization
- [ ] #17: Phase 6.2 - Create SLURM integration

### 🧪 Testing & Quality (Phase 7-8)
- [ ] #18: Phase 7.1 - Write documentation
- [ ] #19: Phase 7.2 - Create test suite
- [ ] #20: Phase 7.3 - Code quality checks
- [ ] #21: Phase 8.1 - End-to-end pipeline test
- [ ] #22: Phase 8.2 - HPC testing
- [ ] #23: Phase 8.3 - Multi-subject testing
- [ ] #24: Phase 8.4 - Documentation verification

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
