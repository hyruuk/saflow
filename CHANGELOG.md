# Changelog

All notable changes to saflow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Statistics module (group-level contrasts, IN/OUT comparisons)
- Classification module (ML decoding, cross-validation)
- Visualization enhancements (topoplots, time-frequency maps)
- Additional feature extraction (connectivity, wavelet)
- Test suite (unit and integration tests)

---

## [0.2.0] - 2026-01-31

### Added
- **Validation functions** for subject/run checking (`code/utils/validation.py`)
  - `validate_subject()`, `validate_run()`, `validate_subject_run()`
  - Fixes import errors in source reconstruction scripts
- **Unified data loaders** (`code/features/loaders.py`)
  - Single `load_data()` function works across sensor/source/atlas spaces
  - `SpatialData` namedtuple for consistent return type
  - Supports continuous and epoched data
- **Complete FOOOF implementation** (`code/features/compute_fooof.py`)
  - Per-trial FOOOF fitting across all spatial units
  - IN/OUT zone averaging for state comparisons
  - Comprehensive parameter extraction (exponent, offset, r², peaks)
  - Multi-space support via unified loader
- **VTC trial classification** in behavioral module
  - `classify_trials_from_vtc()` moved to `code/utils/behavioral.py`
  - Configurable percentile bounds for IN/OUT zones
  - Returns masks, indices, labels, and thresholds
- **Complexity metrics** (`code/features/compute_complexity.py`)
  - Lempel-Ziv Complexity (LZC) via antropy
  - Entropy measures (permutation, sample, approximate, spectral, SVD)
  - Fractal dimensions (Higuchi, Petrosian, Katz, DFA)
- **Comprehensive documentation**
  - README.md with pipeline workflow, VTC framework, design choices
  - CHANGELOG.md for version tracking
  - Updated PROGRESS.md with recent commits

### Changed
- **Architectural improvements**
  - Removed empty `code/features/sensor/` and `code/features/source/` directories
  - VTC/classification functions consolidated in behavioral module
  - All feature scripts now import from `code.utils.behavioral`
- **Import cleanup**
  - Fixed `compute_complexity.py` imports (now uses `from code.utils.*`)
  - Updated all imports to use absolute paths
  - Removed `sys.path.insert` hacks

### Fixed
- Source reconstruction import error (`validate_subject_run` missing)
- Feature extraction import error (`loaders.py` missing)
- Inconsistent import paths across codebase

---

## [0.1.0] - 2026-01-30

### Added

#### Phase 1: Project Infrastructure
- **Project structure** with BIDS-compliant organization
- **Configuration system** (`code/utils/config.py`)
  - YAML-based configuration with validation
  - Placeholder detection for template configs
  - Path expansion (relative → absolute)
- **Logging system** (`code/utils/logging_config.py`)
  - Color-coded console output
  - File logging with timestamps
  - Provenance tracking (git hash, config, timestamp)
- **Dependency management** (`pyproject.toml`)
  - Modern Python packaging (PEP 621)
  - Pinned dependencies for reproducibility
  - Optional dependency groups (dev, test, docs, hpc)
  - Tool configurations (ruff, pytest, mypy)
- **Setup script** (`setup.sh`)
  - Automated environment creation
  - Multiple installation modes (basic, dev, hpc, all)
  - Config generation from template
  - Installation verification
- **Comprehensive .gitignore**
  - Python artifacts, virtual environments, caches
  - Data, logs, reports excluded
  - MEG-specific files, OS artifacts
- **Task runner** (`tasks.py` with invoke)
  - Testing tasks (test, test-fast, test-unit, test-integration)
  - Code quality (lint, format, typecheck, check)
  - Cleaning (clean, clean-all)
  - Environment (setup, rebuild)
  - Documentation (docs, info, validate-config)
  - Workflow (precommit)

#### Phase 2: Core Utilities
- **Behavioral utilities** (`code/utils/behavioral.py`)
  - RT interpolation for missing values
  - VTC computation with Gaussian/Butterworth filtering
  - Performance classification (commission errors, omissions)
  - Signal detection theory (SDT) measures (d', beta, c, A')
- **BIDS utilities** (`code/utils/bids_utils.py`)
  - MEG channel selection helpers
  - Source-level segmentation
  - Statistical mask creation
- **Data loading utilities** (`code/utils/data_loading.py`)
  - Feature loading (PSD, LZC, FOOOF)
  - Dataset balancing
  - Trial quality control
- **Signal processing** (`code/utils/signal_processing.py`)
  - Frequency band averaging
  - Event detection and trimming
  - Hilbert transform for envelopes
- **Statistics** (`code/utils/statistics.py`)
  - Subject averaging
  - Statistical contrasts
  - Single-feature classification
  - Multiple comparison correction
- **Visualization** (`code/utils/visualization.py`)
  - Grid topoplots
  - PSD topomaps
  - Contrast visualization
- **Space utilities** (`code/utils/space.py`)
  - Analysis space validation
  - Sensor/source routing
  - Phase 1 sensor-only enforcement

#### Phase 3: Data Processing Pipeline

**Stage 0: BIDS Generation** (`code/bids/`)
- CTF to BIDS conversion with mne-bids
- Event detection from trigger channels
- **Behavioral enrichment**:
  - VTC computation and filtering
  - RT extraction from logfiles
  - Task performance classification
  - IN/OUT zone classification (50/50, 25/75, 10/90 percentiles)
- Provenance tracking (git hash, subjects processed)
- Progress reporting with rich library

**Stage 1: Preprocessing** (`code/preprocessing/`)
- **Filtering**:
  - Bandpass: 0.1-200 Hz (final), 1-200 Hz (for AutoReject)
  - Notch: 60 Hz harmonics
  - Gradient compensation (grade 3 for source reconstruction)
- **ICA artifact removal**:
  - FastICA with 20 components
  - ECG component detection and removal
  - EOG component detection and removal
- **Two-pass AutoReject**:
  - Pass 1: Fit on 1 Hz highpass data, identify bad epochs
  - Pass 2: Fit+transform on ICA-cleaned data, interpolate bad channels
- **Three output versions**:
  - Continuous ICA-cleaned
  - Epochs ICA-only
  - Epochs ICA+AutoReject
- HTML comparison report with metrics
- Text summary for easy parsing

**Stage 2: Source Reconstruction** (`code/source_reconstruction/`)
- **7-step pipeline**:
  1. Coregistration (MEG ↔ MRI via ICP)
  2. Source space setup (oct6 cortical surfaces)
  3. BEM model creation (single-layer for MEG)
  4. Forward solution computation (leadfield matrix)
  5. Noise covariance estimation (empty-room recording)
  6. Inverse solution (dSPM/MNE/sLORETA)
  7. Morphing to fsaverage template
- Automatic MRI detection (individual or fsaverage fallback)
- Caching of intermediate results (trans, fwd, noise_cov)
- Optional atlas application (aparc.a2009s, aparc, HCPMMP1)
- Provenance metadata with all outputs

**Stage 3: Feature Extraction** (`code/features/`)
- **Welch PSD** (`compute_welch_psd.py`)
  - Multi-taper spectral estimation
  - Trial segmentation with behavioral metadata
  - Unified across sensor/source/atlas spaces
- **FOOOF** (`compute_fooof.py` - initially 29-line stub, completed in 0.2.0)

#### Phase 4: HPC Integration
- **SLURM utilities** (`code/utils/slurm.py`)
  - Job submission with sbatch integration
  - Job status checking with sacct
  - Job cancellation with scancel
  - Job manifest tracking (JSON files)
- **Jinja2 template system** (`slurm/templates/`)
  - Base template with common directives
  - Stage-specific templates (preprocessing, source_reconstruction)
  - Environment setup (venv activation, PYTHONPATH)
- **Invoke task integration**
  - `invoke preprocess --slurm` for distributed processing
  - `invoke source-recon --slurm` for parallel source reconstruction
  - Per-run job distribution (one job per subject-run)
  - Dry-run support for script generation
- **Config-driven resource allocation**
  - Account, partition, email settings
  - Per-stage CPU, memory, time limits

### Design Decisions (0.1.0)

1. **Config-driven architecture**: No hardcoded paths
2. **Logging over print**: All output via Python logging module
3. **Type hints + docstrings**: All public functions documented
4. **BIDS compliance**: Strict adherence to BIDS specification
5. **Provenance tracking**: Git hash + timestamp with all outputs
6. **Separation of raw/derivatives**: Raw data immutable
7. **Modern packaging**: pyproject.toml instead of setup.py
8. **Task runner**: Invoke for user-friendly commands
9. **Gradient compensation**: Grade 3 for source reconstruction compatibility
10. **Two-pass AutoReject**: Better data quality than single pass

---

## Initial Commits (Pre-0.1.0)

- `238a5d8` - Initial commit with phase 1 complete
- `670907d` - Re-implemented bidsification
- `b2a84f3` - Implemented preprocessing
- `ed6d165` - Added SLURM job splitting logic
- `67884c2` - Added source reconstruction pipeline
- `10840a3` - Added FOOOF computation and reworked setup
- `51dc102` - Added complexity module

---

## Migration from cc_saflow

Saflow is a complete refactoring of the original `cc_saflow` codebase with:
- Modern Python packaging and tooling
- No hardcoded paths (all config-driven)
- Comprehensive logging and provenance
- BIDS compliance throughout
- Unified sensor/source/atlas architecture
- HPC-ready with SLURM integration

Key improvements over cc_saflow:
1. Configuration system replaces hardcoded `BIDS_PATH`, `DATA_ROOT`
2. Two-pass AutoReject (second pass was commented out in original)
3. Saves multiple preprocessing versions for comparison
4. Unified loaders eliminate code duplication
5. Comprehensive documentation and type hints
6. Invoke task runner replaces manual script execution
7. Automated testing and code quality checks (ruff, mypy, pytest)

---

## Version Numbering

- **Major (X.0.0)**: Breaking changes to API or data structure
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, no new features

Current: **0.2.0** (feature additions, architectural improvements)
