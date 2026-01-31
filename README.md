# Saflow: MEG Analysis Pipeline for GradCPT Task

A production-ready, config-driven MEG analysis pipeline for processing gradual continuous performance task (gradCPT) data across sensor, source, and atlas analysis spaces.

**Version**: 0.2.0
**Status**: Active Development
**Python**: 3.9-3.12

---

## Table of Contents

- [Overview](#overview)
- [Dataset Specificities](#dataset-specificities)
- [VTC Framework](#vtc-framework)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow)
- [Design Choices](#design-choices)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Known Issues](#known-issues)

---

## Overview

Saflow implements a complete MEG analysis pipeline for the gradCPT (gradual Continuous Performance Task), a sustained attention task designed to study attentional lapses and fluctuations in cognitive engagement. The pipeline supports analysis at three levels:

- **Sensor level**: MEG channel data (~270 channels)
- **Source level**: Cortical surface vertices (~15k vertices)
- **Atlas level**: ROI-averaged parcellations (~80-150 ROIs)

### Key Features

- ✅ **BIDS-compliant** data organization and derivatives
- ✅ **Unified architecture** - same code works across all analysis spaces
- ✅ **VTC-based trial classification** - separate trials by attentional state
- ✅ **Config-driven** - no hardcoded paths, fully reproducible
- ✅ **Comprehensive logging** - provenance tracking with git hash
- ✅ **HPC-ready** - SLURM integration for cluster computing
- ✅ **Two-pass preprocessing** - ICA + AutoReject with comparison reports
- ✅ **Modern Python** - type hints, dataclasses, invoke task runner

---

## Dataset Specificities

### GradCPT Task

The **gradual Continuous Performance Task (gradCPT)** is a sustained attention task where participants monitor a continuous stream of city and mountain scenes, responding only to cities (target stimuli) while withholding responses to mountains (non-target stimuli).

![Figure 1. Gradual onset Continuous Performance Task (gradCPT)](docs/images/gradCPT.png)

**Figure 1. Gradual onset Continuous Performance Task (gradCPT).** **A** Sequence of four consecutive trials illustrating the four possible types of events: baseline trials, correct omissions, omission errors and lapses. The hand icon represents the occurrence of a response, and the nearby arrows show its association with the closest trial. The green and red horizontal bars represent the intensity of each stimulus at every time point, going from 0% (white) to 100% (green or red) and then to 0% again. **B** Experiment structure. The session was split into 8 runs, starting with an eyes-open resting state followed by 6 blocks of the task, and ending with a second resting state.

**Key characteristics**:
- **Gradual transitions**: Scenes morph gradually between cities and mountains over ~850ms
- **Sustained attention**: 8 minutes per run, 6 runs per session
- **Two trial types**:
  - **Frequent (non-target)**: Mountains (90% of trials)
  - **Rare (target)**: Cities (10% of trials)
- **Performance metrics**: Commission errors (lapses), omission errors, reaction times

### Epoch Timing

Epochs are defined relative to stimulus event markers (t=0 = stimulus onset at 0% intensity):

```
                                     ● 100%
                                    / \
                                   /   \
                                  /     \
                                 /       \
                                /         \
                          50%  ●           ●  50%
                              /             \
                             /               \
                            /                 \
                           /                   \
                     0%   ●                     ●   0%
  ────────────────────────┴─────────────────────┴───────────── time
                          ↑    ↑           ↑
                         t=0  tmin        tmax
```

- **t=0**: Event marker at stimulus onset (0% intensity, start of fade-in)
- **tmin (0.426s)**: Epoch starts at 50% intensity (rising phase)
- **midpoint (0.852s)**: 100% stimulus intensity (peak visibility)
- **tmax (1.278s)**: Epoch ends at 50% intensity (falling phase)
- **Duration**: ~852ms, capturing the high-visibility portion of stimulus presentation

These values are configurable in `config.yaml` under `analysis.epochs`.

### Why GradCPT?

Traditional CPT tasks use abrupt scene changes, making it difficult to separate perceptual from attentional effects. The gradCPT's gradual transitions:
1. Minimize low-level visual transients
2. Allow continuous tracking of attention
3. Enable trial-by-trial attentional state estimation via **VTC** (see below)

### Dataset Composition

- **32 subjects** (healthy adults, ages 18-35)
- **6 task runs** per subject (~8 min each, ~48 min total)
- **1 rest run** per subject (eyes open, 5 min)
- **1 empty-room recording** per session (for noise covariance)
- **CTF MEG system**: 275 channels (272 axial gradiometers + 3 reference)
- **Sampling rate**: 1200 Hz native, resampled to 600 Hz during preprocessing

---

## VTC Framework

### What is VTC?

**VTC (Variability Time Course)** quantifies trial-by-trial fluctuations in reaction time variability as a proxy for attentional state. It's computed from behavioral data and used to classify trials into attentional zones.

### Theoretical Foundation

Research shows that:
- **Stable attention (IN zone)**: Low RT variability, fast and consistent responses
- **Fluctuating attention (OUT zone)**: High RT variability, lapses and mind-wandering

VTC provides a **continuous, data-driven measure** of attentional engagement that doesn't rely on subjective reports.

### VTC Computation Pipeline

1. **Load behavioral data** (`.mat` logfiles with trial-by-trial RTs)
2. **Compute raw VTC**: `VTC_raw = |RT - mean(RT)| / std(RT)` (z-scored deviation)
3. **Smooth VTC**: Apply Gaussian filter (FWHM = 9 trials) to reduce noise
4. **Classify trials by percentiles**:
   - **IN zone**: VTC < 25th percentile (stable attention)
   - **OUT zone**: VTC ≥ 75th percentile (fluctuating attention)
   - **MID zone**: Between thresholds (excluded from IN/OUT comparisons)

### Why This Matters

VTC-based trial classification enables:
- **State-dependent analyses**: Compare neural activity during stable vs. fluctuating attention
- **Individual differences**: Each subject's IN/OUT thresholds are personalized
- **Avoid information leakage**: VTC computed during BIDS conversion (Stage 0), thresholds fixed for all downstream analyses

### Configuration

VTC parameters are set in `config.yaml`:
```yaml
behavioral:
  vtc:
    window_size: 20              # Trials for mean/std computation
    filter:
      type: "gaussian"           # Filter type
      gaussian_fwhm: 9           # Full-width half-max (trials)

analysis:
  inout_bounds: [25, 75]         # Percentile thresholds (default: quartiles)
```

Alternative bounds:
- `[50, 50]`: Median split (IN = below median, OUT = above median)
- `[10, 90]`: Conservative (extreme states only)
- `[33, 67]`: Tercile split

---

## Pipeline Architecture

### Unified Multi-Space Design

Saflow uses a **space-agnostic architecture** where the same code processes sensor, source, and atlas data:

```python
# Single loader works for all spaces
from code.features.loaders import load_data

# Load sensor data
sensor_data = load_data("sensor", bids_root, subject, run, "continuous", config)

# Load source data (same interface!)
source_data = load_data("source", bids_root, subject, run, "continuous", config)

# Both return: SpatialData(data, sfreq, spatial_names)
```

**Key insight**: All analysis spaces share the same structure:
- **Data shape**: `(n_spatial, n_times)` where n_spatial varies by space
- **Metadata**: sampling rate, spatial unit names (channels/vertices/ROIs)
- **Operations**: Welch PSD, FOOOF, complexity metrics work identically

### Module Organization

```
code/
├── bids/              # Stage 0: Raw → BIDS conversion
├── preprocessing/     # Stage 1: Filtering, ICA, AutoReject
├── source_reconstruction/  # Stage 2: MNE inverse solutions
├── features/          # Stages 3+: PSD, FOOOF, complexity
├── statistics/        # Group-level stats (planned)
├── classification/    # Decoding analyses (planned)
├── visualization/     # Plotting utilities
└── utils/             # Shared utilities
    ├── behavioral.py     # VTC computation, trial classification
    ├── config.py         # Configuration loading
    ├── logging_config.py # Logging setup
    ├── validation.py     # Input validation
    └── slurm.py          # HPC job submission
```

### No Separate Sensor/Source Code

Unlike many pipelines, saflow **does not duplicate code** for sensor vs. source analysis. Instead:
- Single `load_data()` function with `space` parameter
- Single feature extraction scripts work universally
- Configuration controls which space(s) to process

---

## Installation

### Requirements

- **Python**: 3.9-3.12 (tested on 3.9.5)
- **OS**: Linux (tested), macOS (should work), Windows (untested)
- **Disk space**: ~500GB for full dataset + derivatives
- **RAM**: 32GB recommended (64GB for source reconstruction)

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/saflow.git
cd saflow

# Run setup script (creates venv, installs dependencies)
./setup.sh --all

# Activate environment
source env/bin/activate

# Create configuration from template
cp config.yaml.template config.yaml

# Edit config.yaml with your paths
nano config.yaml
# Replace <PLACEHOLDER> values with actual paths:
#   - data_root: /path/to/data/saflow
#   - freesurfer_subjects_dir: /path/to/fs_subjects
```

### Verify Installation

```bash
# Check package installation
python -c "from code.utils.config import load_config; print('✓ Saflow installed')"

# Validate configuration
invoke validate-config

# Check data availability
invoke validate-inputs
```

---

## Pipeline Workflow

### Expected Task Sequence

The pipeline is designed to run in order, with each stage building on previous outputs:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 0: BIDS Generation (8-10 min)                         │
│ ─────────────────────────────────────────────────────────── │
│ Raw CTF → BIDS format + behavioral enrichment               │
│ Output: BIDS dataset with VTC, RT, performance metrics      │
│ Command: invoke bids                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Preprocessing (30-60 min/run)                      │
│ ─────────────────────────────────────────────────────────── │
│ Filter → ICA (ECG/EOG removal) → AutoReject (2-pass)        │
│ Output: Clean continuous + 2 epoch versions (ICA, ICA+AR)   │
│ Command: invoke preprocess --subject 04                     │
│          invoke preprocess --slurm  (HPC: all subjects)     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Source Reconstruction (30-120 min/run)             │
│ ─────────────────────────────────────────────────────────── │
│ Coregistration → Forward → Inverse → Morph to fsaverage     │
│ Output: Source estimates (vertices), optional atlas ROIs    │
│ Command: invoke source-recon --subject 04                   │
│          invoke source-recon --slurm  (HPC: all subjects)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Feature Extraction (varies by feature)             │
│ ─────────────────────────────────────────────────────────── │
│ Welch PSD → FOOOF → Complexity metrics                      │
│ Output: Trial-level features with IN/OUT classification     │
│ Commands:                                                    │
│   python -m code.features.compute_welch_psd \               │
│     --subject 04 --run 02 --space sensor                    │
│   python -m code.features.compute_fooof \                   │
│     --subject 04 --run 02 --space sensor                    │
│   python -m code.features.compute_complexity \              │
│     --subject 04 --run 02 --space sensor                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4+: Statistics & Classification (planned)             │
│ ─────────────────────────────────────────────────────────── │
│ Group stats, IN/OUT contrasts, ML decoding                  │
└─────────────────────────────────────────────────────────────┘
```

### Typical Workflow

```bash
# 1. Validate inputs
invoke validate-inputs --verbose

# 2. Convert to BIDS (all subjects)
invoke bids

# 3. Preprocess one subject locally (testing)
invoke preprocess --subject 04 --runs "02"

# 4. Preprocess all subjects on HPC
invoke preprocess --slurm

# 5. Source reconstruction on HPC
invoke source-recon --slurm

# 6. Extract features (per subject/run, parallelizable)
# Sensor-level Welch PSD
for subj in 04 05 06; do
  for run in 02 03 04 05 06 07; do
    python -m code.features.compute_welch_psd \
      --subject $subj --run $run --space sensor
  done
done

# FOOOF fitting (loads Welch PSDs)
python -m code.features.compute_fooof \
  --subject 04 --run 02 --space sensor --inout-bounds 25 75
```

### HPC Workflow

For cluster computing (SLURM):

```bash
# Submit all preprocessing jobs (192 jobs: 32 subjects × 6 runs)
invoke preprocess --slurm

# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/slurm/preprocessing/preproc_sub-04_run-02_*.out

# After completion, check job manifest
cat logs/slurm/preprocessing/preprocessing_manifest_*.json
```

---

## Design Choices

### 1. Configuration Over Hardcoding

**Decision**: All paths, parameters, and settings come from `config.yaml`

**Rationale**:
- **Reproducibility**: Different users/environments just update config
- **Flexibility**: Change parameters without editing code
- **Documentation**: Config serves as parameter record

**Alternative rejected**: Hardcoded paths (as in original cc_saflow)

### 2. Unified Multi-Space Architecture

**Decision**: Single codebase with `space` parameter instead of separate sensor/source modules

**Rationale**:
- **DRY principle**: Eliminate code duplication
- **Maintainability**: Bug fixes apply to all spaces
- **Consistency**: Same features computed identically across spaces

**Alternative rejected**: Separate `code/features/sensor/` and `code/features/source/` directories

### 3. VTC in Behavioral Module

**Decision**: VTC computation and trial classification in `code/utils/behavioral.py`

**Rationale**:
- **Separation of concerns**: Behavioral metrics ≠ neural features
- **Reusability**: Multiple analysis scripts use VTC classification
- **Clarity**: Makes VTC dependency explicit

**Alternative rejected**: VTC functions scattered across feature scripts

### 4. Two-Pass AutoReject

**Decision**: Run AutoReject twice with different objectives

**Pass 1** (aggressive filtering):
- Purpose: Identify bad epochs to exclude from ICA fitting
- Settings: 1 Hz highpass (removes slow drifts)
- Action: Fit only, get bad epoch mask

**Pass 2** (final cleaning):
- Purpose: Interpolate bad channels, reject remaining bad epochs
- Settings: 0.1 Hz highpass (preserves low frequencies)
- Data: ICA-cleaned epochs
- Action: Fit + transform, save interpolation log

**Rationale**:
- **Better ICA**: Fitting on clean epochs improves component separation
- **Preserve data**: Second pass can interpolate instead of rejecting
- **Transparency**: Comparison report shows both versions

**Alternative rejected**: Single AutoReject pass (original approach)

### 5. Multiple Preprocessing Outputs

**Decision**: Save 3 versions of preprocessed data

1. **Continuous (ICA-cleaned)**: `*_proc-clean_meg.fif`
2. **Epochs (ICA only)**: `*_proc-ica_meg.fif`
3. **Epochs (ICA+AR)**: `*_proc-icaar_meg.fif`

**Rationale**:
- **Flexibility**: Downstream analyses can choose version
- **Validation**: Compare results across versions
- **Transparency**: Users see impact of each cleaning step

**Alternative rejected**: Save only final ICA+AR version

### 6. FOOOF Per-Trial + Averaged

**Decision**: Fit FOOOF to individual trials AND IN/OUT averages

**Rationale**:
- **Trial-level variability**: Capture dynamics, not just averages
- **State comparisons**: IN vs. OUT contrast for group stats
- **Maximum information**: Averages reduce noise, trials preserve dynamics

**Alternative rejected**: Averaged-only (loses temporal information)

### 7. Provenance Tracking

**Decision**: Save git hash, timestamp, and parameters with every output

**Implementation**:
```python
# Every analysis saves metadata
metadata = {
    "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]),
    "timestamp": datetime.now().isoformat(),
    "script": __file__,
    "parameters": config,
}
```

**Rationale**:
- **Reproducibility**: Know exactly which code version produced results
- **Debugging**: Trace outputs back to code state
- **Publication**: Document methods completely

---

## Usage Examples

### Complete Single-Subject Workflow

```bash
# Setup
source env/bin/activate

# Stage 0: BIDS conversion (just this subject)
invoke bids --subjects "04"

# Stage 1: Preprocessing
invoke preprocess --subject 04 --runs "02 03"

# Check preprocessing report
firefox data/derivatives/preprocessed/sub-04/meg/*_desc-report_meg.html

# Stage 2: Source reconstruction
invoke source-recon --subject 04 --runs "02 03"

# Stage 3a: Welch PSD (sensor level)
python -m code.features.compute_welch_psd \
  --subject 04 --run 02 --space sensor

# Stage 3b: FOOOF (loads Welch PSD)
python -m code.features.compute_fooof \
  --subject 04 --run 02 --space sensor \
  --inout-bounds 25 75

# Stage 3c: Complexity metrics
python -m code.features.compute_complexity \
  --subject 04 --run 02 --space sensor
```

### Multi-Space Analysis

```bash
# Extract same features at all levels
for space in sensor source atlas; do
  python -m code.features.compute_welch_psd \
    --subject 04 --run 02 --space $space

  python -m code.features.compute_fooof \
    --subject 04 --run 02 --space $space
done
```

### Alternative IN/OUT Bounds

```bash
# Median split (50/50)
python -m code.features.compute_fooof \
  --subject 04 --run 02 --space sensor \
  --inout-bounds 50 50

# Conservative (10/90 percentile)
python -m code.features.compute_fooof \
  --subject 04 --run 02 --space sensor \
  --inout-bounds 10 90
```

---

## Configuration

### Essential Settings

Edit `config.yaml` to customize for your environment:

```yaml
paths:
  data_root: /media/storage/DATA/saflow  # Root data directory
  bids: bids/                            # Relative to data_root
  derivatives: derivatives/              # Relative to data_root
  freesurfer_subjects_dir: fs_subjects/  # Relative to data_root

bids:
  subjects:  # List of subject IDs to process
    - "04"
    - "05"
    # ... up to "38" (excluding 16, 25, 27)

  task_runs:  # Task run numbers
    - "02"
    - "03"
    - "04"
    - "05"
    - "06"
    - "07"

analysis:
  inout_bounds: [25, 75]  # VTC percentile thresholds

features:
  fooof:
    freq_range: [2, 120]           # Fitting range (Hz)
    peak_width_limits: [1, 8]      # Peak width (Hz)
    max_n_peaks: 4                 # Max peaks to fit
    aperiodic_mode: "fixed"        # "fixed" or "knee"
```

See `config.yaml.template` for all options.

---

## Directory Structure

```
saflow/
├── code/                      # All analysis code
│   ├── bids/                  # BIDS conversion
│   ├── preprocessing/         # Filtering, ICA, AutoReject
│   ├── source_reconstruction/ # MNE inverse solutions
│   ├── features/              # Feature extraction
│   ├── utils/                 # Shared utilities
│   ├── statistics/            # Stats (planned)
│   └── visualization/         # Plotting
├── config.yaml                # User configuration (gitignored)
├── config.yaml.template       # Config template
├── pyproject.toml             # Package definition
├── tasks.py                   # Invoke task runner
├── setup.sh                   # Environment setup
├── AGENTS.md                  # Development guidelines
├── PROGRESS.md                # Implementation tracker
└── TASKS.md                   # Task documentation
```

### Data Directory (separate)

```
/media/storage/DATA/saflow/
├── sourcedata/                # Raw CTF data (immutable)
│   ├── meg/
│   └── behav/
├── bids/                      # BIDS dataset (ground truth)
│   ├── sub-04/
│   │   └── meg/
│   │       ├── *_meg.ds/              # Raw MEG data
│   │       └── *_events.tsv           # Events with VTC, trial_idx
│   ├── sub-05/
│   └── code/                          # Provenance tracking
├── derivatives/               # Preprocessing & source reconstruction
│   ├── preprocessed/          # Stage 1: ICA-cleaned continuous data
│   │   └── sub-04/meg/
│   │       └── *_proc-clean_meg.fif
│   ├── epochs/                # Stage 1: Epoched data
│   │   └── sub-04/meg/
│   │       ├── *_proc-ica_meg.fif     # ICA-only epochs
│   │       └── *_proc-icaar_meg.fif   # ICA+AutoReject epochs
│   ├── morphed_sources/       # Stage 2: Source estimates (fsaverage)
│   ├── atlased_sources_*/     # Stage 2: Atlas-parcellated sources
│   └── noise_covariance/      # Noise covariance matrices
├── processed/                 # Feature extraction outputs
│   ├── welch_psds_sensor/     # Stage 3: Welch PSDs (sensor-level)
│   │   └── sub-04/
│   │       └── *_desc-welch_psds.npz
│   ├── welch_psds_source/     # Stage 3: Welch PSDs (source-level)
│   ├── features_fooof_sensor/ # Stage 3: FOOOF results (sensor)
│   │   └── sub-04/
│   │       └── *_desc-fooof.pkl
│   └── features_fooof_source/ # Stage 3: FOOOF results (source)
└── fs_subjects/               # FreeSurfer subjects
    └── fsaverage/
```

**Directory philosophy**:
- **bids/**: Immutable ground truth (VTC, trial metadata from behavioral data)
- **derivatives/**: Preprocessing outputs (cleaned signals, source estimates)
- **processed/**: Feature extraction outputs (PSDs, FOOOF, complexity metrics)

---

## Contributing

### Code Style

- **Formatting**: Ruff (line length 100)
- **Type hints**: All public functions
- **Docstrings**: Google style
- **Logging**: Use logger, not print()
- **Imports**: Absolute imports (`from code.utils...`)

### Testing

```bash
# Run tests
invoke test

# Code quality checks
invoke check  # Runs: lint + format + typecheck

# Pre-commit checks
invoke precommit  # Runs: format + lint + test
```

### Guidelines

See `AGENTS.md` for detailed development guidelines including:
- Function size (~10 lines when possible)
- No hardcoded paths
- Config-driven parameters
- Comprehensive logging
- Provenance tracking

---

## Citation

If you use this pipeline, please cite:

```bibtex
@software{saflow2026,
  title = {Saflow: MEG Analysis Pipeline for GradCPT},
  author = {Harel, Yann},
  year = {2026},
  url = {https://github.com/your-org/saflow}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Issues**: https://github.com/your-org/saflow/issues
- **Documentation**: See TASKS.md for detailed command reference
- **Development**: See AGENTS.md and PROGRESS.md

---

## Known Issues

This section documents known data quality issues identified through QC analysis. Run `invoke dev.check.qc` for the latest report.

### Missing Subjects

Subjects **16**, **25**, and **27** are excluded from analysis (no data available).

### ISI Timing Variability

Most subjects have very consistent inter-stimulus intervals (std < 2ms). However, some subjects show higher timing variability during recording:

| Subject | ISI std (ms) | Notes |
|---------|--------------|-------|
| sub-18  | 39.90 | High variability |
| sub-20  | 26.52 | Moderate variability |
| sub-21  | 20.86 | Moderate variability |
| sub-22  | 46.73 | High variability |
| sub-23  | 36.41 | High variability |
| sub-24  | 19.47 | Moderate variability |
| sub-26  | 33.19 | Moderate variability |

These timing inconsistencies may reflect hardware/software issues during data collection. Consider this when interpreting trial-locked analyses for these subjects.

### High Bad Epoch Rates

Bad epochs are identified using a 4000 fT peak-to-peak amplitude threshold. Most subjects have < 5% bad epochs, but several have elevated rates:

| Subject | Bad Epochs % | Notes |
|---------|--------------|-------|
| sub-07  | 23.3% | Review recommended |
| sub-19  | 17.2% | |
| sub-21  | 15.3% | Also has ISI variability |
| sub-38  | 12.1% | |
| sub-37  | 7.7% | |
| sub-26  | 7.4% | Also has ISI variability |
| sub-35  | 7.2% | |
| sub-05  | 6.7% | |

These subjects may have had more movement artifacts or environmental noise. The preprocessing pipeline (ICA + AutoReject) will handle most of these, but results should be interpreted with caution.

### Channel Quality

Channel quality is excellent across all subjects:
- **Bad channels**: 0-1% across all subjects
- **5 channels missing** from the CTF 275 system (270 detected vs 275 expected) - consistent across all recordings

### Behavioral Notes

- **Response rates**: Range from 78% (sub-15) to 95% (sub-04), all within expected range
- **Reaction times**: Range from 592ms (sub-10) to 901ms (sub-15)
- Sub-15 and sub-21 show lower response rates (~78-80%) which may indicate reduced engagement
