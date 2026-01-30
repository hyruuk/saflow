# Saflow Pipeline Task Reference

This document describes the invoke tasks for running the saflow MEG analysis pipeline.

**Quick reference**: Run `invoke --list` to see all available tasks.

---

## Pipeline Tasks

### `invoke validate-inputs`

Validate that raw input data is present and complete before running pipeline.

**What it checks:**
- Raw MEG data directory exists (`data_root/raw/meg/`)
- Behavioral logfiles directory exists (`data_root/raw/behav/`)
- Required `.ds` files are present
- Behavioral `.mat` files are present
- All expected subjects have data
- Shows summary table of available runs per subject

**Arguments:**
- `--data-root PATH`: Override data root from config
- `--verbose`: Show detailed file listings

**Examples:**
```bash
# Basic validation
invoke validate-inputs

# Verbose output with file listings
invoke validate-inputs --verbose

# Override data location
invoke validate-inputs --data-root=/media/storage/DATA/saflow
```

**Output:**
- Colored console report with ✓/✗ indicators
- Table showing runs per subject
- List of missing subjects (if any)
- Summary of behavioral files

---

### `invoke bids`

Run BIDS conversion (Stage 0 of pipeline).

**What it does:**
1. Finds all CTF MEG datasets (`.ds` files) in raw directory
2. Converts to BIDS format using mne-bids
3. For gradCPT task runs:
   - Loads behavioral logfiles
   - Computes VTC (variability of reaction times)
   - Enriches events.tsv with:
     - Trial indices
     - VTC and RT values
     - Task performance (commission_error, correct_omission, etc.)
     - IN/OUT zone classifications (50/50, 25/75, 10/90 percentile bounds)
4. Writes empty-room noise recordings
5. Saves provenance metadata (git hash, timestamp, subjects processed)

**Arguments:**
- `--input-dir PATH`: Override raw data directory from config
- `--output-dir PATH`: Override BIDS output directory from config
- `--subjects "ID1 ID2 ..."`: Process specific subjects only (space-separated)
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--dry-run`: Validate inputs without processing files

**Examples:**
```bash
# Basic usage (uses paths from config.yaml)
invoke bids

# Process specific subjects only
invoke bids --subjects "04 05 06"

# Override input/output paths
invoke bids --input-dir=/path/to/raw --output-dir=/path/to/bids

# Dry run to validate before processing
invoke bids --dry-run

# Debug mode with verbose logging
invoke bids --log-level=DEBUG
```

**Output locations:**
- BIDS dataset: `{config.paths.data_root}/bids/`
- Logs: `logs/bids/bids_conversion_YYYYMMDD_HHMMSS.log`
- Provenance: `{bids_root}/code/provenance_bids.json`

**Expected runtime:**
- ~8-10 minutes for 32 subjects (6 task runs + 1 rest run each)

---

### `invoke preprocess`

Run MEG preprocessing (Stage 1 of pipeline).

**What it does:**
1. Loads raw BIDS data
2. Applies gradient compensation (grade 3, required for source reconstruction)
3. Applies bandpass filtering (0.1-200 Hz) and notch filtering (60 Hz harmonics)
4. Creates epochs for all events
5. Runs AutoReject (first pass) to identify bad epochs for ICA
6. Computes or loads noise covariance from empty-room recording
7. Fits ICA on good epochs
8. Detects and removes ECG and EOG artifact components
9. Runs AutoReject (second pass) with interpolation on ICA-cleaned data
10. Saves:
    - Preprocessed continuous data (ICA-cleaned, FIF format)
    - Clean epochs (ICA only version)
    - Clean epochs (ICA+AR version with interpolation)
    - Both AutoReject logs (pickle)
    - HTML report with diagnostic plots
    - Text summary with all metrics
    - Processing metadata (JSON)

**Arguments:**
- `--subject ID`: Subject ID to process (default: all subjects from config when using --slurm)
- `--runs "R1 R2 ..."`: Run numbers to process (space-separated, default: all task runs from config)
- `--bids-root PATH`: Override BIDS root directory from config
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--skip-existing` / `--no-skip-existing`: Skip/reprocess existing files (default: skip)
- `--slurm`: Submit jobs to SLURM cluster (one job per run)
- `--dry-run`: Generate SLURM scripts without submitting (requires --slurm)

**Examples:**
```bash
# Local execution (single subject)
invoke preprocess --subject=04
invoke preprocess --subject=04 --runs="02 03"
invoke preprocess --subject=04 --log-level=DEBUG

# SLURM execution (distributed processing)
invoke preprocess --subject=04 --slurm              # One job per run for subject 04
invoke preprocess --slurm                           # Process ALL subjects (192 jobs)
invoke preprocess --subject=04 --runs="02 03" --slurm  # Specific runs on SLURM
invoke preprocess --slurm --dry-run                 # Generate scripts without submitting
```

**Output locations:**
- Continuous (ICA-cleaned): `{data_root}/derivatives/preprocessed/sub-{subject}/meg/*_proc-clean_meg.fif`
- Epochs (ICA only): `{data_root}/derivatives/epochs/sub-{subject}/meg/*_proc-ica_meg.fif`
- Epochs (ICA+AR): `{data_root}/derivatives/epochs/sub-{subject}/meg/*_proc-icaar_meg.fif`
- HTML reports: `{derivatives}/preprocessed/sub-{subject}/meg/*_desc-report_meg.html`
- Text summaries: `{derivatives}/preprocessed/sub-{subject}/meg/*_desc-report_meg_summary.txt`
- AutoReject logs: `{derivatives}/preprocessed/sub-{subject}/meg/*_desc-ARlog{1,2}_meg.pkl`
- Metadata: `{derivatives}/preprocessed/sub-{subject}/meg/*_params.json`
- Logs: `logs/preprocessing/preprocessing_sub-{subject}_YYYYMMDD_HHMMSS.log`

**Expected runtime:**
- ~30-60 minutes per run (depends on n_jobs)
- Longer for first run (computes noise covariance)
- Can run multiple subjects in parallel

**Configuration parameters used:**
```yaml
preprocessing:
  filter:
    lowcut: 0.1          # Hz
    highcut: 200         # Hz
    notch: [60]          # Hz
  ica:
    method: "fastica"
    n_components: 20
    random_state: 42
  autoreject:
    n_interpolate: [1, 4, 32]
    consensus: [0.1, 0.2, 0.3, 0.5]

computing:
  n_jobs: -1  # Use all CPUs
```

---

### `invoke source-recon`

Run source reconstruction (Stage 2 of pipeline).

**What it does:**
1. Computes coregistration between MEG and MRI coordinate systems
2. Sets up source space (cortical surface model)
3. Creates BEM (Boundary Element Model) for forward modeling
4. Computes forward solution (leadfield matrix)
5. Estimates noise covariance from empty-room recording
6. Applies inverse operator to compute source estimates
7. Morphs source estimates to fsaverage template for group analysis

Requires:
- Preprocessed data from `invoke preprocess` (Stage 1)
- FreeSurfer subjects directory with fsaverage (individual MRIs optional)
- Empty-room noise recordings in BIDS dataset

**Arguments:**
- `--subject ID`: Subject ID to process (default: all subjects from config when using --slurm)
- `--runs "R1 R2 ..."`: Run numbers to process (space-separated, default: all task runs from config)
- `--input-type TYPE`: Input data type: `continuous` (ICA-cleaned) or `epochs` (ICA+AutoReject) (default: continuous)
- `--processing STATE`: Processing state: `clean` (continuous), `ica` or `icaar` (epochs) (default: clean)
- `--bids-root PATH`: Override BIDS root directory from config
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--skip-existing` / `--no-skip-existing`: Skip/reprocess existing files (default: skip)
- `--slurm`: Submit jobs to SLURM cluster (one job per run)
- `--dry-run`: Generate SLURM scripts without submitting (requires --slurm)

**TWO-TRACK PROCESSING:**
Source reconstruction now supports both continuous and epoched data, enabling comparison between:
- **Track 1 (continuous)**: ICA-cleaned continuous data → Single long source estimate
- **Track 2 (epochs)**: ICA+AutoReject epoched data → Multiple epoched source estimates

This allows comparing results from both preprocessing approaches for robustness validation.

**Examples:**
```bash
# Local execution - continuous (default)
invoke source-recon --subject=04 --runs="02 03"
invoke source-recon --subject=04 --log-level=DEBUG

# Local execution - epochs
invoke source-recon --subject=04 --runs="02" --input-type=epochs --processing=icaar

# SLURM execution (distributed processing)
invoke source-recon --subject=04 --slurm              # One job per run for subject 04
invoke source-recon --slurm                           # Process ALL subjects (192 jobs)
invoke source-recon --subject=04 --runs="02 03" --slurm  # Specific runs on SLURM
invoke source-recon --slurm --dry-run                 # Generate scripts without submitting

# Two-track processing (both continuous and epochs)
invoke source-recon --subject=04 --runs="02" --input-type=continuous
invoke source-recon --subject=04 --runs="02" --input-type=epochs --processing=icaar
```

**Output locations:**
- Coregistration transforms: `{derivatives}/trans/sub-{subject}/meg/*_proc-trans_meg.fif`
- Forward solutions: `{derivatives}/fwd/sub-{subject}/meg/*_proc-forward_meg.fif`
- Noise covariance: `{derivatives}/noise_cov/sub-emptyroom/meg/sub-emptyroom_*.fif`
- Source estimates: `{derivatives}/minimum-norm-estimate/sub-{subject}/meg/*_desc-sources_meg-stc.h5`
- Morphed sources: `{derivatives}/morphed_sources/sub-{subject}/meg/*_desc-morphed_meg-stc.h5`
- Metadata: `{derivatives}/morphed_sources/sub-{subject}/meg/*_params.json`
- Logs: `logs/source_reconstruction/source_recon_sub-{subject}_YYYYMMDD_HHMMSS.log`

**Expected runtime:**
- ~30-120 minutes per run (depends on MRI availability and hardware)
- Coregistration: ~2-5 minutes
- Forward solution: ~5-10 minutes
- Inverse solution: ~10-30 minutes
- Morphing: ~10-30 minutes
- First run takes longer (computes noise covariance)
- Individual MRI processing is slower than fsaverage

**Configuration parameters used:**
```yaml
source_reconstruction:
  method: dSPM          # Inverse method (dSPM, MNE, sLORETA)
  snr: 3.0              # Signal-to-noise ratio
  atlas: aparc.a2009s   # Atlas for parcellation

paths:
  freesurfer_subjects_dir: fs_subjects/  # FreeSurfer directory

computing:
  n_jobs: -1  # Parallel jobs for forward solution
  slurm:
    source_reconstruction:
      cpus: 1
      mem: 256G           # Large memory for morphing
      time: "2:00:00"
```

---

### `invoke apply-atlas` (Optional)

Apply cortical parcellation to source estimates (Stage 2b).

This is an optional post-processing step that averages source time series within
ROIs defined by a cortical atlas (e.g., aparc.a2009s). Useful for ROI-based
analyses in source space.

**Arguments:**
- `--subject ID`: Subject ID to process (required)
- `--run ID` or `--runs "R1 R2 ..."`: Run(s) to process
- `--atlas NAME`: Atlas name (default: from config, e.g., aparc.a2009s)
- `--processing STATE`: Processing state (clean, ica, icaar; default: clean)

**Examples:**
```bash
invoke apply-atlas --subject=04 --runs="02 03"
invoke apply-atlas --subject=04 --run=02 --atlas=aparc
```

**Output:**
- ROI-averaged time series (pickle format): `{derivatives}/atlased_sources_{atlas}/*-avg.pkl`

---

## Utility Tasks

### `invoke test`

Run test suite with pytest.

**Arguments:**
- `--verbose`: Enable verbose output
- `--coverage` / `--no-coverage`: Generate coverage report (default: True)

**Examples:**
```bash
# Run all tests with coverage
invoke test

# Verbose output
invoke test --verbose

# Without coverage report
invoke test --no-coverage
```

---

### `invoke info`

Display project information and configuration status.

Shows:
- Project root and directory paths
- Configuration file status
- Virtual environment status
- Python version

**Examples:**
```bash
invoke info
```

---

## SLURM Integration (HPC)

Pipeline tasks support distributed execution on HPC clusters via SLURM.

### Using SLURM

Add `--slurm` to any pipeline task to submit jobs to the cluster:

```bash
# Process all subjects in parallel (one job per run)
invoke preprocess --slurm

# Process specific subject on cluster
invoke preprocess --subject=04 --slurm

# Dry run: generate scripts without submitting
invoke preprocess --slurm --dry-run
```

### Configuration

SLURM settings are configured in `config.yaml`:

```yaml
computing:
  slurm:
    enabled: true
    account: def-kjerbi           # Your SLURM account
    partition: standard           # Partition to use
    email: user@example.com       # Optional email notifications

    preprocessing:
      cpus: 12
      mem: 32G
      time: "12:00:00"
```

### Job Management

**Check job status:**
```bash
squeue -u $USER                   # View your jobs
sacct -j JOBID                    # Check specific job status
```

**Cancel jobs:**
```bash
scancel JOBID                     # Cancel specific job
scancel -u $USER                  # Cancel all your jobs
```

**Job manifests:**

Each batch submission creates a manifest file:
```bash
logs/slurm/preprocessing/preprocessing_manifest_YYYYMMDD_HHMMSS.json
```

Contains:
- List of all job IDs
- Subjects and runs processed
- Timestamps and metadata

**Logs:**

SLURM output logs are saved to:
```bash
logs/slurm/preprocessing/preproc_sub-{subject}_run-{run}_{jobid}.out
logs/slurm/preprocessing/preproc_sub-{subject}_run-{run}_{jobid}.err
```

---

## Notes

- All pipeline tasks support `--help` to see detailed options
- Logs are saved to `logs/{stage}/` for each pipeline stage
- Most tasks are idempotent (can be re-run safely)
- Use `--skip-existing` (default) to avoid reprocessing
- Configuration is loaded from `config.yaml` (see `config.yaml.template`)
