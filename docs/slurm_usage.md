# SLURM Usage Guide

This guide explains how to use the SLURM integration for distributed computing on HPC clusters.

## Quick Start

### Local Execution (Single Subject)
```bash
# Process one subject locally
invoke preprocess --subject=04

# Process specific runs locally
invoke preprocess --subject=04 --runs="02 03"
```

### SLURM Execution (Distributed)
```bash
# Process one subject on cluster (6 jobs, one per run)
invoke preprocess --subject=04 --slurm

# Process ALL subjects on cluster (192 jobs: 32 subjects × 6 runs)
invoke preprocess --slurm

# Process specific runs for one subject
invoke preprocess --subject=04 --runs="02 03" --slurm

# Dry run (generate scripts without submitting)
invoke preprocess --subject=04 --slurm --dry-run
```

## Configuration

SLURM settings are in `config.yaml`:

```yaml
computing:
  slurm:
    enabled: true
    account: def-kjerbi           # Your SLURM account
    partition: standard           # Partition to use
    email: user@example.com       # Optional: email notifications

    preprocessing:
      cpus: 12                    # CPUs per job
      mem: 32G                    # Memory per job
      time: "12:00:00"            # Wall time (HH:MM:SS)
```

## Job Distribution Strategy

The pipeline uses **per-run distribution**:
- One SLURM job per (subject, run) combination
- Maximizes parallelization across cluster
- Example: `invoke preprocess --slurm` submits 192 jobs (32 subjects × 6 runs)

This is more efficient than per-subject distribution because:
- Runs are independent and can be processed in parallel
- Better resource utilization
- Faster total processing time
- Each job completes in ~30-60 minutes instead of 3-6 hours

## Job Management

### Check Job Status
```bash
# View all your jobs
squeue -u $USER

# Check specific job
sacct -j JOBID

# Check job output in real-time
tail -f logs/slurm/preprocessing/preproc_sub-04_run-02_*.out
```

### Cancel Jobs
```bash
# Cancel specific job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# Cancel all preprocessing jobs (by name pattern)
scancel -u $USER --name="preproc_sub-*"
```

### Job Manifests

Each batch submission creates a manifest file tracking all submitted jobs:

```bash
logs/slurm/preprocessing/preprocessing_manifest_20240101_120000.json
```

Contains:
```json
{
  "job_ids": ["12345", "12346", "12347", ...],
  "num_jobs": 192,
  "metadata": {
    "stage": "preprocessing",
    "timestamp": "20240101_120000",
    "subjects": ["04", "05", "06", ...],
    "runs": ["02", "03", "04", "05", "06", "07"],
    "num_subjects": 32,
    "num_runs": 6
  }
}
```

Use this to:
- Track batch submissions
- Implement job dependencies for next stages
- Re-run failed jobs
- Generate reports

## Output Locations

### SLURM Scripts
Generated scripts are saved for reference:
```
slurm/scripts/preprocessing/
├── preproc_sub-04_run-02_20240101_120000.sh
├── preproc_sub-04_run-03_20240101_120000.sh
└── ...
```

### Job Logs
SLURM stdout/stderr logs:
```
logs/slurm/preprocessing/
├── preproc_sub-04_run-02_12345.out
├── preproc_sub-04_run-02_12345.err
└── ...
```

### Processing Outputs
Same locations as local execution:
```
derivatives/
├── preprocessed/sub-04/meg/*_proc-clean_meg.fif
└── epochs/sub-04/meg/*_proc-ica_meg.fif
```

## Templates

SLURM scripts are generated from Jinja2 templates:

- `slurm/templates/base.sh.j2` - Base template (inherited by all stages)
- `slurm/templates/preprocessing.sh.j2` - Preprocessing-specific

To add SLURM support to a new stage:
1. Create template (e.g., `features.sh.j2`) that extends `base.sh.j2`
2. Add `--slurm` support to invoke task
3. Configure resources in `config.yaml`

## Tips

1. **Start small**: Test with one subject before submitting all 192 jobs
   ```bash
   invoke preprocess --subject=04 --slurm
   ```

2. **Use dry-run**: Verify scripts before submitting
   ```bash
   invoke preprocess --slurm --dry-run
   ```

3. **Monitor progress**: Check logs directory for job outputs
   ```bash
   ls -lt logs/slurm/preprocessing/*.out | head
   ```

4. **Resource optimization**: Adjust CPU/memory in config based on actual usage
   - Check job efficiency with `seff JOBID`
   - Reduce resources if overallocated
   - Increase if jobs fail due to OOM or timeout

5. **Email notifications**: Enable in config to get notified on job completion/failure
   ```yaml
   slurm:
     email: your-email@example.com
   ```

## Troubleshooting

### Jobs fail immediately
- Check SLURM account is correct in config
- Verify partition exists: `sinfo`
- Check log files for errors

### Jobs timeout
- Increase time allocation in config
- Consider if data quality issues are causing slow processing
- Check if specific subjects/runs are problematic

### Out of memory errors
- Increase mem allocation in config
- Check if specific subjects have unusually large files
- Consider reducing n_jobs in preprocessing config

### Can't find venv
- Verify venv path in config.yaml
- Ensure venv was created on shared filesystem visible to compute nodes
- Check venv is activated in job output logs

## Future Stages

The same SLURM system will be used for:
- Feature extraction (`invoke extract-features --slurm`)
- Statistical tests (`invoke run-statistics --slurm`)
- Classification (`invoke run-classification --slurm`)

Each stage will have its own:
- Template in `slurm/templates/`
- Resource allocation in `config.yaml`
- Job manifest tracking
