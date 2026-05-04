#!/bin/bash
# Download the fsaverage FreeSurfer template (surfaces, BEM, head model).
# Required for source reconstruction when no individual MRI is available:
# subjects without an MRI are coregistered against a per-subject scaled copy
# of fsaverage (see code/source_reconstruction/utils.ensure_subject_anatomy).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config.yaml not found at $CONFIG_FILE"
    exit 1
fi

DATA_ROOT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['paths']['data_root'])")
FS_SUBJECTS_REL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['paths']['freesurfer_subjects_dir'])")

# freesurfer_subjects_dir may be absolute or relative to data_root
if [[ "$FS_SUBJECTS_REL" = /* ]]; then
    FS_SUBJECTS_DIR="$FS_SUBJECTS_REL"
else
    FS_SUBJECTS_DIR="$DATA_ROOT/$FS_SUBJECTS_REL"
fi

echo "=============================================="
echo "fsaverage Template Downloader"
echo "=============================================="
echo "Target subjects dir: $FS_SUBJECTS_DIR"
echo ""

mkdir -p "$FS_SUBJECTS_DIR"

export FS_SUBJECTS_DIR
python3 - <<PY
import os
from pathlib import Path
import mne

subjects_dir = Path(os.environ["FS_SUBJECTS_DIR"])
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)

# Sanity check: head BEM must be present, otherwise coregistration fails
fsaverage = subjects_dir / "fsaverage"
head_bems = list((fsaverage / "bem").glob("*-head*.fif"))
if not head_bems:
    raise SystemExit(
        f"fsaverage downloaded but no head BEM found in {fsaverage}/bem. "
        "Try removing the directory and re-running."
    )
print(f"OK: fsaverage at {fsaverage}")
print(f"Head BEM files: {[p.name for p in head_bems]}")
PY

echo ""
echo "=============================================="
echo "fsaverage ready"
echo "=============================================="
