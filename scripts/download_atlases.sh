#!/bin/bash
# Download atlas annotation files for FreeSurfer fsaverage
# Includes: Schaefer 2018 (100, 200, 400 parcels) and Destrieux (aparc.a2009s)
# These are required for source-level parcellation with MNE-Python

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config.yaml not found at $CONFIG_FILE"
    exit 1
fi

# Parse paths from config.yaml using Python (more reliable than grep for YAML)
DATA_ROOT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['paths']['data_root'])")
FS_SUBJECTS_REL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['paths']['freesurfer_subjects_dir'])")

# Construct full path to fsaverage label directory
LABEL_DIR="$DATA_ROOT/$FS_SUBJECTS_REL/fsaverage/label"

echo "=============================================="
echo "FreeSurfer Atlas Downloader"
echo "=============================================="
echo "Config file: $CONFIG_FILE"
echo "Data root: $DATA_ROOT"
echo "Target directory: $LABEL_DIR"
echo ""

# Check if target directory exists
if [[ ! -d "$LABEL_DIR" ]]; then
    echo "Error: Label directory does not exist: $LABEL_DIR"
    echo "Make sure fsaverage is installed in your FreeSurfer subjects directory."
    exit 1
fi

cd "$LABEL_DIR"

# Base URLs
SCHAEFER_URL="https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label"
FREESURFER_URL="https://raw.githubusercontent.com/freesurfer/freesurfer/dev/subjects/fsaverage/label"

# Generic download function
download_file() {
    local url=$1
    local filename=$2

    if [[ -f "$filename" ]]; then
        echo "  [SKIP] $filename already exists"
    else
        echo "  [DOWNLOAD] $filename"
        wget -q "$url/$filename" -O "$filename"
        if [[ $? -eq 0 ]]; then
            echo "  [OK] $filename"
        else
            echo "  [FAIL] $filename"
            rm -f "$filename"  # Clean up partial download
            return 1
        fi
    fi
}

# Download Destrieux atlas (aparc.a2009s)
echo ""
echo "Downloading Destrieux atlas (aparc.a2009s)..."
download_file "$FREESURFER_URL" "lh.aparc.a2009s.annot"
download_file "$FREESURFER_URL" "rh.aparc.a2009s.annot"

# Download all Schaefer atlases (100, 200, 400 parcels)
for parcels in 100 200 400; do
    echo ""
    echo "Downloading Schaefer ${parcels} parcels..."
    download_file "$SCHAEFER_URL" "lh.Schaefer2018_${parcels}Parcels_7Networks_order.annot"
    download_file "$SCHAEFER_URL" "rh.Schaefer2018_${parcels}Parcels_7Networks_order.annot"
done

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="
echo ""
echo "Installed atlases:"
echo ""
echo "Destrieux (aparc.a2009s):"
ls -la "$LABEL_DIR"/*aparc.a2009s* 2>/dev/null || echo "  (none found)"
echo ""
echo "Schaefer 2018:"
ls -la "$LABEL_DIR"/*Schaefer* 2>/dev/null || echo "  (none found)"
