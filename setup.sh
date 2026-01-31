#!/bin/bash
# ==============================================================================
# Saflow Setup Script
# ==============================================================================
# This script sets up the saflow development environment with interactive
# configuration prompts for data paths and HPC settings.
#
# Usage:
#   ./setup.sh [OPTIONS]
#
# Options:
#   --python      Specify Python executable (default: auto-detect best 3.9-3.12)
#   --force       Force reinstall if venv already exists
#   --help        Show this help message
#
# Examples:
#   ./setup.sh                      # Standard installation (auto-detects Python)
#   ./setup.sh --python python3.10  # Use specific Python version
#   ./setup.sh --force              # Force reinstall
#
# Interactive Configuration:
#   The script will prompt you to configure:
#   - Data root directory (where your MEG data lives)
#   - SLURM account (optional, leave empty if not using HPC)
#   You can skip prompts and edit config.yaml manually later.
# ==============================================================================

set -e  # Exit on error

# No colors for output

# Default values
PYTHON_CMD=""  # Will be auto-detected if not specified
FORCE_REINSTALL=false
VENV_DIR="env"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo "==================================================================="
    echo "$1"
    echo "==================================================================="
}

print_success() {
    echo "[OK] $1"
}

print_error() {
    echo "[ERROR] $1"
}

print_warning() {
    echo "[WARNING] $1"
}

print_info() {
    echo "[INFO] $1"
}

show_help() {
    head -n 21 "$0" | tail -n 15
    exit 0
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# ==============================================================================
# Python Auto-Detection
# ==============================================================================

# Find the best available Python version (prefer 3.11 > 3.10 > 3.9)
find_best_python() {
    local candidates=("python3.11" "python3.10" "python3.9" "python3")

    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            # Verify it's actually a supported version (3.9-3.12)
            local version_info=$("$cmd" -c 'import sys; print(sys.version_info.major, sys.version_info.minor)' 2>/dev/null)
            local major=$(echo "$version_info" | cut -d' ' -f1)
            local minor=$(echo "$version_info" | cut -d' ' -f2)

            if [[ "$major" == "3" ]] && [[ "$minor" -ge 9 ]] && [[ "$minor" -le 12 ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done

    return 1
}

# ==============================================================================
# Preflight Checks
# ==============================================================================

print_header "Saflow Environment Setup"

# Auto-detect Python if not specified
if [[ -z "$PYTHON_CMD" ]]; then
    PYTHON_CMD=$(find_best_python)
    if [[ -z "$PYTHON_CMD" ]]; then
        print_error "No suitable Python found (requires 3.9-3.12)"
        echo "  Please install Python 3.9+ or specify a different executable with --python"
        exit 1
    fi
    print_info "Auto-detected Python: $PYTHON_CMD"
fi

# Check if Python is available
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    print_error "Python executable '$PYTHON_CMD' not found"
    echo "  Please install Python 3.9+ or specify a different executable with --python"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

# Validate version numbers are integers
if ! [[ "$PYTHON_MAJOR" =~ ^[0-9]+$ ]] || ! [[ "$PYTHON_MINOR" =~ ^[0-9]+$ ]]; then
    print_error "Could not parse Python version (got: $PYTHON_VERSION)"
    exit 1
fi

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 9 ]]; then
    print_error "Python 3.9+ is required (found: $PYTHON_VERSION)"
    exit 1
fi

print_success "Found Python $PYTHON_VERSION"

# Check if venv module is available
if ! "$PYTHON_CMD" -c "import venv" 2>/dev/null; then
    print_error "Python venv module not found"
    echo "  Please install python3-venv package:"
    echo "    Ubuntu/Debian: sudo apt-get install python3-venv"
    echo "    CentOS/RHEL: sudo yum install python3-venv"
    exit 1
fi

# ==============================================================================
# Virtual Environment Setup
# ==============================================================================

print_header "Setting up virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    if [[ "$FORCE_REINSTALL" == true ]]; then
        print_warning "Removing existing virtual environment"
        rm -rf "$VENV_DIR"
    else
        print_warning "Virtual environment already exists at: $VENV_DIR"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            print_info "Using existing virtual environment"
        fi
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    print_info "Creating virtual environment with $PYTHON_CMD"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    print_success "Virtual environment created at: $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip, setuptools, wheel
print_info "Upgrading pip, setuptools, and wheel"
pip install --upgrade pip setuptools wheel > /dev/null
print_success "Build tools updated"

# ==============================================================================
# Package Installation
# ==============================================================================

print_header "Installing saflow package"

print_info "Installing saflow with all dependencies"
pip install -e ".[all]"

print_success "Package installation complete"

# ==============================================================================
# Configuration Setup
# ==============================================================================

print_header "Configuration setup"

if [[ ! -f "config.yaml" ]]; then
    if [[ -f "config.yaml.template" ]]; then
        print_info "Creating config.yaml from template"
        cp config.yaml.template config.yaml
        print_success "config.yaml created"
        echo ""

        # Interactive configuration prompts
        print_info "Let's configure your environment"
        echo ""

        # Prompt for data root path
        echo "Data root directory:"
        echo "  This is where your data lives (sourcedata/, derivatives/, etc.)"
        read -p "  Enter full path (or press Enter to configure manually later): " DATA_ROOT
        echo ""

        if [[ -n "$DATA_ROOT" ]]; then
            # Expand tilde and remove trailing slash
            DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
            DATA_ROOT="${DATA_ROOT%/}"

            # Validate path exists
            if [[ -d "$DATA_ROOT" ]]; then
                # Escape special characters for sed
                DATA_ROOT_ESCAPED=$(echo "$DATA_ROOT" | sed 's/[\/&]/\\&/g')
                sed -i "s|<DATA_ROOT>|$DATA_ROOT|g" config.yaml
                print_success "Data root set to: $DATA_ROOT"
            else
                print_warning "Directory does not exist: $DATA_ROOT"
                read -p "  Create it now? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    mkdir -p "$DATA_ROOT"
                    print_success "Created directory: $DATA_ROOT"
                    DATA_ROOT_ESCAPED=$(echo "$DATA_ROOT" | sed 's/[\/&]/\\&/g')
                    sed -i "s|<DATA_ROOT>|$DATA_ROOT|g" config.yaml
                    print_success "Data root set to: $DATA_ROOT"
                else
                    print_info "Placeholder <DATA_ROOT> left in config.yaml for manual editing"
                fi
            fi
        else
            print_info "Skipped - you'll need to edit config.yaml manually"
        fi

        echo ""

        # Prompt for SLURM account (optional)
        echo "SLURM account (for HPC job submissions):"
        echo "  Leave empty if you don't use HPC/SLURM"
        read -p "  Enter your SLURM account (or press Enter to skip): " SLURM_ACCOUNT
        echo ""

        if [[ -n "$SLURM_ACCOUNT" ]]; then
            # User provided SLURM account
            sed -i "s|account:.*|account: \"$SLURM_ACCOUNT\"|g" config.yaml
            print_success "SLURM account set to: $SLURM_ACCOUNT"
        else
            # User skipped - set to empty string (not placeholder)
            sed -i "s|account:.*|account: \"\"|g" config.yaml
            print_info "SLURM account left empty (HPC features disabled)"
            print_info "  You can set it later in config.yaml if needed"
        fi

        echo ""

        # Check if any placeholders remain
        if grep -q "<.*>" config.yaml; then
            print_warning "Some placeholders still remain in config.yaml"
            echo ""
            echo "  You can edit them manually later with:"
            echo "    nano config.yaml"
        else
            print_success "All placeholders configured!"
        fi
        echo ""
    else
        print_error "config.yaml.template not found"
        exit 1
    fi
else
    print_info "config.yaml already exists (not overwriting)"
fi

# ==============================================================================
# Directory Creation
# ==============================================================================

print_header "Creating project directories"

# Create directories that should exist but are empty in git
mkdir -p logs
mkdir -p reports/figures
mkdir -p reports/tables
mkdir -p reports/statistics

print_success "Project directories created"

# ==============================================================================
# Development Tools Setup
# ==============================================================================

print_header "Setting up development tools"

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [[ -f ".pre-commit-config.yaml" ]]; then
    print_info "Installing pre-commit hooks"
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_info "No .pre-commit-config.yaml found (skipping pre-commit setup)"
fi

# ==============================================================================
# Verification
# ==============================================================================

print_header "Verifying installation"

# Check if code package is importable
if python -c "import code.utils.config" 2>/dev/null; then
    print_success "code.utils package can be imported"
else
    print_error "code.utils package cannot be imported"
    exit 1
fi

# Check if configuration can be loaded (if config.yaml has no placeholders)
if python -c "from code.utils.config import load_config; load_config()" 2>/dev/null; then
    print_success "Configuration loads successfully"
else
    print_warning "Configuration cannot be loaded yet (placeholders need to be replaced)"
fi

# ==============================================================================
# Summary
# ==============================================================================

print_header "Setup complete!"

echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source env/bin/activate"
echo ""
echo "  2. Edit config.yaml if needed (paths, SLURM account, etc.):"
echo "     nano config.yaml"
echo ""
echo "  3. Verify configuration:"
echo "     python -c 'from code.utils.config import load_config; load_config()'"
echo ""
echo "  4. Run tests to verify everything works:"
echo "     pytest tests/"
echo ""
echo "  5. Run linting:"
echo "     ruff check saflow/ code/"
echo ""
echo ""
echo "For more information, see:"
echo "  - README.md: Project overview and usage"
echo "  - TASKS.md: Pipeline scripts and examples"
echo "  - docs/workflow.md: Detailed pipeline documentation"
echo ""

print_success "Setup script finished successfully!"
