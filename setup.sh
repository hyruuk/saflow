#!/bin/bash
# ==============================================================================
# Saflow Setup Script
# ==============================================================================
# This script sets up the saflow development environment.
#
# Usage:
#   ./setup.sh [OPTIONS]
#
# Options:
#   --dev         Install development dependencies ([dev,test,docs])
#   --hpc         Install HPC dependencies (for SLURM)
#   --all         Install all optional dependencies
#   --python      Specify Python executable (default: python3.9)
#   --force       Force reinstall if venv already exists
#   --help        Show this help message
#
# Examples:
#   ./setup.sh                  # Basic installation
#   ./setup.sh --dev            # Install with dev tools
#   ./setup.sh --all            # Install everything
#   ./setup.sh --python python3.10  # Use specific Python version
# ==============================================================================

set -e  # Exit on error

# No colors for output

# Default values
PYTHON_CMD="python3.9"
INSTALL_MODE="basic"
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
    head -n 25 "$0" | tail -n 19
    exit 0
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            INSTALL_MODE="dev"
            shift
            ;;
        --hpc)
            INSTALL_MODE="hpc"
            shift
            ;;
        --all)
            INSTALL_MODE="all"
            shift
            ;;
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
# Preflight Checks
# ==============================================================================

print_header "Saflow Environment Setup"

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

case "$INSTALL_MODE" in
    basic)
        print_info "Installing basic dependencies"
        pip install -e .
        ;;
    dev)
        print_info "Installing with development dependencies"
        pip install -e ".[dev,test,docs]"
        ;;
    hpc)
        print_info "Installing with HPC dependencies"
        pip install -e ".[hpc]"
        ;;
    all)
        print_info "Installing all dependencies"
        pip install -e ".[all]"
        ;;
esac

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
        print_warning "IMPORTANT: Edit config.yaml and replace all <PLACEHOLDER> values with actual paths"
        echo ""
        echo "  Required placeholders to update:"
        echo "    - paths.data_root: Location of your data directory"
        echo "    - computing.slurm.account: Your SLURM account (if using HPC)"
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
# Development Tools Setup (if dev mode)
# ==============================================================================

if [[ "$INSTALL_MODE" == "dev" || "$INSTALL_MODE" == "all" ]]; then
    print_header "Setting up development tools"

    # Install pre-commit hooks if .pre-commit-config.yaml exists
    if [[ -f ".pre-commit-config.yaml" ]]; then
        print_info "Installing pre-commit hooks"
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_info "No .pre-commit-config.yaml found (skipping pre-commit setup)"
    fi
fi

# ==============================================================================
# Verification
# ==============================================================================

print_header "Verifying installation"

# Check if saflow package is importable
if python -c "import saflow" 2>/dev/null; then
    print_success "saflow package can be imported"
else
    print_error "saflow package cannot be imported"
    exit 1
fi

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
echo "  2. Edit config.yaml and replace all <PLACEHOLDER> values:"
echo "     nano config.yaml"
echo ""
echo "  3. Verify configuration:"
echo "     python -c 'from code.utils.config import load_config; load_config()'"
echo ""

if [[ "$INSTALL_MODE" == "dev" || "$INSTALL_MODE" == "all" ]]; then
    echo "  4. Run tests to verify everything works:"
    echo "     pytest tests/"
    echo ""
    echo "  5. Run linting:"
    echo "     ruff check saflow/ code/"
    echo ""
fi

echo "For more information, see:"
echo "  - README.md: Project overview and usage"
echo "  - TASKS.md: Pipeline scripts and examples"
echo "  - docs/workflow.md: Detailed pipeline documentation"
echo ""

print_success "Setup script finished successfully!"
