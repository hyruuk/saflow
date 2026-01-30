"""Analysis space utilities for saflow.

This module provides utilities for handling sensor-level vs source-level
analysis spaces. It ensures that the correct code paths and file locations
are used based on the configured analysis space.
"""

from pathlib import Path
from typing import Dict, Any, Optional


class AnalysisSpaceError(Exception):
    """Raised when analysis space configuration is invalid or incompatible."""

    pass


def validate_analysis_space(config: Dict[str, Any], required_space: Optional[str] = None) -> None:
    """Validate that the analysis space is correctly configured.

    Args:
        config: Configuration dictionary.
        required_space: Optional required analysis space ('sensor' or 'source').
            If provided, raises error if config space doesn't match.

    Raises:
        AnalysisSpaceError: If analysis space is invalid or doesn't match required.
    """
    current_space = config["analysis"]["space"]
    valid_spaces = ["sensor", "source"]

    if current_space not in valid_spaces:
        raise AnalysisSpaceError(
            f"Invalid analysis.space: {current_space}. Must be one of {valid_spaces}"
        )

    if required_space and current_space != required_space:
        raise AnalysisSpaceError(
            f"This script requires analysis.space='{required_space}', "
            f"but config has '{current_space}'"
        )


def get_space_suffix(config: Dict[str, Any]) -> str:
    """Get suffix for filenames based on analysis space.

    Args:
        config: Configuration dictionary.

    Returns:
        Space suffix: '_sensor' or '_source'.
    """
    space = config["analysis"]["space"]
    return f"_{space}"


def get_space_dir(base_dir: Path, space: str) -> Path:
    """Get space-specific subdirectory.

    Args:
        base_dir: Base directory path.
        space: Analysis space ('sensor' or 'source').

    Returns:
        Path to space-specific subdirectory.

    Example:
        >>> base_dir = Path('processed/features')
        >>> get_space_dir(base_dir, 'sensor')
        Path('processed/features_sensor')
    """
    return base_dir.parent / f"{base_dir.name}_{space}"


def route_to_space_module(
    config: Dict[str, Any], module_base: str, function_name: str
) -> str:
    """Route to the appropriate module based on analysis space.

    This is a helper for dynamically importing the correct module
    (sensor vs source) based on configuration.

    Args:
        config: Configuration dictionary.
        module_base: Base module path (e.g., 'code.features').
        function_name: Function name to import.

    Returns:
        Full module path for import.

    Example:
        >>> route_to_space_module(config, 'code.features', 'compute_psd')
        'code.features.sensor.compute_psd'  # if space='sensor'
    """
    space = config["analysis"]["space"]
    return f"{module_base}.{space}.{function_name}"


def check_source_prerequisites(config: Dict[str, Any]) -> None:
    """Check that prerequisites for source-level analysis are met.

    Args:
        config: Configuration dictionary.

    Raises:
        AnalysisSpaceError: If source-level prerequisites are not met.
    """
    if config["analysis"]["space"] != "source":
        return

    # Check FreeSurfer subjects directory is configured
    if "freesurfer_subjects_dir" not in config["paths"]:
        raise AnalysisSpaceError(
            "Source-level analysis requires paths.freesurfer_subjects_dir "
            "to be configured"
        )

    fs_dir = Path(config["paths"]["freesurfer_subjects_dir"])
    if not fs_dir.exists():
        raise AnalysisSpaceError(
            f"FreeSurfer subjects directory does not exist: {fs_dir}"
        )

    # Check source reconstruction settings
    if "use_atlas" in config["analysis"]["source"]:
        if config["analysis"]["source"]["use_atlas"]:
            if "atlas_name" not in config["analysis"]["source"]:
                raise AnalysisSpaceError(
                    "Source-level analysis with atlas requires "
                    "analysis.source.atlas_name to be configured"
                )


def get_space_label(config: Dict[str, Any]) -> str:
    """Get human-readable label for current analysis space.

    Args:
        config: Configuration dictionary.

    Returns:
        Space label: 'Sensor-level' or 'Source-level'.
    """
    space = config["analysis"]["space"]
    return f"{space.capitalize()}-level"


def format_output_path(
    base_path: Path, config: Dict[str, Any], include_space: bool = True
) -> Path:
    """Format output path to include analysis space if needed.

    Args:
        base_path: Base output path.
        config: Configuration dictionary.
        include_space: Whether to include space in path (default: True).

    Returns:
        Formatted path with space suffix if include_space=True.
    """
    if not include_space:
        return base_path

    space = config["analysis"]["space"]
    stem = base_path.stem
    suffix = base_path.suffix

    # Add space to filename stem
    new_stem = f"{stem}_{space}"
    return base_path.parent / f"{new_stem}{suffix}"


def is_source_implemented() -> bool:
    """Check if source-level analysis is implemented.

    This is a temporary function for the refactoring phase.
    During Phase 1, source-level analysis is not yet implemented.

    Returns:
        True if source-level analysis is implemented, False otherwise.
    """
    # TODO: Update this to True when Phase 2 (source-level) is implemented
    return False


def require_sensor_space(config: Dict[str, Any]) -> None:
    """Require sensor-level analysis space (for Phase 1).

    This is a temporary check during the refactoring. Scripts that are
    only implemented for sensor-level should call this function.

    Args:
        config: Configuration dictionary.

    Raises:
        AnalysisSpaceError: If analysis space is not 'sensor'.
    """
    if config["analysis"]["space"] != "sensor":
        raise AnalysisSpaceError(
            "This script currently only supports sensor-level analysis. "
            "Source-level analysis will be implemented in Phase 2. "
            "Please set analysis.space='sensor' in config.yaml"
        )
