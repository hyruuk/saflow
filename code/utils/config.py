"""Configuration management for saflow.

This module provides configuration loading, validation, and access utilities.
All configuration is loaded from a YAML file (config.yaml) which should be
created from config.yaml.template.

Usage:
    from code.utils.config import load_config, get_config

    # Load configuration (typically done once at startup)
    config = load_config('config.yaml')

    # Access configuration values
    data_root = config['paths']['data_root']
    subjects = config['bids']['subjects']
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""

    pass


def find_config_file(config_path: Optional[str] = None) -> Path:
    """Find the configuration file.

    Args:
        config_path: Optional path to config file. If not provided, searches
            for config.yaml in standard locations.

    Returns:
        Path to configuration file.

    Raises:
        ConfigurationError: If config file not found.
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        return path

    # Search for config.yaml in standard locations
    search_paths = [
        Path.cwd() / "config.yaml",  # Current directory
        Path.cwd().parent / "config.yaml",  # Parent directory
        Path(__file__).parent.parent.parent / "config.yaml",  # Project root
    ]

    for path in search_paths:
        if path.exists():
            return path

    raise ConfigurationError(
        "config.yaml not found. Please create it from config.yaml.template"
    )


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and required fields.

    Args:
        config: Configuration dictionary.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    # Required top-level sections
    required_sections = [
        "paths",
        "bids",
        "preprocessing",
        "source_reconstruction",
        "analysis",
        "features",
        "computing",
        "logging",
        "reproducibility",
    ]

    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required section: {section}")

    # Validate paths section
    required_paths = ["data_root", "raw", "derivatives", "features"]
    for path_key in required_paths:
        if path_key not in config["paths"]:
            raise ConfigurationError(f"Missing required path: paths.{path_key}")

    # Check for placeholders
    placeholders_found = []
    for key, value in config["paths"].items():
        if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
            placeholders_found.append(f"paths.{key}")

    # Check SLURM account only if SLURM is enabled and account is not empty
    slurm_enabled = config["computing"]["slurm"].get("enabled", False)
    slurm_account = config["computing"]["slurm"].get("account", "")

    if slurm_enabled and slurm_account:
        # SLURM is enabled and account is provided - check for placeholder
        if slurm_account.startswith("<") and slurm_account.endswith(">"):
            placeholders_found.append("computing.slurm.account")

    if placeholders_found:
        raise ConfigurationError(
            f"Configuration contains unresolved placeholders: {', '.join(placeholders_found)}\n"
            "Please update config.yaml with actual values."
        )

    # Validate analysis space
    valid_spaces = ["sensor", "source"]
    analysis_space = config["analysis"]["space"]
    if analysis_space not in valid_spaces:
        raise ConfigurationError(
            f"Invalid analysis.space: {analysis_space}. Must be one of {valid_spaces}"
        )

    # Validate BIDS configuration
    if not config["bids"]["subjects"]:
        raise ConfigurationError("No subjects specified in bids.subjects")

    if not config["bids"]["task_runs"]:
        raise ConfigurationError("No task runs specified in bids.task_runs")


def expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand relative paths in configuration to absolute paths.

    Args:
        config: Configuration dictionary.

    Returns:
        Configuration with expanded paths.
    """
    # Expand data_root
    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    config["paths"]["data_root"] = str(data_root)

    # Expand paths relative to data_root
    for key in ["raw", "derivatives", "features"]:
        path = Path(config["paths"][key])
        if not path.is_absolute():
            path = data_root / path
        config["paths"][key] = str(path)

    # Expand project-specific paths
    for key in ["reports", "logs", "venv", "slurm_output", "tmp"]:
        if key in config["paths"]:
            path = Path(config["paths"][key]).expanduser().resolve()
            config["paths"][key] = str(path)

    # Expand FreeSurfer subjects directory if present (relative to data_root)
    if "freesurfer_subjects_dir" in config["paths"]:
        fs_dir = Path(config["paths"]["freesurfer_subjects_dir"])
        if not fs_dir.is_absolute():
            fs_dir = data_root / fs_dir
        config["paths"]["freesurfer_subjects_dir"] = str(fs_dir)

    return config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Optional path to config file. If not provided, searches
            for config.yaml in standard locations.

    Returns:
        Configuration dictionary with validated and expanded paths.

    Raises:
        ConfigurationError: If configuration is invalid or incomplete.
    """
    config_file = find_config_file(config_path)

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing config file: {e}")

    if config is None:
        raise ConfigurationError("Config file is empty")

    # Validate configuration
    validate_config(config)

    # Expand paths
    config = expand_paths(config)

    return config


def get_subjects(config: Dict[str, Any]) -> List[str]:
    """Get list of subjects from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of subject IDs.
    """
    return config["bids"]["subjects"]


def get_task_runs(config: Dict[str, Any]) -> List[str]:
    """Get list of task runs from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of task run IDs.
    """
    return config["bids"]["task_runs"]


def get_rest_runs(config: Dict[str, Any]) -> List[str]:
    """Get list of rest runs from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of rest run IDs.
    """
    return config["bids"].get("rest_runs", [])


def get_analysis_space(config: Dict[str, Any]) -> str:
    """Get the current analysis space (sensor or source).

    Args:
        config: Configuration dictionary.

    Returns:
        Analysis space: 'sensor' or 'source'.
    """
    return config["analysis"]["space"]


def is_sensor_space(config: Dict[str, Any]) -> bool:
    """Check if analysis is in sensor space.

    Args:
        config: Configuration dictionary.

    Returns:
        True if analysis space is 'sensor', False otherwise.
    """
    return get_analysis_space(config) == "sensor"


def is_source_space(config: Dict[str, Any]) -> bool:
    """Check if analysis is in source space.

    Args:
        config: Configuration dictionary.

    Returns:
        True if analysis space is 'source', False otherwise.
    """
    return get_analysis_space(config) == "source"


# Global configuration instance (loaded on first access)
_config: Optional[Dict[str, Any]] = None


def get_config(reload: bool = False) -> Dict[str, Any]:
    """Get the global configuration instance.

    This function maintains a singleton configuration instance. The first call
    loads the configuration; subsequent calls return the cached instance.

    Args:
        reload: If True, force reload configuration from file.

    Returns:
        Configuration dictionary.
    """
    global _config

    if _config is None or reload:
        _config = load_config()

    return _config
