"""Quality control module for saflow.

This module provides data validation utilities:

- check_dataset: Check dataset completeness across all pipeline stages
- check_qc: Run data quality checks on raw MEG sourcedata:
  - Recording parameters (sampling rate, duration, channel counts)
  - ISI (inter-stimulus interval) from event triggers
  - Event/trigger validation
  - Channel quality (flat, noisy, missing)
  - Response detection and RT statistics

Usage:
    python -m code.qc.check_dataset
    python -m code.qc.check_qc                    # All subjects
    python -m code.qc.check_qc --subject 04       # Single subject
    python -m code.qc.check_qc --verbose          # Detailed output
"""
