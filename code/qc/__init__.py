"""Quality control module for saflow.

This module provides data validation utilities:

- check_dataset: Check dataset completeness across all pipeline stages
- check_qc: Run data quality checks (ISI, responses, channels, events)

Usage:
    python -m code.qc.check_dataset
    python -m code.qc.check_qc --subject 04
"""
