"""BIDS conversion module for saflow.

Stage 0: Raw MEG data → BIDS format conversion.
"""

from code.bids.utils import (
    add_behavioral_info,
    add_trial_indices,
    detect_events,
    find_trial_type,
    load_meg_recording,
    parse_info_from_name,
)

__all__ = [
    "add_behavioral_info",
    "add_trial_indices",
    "detect_events",
    "find_trial_type",
    "load_meg_recording",
    "parse_info_from_name",
]
