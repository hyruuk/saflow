"""Color definitions for saflow visualizations.

This module provides consistent color schemes across all visualizations.
Import colors from here to ensure visual consistency.

Usage:
    from code.utils.colors import ZONE_COLORS, EVENT_COLORS
"""

# Zone colors (IN/OUT classification)
ZONE_COLORS = {
    "IN": "#4878A8",      # Steel blue
    "OUT": "#E07850",     # Coral orange
    "MID": "#888888",     # Gray (for excluded trials)
}

# Event type colors (behavioral markers)
EVENT_COLORS = {
    "lapse": "#D62728",              # Red - commission errors
    "correct_omission": "#2CA02C",   # Green - correct omissions
    "omission_error": "#FF7F0E",     # Orange - omission errors
    "correct_commission": "#1F77B4", # Blue - correct commissions
}

# VTC trace colors
VTC_COLORS = {
    "raw": "#808080",     # Gray for raw VTC
    "filtered": "#404040", # Dark gray for filtered VTC (if used)
}

# Palette for seaborn (ordered for IN, OUT)
ZONE_PALETTE = {
    "IN": ZONE_COLORS["IN"],
    "OUT": ZONE_COLORS["OUT"],
}
