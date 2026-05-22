"""Yeo network parsing and aggregation for Schaefer parcellations.

Schaefer label format (both 7Networks and 17Networks variants):
    <prefix>_<HEMI>_<network>_<subregion>_<idx>-<hemi>
    e.g.  7Networks_LH_Cont_Cing_1-lh
          17Networks_RH_DefaultA_pCun_1-rh

This module is the single source of truth for:
- Parsing Schaefer label strings into network/hemisphere/subregion
- Mapping per-parcel arrays to per-Yeo-network summaries
- The canonical Yeo color palette and display names used across plots
- The 17→7 grouping (Schaefer's official assignment)
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# Canonical Yeo-7 networks in the standard order (Yeo et al. 2011).
YEO7_NETWORKS: Tuple[str, ...] = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)

# Display labels (full names for figure titles/axes).
YEO7_FULL_NAMES: Dict[str, str] = {
    "Vis":         "Visual",
    "SomMot":      "Somatomotor",
    "DorsAttn":    "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Limbic":      "Limbic",
    "Cont":        "Frontoparietal Control",
    "Default":     "Default Mode",
}

# Canonical Yeo-7 palette (Yeo 2011 / Buckner Lab). Keep these stable —
# every network plot in the project picks colors from this dict so figures
# stay visually consistent.
YEO7_COLORS: Dict[str, str] = {
    "Vis":         "#781286",
    "SomMot":      "#4682B4",
    "DorsAttn":    "#00760E",
    "SalVentAttn": "#C43AFA",
    "Limbic":      "#DCF8A4",
    "Cont":        "#E69422",
    "Default":     "#CD3E4E",
}

# Yeo-17 networks in Schaefer's 17Networks_order ordering. Each maps back to
# its Yeo-7 parent (YEO17_TO_YEO7) so 17-resolution arrays can be regrouped
# to 7 without re-parsing the labels.
YEO17_NETWORKS: Tuple[str, ...] = (
    "VisCent", "VisPeri",
    "SomMotA", "SomMotB",
    "DorsAttnA", "DorsAttnB",
    "SalVentAttnA", "SalVentAttnB",
    "LimbicB", "LimbicA",
    "ContA", "ContB", "ContC",
    "DefaultA", "DefaultB", "DefaultC",
    "TempPar",
)

YEO17_TO_YEO7: Dict[str, str] = {
    "VisCent": "Vis", "VisPeri": "Vis",
    "SomMotA": "SomMot", "SomMotB": "SomMot",
    "DorsAttnA": "DorsAttn", "DorsAttnB": "DorsAttn",
    "SalVentAttnA": "SalVentAttn", "SalVentAttnB": "SalVentAttn",
    "LimbicB": "Limbic", "LimbicA": "Limbic",
    "ContA": "Cont", "ContB": "Cont", "ContC": "Cont",
    "DefaultA": "Default", "DefaultB": "Default", "DefaultC": "Default",
    "TempPar": "Default",   # Schaefer groups TempParietal under Default at Yeo-7
}

# Parses prefix (7Networks|17Networks), hemisphere (LH|RH), network token,
# and the remaining subregion+index+hemi-suffix in one shot.
_LABEL_RE = re.compile(
    r"^(?P<prefix>(?:7|17)Networks)_(?P<hemi>LH|RH)_(?P<network>[A-Za-z]+)_(?P<rest>.+)$"
)

# Sentinel used for non-Schaefer labels (e.g. Schaefer atlases ship 2
# "Background+FreeSurfer_Defined_Medial_Wall" entries alongside the 400 cortical
# parcels). Parcels tagged Unknown are silently dropped by aggregation/indexing.
UNKNOWN_NETWORK = "Unknown"


def parse_schaefer_label(name: str, strict: bool = False) -> Optional[Dict[str, str]]:
    """Parse a Schaefer label into prefix, hemi, network, subregion.

    Returns None for non-Schaefer labels (e.g. medial-wall placeholders)
    unless ``strict=True``, in which case a ValueError is raised.
    """
    m = _LABEL_RE.match(str(name))
    if not m:
        if strict:
            raise ValueError(f"Not a Schaefer-style label: {name!r}")
        return None
    return {
        "prefix":    m.group("prefix"),
        "hemi":      m.group("hemi"),
        "network":   m.group("network"),
        "subregion": m.group("rest"),
    }


def network_order(n_networks: int = 7) -> Tuple[str, ...]:
    """Canonical ordering for plotting and indexing."""
    if n_networks == 7:
        return YEO7_NETWORKS
    if n_networks == 17:
        return YEO17_NETWORKS
    raise ValueError(f"n_networks must be 7 or 17, got {n_networks}")


def network_palette(n_networks: int = 7) -> Dict[str, str]:
    """Color per network. Yeo-17 reuses the Yeo-7 color of its parent."""
    if n_networks == 7:
        return dict(YEO7_COLORS)
    return {n17: YEO7_COLORS[YEO17_TO_YEO7[n17]] for n17 in YEO17_NETWORKS}


def network_display_name(network: str) -> str:
    """Long-form name for a Yeo-7 (or Yeo-17, falling back to itself) network."""
    if network in YEO7_FULL_NAMES:
        return YEO7_FULL_NAMES[network]
    return network


def get_network_assignments(
    ch_names: Sequence[str],
    n_networks: int = 7,
) -> np.ndarray:
    """Return an array of network labels aligned with ``ch_names``.

    For ``n_networks=7``: works on both 7Networks_* (direct read) and
    17Networks_* labels (mapped down via YEO17_TO_YEO7).
    For ``n_networks=17``: requires 17Networks_* labels and raises if a
    7Networks_* label is encountered (cannot upsample 7→17).
    """
    if n_networks not in (7, 17):
        raise ValueError(f"n_networks must be 7 or 17, got {n_networks}")

    out = []
    for ch in ch_names:
        parsed = parse_schaefer_label(ch)
        if parsed is None:
            out.append(UNKNOWN_NETWORK)
            continue
        net = parsed["network"]
        prefix = parsed["prefix"]
        if n_networks == 7:
            if prefix == "17Networks":
                net = YEO17_TO_YEO7.get(net, net)
        else:
            if prefix != "17Networks":
                raise ValueError(
                    f"n_networks=17 requires 17Networks_* labels, got "
                    f"prefix={prefix!r} from {ch!r}. Re-run atlas with the "
                    f"Schaefer*_17Networks_order variant."
                )
        out.append(net)
    return np.asarray(out)


def network_parcel_indices(
    ch_names: Sequence[str],
    n_networks: int = 7,
) -> Dict[str, np.ndarray]:
    """``{network_name: parcel_indices}`` grouping for spatial-axis masking.

    Used by network-restricted classification to subset the spatial axis to
    one network at a time.
    """
    assignments = get_network_assignments(ch_names, n_networks=n_networks)
    return {
        net: np.where(assignments == net)[0]
        for net in network_order(n_networks)
    }


def aggregate_to_networks(
    values: np.ndarray,
    ch_names: Sequence[str],
    n_networks: int = 7,
    agg: str = "mean",
    networks: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Aggregate per-parcel values to per-network values along the last axis.

    Args:
        values: shape ``(..., n_parcels)``. Last axis is reduced to networks.
        ch_names: length-n_parcels parcel names.
        n_networks: 7 or 17.
        agg: aggregation rule applied within each network:
            ``mean``         — nanmean of parcel values
            ``median``       — nanmedian
            ``sum``          — nansum
            ``signed_count`` — (count > 0) − (count < 0). Pass a t-value
                array zeroed where non-significant to get net signed count
                of significant parcels per network.
        networks: optional override of the network ordering (defaults to
            ``network_order(n_networks)``).

    Returns:
        ``(out, network_names)`` where ``out`` has shape
        ``(..., len(network_names))``.
    """
    values = np.asarray(values)
    if values.shape[-1] != len(ch_names):
        raise ValueError(
            f"Last axis of values ({values.shape[-1]}) must equal "
            f"len(ch_names) ({len(ch_names)})."
        )

    assignments = get_network_assignments(ch_names, n_networks=n_networks)
    net_order = tuple(networks) if networks is not None else network_order(n_networks)

    out = np.full(values.shape[:-1] + (len(net_order),), np.nan, dtype=float)

    for k, net in enumerate(net_order):
        idx = np.where(assignments == net)[0]
        if idx.size == 0:
            continue
        block = values[..., idx]
        if agg == "mean":
            out[..., k] = np.nanmean(block, axis=-1)
        elif agg == "median":
            out[..., k] = np.nanmedian(block, axis=-1)
        elif agg == "sum":
            out[..., k] = np.nansum(block, axis=-1)
        elif agg == "signed_count":
            pos = np.sum(block > 0, axis=-1)
            neg = np.sum(block < 0, axis=-1)
            out[..., k] = pos.astype(float) - neg.astype(float)
        else:
            raise ValueError(f"Unknown agg: {agg!r}")

    return out, net_order
