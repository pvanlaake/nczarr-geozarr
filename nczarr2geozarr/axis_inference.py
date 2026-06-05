"""
axis_inference.py — Infer cs axis abbreviation and direction from CF metadata.

The cs convention uses OGC 19111-2 axis abbreviations (X, Y, Z, T) and
direction strings (east, north, up, down, future, past, …).

Priority chain for each dimension:
  1. CF `axis` attribute on the coordinate array (X/Y/Z/T)
  2. CF `standard_name` matched against known names
  3. CF `units` pattern matching (degrees_east, degrees_north, Pa, …)
  4. Fallback: AxisAbbrev.OTHER, direction "unspecified"

References
----------
CF Conventions §4 — Coordinate Types
OGC Abstract Specification Topic 2 (19111) — axis direction vocabulary
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

from .models import AxisAbbrev, NCZarrArray

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AxisInfo:
    abbreviation: AxisAbbrev
    direction: str          # OGC direction string, lower-case


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# CF standard_name → (abbreviation, direction)
_STANDARD_NAME_MAP: dict[str, AxisInfo] = {
    # Longitude / X
    "longitude":                              AxisInfo(AxisAbbrev.X, "east"),
    "grid_longitude":                         AxisInfo(AxisAbbrev.X, "east"),
    "projection_x_coordinate":               AxisInfo(AxisAbbrev.X, "east"),
    "projection_x_angular_coordinate":       AxisInfo(AxisAbbrev.X, "east"),
    "grid_cell_x_coordinate":                AxisInfo(AxisAbbrev.X, "east"),
    # Latitude / Y
    "latitude":                               AxisInfo(AxisAbbrev.Y, "north"),
    "grid_latitude":                          AxisInfo(AxisAbbrev.Y, "north"),
    "projection_y_coordinate":               AxisInfo(AxisAbbrev.Y, "north"),
    "projection_y_angular_coordinate":       AxisInfo(AxisAbbrev.Y, "north"),
    "grid_cell_y_coordinate":                AxisInfo(AxisAbbrev.Y, "north"),
    # Vertical / Z — upward
    "altitude":                               AxisInfo(AxisAbbrev.Z, "up"),
    "height":                                 AxisInfo(AxisAbbrev.Z, "up"),
    "height_above_geoid":                     AxisInfo(AxisAbbrev.Z, "up"),
    "height_above_mean_sea_level":            AxisInfo(AxisAbbrev.Z, "up"),
    "geopotential_height":                    AxisInfo(AxisAbbrev.Z, "up"),
    # Vertical / Z — downward
    "depth":                                  AxisInfo(AxisAbbrev.Z, "down"),
    "depth_below_geoid":                      AxisInfo(AxisAbbrev.Z, "down"),
    "depth_below_sea_floor":                  AxisInfo(AxisAbbrev.Z, "down"),
    # Vertical / Z — pressure (decreasing with height → down in index space
    # but the OGC direction for pressure axes is "up" by convention when the
    # axis is oriented toward lower pressure = higher altitude)
    "air_pressure":                           AxisInfo(AxisAbbrev.Z, "up"),
    "atmosphere_ln_pressure_coordinate":     AxisInfo(AxisAbbrev.Z, "up"),
    "atmosphere_sigma_coordinate":           AxisInfo(AxisAbbrev.Z, "up"),
    "atmosphere_hybrid_sigma_pressure_coordinate": AxisInfo(AxisAbbrev.Z, "up"),
    "ocean_s_coordinate":                    AxisInfo(AxisAbbrev.Z, "up"),
    "ocean_s_coordinate_g1":                 AxisInfo(AxisAbbrev.Z, "up"),
    "ocean_s_coordinate_g2":                 AxisInfo(AxisAbbrev.Z, "up"),
    "ocean_sigma_coordinate":               AxisInfo(AxisAbbrev.Z, "up"),
    "ocean_double_sigma_coordinate":        AxisInfo(AxisAbbrev.Z, "up"),
    # Time
    "time":                                   AxisInfo(AxisAbbrev.T, "future"),
    "forecast_reference_time":               AxisInfo(AxisAbbrev.T, "future"),
    "forecast_period":                       AxisInfo(AxisAbbrev.T, "future"),
    "lead_time":                             AxisInfo(AxisAbbrev.T, "future"),
}

# CF `axis` attribute value → AxisInfo (direction is the most common default)
_AXIS_ATTR_MAP: dict[str, AxisInfo] = {
    "X": AxisInfo(AxisAbbrev.X, "east"),
    "Y": AxisInfo(AxisAbbrev.Y, "north"),
    "Z": AxisInfo(AxisAbbrev.Z, "up"),      # overridden below if units hint
    "T": AxisInfo(AxisAbbrev.T, "future"),
}

# Units patterns for last-resort inference
_UNITS_PATTERNS: list[tuple[re.Pattern[str], AxisInfo]] = [
    (re.compile(r"degrees?[_\s]*east",  re.I), AxisInfo(AxisAbbrev.X, "east")),
    (re.compile(r"degrees?[_\s]*north", re.I), AxisInfo(AxisAbbrev.Y, "north")),
    (re.compile(r"degrees?[_\s]*west",  re.I), AxisInfo(AxisAbbrev.X, "west")),
    (re.compile(r"degrees?[_\s]*south", re.I), AxisInfo(AxisAbbrev.Y, "south")),
    (re.compile(r"\bm\b|\bmeters?\b|\bmetres?\b", re.I),
                                                AxisInfo(AxisAbbrev.Z, "up")),
    (re.compile(r"\bpa\b|\bpascals?\b|\bhpa\b|\bmbar\b", re.I),
                                                AxisInfo(AxisAbbrev.Z, "up")),
    # Time units: "days since …", "hours since …", etc.
    (re.compile(r"\bsince\b", re.I),           AxisInfo(AxisAbbrev.T, "future")),
]

_FALLBACK = AxisInfo(AxisAbbrev.OTHER, "unspecified")


# ---------------------------------------------------------------------------
# Depth / down override
# ---------------------------------------------------------------------------

_POSITIVE_DOWN_UNITS = re.compile(
    r"\bm\b|\bmeters?\b|\bmetres?\b|\bfeet\b|\bft\b", re.I
)


def _maybe_down(info: AxisInfo, arr: NCZarrArray) -> AxisInfo:
    """
    If `info` says Z/up but the coordinate carries `positive=down` or
    its standard_name is an explicitly downward one, flip to down.
    """
    if info.abbreviation is not AxisAbbrev.Z:
        return info
    positive = arr.extra_attrs.get("positive", "").lower()
    if positive == "down":
        return AxisInfo(AxisAbbrev.Z, "down")
    return info


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def infer_axis(coord_arr: NCZarrArray) -> AxisInfo:
    """
    Infer the cs AxisInfo for a coordinate array using CF metadata.

    Parameters
    ----------
    coord_arr:
        The NCZarrArray for the coordinate variable (not the data variable).

    Returns
    -------
    AxisInfo with .abbreviation and .direction populated.
    """

    # 1. CF `standard_name`
    if coord_arr.standard_name:
        sn = coord_arr.standard_name.strip().lower()
        if sn in _STANDARD_NAME_MAP:
            info = _STANDARD_NAME_MAP[sn]
            return _maybe_down(info, coord_arr)

    # 2. CF `axis` attribute
    if coord_arr.axis:
        ax = coord_arr.axis.strip().upper()
        if ax in _AXIS_ATTR_MAP:
            info = _AXIS_ATTR_MAP[ax]
            return _maybe_down(info, coord_arr)

    # 3. Units pattern matching
    if coord_arr.units:
        for pattern, info in _UNITS_PATTERNS:
            if pattern.search(coord_arr.units):
                return _maybe_down(info, coord_arr)

    # 4. Fallback
    log.debug(
        "%s: cannot infer axis from CF metadata "
        "(standard_name=%r, axis=%r, units=%r) — using OTHER/unspecified",
        coord_arr.path, coord_arr.standard_name, coord_arr.axis, coord_arr.units,
    )
    return _FALLBACK


def effective_unit(coord_arr: NCZarrArray) -> str | None:
    """
    Return the best unit string for a cs coordinate entry.

    Prefers the CF `units` attribute; falls back to None (the cs spec
    allows omitting unit when it is unknown or dimensionless).
    """
    u = coord_arr.units
    if u in (None, "", "1", "none", "None"):
        return None
    return u
