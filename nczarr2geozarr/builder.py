"""
builder.py — Construct cs attribute trees for data variables.

For each DATA variable in the resolved store this module produces:
  • A CsSpec (the value of the `cs` key in zarr.json attributes)
  • A zarr_conventions list containing the cs CMO

v0.1 scope
──────────
  • Two CRS objects per data variable: one temporal, one engineering/spatial
  • Coordinate values as explicit inline list when n < EXPLICIT_THRESHOLD,
    otherwise as a store-relative external ref
  • Cell bounds as store-relative external ref
  • proj:code populated from the grid_mapping variable when present
  • CF time unit string split into unit + epoch
  • Parametric coordinates → TODO stub
  • Geolocation / curvilinear grids → TODO stub
"""

from __future__ import annotations

import logging
import re
import struct
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .axis_inference import AxisAbbrev, effective_unit, infer_axis
from .models import (
    CsCoordinate,
    CsAxis,
    CsCrs,
    CsSpec,
    NCZarrArray,
    NCZarrStore,
    ResolvedVar,
    VarKind,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cs Convention Metadata Object
# ---------------------------------------------------------------------------

CS_CMO: dict[str, str] = {
    "schema_url": (
        "https://raw.githubusercontent.com/R-CF/zarr_convention_cs"
        "/main/schema.json"
    ),
    "spec_url": (
        "https://raw.githubusercontent.com/R-CF/zarr_convention_cs"
        "/main/README.md"
    ),
    "uuid":        "e4dbf0b7-7a00-4ce6-b23e-484292014ab4",
    "name":        "cs",
    "description": "Coordinate set convention for Zarr arrays",
}

# Arrays with this many values or fewer are inlined as explicit values
EXPLICIT_THRESHOLD = 30

# Temporal axis abbreviations — these go in the temporal CRS
_TEMPORAL_ABBREVS = {AxisAbbrev.T}

# Spatial/engineering axis abbreviations — these go in the spatial CRS
_SPATIAL_ABBREVS = {AxisAbbrev.X, AxisAbbrev.Y, AxisAbbrev.Z, AxisAbbrev.OTHER}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _name(path: str) -> str:
    return PurePosixPath(path).name


def _relative_ref(from_array_path: str, to_array_path: str) -> str:
    """
    Compute a relative path from a data array to a coordinate array,
    both given as store-absolute paths.

    E.g. from "/grp/temp" to "/time" → "../time"
         from "/grp/temp" to "/grp/lat" → "../lat"  (sibling)
         from "/temp" to "/time" → "../time"
    """
    from_group = str(PurePosixPath(from_array_path).parent)
    # PurePosixPath relative_to only works for true parents; use manual approach
    from_parts = [p for p in from_group.split("/") if p]
    to_parts   = [p for p in to_array_path.strip("/").split("/") if p]

    # Find common prefix length
    common = 0
    for a, b in zip(from_parts, to_parts):
        if a == b:
            common += 1
        else:
            break

    # Steps up from from_group to common ancestor
    up = len(from_parts) - common
    down = to_parts[common:]

    parts = [".."] * up + down
    rel = "/".join(parts) if parts else "."
    # Always needs at least one "../" to exit the array's own "directory"
    if not rel.startswith(".."):
        rel = "../" + rel
    return rel


# ---------------------------------------------------------------------------
# Time unit parsing
# ---------------------------------------------------------------------------

_TIME_UNIT_RE = re.compile(
    r"^\s*(?P<unit>\S+)\s+since\s+(?P<epoch>\S+(?:\s+\S+)*)\s*$",
    re.IGNORECASE,
)


def _split_time_units(units_str: str) -> dict[str, str] | None:
    """
    Split a CF time units string into a cs time object.

    "months since 2000-01-01" → {"unit": "months", "epoch": "2000-01-01"}
    "days since 1900-01-01 00:00:00" → {"unit": "days", "epoch": "1900-01-01T00:00:00"}

    Returns None if the string doesn't match the CF pattern.
    """
    if not units_str:
        return None
    m = _TIME_UNIT_RE.match(units_str)
    if not m:
        return None
    unit  = m.group("unit").lower()
    epoch = m.group("epoch").strip()
    # Normalise epoch: replace space between date and time with T
    epoch = re.sub(r"(\d{4}-\d{2}-\d{2})\s+(\d)", r"\1T\2", epoch)
    return {"unit": unit, "epoch": epoch}


# ---------------------------------------------------------------------------
# Explicit value reading
# ---------------------------------------------------------------------------

def _read_chunk_values(
    src_root: Path,
    arr: NCZarrArray,
) -> list[Any] | None:
    """
    Read the first (and typically only) chunk of a small coordinate array
    and return its values as a Python list suitable for JSON embedding.

    Returns None if reading fails or the array is too large.
    """
    n = int(np.prod(arr.shape)) if arr.shape else 0
    if n == 0 or n > EXPLICIT_THRESHOLD:
        return None

    # Chunk key for a 1-D array with a single chunk is always "0"
    # For 0-D arrays it's just "0" as well
    chunk_key = ".".join(["0"] * max(len(arr.shape), 1))
    chunk_path = src_root / arr.path.lstrip("/") / chunk_key

    if not chunk_path.exists():
        log.debug("Chunk file not found for explicit read: %s", chunk_path)
        return None

    try:
        raw = chunk_path.read_bytes()

        # Decompress if compressor is zlib/gzip
        comp = arr.compressor or {}
        if isinstance(comp, dict) and comp.get("id") in ("zlib", "gzip"):
            raw = zlib.decompress(raw)

        dt = np.dtype(arr.dtype)
        values = np.frombuffer(raw, dtype=dt).reshape(arr.shape)

        # Convert to plain Python scalars for JSON serialisation
        flat = values.flatten().tolist()
        # Round floats to avoid spurious precision in the JSON
        if dt.kind == "f":
            flat = [round(v, 10) for v in flat]
        return flat

    except Exception as exc:
        log.debug("Could not read chunk for explicit values: %s", exc)
        return None

def _grid_mapping_proj_code(
    gm_path: str | None,
    store: NCZarrStore,
) -> str | None:
    if gm_path is None:
        return None
    gm_arr = store.array(gm_path)
    if gm_arr is None:
        return None
    gmn = gm_arr.extra_attrs.get("grid_mapping_name", "")
    if gmn == "latitude_longitude":
        return "EPSG:4326"
    for key in ("epsg_code", "EPSG_code", "epsg"):
        val = gm_arr.extra_attrs.get(key)
        if val:
            code = str(val).replace("EPSG:", "").strip()
            if code.isdigit():
                return f"EPSG:{code}"
    if "crs_wkt" in gm_arr.extra_attrs or "spatial_ref" in gm_arr.extra_attrs:
        log.debug("%s: WKT→proj:code not yet implemented (v0.2 TODO)", gm_path)
    return None


# ---------------------------------------------------------------------------
# Axis builder
# ---------------------------------------------------------------------------

def _build_axis(
    dim_name: str,
    coord_arr: NCZarrArray | None,
    coord_path: str | None,
    bounds_path: str | None,
    data_var_path: str,
    src_root: Path | None,
) -> CsAxis | None:
    if coord_arr is None:
        return CsAxis(
            dim_name     = dim_name,
            abbreviation = AxisAbbrev.OTHER,
            direction    = "unspecified",
            coordinates  = [],
        )

    info = infer_axis(coord_arr)

    # Parametric vertical stub
    sn = coord_arr.standard_name or ""
    PARAMETRIC_SNS = {
        "atmosphere_sigma_coordinate",
        "atmosphere_hybrid_sigma_pressure_coordinate",
        "atmosphere_ln_pressure_coordinate",
        "ocean_s_coordinate", "ocean_s_coordinate_g1", "ocean_s_coordinate_g2",
        "ocean_sigma_coordinate", "ocean_double_sigma_coordinate",
    }
    if sn in PARAMETRIC_SNS:
        log.warning(
            "%s: parametric vertical coordinate (%s) — parametric block not "
            "yet emitted (v0.2 TODO); stored values used as-is",
            coord_path, sn,
        )

    coord_entry = CsCoordinate(
        array_path  = coord_path,
        unit        = effective_unit(coord_arr),
        bounds_path = bounds_path,
    )

    return CsAxis(
        dim_name     = dim_name,
        abbreviation = info.abbreviation,
        direction    = info.direction,
        coordinates  = [coord_entry],
    )


# ---------------------------------------------------------------------------
# CRS splitting
# ---------------------------------------------------------------------------

def _split_into_crs(axes: dict[str, CsAxis], proj_code: str | None) -> list[CsCrs]:
    """
    Partition axes into at most two CRS objects:
      • temporal CRS  (T axes)
      • spatial / engineering CRS  (X, Y, Z, OTHER axes)

    If one partition is empty it is omitted.
    """
    temporal: dict[str, CsAxis] = {}
    spatial:  dict[str, CsAxis] = {}

    for dim_name, axis in axes.items():
        if axis.abbreviation in _TEMPORAL_ABBREVS:
            temporal[dim_name] = axis
        else:
            spatial[dim_name] = axis

    crs_list: list[CsCrs] = []
    if temporal:
        crs_list.append(CsCrs(axes=temporal, proj_code=None))
    if spatial:
        crs_list.append(CsCrs(axes=spatial, proj_code=proj_code))
    if not crs_list:
        crs_list.append(CsCrs(axes=axes, proj_code=proj_code))

    return crs_list


# ---------------------------------------------------------------------------
# Builder class
# ---------------------------------------------------------------------------

class Builder:
    def __init__(
        self,
        store: NCZarrStore,
        resolved: dict[str, ResolvedVar],
        kinds: dict[str, VarKind],
        src_root: Path | None = None,
    ) -> None:
        self._store    = store
        self._resolved = resolved
        self._kinds    = kinds
        self._src_root = src_root   # needed for explicit value reading

    def build_all(self) -> dict[str, CsSpec]:
        result: dict[str, CsSpec] = {}
        for path, rv in self._resolved.items():
            if self._kinds.get(path) is not VarKind.DATA:
                continue
            spec = self._build_one(rv)
            if spec is not None:
                result[path] = spec
        return result

    def _build_one(self, rv: ResolvedVar) -> CsSpec | None:
        arr = rv.raw
        if not arr.dimrefs:
            log.debug("%s: no dimrefs → skipping cs generation", arr.path)
            return None

        proj_code = _grid_mapping_proj_code(rv.grid_mapping_path, self._store)

        # Bounds per dim coord
        dim_bounds: dict[str, str | None] = {}
        for dim_path, coord_path in zip(arr.dimrefs, rv.dim_coord_paths):
            if coord_path is None:
                dim_bounds[dim_path] = None
                continue
            coord_rv = self._resolved.get(coord_path)
            dim_bounds[dim_path] = coord_rv.bounds_path if coord_rv else None

        # Build axes
        axes: dict[str, CsAxis] = {}
        for dim_ref, coord_path in zip(arr.dimrefs, rv.dim_coord_paths):
            dim_name  = _name(dim_ref)
            coord_arr = self._store.array(coord_path) if coord_path else None
            axis = _build_axis(
                dim_name       = dim_name,
                coord_arr      = coord_arr,
                coord_path     = coord_path,
                bounds_path    = dim_bounds.get(dim_ref),
                data_var_path  = arr.path,
                src_root       = self._src_root,
            )
            if axis is not None:
                axes[dim_name] = axis

        # Attach 1-D aux coords
        for aux_path in rv.coord_paths:
            aux_arr = self._store.array(aux_path)
            if aux_arr is None:
                continue
            if len(aux_arr.shape) != 1 or len(aux_arr.dimrefs) != 1:
                log.debug(
                    "%s: aux coord %s is %d-D — geolocation not yet implemented (v0.2)",
                    arr.path, aux_path, len(aux_arr.shape),
                )
                continue
            aux_dim_name = _name(aux_arr.dimrefs[0])
            if aux_dim_name not in axes:
                continue
            info = infer_axis(aux_arr)
            existing = axes[aux_dim_name]
            already_has = any(
                self._store.array(c.array_path) is not None
                and infer_axis(self._store.array(c.array_path)).abbreviation
                   == info.abbreviation
                for c in existing.coordinates
            )
            if not already_has:
                aux_rv = self._resolved.get(aux_path)
                existing.coordinates.append(
                    CsCoordinate(
                        array_path  = aux_path,
                        unit        = effective_unit(aux_arr),
                        bounds_path = aux_rv.bounds_path if aux_rv else None,
                    )
                )

        crs_list = _split_into_crs(axes, proj_code)
        return CsSpec(crs=crs_list)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_values(
    coord: CsCoordinate,
    data_var_path: str,
    store: NCZarrStore,
    src_root: Path | None,
) -> dict[str, Any]:
    """
    Produce the cs `values` object for one coordinate entry.

    Strategy:
      • n <= EXPLICIT_THRESHOLD and src_root available → {"explicit": [...]}
      • otherwise → {"external": {"ref": "<relative-path>"}}
    """
    coord_arr = store.array(coord.array_path)
    n = int(np.prod(coord_arr.shape)) if (coord_arr and coord_arr.shape) else 0

    if src_root is not None and 0 < n <= EXPLICIT_THRESHOLD:
        values = _read_chunk_values(src_root, coord_arr)
        if values is not None:
                return {"explicit": values}

    # Fallback: store-relative external ref
    rel = _relative_ref(data_var_path, coord.array_path)
    return {"external": {"ref": {"node": rel}}}


def _serialise_time(coord_arr: NCZarrArray | None) -> dict[str, str] | None:
    """
    Return a cs `time` object if this coordinate has a CF time units string,
    otherwise None.  Calendar is included when present.
    """
    if coord_arr is None or not coord_arr.units:
        return None
    time_obj = _split_time_units(coord_arr.units)
    if time_obj is None:
        return None
    if coord_arr.calendar:
        time_obj["calendar"] = coord_arr.calendar
    return time_obj


def _serialise_unit(coord_arr: NCZarrArray | None) -> str | None:
    """
    Return the plain unit string for non-temporal coordinates, or None.
    Time coordinates use the `time` key instead (see _serialise_time).
    """
    if coord_arr is None or not coord_arr.units:
        return None
    # If this looks like a CF time string, unit goes into the time object
    if _split_time_units(coord_arr.units) is not None:
        return None
    return coord_arr.units


def spec_to_dict(
    spec: CsSpec,
    data_var_path: str,
    store: NCZarrStore,
    src_root: Path | None,
) -> dict[str, Any]:
    """
    Serialise a CsSpec to the plain dict for zarr.json attributes["cs"].
    """
    crs_list = []
    for crs in spec.crs:
        axes_dict: dict[str, Any] = {}
        for dim_name, axis in crs.axes.items():
            coords_list = []
            for coord in axis.coordinates:
                coord_arr = store.array(coord.array_path)
                values_obj = _serialise_values(coord, data_var_path, store, src_root)
                unit_val   = _serialise_unit(coord_arr)
                time_val   = _serialise_time(coord_arr)

                entry: dict[str, Any] = {"values": values_obj}
                if unit_val is not None:
                    entry["unit"] = unit_val
                if time_val is not None:
                    entry["time"] = time_val
                if coord.bounds_path is not None:
                    rel = _relative_ref(data_var_path, coord.bounds_path)
                    entry["boundaries"] = {"external": {"ref": {"node": rel}}}
                coords_list.append(entry)

            axis_dict: dict[str, Any] = {
                "abbreviation": axis.abbreviation.value,
                "direction":    axis.direction,
            }
            if coords_list:
                axis_dict["coordinates"] = coords_list

            axes_dict[dim_name] = axis_dict

        crs_dict: dict[str, Any] = {"axes": axes_dict}
        if crs.proj_code is not None:
            crs_dict["proj:code"] = crs.proj_code

        crs_list.append(crs_dict)

    result: dict[str, Any] = {"crs": crs_list}
    if spec.name is not None:
        result["name"] = spec.name
    return result


def build_attributes(
    rv: ResolvedVar,
    cs_spec: CsSpec | None,
    store: NCZarrStore,
    src_root: Path | None,
) -> dict[str, Any]:
    arr = rv.raw
    attrs: dict[str, Any] = {}

    for key, val in [
        ("units",         arr.units),
        ("axis",          arr.axis),
        ("standard_name", arr.standard_name),
        ("long_name",     arr.long_name),
        ("calendar",      arr.calendar),
        ("coordinates",   arr.coordinates),
        ("bounds",        arr.bounds),
        ("grid_mapping",  arr.grid_mapping),
    ]:
        if val is not None:
            attrs[key] = val

    attrs.update(arr.extra_attrs)

    if cs_spec is not None:
        attrs["cs"] = spec_to_dict(cs_spec, arr.path, store, src_root)
        attrs["zarr_conventions"] = [CS_CMO]

    return attrs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_store(
    store: NCZarrStore,
    resolved: dict[str, ResolvedVar],
    kinds: dict[str, VarKind],
    src_root: Path | None = None,
) -> tuple[dict[str, CsSpec], dict[str, dict[str, Any]]]:
    builder = Builder(store, resolved, kinds, src_root=src_root)
    specs   = builder.build_all()

    attrs: dict[str, dict[str, Any]] = {}
    for path, rv in resolved.items():
        attrs[path] = build_attributes(rv, specs.get(path), store, src_root)

    return specs, attrs