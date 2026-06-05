"""
Shared dataclasses for the NCZarr → GeoZarr/cs translator.

These are plain data containers; no business logic lives here.
All paths are absolute within the store (leading slash, e.g. "/group/var").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VarKind(Enum):
    """Classification of a variable's role in the dataset."""
    DATA         = auto()   # primary data variable
    DIM_COORD    = auto()   # 1-D coordinate sharing its dimension's name
    AUX_COORD    = auto()   # auxiliary coordinate (e.g. 2-D lat/lon)
    BOUNDS       = auto()   # cell-boundary array (CF bounds)
    GRID_MAPPING = auto()   # grid-mapping container variable (no data)
    ANCILLARY    = auto()   # ancillary variable (quality flags, etc.)
    SCALAR_COORD = auto()   # 0-D scalar coordinate


class AxisAbbrev(str, Enum):
    """OGC / CF axis abbreviations used by the cs convention."""
    X = "X"
    Y = "Y"
    Z = "Z"
    T = "T"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Raw NCZarr layer (output of reader.py)
# ---------------------------------------------------------------------------

@dataclass
class NCZarrArray:
    """Everything read directly from a single NCZarr v2 array node."""

    # Store-absolute path, e.g. "/ocean/temp"
    path: str

    # Zarr v2 .zarray fields needed verbatim for the v3 output
    shape:     list[int]
    dtype:     str          # Zarr dtype string, e.g. "<f4"
    chunks:    list[int]
    compressor: dict[str, Any] | None     # e.g. {"id": "zlib", "level": 4}
    filters:   list[dict[str, Any]]       # additional filters (may be empty)
    fill_value: Any                        # numeric, null, or "NaN"
    order:     str                         # "C" or "F"

    # Dimension references from _NCZARR_ARRAY.dimrefs
    # Each entry is a store-absolute path to the dimension definition, e.g.
    # "/ocean/xi_rho"  (NCZarr fully-qualifies these)
    dimrefs: list[str]

    # CF attributes (absent keys → None, not KeyError)
    units:         str | None = None
    axis:          str | None = None       # "X", "Y", "Z", or "T"
    standard_name: str | None = None
    long_name:     str | None = None
    calendar:      str | None = None       # for time axes

    # Space-separated names as they appear in the source .zattrs
    coordinates:   str | None = None       # e.g. "lat lon time"
    bounds:        str | None = None       # e.g. "lat_bnds"
    grid_mapping:  str | None = None       # e.g. "crs"

    # All remaining attributes, passed through verbatim to the output
    extra_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class NCZarrGroup:
    """Metadata for a single NCZarr group node."""

    # Store-absolute path; root is "/"
    path: str

    # Attributes from .zattrs (global metadata, Conventions string, etc.)
    attrs: dict[str, Any] = field(default_factory=dict)

    # Dimension names declared in this group via _NCZARR_GROUP.dims,
    # mapping local dim name → length.  NCZarr declares dimensions at the
    # group level; the dimref paths in NCZarrArray point back here.
    dims: dict[str, int] = field(default_factory=dict)


@dataclass
class NCZarrStore:
    """The complete raw metadata snapshot of an NCZarr v2 store."""

    # All groups, keyed by store-absolute path
    groups: dict[str, NCZarrGroup] = field(default_factory=dict)

    # All arrays, keyed by store-absolute path
    arrays: dict[str, NCZarrArray] = field(default_factory=dict)

    def array(self, path: str) -> NCZarrArray | None:
        return self.arrays.get(path)

    def group(self, path: str) -> NCZarrGroup | None:
        return self.groups.get(path)


# ---------------------------------------------------------------------------
# Resolved layer (output of resolver.py)
# ---------------------------------------------------------------------------

@dataclass
class ResolvedVar:
    """
    A single variable after cross-group reference resolution.

    All name references (coordinates, bounds, grid_mapping) have been
    converted to store-absolute paths pointing into NCZarrStore.arrays.
    """

    raw: NCZarrArray

    # Ordered list of absolute paths to coordinate arrays for this variable,
    # preserving the order from the CF `coordinates` attribute.
    coord_paths: list[str] = field(default_factory=list)

    # Absolute path to the bounds array, or None.
    bounds_path: str | None = None

    # Absolute path to the grid-mapping variable, or None.
    grid_mapping_path: str | None = None

    # Absolute paths of the dimension coordinate arrays, in axis order.
    # Index i corresponds to raw.dimrefs[i].
    dim_coord_paths: list[str | None] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.raw.path


# ---------------------------------------------------------------------------
# cs convention layer (output of classifier.py + axis_inference.py)
# ---------------------------------------------------------------------------

@dataclass
class CsCoordinate:
    """
    One entry in the cs `coordinates` array for a single axis.

    Covers the v0.1 scope: explicit stored values via a store-internal ref.
    Parametric and geolocation hooks are marked TODO.
    """

    # Store-absolute path to the coordinate array
    array_path: str

    # Physical unit string (CF UDUNITS-compatible), e.g. "degrees_east"
    unit: str | None = None

    # Absolute path to the bounds array, or None
    bounds_path: str | None = None

    # TODO v0.2: parametric: dict | None = None
    # TODO v0.2: geolocation: dict | None = None


@dataclass
class CsAxis:
    """
    The cs representation of one dimension axis.

    One CsAxis is created per dimension of a data variable.
    """

    # The dimension name as it appears in dimension_names / dimrefs
    dim_name: str

    # OGC axis abbreviation for the cs convention
    abbreviation: AxisAbbrev

    # OGC axis direction string for the cs convention, e.g. "east", "north",
    # "up", "down", "future"
    direction: str

    # Coordinate objects for this axis (usually one; multiples are allowed
    # by the spec for alternative realisations)
    coordinates: list[CsCoordinate] = field(default_factory=list)


@dataclass
class CsCrs:
    """
    One CRS object in the cs `crs` array.

    In v0.1 we emit one CRS per data variable covering all its axes.
    Splitting into separate horizontal / vertical / temporal CRS objects
    is a future refinement.
    """

    # Keyed by dim_name
    axes: dict[str, CsAxis] = field(default_factory=dict)

    # Optional EPSG or other authority code inferred from grid_mapping,
    # e.g. "EPSG:4326".  None if no grid_mapping was present or parseable.
    proj_code: str | None = None

    # TODO v0.2: geolocation object for curvilinear grids


@dataclass
class CsSpec:
    """
    The complete cs attribute value for one array.

    Serialises directly to the `cs` key in the output zarr.json attributes.
    """

    crs: list[CsCrs] = field(default_factory=list)

    # Optional human-readable name for the coordinate set
    name: str | None = None


# ---------------------------------------------------------------------------
# Output layer (input to writer.py)
# ---------------------------------------------------------------------------

@dataclass
class OutputArray:
    """
    Everything needed to write one Zarr v3 array node.
    """

    # Store-absolute path in the output store
    path: str

    # Zarr v3 core metadata
    shape:      list[int]
    dtype:      str
    chunks:     list[int]
    fill_value: Any
    dimension_names: list[str]

    # Codec chain expressed as Zarr v3 codec dicts
    codecs: list[dict[str, Any]]

    # Final merged attributes (CF pass-through + cs + zarr_conventions)
    attributes: dict[str, Any]

    # Source path in the NCZarr store, used to locate chunk files to copy
    source_path: str


@dataclass
class OutputGroup:
    """
    Everything needed to write one Zarr v3 group node.
    """

    path: str
    attributes: dict[str, Any]


@dataclass
class OutputStore:
    """The complete description of the Zarr v3 output store."""

    groups: list[OutputGroup] = field(default_factory=list)
    arrays: list[OutputArray] = field(default_factory=list)
