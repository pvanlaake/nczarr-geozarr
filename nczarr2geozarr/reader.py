"""
reader.py — NCZarr v2 metadata discovery.

Walks the store tree directly via pathlib (local) or fsspec (HTTP), reads
.zgroup / .zarray / .zattrs per node, and returns a fully-populated
NCZarrStore.  Deliberately avoids the zarr-python store API, which changed
incompatibly between v2 and v3.

NCZarr v2 encoding recap
─────────────────────────
Groups
  <path>/.zgroup   → {"zarr_format": 2}
  <path>/.zattrs   → group attributes, which MAY contain:
      "_NCZARR_GROUP": {
          "dims": {"xi_rho": 100, "eta_rho": 120, ...},
          "vars": ["temp", "salt", ...],
          "groups": ["coords", ...]
      }

Arrays
  <path>/.zarray   → zarr v2 array metadata (shape, dtype, chunks, …)
                      MAY contain "_NCZARR_ARRAY": {"dimrefs": [...], ...}
  <path>/.zattrs   → CF attributes (units, axis, standard_name, coordinates,
                      bounds, grid_mapping, …)

Dimension references (dimrefs) in _NCZARR_ARRAY are fully-qualified paths
within the store, e.g. "/coords/xi_rho".  We normalise all paths to
store-absolute form with a leading slash.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any

from .models import NCZarrArray, NCZarrGroup, NCZarrStore

log = logging.getLogger(__name__)

# CF attributes we lift into dedicated fields on NCZarrArray
_CF_ATTRS = {
    "units", "axis", "standard_name", "long_name",
    "calendar", "coordinates", "bounds", "grid_mapping",
}

# Keys that are NCZarr-internal and should not be passed through
_NCZARR_KEYS = {"_NCZARR_ARRAY", "_NCZARR_GROUP", "_NCZARR_SUPERBLOCK", "_nczarr_attr", "_nczarr_array", "_nczarr_group", "_nczarr_superblock"}
_XARRAY_KEYS = {"_ARRAY_DIMENSIONS"}
_STRIP_KEYS  = _NCZARR_KEYS | _XARRAY_KEYS


def _abs(path: str) -> str:
    """Normalise a path to store-absolute form (leading slash, no trailing)."""
    p = "/" + path.strip("/")
    return p if p != "/" else "/"


def _parent(path: str) -> str:
    parent = str(PurePosixPath(_abs(path)).parent)
    return parent if parent else "/"


def _read_json_file(p: Path) -> dict[str, Any]:
    """Read a JSON file from disk; return {} on missing/invalid."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _parse_zarray(meta: dict[str, Any], attrs: dict[str, Any], path: str) -> NCZarrArray:
    """Build an NCZarrArray from .zarray + .zattrs dicts."""

    # --- dimension references -------------------------------------------
    nczarr_array = attrs.get("_NCZARR_ARRAY") or attrs.get("_nczarr_array") or {}
    # Also check inside .zarray itself (some NCO versions put it there)
    if not nczarr_array:
        nczarr_array = meta.get("_NCZARR_ARRAY") or meta.get("_nczarr_array") or {}

    raw_dimrefs: list[str] = nczarr_array.get("dimrefs", [])
    dimrefs = [_abs(d) for d in raw_dimrefs]

    # If NCZarr dimrefs are absent, fall back to XArray _ARRAY_DIMENSIONS
    if not dimrefs:
        xarray_dims: list[str] = attrs.get("_ARRAY_DIMENSIONS", [])
        if xarray_dims:
            log.debug("%s: no dimrefs, falling back to _ARRAY_DIMENSIONS", path)
            dimrefs = [_abs(d) for d in xarray_dims]

    # --- CF attributes --------------------------------------------------
    cf: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in attrs.items():
        if k in _STRIP_KEYS:
            continue
        if k in _CF_ATTRS:
            cf[k] = v
        else:
            extra[k] = v

    compressor = meta.get("compressor")
    filters    = meta.get("filters") or []

    return NCZarrArray(
        path       = path,
        shape      = meta.get("shape", []),
        dtype      = meta.get("dtype", "|u1"),
        chunks     = meta.get("chunks", meta.get("shape", [])),
        compressor = compressor,
        filters    = filters,
        fill_value = meta.get("fill_value"),
        order      = meta.get("order", "C"),
        dimrefs    = dimrefs,
        units          = cf.get("units"),
        axis           = cf.get("axis"),
        standard_name  = cf.get("standard_name"),
        long_name      = cf.get("long_name"),
        calendar       = cf.get("calendar"),
        coordinates    = cf.get("coordinates"),
        bounds         = cf.get("bounds"),
        grid_mapping   = cf.get("grid_mapping"),
        extra_attrs    = extra,
    )


def _parse_zgroup(attrs: dict[str, Any], path: str) -> NCZarrGroup:
    """Build an NCZarrGroup from .zattrs dict."""
    nczarr_group = attrs.get("_NCZARR_GROUP") or attrs.get("_nczarr_group") or {}
    raw_dims: dict[str, int] = nczarr_group.get("dims", {})
    clean_attrs = {k: v for k, v in attrs.items() if k not in _STRIP_KEYS}
    return NCZarrGroup(path=path, attrs=clean_attrs, dims=raw_dims)


def _discover_nodes(root: Path) -> tuple[set[str], set[str]]:
    """
    Walk the filesystem tree and collect group and array paths.

    Returns (group_paths, array_paths) as sets of store-absolute paths.
    """
    group_paths: set[str] = {"/"}
    array_paths: set[str] = set()

    for item in root.rglob("*"):
        if item.name == ".zgroup":
            rel = item.parent.relative_to(root)
            gpath = _abs(str(rel)) if str(rel) != "." else "/"
            group_paths.add(gpath)
        elif item.name == ".zarray":
            rel = item.parent.relative_to(root)
            apath = _abs(str(rel))
            array_paths.add(apath)

    return group_paths, array_paths


def read_nczarr_store(source: str) -> NCZarrStore:
    """
    Open an NCZarr v2 store (local path or HTTP URL) and return an
    NCZarrStore populated with all group and array metadata.

    Parameters
    ----------
    source:
        A local filesystem path or an HTTP(S) URL pointing to the root of
        a Zarr v2 / NCZarr store.  HTTP stores are downloaded to a temp
        directory by cli.py before this function is called; passing an
        HTTP URL directly here will raise NotImplementedError.

    Returns
    -------
    NCZarrStore
    """
    if source.startswith("http://") or source.startswith("https://"):
        raise NotImplementedError(
            "Pass a local path; HTTP sources are downloaded to a temp "
            "directory by cli.py before read_nczarr_store is called."
        )

    root = Path(source)
    log.info("Opening NCZarr store: %s", root)

    group_paths, array_paths = _discover_nodes(root)
    log.info("Discovered %d group(s), %d array(s)", len(group_paths), len(array_paths))

    ncs = NCZarrStore()

    # --- Read groups ----------------------------------------------------
    for gpath in sorted(group_paths):
        rel = gpath.lstrip("/")
        node_dir = root / rel if rel else root
        attrs = _read_json_file(node_dir / ".zattrs")
        ncs.groups[gpath] = _parse_zgroup(attrs, gpath)
        log.debug("Group: %s  dims=%s", gpath, ncs.groups[gpath].dims)

    # --- Read arrays ----------------------------------------------------
    for apath in sorted(array_paths):
        rel = apath.lstrip("/")
        node_dir = root / rel

        meta  = _read_json_file(node_dir / ".zarray")
        attrs = _read_json_file(node_dir / ".zattrs")

        if not meta:
            log.warning("Empty .zarray for %s — skipping", apath)
            continue

        arr = _parse_zarray(meta, attrs, apath)
        ncs.arrays[apath] = arr
        log.debug(
            "Array: %s  shape=%s  dtype=%s  dimrefs=%s",
            apath, arr.shape, arr.dtype, arr.dimrefs,
        )

    # Sanity: every array's parent should be a known group
    for apath in ncs.arrays:
        parent = _parent(apath)
        if parent not in ncs.groups:
            log.warning(
                "Array %s has no matching group for parent %s — "
                "adding implicit group", apath, parent,
            )
            ncs.groups[parent] = NCZarrGroup(path=parent)

    log.info("Read complete: %d groups, %d arrays", len(ncs.groups), len(ncs.arrays))
    return ncs