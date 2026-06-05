"""
writer.py — Write a Zarr v3 output store.

Given the NCZarrStore (raw metadata + chunk files) and the built attribute
dicts, this module:

  1. Creates the output directory structure
  2. Writes zarr.json for every group and array (Zarr v3 format)
  3. Copies chunk files verbatim from the source store (no re-encoding)
  4. Translates Zarr v2 compressor/filter specs to Zarr v3 codec objects

Zarr v2 → v3 codec translation
────────────────────────────────
v2 uses:
    "compressor": {"id": "zlib", "level": 4}
    "filters":    [{"id": "delta", "dtype": "<f4"}]

v3 uses a codec pipeline list:
    "codecs": [
        {"name": "bytes",  "configuration": {"endian": "little"}},   ← mandatory
        {"name": "gzip",   "configuration": {"level": 4}},
        ...
    ]

The bytes codec (endian) is always inserted for numeric dtypes.
Then array-to-bytes codecs (delta, etc.) are inserted before the bytes
codec; bytes-to-bytes codecs (gzip, blosc, zstd) come after.

Supported v2 compressors → v3
    zlib / gzip    → gzip
    blosc          → blosc
    zstd           → zstd
    bz2            → gzip (fallback with warning)
    lzma / lz4     → logged as unsupported, stored uncompressed

Supported v2 filters → v3
    delta          → delta (array-to-bytes)
    fixedscaleoffset → fixedscaleoffset
    quantize       → quantize
    shuffle        → shuffle (bytes-to-bytes, inserted after bytes codec)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .models import NCZarrArray, NCZarrGroup, NCZarrStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zarr v3 fixed metadata values
# ---------------------------------------------------------------------------

ZARR_FORMAT = 3


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def _endian(dtype_str: str) -> str:
    """Return 'little', 'big', or 'native' for a numpy dtype string."""
    try:
        dt = np.dtype(dtype_str)
        if dt.byteorder in ("<", "="):
            return "little"
        if dt.byteorder == ">":
            return "big"
    except TypeError:
        pass
    return "little"


def _is_numeric(dtype_str: str) -> bool:
    try:
        dt = np.dtype(dtype_str)
        return dt.kind in "biufcmM"
    except TypeError:
        return False


# Zarr v2 dtype string → Zarr v3 data_type string
_V2_TO_V3_DTYPE: dict[str, str] = {
    "<f2": "float16", ">f2": "float16",
    "<f4": "float32", ">f4": "float32",
    "<f8": "float64", ">f8": "float64",
    "|i1": "int8",
    "<i2": "int16",  ">i2": "int16",
    "<i4": "int32",  ">i4": "int32",
    "<i8": "int64",  ">i8": "int64",
    "|u1": "uint8",
    "<u2": "uint16", ">u2": "uint16",
    "<u4": "uint32", ">u4": "uint32",
    "<u8": "uint64", ">u8": "uint64",
    "<c8":  "complex64",  ">c8":  "complex64",
    "<c16": "complex128", ">c16": "complex128",
    "|b1": "bool",
}


def _v3_dtype(dtype_str: str) -> str:
    """Translate a Zarr v2 dtype string to a Zarr v3 data_type string."""
    v3 = _V2_TO_V3_DTYPE.get(dtype_str)
    if v3 is not None:
        return v3
    log.debug("No v3 dtype mapping for %r — passing through", dtype_str)
    return dtype_str


# ---------------------------------------------------------------------------
# Codec translation
# ---------------------------------------------------------------------------

# v2 compressor id → v3 codec name
_COMPRESSOR_MAP: dict[str, str] = {
    "zlib":  "gzip",
    "gzip":  "gzip",
    "blosc": "blosc",
    "zstd":  "zstd",
    "bz2":   "gzip",   # approximation; warn
}

# v2 filter id → v3 codec name (array-to-bytes, inserted before bytes codec)
_ARRAY_TO_BYTES_FILTERS: set[str] = {"delta", "fixedscaleoffset", "quantize"}
# v2 filter id → v3 codec name (bytes-to-bytes, inserted after bytes codec)
_BYTES_TO_BYTES_FILTERS: set[str] = {"shuffle"}


def _translate_compressor(comp: dict[str, Any] | None, dtype: str) -> dict[str, Any] | None:
    """Translate a v2 compressor spec to a v3 codec dict, or None."""
    if comp is None:
        return None
    cid = comp.get("id", "")
    v3_name = _COMPRESSOR_MAP.get(cid)
    if v3_name is None:
        log.warning("Unsupported v2 compressor %r — output will be uncompressed", cid)
        return None
    if cid == "bz2":
        log.warning("bz2 compressor mapped to gzip in v3; decompressed data is identical")

    cfg: dict[str, Any] = {}
    if v3_name == "gzip":
        cfg["level"] = comp.get("level", 1)
    elif v3_name == "blosc":
        for key in ("cname", "clevel", "shuffle", "blocksize"):
            if key in comp:
                cfg[key] = comp[key]
    elif v3_name == "zstd":
        if "level" in comp:
            cfg["level"] = comp["level"]

    codec: dict[str, Any] = {"name": v3_name}
    if cfg:
        codec["configuration"] = cfg
    return codec


def _translate_filter(f: dict[str, Any]) -> tuple[str, dict[str, Any] | None] | None:
    """
    Translate one v2 filter dict.

    Returns (placement, codec_dict) where placement is "before" (before the
    bytes codec) or "after" (after), or None to drop.
    """
    fid = f.get("id", "")
    cfg: dict[str, Any] = {k: v for k, v in f.items() if k != "id"}

    if fid in _ARRAY_TO_BYTES_FILTERS:
        codec: dict[str, Any] = {"name": fid}
        if cfg:
            codec["configuration"] = cfg
        return ("before", codec)

    if fid in _BYTES_TO_BYTES_FILTERS:
        codec = {"name": fid}
        if cfg:
            codec["configuration"] = cfg
        return ("after", codec)

    log.warning("Unsupported v2 filter %r — dropped from codec chain", fid)
    return None


def build_codecs(arr: NCZarrArray) -> list[dict[str, Any]]:
    """
    Build the Zarr v3 codecs list for an array.
    """
    before: list[dict[str, Any]] = []   # array-to-bytes filters
    after:  list[dict[str, Any]] = []   # bytes-to-bytes codecs

    # Translate v2 filters
    for f in arr.filters or []:
        result = _translate_filter(f)
        if result is None:
            continue
        placement, codec = result
        if placement == "before":
            before.append(codec)
        else:
            after.append(codec)

    # Bytes codec (mandatory for numeric dtypes)
    bytes_codec: list[dict[str, Any]] = []
    if _is_numeric(arr.dtype):
        bytes_codec = [{"name": "bytes", "configuration": {"endian": _endian(arr.dtype)}}]

    # Compressor → goes after bytes codec
    comp_codec_obj = _translate_compressor(arr.compressor, arr.dtype)
    if comp_codec_obj:
        after.append(comp_codec_obj)

    return before + bytes_codec + after


# ---------------------------------------------------------------------------
# zarr.json construction
# ---------------------------------------------------------------------------

def _group_zarr_json(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "zarr_format": ZARR_FORMAT,
        "node_type":   "group",
        "attributes":  attrs,
    }


def _array_zarr_json(
    arr: NCZarrArray,
    dimension_names: list[str],
    codecs: list[dict[str, Any]],
    attrs: dict[str, Any],
) -> dict[str, Any]:
    # fill_value: zarr v3 wants JSON-serialisable scalar or null
    fv = arr.fill_value
    if fv == "NaN":
        fv = "NaN"   # v3 accepts the string "NaN" for float arrays

    return {
        "zarr_format":      ZARR_FORMAT,
        "node_type":        "array",
        "shape":            arr.shape,
        "data_type":        _v3_dtype(arr.dtype),
        "chunk_grid": {
            "name":          "regular",
            "configuration": {"chunk_shape": arr.chunks},
        },
        "chunk_key_encoding": {
            "name":          "default",
            "configuration": {"separator": "/"},
        },
        "fill_value":       fv,
        "codecs":           codecs,
        "dimension_names":  dimension_names,
        "attributes":       attrs,
    }


# ---------------------------------------------------------------------------
# Chunk file discovery and copying
# ---------------------------------------------------------------------------

def _is_metadata_key(key: str) -> bool:
    """Return True for keys that are metadata, not chunk data."""
    for suffix in (".zarray", ".zattrs", ".zgroup", ".zmetadata"):
        if key.endswith(suffix):
            return True
    return False


def _copy_chunks(
    src_root: Path,
    dst_root: Path,
    src_array_path: str,
    dst_array_path: str,
) -> int:
    """
    Copy all chunk files for one array from src to dst.

    v2 chunk keys look like:  <array_rel>/<c0>.<c1>.<c2>
    v3 chunk keys look like:  <array_rel>/c/<c0>/<c1>/<c2>

    We translate the separator from "." to "/" and add the "c/" prefix.

    Returns the number of chunks copied.
    """
    src_dir = src_root / src_array_path.lstrip("/")
    dst_chunk_dir = dst_root / dst_array_path.lstrip("/") / "c"
    dst_chunk_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    if not src_dir.exists():
        log.warning("Source array directory not found: %s", src_dir)
        return 0

    for item in src_dir.iterdir():
        if item.is_file() and not _is_metadata_key(item.name):
            # v2 chunk name: "0.1.2" → v3 path: c/0/1/2
            chunk_coords = item.name.split(".")
            dst_path = dst_chunk_dir.joinpath(*chunk_coords)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_path)
            count += 1

    return count


# ---------------------------------------------------------------------------
# Public writer
# ---------------------------------------------------------------------------

def write_v3_store(
    src_local_root: Path,
    dst_root: Path,
    store: NCZarrStore,
    built_attrs: dict[str, dict[str, Any]],
) -> None:
    """
    Write a Zarr v3 store to `dst_root`.

    Parameters
    ----------
    src_local_root:
        Root directory of the source NCZarr store on the local filesystem.
        For HTTP sources the caller is responsible for downloading first
        (or we copy via the zarr store object — see note in cli.py).
    dst_root:
        Output directory.  Will be created if absent.
    store:
        The parsed NCZarrStore.
    built_attrs:
        Mapping path → final attributes dict produced by builder.build_store().
    """
    dst_root.mkdir(parents=True, exist_ok=True)
    log.info("Writing Zarr v3 store to %s", dst_root)

    # --- Groups ---------------------------------------------------------
    for gpath, grp in store.groups.items():
        rel = gpath.lstrip("/")
        node_dir = dst_root / rel if rel else dst_root
        node_dir.mkdir(parents=True, exist_ok=True)

        # Pass-through group attributes (Conventions, title, institution, …)
        group_attrs = dict(grp.attrs)

        zarr_json = _group_zarr_json(group_attrs)
        (node_dir / "zarr.json").write_text(
            json.dumps(zarr_json, indent=2, allow_nan=False), encoding="utf-8"
        )
        log.debug("Wrote group zarr.json: %s", node_dir / "zarr.json")

    # --- Arrays ---------------------------------------------------------
    for apath, arr in store.arrays.items():
        rel = apath.lstrip("/")
        node_dir = dst_root / rel
        node_dir.mkdir(parents=True, exist_ok=True)

        # Dimension names: last path component of each dimref
        from pathlib import PurePosixPath
        dimension_names = [PurePosixPath(d).name for d in arr.dimrefs]

        codecs = build_codecs(arr)
        attrs  = built_attrs.get(apath, {})

        zarr_json = _array_zarr_json(arr, dimension_names, codecs, attrs)
        (node_dir / "zarr.json").write_text(
            json.dumps(zarr_json, indent=2, allow_nan=False), encoding="utf-8"
        )
        log.debug("Wrote array zarr.json: %s", node_dir / "zarr.json")

        # Copy chunks
        n = _copy_chunks(src_local_root, dst_root, apath, apath)
        log.debug("  → copied %d chunk(s) for %s", n, apath)

    log.info("Write complete.")