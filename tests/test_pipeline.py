"""
tests/test_pipeline.py

Creates a minimal NCZarr v2 store on disk (mimicking what NCO / libnetcdf
produce) and exercises the full reader → resolver → classifier → builder
→ writer pipeline.

The synthetic store has:
    /                           root group
    /lon    (xi_rho,)           dim coord, degrees_east
    /lat    (eta_rho,)          dim coord, degrees_north
    /temp   (eta_rho, xi_rho)   data variable with coordinates="lat lon"
    /time   (time,)             dim coord, "days since 2000-01-01"
    /crs                        grid_mapping variable (latitude_longitude)

This exercises:
  • multi-group resolution (all in root here, but resolver is generic)
  • dim coord classification
  • aux coord attachment via `coordinates`
  • proj:code inference from grid_mapping_name=latitude_longitude
  • codec translation (zlib → gzip)
  • Zarr v3 zarr.json output format
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers to write an NCZarr v2 store by hand
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_chunk_zlib(path: Path, data: list[float], dtype: str = "<f4") -> None:
    """Write a zlib-compressed chunk file."""
    import numpy as np
    arr = np.array(data, dtype=dtype)
    raw = arr.tobytes()
    compressed = zlib.compress(raw, level=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)


def make_nczarr_store(root: Path) -> None:
    """Write a minimal NCZarr v2 store under `root`."""

    # --- Root group -----------------------------------------------------
    _write_json(root / ".zgroup", {"zarr_format": 2})
    _write_json(root / ".zattrs", {
        "_NCZARR_GROUP": {
            "dims":   {"xi_rho": 4, "eta_rho": 3, "time": 2},
            "vars":   ["lon", "lat", "temp", "time", "crs"],
            "groups": [],
        },
        "Conventions": "CF-1.9",
        "title":       "Synthetic NCZarr test store",
    })

    # --- lon (xi_rho,) --------------------------------------------------
    _write_json(root / "lon" / ".zarray", {
        "zarr_format": 2,
        "shape":       [4],
        "chunks":      [4],
        "dtype":       "<f8",
        "compressor":  {"id": "zlib", "level": 4},
        "filters":     None,
        "fill_value":  "NaN",
        "order":       "C",
    })
    _write_json(root / "lon" / ".zattrs", {
        "_NCZARR_ARRAY": {"dimrefs": ["/lon"]},
        "units":         "degrees_east",
        "standard_name": "longitude",
        "axis":          "X",
        "long_name":     "Longitude",
    })
    _write_chunk_zlib(root / "lon" / "0",
                      [-10.0, -9.5, -9.0, -8.5], dtype="<f8")

    # --- lat (eta_rho,) -------------------------------------------------
    _write_json(root / "lat" / ".zarray", {
        "zarr_format": 2,
        "shape":       [3],
        "chunks":      [3],
        "dtype":       "<f8",
        "compressor":  {"id": "zlib", "level": 4},
        "filters":     None,
        "fill_value":  "NaN",
        "order":       "C",
    })
    _write_json(root / "lat" / ".zattrs", {
        "_NCZARR_ARRAY": {"dimrefs": ["/lat"]},
        "units":         "degrees_north",
        "standard_name": "latitude",
        "axis":          "Y",
        "long_name":     "Latitude",
    })
    _write_chunk_zlib(root / "lat" / "0",
                      [20.0, 20.5, 21.0], dtype="<f8")

    # --- time (time,) ---------------------------------------------------
    _write_json(root / "time" / ".zarray", {
        "zarr_format": 2,
        "shape":       [2],
        "chunks":      [2],
        "dtype":       "<f8",
        "compressor":  {"id": "zlib", "level": 4},
        "filters":     None,
        "fill_value":  "NaN",
        "order":       "C",
    })
    _write_json(root / "time" / ".zattrs", {
        "_NCZARR_ARRAY": {"dimrefs": ["/time"]},
        "units":    "days since 2000-01-01",
        "calendar": "proleptic_gregorian",
        "standard_name": "time",
        "axis":     "T",
        "long_name": "Time",
    })
    _write_chunk_zlib(root / "time" / "0", [0.0, 1.0], dtype="<f8")

    # --- crs (grid mapping, shape []) -----------------------------------
    _write_json(root / "crs" / ".zarray", {
        "zarr_format": 2,
        "shape":       [],
        "chunks":      [],
        "dtype":       "|i4",
        "compressor":  None,
        "filters":     None,
        "fill_value":  0,
        "order":       "C",
    })
    _write_json(root / "crs" / ".zattrs", {
        "_NCZARR_ARRAY":    {"dimrefs": []},
        "grid_mapping_name": "latitude_longitude",
        "longitude_of_prime_meridian": 0.0,
        "semi_major_axis":  6378137.0,
        "inverse_flattening": 298.257223563,
    })

    # --- temp (time, eta_rho, xi_rho) -----------------------------------
    import numpy as np
    _write_json(root / "temp" / ".zarray", {
        "zarr_format": 2,
        "shape":       [2, 3, 4],
        "chunks":      [2, 3, 4],
        "dtype":       "<f4",
        "compressor":  {"id": "zlib", "level": 4},
        "filters":     None,
        "fill_value":  9.969209968386869e+36,
        "order":       "C",
    })
    _write_json(root / "temp" / ".zattrs", {
        "_NCZARR_ARRAY": {"dimrefs": ["/time", "/lat", "/lon"]},
        "standard_name": "sea_water_potential_temperature",
        "long_name":     "Potential Temperature",
        "units":         "degree_C",
        "grid_mapping":  "crs",
        "coordinates":   "lat lon time",
    })
    data = np.random.default_rng(42).uniform(15, 25, (2, 3, 4)).astype("<f4")
    _write_chunk_zlib(root / "temp" / "0.0.0", data.flatten().tolist(), dtype="<f4")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nczarr_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("nczarr_store")
    make_nczarr_store(root)
    return root


@pytest.fixture(scope="module")
def output_root(tmp_path_factory, nczarr_root) -> Path:
    """Run the full pipeline and return the output store root."""
    from nczarr_to_cs.reader     import read_nczarr_store
    from nczarr_to_cs.resolver   import resolve_store
    from nczarr_to_cs.classifier import classify
    from nczarr_to_cs.builder    import build_store
    from nczarr_to_cs.writer     import write_v3_store

    store    = read_nczarr_store(str(nczarr_root))
    resolved = resolve_store(store)
    kinds    = classify(store, resolved)
    _, attrs = build_store(store, resolved, kinds, src_root=nczarr_root)

    out = tmp_path_factory.mktemp("v3_store")
    write_v3_store(nczarr_root, out, store, attrs)
    return out


# --- Reader tests ----------------------------------------------------------

class TestReader:
    def test_groups_discovered(self, nczarr_root):
        from nczarr_to_cs.reader import read_nczarr_store
        s = read_nczarr_store(str(nczarr_root))
        assert "/" in s.groups

    def test_arrays_discovered(self, nczarr_root):
        from nczarr_to_cs.reader import read_nczarr_store
        s = read_nczarr_store(str(nczarr_root))
        assert {"/lon", "/lat", "/time", "/temp", "/crs"} <= s.arrays.keys()

    def test_dimrefs_parsed(self, nczarr_root):
        from nczarr_to_cs.reader import read_nczarr_store
        s = read_nczarr_store(str(nczarr_root))
        assert s.arrays["/lon"].dimrefs == ["/lon"]
        assert s.arrays["/temp"].dimrefs == ["/time", "/lat", "/lon"]

    def test_cf_attrs_lifted(self, nczarr_root):
        from nczarr_to_cs.reader import read_nczarr_store
        s = read_nczarr_store(str(nczarr_root))
        assert s.arrays["/lon"].units == "degrees_east"
        assert s.arrays["/lon"].standard_name == "longitude"
        assert s.arrays["/temp"].grid_mapping == "crs"
        assert s.arrays["/temp"].coordinates == "lat lon time"

    def test_nczarr_keys_stripped(self, nczarr_root):
        from nczarr_to_cs.reader import read_nczarr_store
        s = read_nczarr_store(str(nczarr_root))
        assert "_NCZARR_ARRAY" not in s.arrays["/temp"].extra_attrs


# --- Resolver tests --------------------------------------------------------

class TestResolver:
    def test_coord_paths_resolved(self, nczarr_root):
        from nczarr_to_cs.reader   import read_nczarr_store
        from nczarr_to_cs.resolver import resolve_store
        s  = read_nczarr_store(str(nczarr_root))
        rv = resolve_store(s)
        temp_rv = rv["/temp"]
        assert set(temp_rv.coord_paths) >= {"/lat", "/lon", "/time"}

    def test_grid_mapping_resolved(self, nczarr_root):
        from nczarr_to_cs.reader   import read_nczarr_store
        from nczarr_to_cs.resolver import resolve_store
        s  = read_nczarr_store(str(nczarr_root))
        rv = resolve_store(s)
        assert rv["/temp"].grid_mapping_path == "/crs"

    def test_dim_coord_paths(self, nczarr_root):
        from nczarr_to_cs.reader   import read_nczarr_store
        from nczarr_to_cs.resolver import resolve_store
        s  = read_nczarr_store(str(nczarr_root))
        rv = resolve_store(s)
        # temp has dimrefs [/time, /lat, /lon]; all are coordinate arrays
        assert rv["/temp"].dim_coord_paths == ["/time", "/lat", "/lon"]


# --- Classifier tests ------------------------------------------------------

class TestClassifier:
    def test_temp_is_data(self, nczarr_root):
        from nczarr_to_cs.reader     import read_nczarr_store
        from nczarr_to_cs.resolver   import resolve_store
        from nczarr_to_cs.classifier import classify
        from nczarr_to_cs.models     import VarKind
        s = read_nczarr_store(str(nczarr_root))
        r = resolve_store(s)
        k = classify(s, r)
        assert k["/temp"] == VarKind.DATA

    def test_lon_lat_are_dim_coords(self, nczarr_root):
        from nczarr_to_cs.reader     import read_nczarr_store
        from nczarr_to_cs.resolver   import resolve_store
        from nczarr_to_cs.classifier import classify
        from nczarr_to_cs.models     import VarKind
        s = read_nczarr_store(str(nczarr_root))
        r = resolve_store(s)
        k = classify(s, r)
        assert k["/lon"] == VarKind.DIM_COORD
        assert k["/lat"] == VarKind.DIM_COORD

    def test_crs_is_grid_mapping(self, nczarr_root):
        from nczarr_to_cs.reader     import read_nczarr_store
        from nczarr_to_cs.resolver   import resolve_store
        from nczarr_to_cs.classifier import classify
        from nczarr_to_cs.models     import VarKind
        s = read_nczarr_store(str(nczarr_root))
        r = resolve_store(s)
        k = classify(s, r)
        assert k["/crs"] == VarKind.GRID_MAPPING


# --- Builder tests ---------------------------------------------------------

class TestBuilder:
    def test_cs_spec_generated_for_temp(self, nczarr_root):
        from nczarr_to_cs.reader     import read_nczarr_store
        from nczarr_to_cs.resolver   import resolve_store
        from nczarr_to_cs.classifier import classify
        from nczarr_to_cs.builder    import build_store
        s = read_nczarr_store(str(nczarr_root))
        r = resolve_store(s)
        k = classify(s, r)
        specs, _ = build_store(s, r, k, src_root=nczarr_root)
        assert "/temp" in specs

    def test_proj_code_epsg4326(self, nczarr_root):
        from nczarr_to_cs.reader     import read_nczarr_store
        from nczarr_to_cs.resolver   import resolve_store
        from nczarr_to_cs.classifier import classify
        from nczarr_to_cs.builder    import build_store
        s = read_nczarr_store(str(nczarr_root))
        r = resolve_store(s)
        k = classify(s, r)
        specs, _ = build_store(s, r, k, src_root=nczarr_root)
        # proj_code goes on the spatial CRS (last one)
        spatial_crs = specs["/temp"].crs[-1]
        assert spatial_crs.proj_code == "EPSG:4326"

    def test_axes_abbreviations(self, nczarr_root):
        from nczarr_to_cs.reader     import read_nczarr_store
        from nczarr_to_cs.resolver   import resolve_store
        from nczarr_to_cs.classifier import classify
        from nczarr_to_cs.builder    import build_store
        from nczarr_to_cs.models     import AxisAbbrev
        s = read_nczarr_store(str(nczarr_root))
        r = resolve_store(s)
        k = classify(s, r)
        specs, _ = build_store(s, r, k, src_root=nczarr_root)
        # Collect all axes across all CRS objects
        all_axes = {}
        for crs in specs["/temp"].crs:
            all_axes.update(crs.axes)
        assert all_axes["lon"].abbreviation == AxisAbbrev.X
        assert all_axes["lat"].abbreviation == AxisAbbrev.Y
        assert all_axes["time"].abbreviation == AxisAbbrev.T

    def test_temporal_and_spatial_crs(self, nczarr_root):
        from nczarr_to_cs.reader     import read_nczarr_store
        from nczarr_to_cs.resolver   import resolve_store
        from nczarr_to_cs.classifier import classify
        from nczarr_to_cs.builder    import build_store
        from nczarr_to_cs.models     import AxisAbbrev
        s = read_nczarr_store(str(nczarr_root))
        r = resolve_store(s)
        k = classify(s, r)
        specs, _ = build_store(s, r, k, src_root=nczarr_root)
        crs_list = specs["/temp"].crs
        assert len(crs_list) == 2
        abbrevs = [list(crs.axes.values())[0].abbreviation for crs in crs_list]
        assert AxisAbbrev.T in abbrevs

    def test_zarr_conventions_cmo_present(self, nczarr_root):
        from nczarr_to_cs.reader     import read_nczarr_store
        from nczarr_to_cs.resolver   import resolve_store
        from nczarr_to_cs.classifier import classify
        from nczarr_to_cs.builder    import build_store
        s = read_nczarr_store(str(nczarr_root))
        r = resolve_store(s)
        k = classify(s, r)
        _, attrs = build_store(s, r, k, src_root=nczarr_root)
        temp_attrs = attrs["/temp"]
        assert "zarr_conventions" in temp_attrs
        cmos = temp_attrs["zarr_conventions"]
        assert any(c.get("name") == "cs" for c in cmos)


# --- Writer tests ----------------------------------------------------------

class TestWriter:
    def test_root_zarr_json_exists(self, output_root):
        assert (output_root / "zarr.json").exists()

    def test_root_is_group(self, output_root):
        meta = json.loads((output_root / "zarr.json").read_text())
        assert meta["zarr_format"] == 3
        assert meta["node_type"]   == "group"

    def test_temp_zarr_json_exists(self, output_root):
        assert (output_root / "temp" / "zarr.json").exists()

    def test_temp_is_array_v3(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        assert meta["zarr_format"] == 3
        assert meta["node_type"]   == "array"

    def test_dimension_names_written(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        assert meta["dimension_names"] == ["time", "lat", "lon"]

    def test_cs_in_output_attributes(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        assert "cs" in meta["attributes"]
        cs = meta["attributes"]["cs"]
        assert "crs" in cs
        assert len(cs["crs"]) == 2   # temporal + spatial

    def test_proj_code_in_output(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        # proj_code on the spatial (last) CRS
        crs = meta["attributes"]["cs"]["crs"][-1]
        assert crs.get("proj:code") == "EPSG:4326"

    def test_time_explicit_values(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        temporal_crs = meta["attributes"]["cs"]["crs"][0]
        time_axis = temporal_crs["axes"]["time"]
        coord = time_axis["coordinates"][0]
        # Small time array → explicit values, always a list
        assert "explicit" in coord["values"]
        assert isinstance(coord["values"]["explicit"], list)

    def test_time_unit_split(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        temporal_crs = meta["attributes"]["cs"]["crs"][0]
        time_axis = temporal_crs["axes"]["time"]
        coord = time_axis["coordinates"][0]
        # Time info goes in "time" key, not "unit"
        assert "time" in coord
        assert "unit" not in coord
        assert "unit" in coord["time"] and "epoch" in coord["time"]

    def test_spatial_coord_values_present(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        spatial_crs = meta["attributes"]["cs"]["crs"][-1]
        for axis in spatial_crs["axes"].values():
            for coord in axis.get("coordinates", []):
                values = coord["values"]
                # Must be either explicit (small array) or external ref
                assert "explicit" in values or "external" in values

    def test_gzip_codec_in_output(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        codec_names = [c["name"] for c in meta["codecs"]]
        assert "gzip" in codec_names

    def test_bytes_codec_present(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        codec_names = [c["name"] for c in meta["codecs"]]
        assert "bytes" in codec_names

    def test_nczarr_keys_absent_from_output(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        attrs = meta["attributes"]
        assert "_NCZARR_ARRAY" not in attrs
        assert "_ARRAY_DIMENSIONS" not in attrs

    def test_chunk_file_copied(self, output_root):
        # Chunk "0.0.0" in v2 → c/0/0/0 in v3
        chunk_path = output_root / "temp" / "c" / "0" / "0" / "0"
        assert chunk_path.exists(), f"Chunk not found: {chunk_path}"

    def test_cf_passthrough_attrs(self, output_root):
        meta = json.loads((output_root / "temp" / "zarr.json").read_text())
        attrs = meta["attributes"]
        assert attrs.get("standard_name") == "sea_water_potential_temperature"
        assert attrs.get("units") == "degree_C"