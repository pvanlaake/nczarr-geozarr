"""
classifier.py — Assign a VarKind role to every variable in the store.

Rules (applied in priority order):
  1. GRID_MAPPING  — has a `grid_mapping_name` attribute (it IS a GM variable)
  2. BOUNDS        — is referenced as `bounds` by any other variable
  3. SCALAR_COORD  — shape == [] or shape == [1] AND is referenced as a coordinate
  4. DIM_COORD     — 1-D array whose name == its sole dimension name
                     (name is the last path component, dim is the last component
                     of its sole dimref)
  5. AUX_COORD     — referenced as a coordinate by any data variable but not
                     satisfying rule 4
  6. ANCILLARY     — referenced via `ancillary_variables` by any data variable
  7. DATA          — everything else

The classifier operates on the resolved store (dict of ResolvedVar) plus the
raw NCZarrStore so it can inspect array metadata not carried by ResolvedVar.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from .models import NCZarrStore, ResolvedVar, VarKind

log = logging.getLogger(__name__)


def _name(path: str) -> str:
    """Last path component."""
    return PurePosixPath(path).name


def classify(
    store: NCZarrStore,
    resolved: dict[str, ResolvedVar],
) -> dict[str, VarKind]:
    """
    Return a mapping of store-absolute path → VarKind for every array.
    """

    kinds: dict[str, VarKind] = {}

    # --- Pre-compute sets of referenced paths ---------------------------

    # Paths that serve as bounds arrays
    bounds_paths: set[str] = set()
    for rv in resolved.values():
        if rv.bounds_path:
            bounds_paths.add(rv.bounds_path)

    # Paths that serve as coordinate arrays (aux coords)
    coord_paths: set[str] = set()
    for rv in resolved.values():
        coord_paths.update(rv.coord_paths)

    # Paths that serve as grid-mapping variables
    gm_paths: set[str] = set()
    for rv in resolved.values():
        if rv.grid_mapping_path:
            gm_paths.add(rv.grid_mapping_path)

    # Dimension-coordinate paths (pointed to by dimrefs)
    dim_coord_paths: set[str] = set()
    for rv in resolved.values():
        for p in rv.dim_coord_paths:
            if p is not None:
                dim_coord_paths.add(p)

    # --- Assign roles ---------------------------------------------------

    for path, rv in resolved.items():
        arr = rv.raw

        # Rule 1: grid-mapping variable (has grid_mapping_name, no data role)
        if "grid_mapping_name" in arr.extra_attrs or path in gm_paths:
            if "grid_mapping_name" in arr.extra_attrs:
                kinds[path] = VarKind.GRID_MAPPING
                continue

        # Rule 2: bounds array
        if path in bounds_paths:
            kinds[path] = VarKind.BOUNDS
            continue

        # Rule 3: dimension coordinate — 1-D, name == dim name.
        if len(arr.shape) == 1 and len(arr.dimrefs) == 1:
            var_name = _name(path)
            dim_name = _name(arr.dimrefs[0])
            if var_name == dim_name:
                kinds[path] = VarKind.DIM_COORD
                continue

        # Rule 4: scalar coordinate — CF definition: no dimensions (shape [])
        # A shape-[1] array with a dimref is a regular 1-D variable, not scalar.
        if arr.shape == [] or arr.dimrefs == []:
            if path in coord_paths or path in dim_coord_paths:
                kinds[path] = VarKind.SCALAR_COORD
                continue

        # Rule 5: auxiliary coordinate (referenced by another variable)
        if path in coord_paths:
            kinds[path] = VarKind.AUX_COORD
            continue

        # Rule 6: grid-mapping variable discovered only by reference
        if path in gm_paths:
            kinds[path] = VarKind.GRID_MAPPING
            continue

        # Rule 7: data variable (default)
        kinds[path] = VarKind.DATA

    log.info(
        "Classification: %s",
        {k.name: sum(1 for v in kinds.values() if v == k) for k in VarKind},
    )
    return kinds