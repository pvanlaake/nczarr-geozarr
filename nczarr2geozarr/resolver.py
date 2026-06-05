"""
resolver.py — Cross-group reference resolution.

Takes the flat NCZarrStore (raw metadata snapshot) produced by reader.py
and resolves all symbolic references into store-absolute paths:

  • dimrefs          → absolute paths to dimension-coordinate arrays
  • coordinates      → list of absolute paths to auxiliary coordinate arrays
  • bounds           → absolute path to bounds array
  • grid_mapping     → absolute path to grid-mapping variable

The resolution rules follow CF conventions + NCZarr path conventions:

  1. dimrefs are already store-absolute (normalised by reader.py).

  2. `coordinates`, `bounds`, and `grid_mapping` are bare names relative
     to the variable's own group.  We search outward through ancestor
     groups (CF §2.6.2 scoping rule) until a matching array is found.

  3. If a name cannot be resolved we log a warning and skip it; we never
     raise — partial resolution is always better than a hard failure.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from .models import NCZarrArray, NCZarrStore, ResolvedVar

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _abs(path: str) -> str:
    p = "/" + path.strip("/")
    return "/" if p == "//" else p


def _parent(path: str) -> str:
    p = str(PurePosixPath(_abs(path)).parent)
    return p or "/"


def _join(group: str, name: str) -> str:
    """Join a group path and a bare name into a store-absolute path."""
    if group == "/":
        return f"/{name}"
    return f"{group.rstrip('/')}/{name}"


def _ancestor_groups(path: str) -> list[str]:
    """
    Return [path, parent(path), grandparent, …, "/"] for a given path,
    i.e. the chain of ancestor groups from nearest to root.
    """
    ancestors: list[str] = []
    current = _abs(path)
    while True:
        ancestors.append(current)
        if current == "/":
            break
        current = _parent(current)
    return ancestors


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------

class Resolver:
    """
    Resolves all cross-group CF references in an NCZarrStore.

    Usage
    -----
    resolver = Resolver(store)
    resolved_vars = resolver.resolve_all()
    """

    def __init__(self, store: NCZarrStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_all(self) -> dict[str, ResolvedVar]:
        """
        Resolve every array in the store and return a dict mapping
        store-absolute path → ResolvedVar.
        """
        result: dict[str, ResolvedVar] = {}
        for path, arr in self._store.arrays.items():
            result[path] = self._resolve_one(arr)
        return result

    # ------------------------------------------------------------------
    # Per-variable resolution
    # ------------------------------------------------------------------

    def _resolve_one(self, arr: NCZarrArray) -> ResolvedVar:
        group = _parent(arr.path)

        coord_paths      = self._resolve_names_attr(arr.coordinates, group, arr.path, "coordinates")
        bounds_path      = self._resolve_single(arr.bounds,       group, arr.path, "bounds")
        grid_mapping_path = self._resolve_single(arr.grid_mapping, group, arr.path, "grid_mapping")
        dim_coord_paths  = self._resolve_dimcoords(arr.dimrefs, arr.path)

        return ResolvedVar(
            raw               = arr,
            coord_paths       = coord_paths,
            bounds_path       = bounds_path,
            grid_mapping_path = grid_mapping_path,
            dim_coord_paths   = dim_coord_paths,
        )

    # ------------------------------------------------------------------
    # Dimension coordinate resolution
    # ------------------------------------------------------------------

    def _resolve_dimcoords(
        self, dimrefs: list[str], var_path: str
    ) -> list[str | None]:
        """
        For each dimref (absolute path to the dimension's defining array),
        attempt to find the array at that path.

        NCZarr's dimrefs point directly to the coordinate array, so in
        most cases this is a simple lookup.  If the array is absent (the
        dimension is declared but has no coordinate variable), we return
        None for that position.
        """
        result: list[str | None] = []
        for ref in dimrefs:
            if ref in self._store.arrays:
                result.append(ref)
            else:
                log.debug(
                    "%s: dimref %s not found as array (dimension without coordinate)",
                    var_path, ref,
                )
                result.append(None)
        return result

    # ------------------------------------------------------------------
    # Auxiliary coordinate / bounds / grid_mapping resolution
    # ------------------------------------------------------------------

    def _resolve_names_attr(
        self,
        names_str: str | None,
        group: str,
        var_path: str,
        attr_name: str,
    ) -> list[str]:
        """
        Resolve a space-separated CF names attribute (e.g. `coordinates`)
        to a list of store-absolute paths.
        """
        if not names_str:
            return []
        resolved = []
        for name in names_str.split():
            path = self._find_array(name, group, var_path, attr_name)
            if path is not None:
                resolved.append(path)
        return resolved

    def _resolve_single(
        self,
        name: str | None,
        group: str,
        var_path: str,
        attr_name: str,
    ) -> str | None:
        """Resolve a single bare name to a store-absolute path, or None."""
        if not name:
            return None
        return self._find_array(name, group, var_path, attr_name)

    def _find_array(
        self,
        name: str,
        start_group: str,
        var_path: str,
        attr_name: str,
    ) -> str | None:
        """
        Search for an array named `name` by walking up the group hierarchy
        from `start_group` to the store root.

        CF §2.6.2: a referenced variable must be accessible from the same
        group or an ancestor group.  NCZarr serialises references as bare
        names within a group scope, so we must search upward.
        """
        for ancestor in _ancestor_groups(start_group):
            candidate = _join(ancestor, name)
            if candidate in self._store.arrays:
                if ancestor != start_group:
                    log.debug(
                        "%s attr '%s': resolved '%s' → %s (found in ancestor %s)",
                        var_path, attr_name, name, candidate, ancestor,
                    )
                return candidate

        log.warning(
            "%s attr '%s': cannot resolve '%s' — no array found in "
            "%s or any ancestor group",
            var_path, attr_name, name, start_group,
        )
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_store(store: NCZarrStore) -> dict[str, ResolvedVar]:
    """
    Convenience wrapper: resolve all references in `store` and return the
    mapping of store-absolute path → ResolvedVar.
    """
    return Resolver(store).resolve_all()
