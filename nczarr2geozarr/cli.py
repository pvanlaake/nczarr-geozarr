"""
cli.py — Command-line entry point for the NCZarr → GeoZarr/cs translator.

Usage
─────
    nczarr2geozarr <source> <destination> [options]

    source       Local path or HTTP(S) URL to an NCZarr v2 store.
    destination  Local path for the output Zarr v3 store (must not exist,
                 or use --overwrite).

Options
───────
    --overwrite        Remove destination if it already exists.
    --log-level LEVEL  Logging verbosity: DEBUG, INFO (default), WARNING, ERROR.
    --no-copy-chunks   Write metadata only (useful for testing / dry-run).
    --help / -h        Show this message.

Pipeline
────────
    reader    → NCZarrStore       (raw metadata)
    resolver  → ResolvedVar map   (cross-group refs resolved)
    classifier→ VarKind map       (role of each variable)
    builder   → cs attribute dicts
    writer    → Zarr v3 on disk
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from .reader     import read_nczarr_store
from .resolver   import resolve_store
from .classifier import classify
from .builder    import build_store
from .writer     import write_v3_store

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP source support: download to a temp directory if needed
# ---------------------------------------------------------------------------

def _is_http(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _ensure_local(source: str) -> tuple[Path, bool]:
    """
    Return (local_path, is_temp).

    For HTTP sources we cannot copy chunks file-by-file unless the server
    supports directory listing.  Instead we use zarr-python's FSStore to
    enumerate all keys and download them into a temp directory.

    Returns is_temp=True when a temp directory was created (caller must
    clean it up).
    """
    if not _is_http(source):
        return Path(source), False

    log.info("HTTP source detected — downloading store to temporary directory")
    import zarr
    import fsspec

    tmp = Path(tempfile.mkdtemp(prefix="nczarr_to_cs_"))
    log.info("Temporary directory: %s", tmp)

    try:
        src_store = zarr.storage.FSStore(source, mode="r")
        keys = list(src_store.keys())
        log.info("Downloading %d keys…", len(keys))
        for key in keys:
            local_key = tmp / key
            local_key.parent.mkdir(parents=True, exist_ok=True)
            local_key.write_bytes(bytes(src_store[key]))
        log.info("Download complete.")
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Failed to download HTTP store: {exc}") from exc

    return tmp, True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(source: str, destination: str, *, overwrite: bool, copy_chunks: bool) -> None:
    """Execute the full translation pipeline."""

    dst = Path(destination)
    if dst.exists():
        if overwrite:
            log.info("Removing existing destination: %s", dst)
            shutil.rmtree(dst)
        else:
            log.error(
                "Destination already exists: %s  (use --overwrite to replace)", dst
            )
            sys.exit(1)

    # Step 0: ensure local copy for chunk copying
    local_root, is_temp = _ensure_local(source)

    try:
        # Step 1: read raw NCZarr metadata
        log.info("=== Step 1/5: Reading NCZarr metadata ===")
        store = read_nczarr_store(str(local_root) if not _is_http(source) else source)

        # Step 2: resolve cross-group references
        log.info("=== Step 2/5: Resolving cross-group references ===")
        resolved = resolve_store(store)

        # Step 3: classify variables
        log.info("=== Step 3/5: Classifying variables ===")
        kinds = classify(store, resolved)

        # Step 4: build cs attribute trees
        log.info("=== Step 4/5: Building cs attributes ===")
        specs, attrs = build_store(store, resolved, kinds, src_root=local_root)
        log.info("Generated cs specs for %d data variable(s)", len(specs))

        # Step 5: write Zarr v3 output
        log.info("=== Step 5/5: Writing Zarr v3 output ===")
        if copy_chunks:
            write_v3_store(local_root, dst, store, attrs)
        else:
            log.info("--no-copy-chunks: writing metadata only")
            from .writer import (
                _group_zarr_json, _array_zarr_json, build_codecs,
            )
            import json
            from pathlib import PurePosixPath

            dst.mkdir(parents=True, exist_ok=True)
            for gpath, grp in store.groups.items():
                rel = gpath.lstrip("/")
                nd = dst / rel if rel else dst
                nd.mkdir(parents=True, exist_ok=True)
                (nd / "zarr.json").write_text(
                    json.dumps(_group_zarr_json(grp.attrs), indent=2), encoding="utf-8"
                )
            for apath, arr in store.arrays.items():
                rel = apath.lstrip("/")
                nd = dst / rel
                nd.mkdir(parents=True, exist_ok=True)
                dim_names = [PurePosixPath(d).name for d in arr.dimrefs]
                codecs = build_codecs(arr)
                zarr_json = _array_zarr_json(arr, dim_names, codecs, attrs.get(apath, {}))
                (nd / "zarr.json").write_text(
                    json.dumps(zarr_json, indent=2), encoding="utf-8"
                )

        import zarr
        log.info("Consolidating metadata...")
        zarr.consolidate_metadata(str(dst))

        log.info("Done. Output store: %s", dst)

    finally:
        if is_temp:
            log.info("Cleaning up temporary directory: %s", local_root)
            shutil.rmtree(local_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="nczarr-to-cs",
        description=(
            "Translate an NCZarr v2 store to Zarr v3 with GeoZarr cs convention."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source",      help="NCZarr store path or HTTP(S) URL")
    parser.add_argument("destination", help="Output Zarr v3 store path")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Remove destination if it already exists",
    )
    parser.add_argument(
        "--no-copy-chunks", dest="copy_chunks",
        action="store_false", default=True,
        help="Write metadata only (dry-run / testing)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    run(
        source      = args.source,
        destination = args.destination,
        overwrite   = args.overwrite,
        copy_chunks = args.copy_chunks,
    )


if __name__ == "__main__":
    main()