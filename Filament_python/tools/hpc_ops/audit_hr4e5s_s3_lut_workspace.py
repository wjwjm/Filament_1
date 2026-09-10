#!/usr/bin/env python3
"""Build or attest the frozen S3 ionization LUTs in a private workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_private_workspace(workspace: Path) -> Path:
    """Create a new private workspace or validate an explicitly seeded one."""
    if workspace.exists():
        if workspace.is_symlink() or not workspace.is_dir():
            raise RuntimeError(f"LUT workspace is not a regular directory: {workspace}")
        if (workspace.stat().st_mode & 0o777) != 0o700:
            raise PermissionError(f"LUT workspace must have mode 700: {workspace}")
    else:
        workspace.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(workspace, 0o700)
    return workspace.resolve()


def audit(config_path: Path, workspace: Path, out_path: Path) -> dict:
    """Exercise the frozen relative cache path and return persistent evidence."""
    repo_root = Path(__file__).resolve().parents[3]
    filament_root = repo_root / "Filament_python"
    for candidate in (str(repo_root), str(filament_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    from Filament_python.KHz_filament.confio import load_all
    from Filament_python.KHz_filament.constants import c0
    from Filament_python.KHz_filament.ionization import prepare_ionization_lut_cache
    from Filament_python.KHz_filament.ionization.lut import _canonical_table_metadata, _ion_rate_table_defaults, _table_path, _table_signature
    from Filament_python.KHz_filament.ionization.runtime import _resolve_rate

    if out_path.exists():
        raise FileExistsError(out_path)
    workspace = _prepare_private_workspace(workspace)
    probe = workspace / ".hr4e5s_s3_lut_write_probe"
    with probe.open("x", encoding="utf-8") as handle:
        handle.write("HR-4E-5S S3 LUT workspace write probe\n")
        handle.flush()
        os.fsync(handle.fileno())
    probe.unlink()

    os.chdir(workspace)
    _grid, beam, _prop, ion, _heat, _run, _raman = load_all(config_path)
    table_cfg = _ion_rate_table_defaults(ion)
    cache_dir = Path(str(table_cfg["cache_dir"]))
    if cache_dir.is_absolute():
        raise ValueError("S3 frozen LUT cache_dir must remain relative")
    resolved_cache_dir = (workspace / cache_dir).resolve()
    if not _within(resolved_cache_dir, workspace):
        raise ValueError("S3 LUT cache_dir escapes its private workspace")
    omega0 = 2.0 * math.pi * float(c0) / float(beam.lam0)
    n0 = float(beam.n0)

    tables = prepare_ionization_lut_cache(ion, omega0_SI=omega0, n0=n0)
    records = []
    for species in list(getattr(ion, "species", None) or []):
        resolved_rate = _resolve_rate(species, ion)
        if resolved_rate not in ("ppt_talebpour_i_lut", "popruzhenko_atom_i_lut"):
            continue
        local_species = dict(species)
        local_species["rate"] = resolved_rate
        reference_model = str(local_species.get("reference_model", "")).lower()
        if not reference_model:
            reference_model = "ppt_talebpour_i_full_reference" if resolved_rate == "ppt_talebpour_i_lut" else "popruzhenko_atom_i_full_reference"
            local_species["reference_model"] = reference_model
        metadata = _canonical_table_metadata(reference_model, str(local_species.get("name", "species")), local_species, omega0, n0, table_cfg)
        signature = _table_signature(metadata)
        lut_path = Path(_table_path(str(cache_dir), metadata["species_name"], signature)).resolve()
        if not _within(lut_path, workspace) or not lut_path.is_file():
            raise RuntimeError(f"missing or unsafe LUT artifact: {lut_path}")
        matching = [table for table in tables if table.get("metadata") == metadata]
        if len(matching) != 1:
            raise RuntimeError(f"metadata validation failed for LUT species={metadata['species_name']}")
        validation = dict(matching[0].get("validation") or {})
        if not validation or not all(math.isfinite(float(value)) for value in validation.values()):
            raise RuntimeError(f"missing or non-finite LUT validation for species={metadata['species_name']}")
        records.append({"species": metadata["species_name"], "resolved_rate": resolved_rate, "canonical_metadata": metadata, "canonical_signature": signature, "lut_path": str(lut_path), "lut_sha256": _sha256(lut_path), "validation": validation})
    if not records:
        raise RuntimeError("frozen S3 config did not produce any LUT species")

    filesystem = os.statvfs(workspace)
    result = {"schema": "khz_filament.hr4e5s.s3.lut_workspace.v1", "status": "PASS", "config": str(config_path.resolve()), "config_sha256": _sha256(config_path), "workspace": str(workspace), "workspace_mode_octal": format(workspace.stat().st_mode & 0o777, "03o"), "write_probe": "PASS", "f_bavail_bytes": int(filesystem.f_bavail * filesystem.f_frsize), "cache_dir_relative": str(cache_dir), "cache_dir_resolved": str(resolved_cache_dir), "lut_records": records}
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.config, args.workspace, args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
