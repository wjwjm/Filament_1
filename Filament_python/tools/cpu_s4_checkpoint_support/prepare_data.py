"""Materialize the explicit, hash-verified small-grid external-data layout."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from tools.hpc_ops import provenance_v2 as pv

LUTS = (
    ("LUT_MANIFEST.json", 1935, "00d341f6e1cc49b56a38ce24771119a6a28665497068521140ab07dbb13b2ee5"),
    ("cache/rate_tables/ionlut_N2_76585af6514cdd04.npz", 49064, "1f70691ab004c410123f5e06a0dbfa626beecf4059c09b715a8052feeb4e128e"),
    ("cache/rate_tables/ionlut_O2_1446bb99e8f9fd80.npz", 49013, "57617bc72acbb794ae2c939241beaa57fa313eaa27a3d2981882f1fbe68d6cf8"),
)

def materialize(source: Path, target: Path):
    if not source.is_absolute() or not target.is_absolute():
        raise RuntimeError("EXPLICIT_ABSOLUTE_DATA_PATHS_REQUIRED")
    source = source.resolve(strict=True)
    target = target.absolute()
    if target.exists():
        raise FileExistsError("DATA_ROOT_MUST_BE_NEW")
    records = []
    for name, size, sha in LUTS:
        path = source / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size or pv.raw_sha256_file(path) != sha:
            raise RuntimeError("EXTERNAL_LUT_MISSING_OR_CHANGED:" + name)
        records.append(dict(path=name, classification="external", hash_scope="raw_bytes", raw_sha256=sha, size_bytes=size))
    target.mkdir(parents=True, exist_ok=False)
    for row in records:
        destination = target / "inputs" / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / row["path"], destination)
    shutil.copyfile(Path(__file__).parent / "config/SMALLGRID_CONFIG.json", target / "SMALLGRID_CONFIG.json")
    (target / "raw").mkdir()
    with (target / "raw/FROZEN_LUT_RETRIEVAL.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(records, stream, indent=2); stream.write("\n")
    result = dict(status="PASS_EXPLICIT_EXTERNAL_DATA_MATERIALIZATION", inputs=records,
                  config_raw_sha256=pv.raw_sha256_file(target / "SMALLGRID_CONFIG.json"),
                  source="Three externally supplied frozen LUT artifacts and tracked config; no code or arrays fetched implicitly")
    with (target / "DATA_RECEIPT.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lut-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_lut_root, args.data_root)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
