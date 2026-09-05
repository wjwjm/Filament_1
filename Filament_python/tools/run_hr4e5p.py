#!/usr/bin/env python3
"""Controlled HR-4E-5P preparation, worker, gather, and comparison commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from KHz_filament.hr4c_state import HR4CThreeFieldStore, evolve_hr4_full_z
from KHz_filament.hr4e3_domain import e3_geometry
from KHz_filament.hr4e5_parallel import (
    build_screen_blocks,
    compare_store_states,
    execute_worker,
    gather_worker_outputs,
    open_store_from_spec,
    screen_independence_audit,
    store_spec,
)
from KHz_filament.hr4e_timestep import json_safe, sha256_array, sha256_file


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(value)), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _parse_indices(values: Sequence[str]) -> list[int]:
    result: list[int] = []
    for raw in values:
        for item in raw.split(","):
            value = int(item)
            if value < 0 or value in result:
                raise ValueError("screen indices must be unique non-negative integers")
            result.append(value)
    if not result:
        raise ValueError("at least one screen index is required")
    return result


def _initialize_store(path: Path, *, state: Mapping[str, np.ndarray], geometry: Mapping[str, Any]) -> dict[str, Any]:
    count = int(state["delta_n"].shape[0])
    store = HR4CThreeFieldStore(
        output_path=str(path), n_intervals=count, shape=tuple(state["delta_n"].shape[1:]), dtype=np.float64,
        z_edges=np.arange(count + 1, dtype=np.float64), dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]),
        authoritative_metadata={
            "schema": "khz_filament.hr4e5p.validation_subset.v1",
            "z_layout": "ordinal_slots_with_original_z_identity_in_records",
            "geometry": dict(geometry),
        },
    )
    try:
        store.begin_staging()
        for index in range(count):
            store.write_staging_batch(index, {field: state[field][index:index + 1] for field in ("delta_n", "vx", "vy")})
        store.commit_staging({"operation": "e5p_validation_subset_initialization", "batch_intervals": 1})
        return store_spec(store)
    finally:
        store.close()


def command_audit(args: argparse.Namespace) -> int:
    audit = screen_independence_audit()
    _write_json(args.out, audit)
    lines = ["# HR-4E-5P Screen Independence Audit", "", f"**{audit['status']}**", ""]
    for key, value in audit["evidence"].items():
        lines.append(f"- `{key}`: `{value}`")
    markdown = args.out.with_name("E5P_SCREEN_INDEPENDENCE_AUDIT.md")
    if markdown.exists():
        raise FileExistsError(markdown)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    source_manifest = _read_json(args.source_manifest)
    geometry = e3_geometry("D0")
    target = dict(source_manifest.get("target_grid", {}))
    if target != {key: geometry[key] for key in geometry if key != "domain_id"}:
        raise ValueError("source manifest does not prove the accepted D0 target grid")
    indices = _parse_indices(args.screen_index)
    state_file = Path(args.source_state)
    if sha256_file(state_file) != str(source_manifest.get("hr3b_state_file_sha256", "")):
        raise ValueError("source HR-3B state file hash does not match source manifest")
    source = np.load(state_file, mmap_mode="r", allow_pickle=False)
    try:
        if source.dtype != np.dtype("float64") or source.ndim != 3 or source.shape[1:] != (int(geometry["Ny"]), int(geometry["Nx"])):
            raise ValueError("source HR-3B state is not the accepted float64 D0 layout")
        if max(indices) >= source.shape[0]:
            raise ValueError("requested screen index exceeds source state")
        z_positions = source_manifest.get("source_z_positions_m")
        if not isinstance(z_positions, list) or len(z_positions) != source.shape[0]:
            raise ValueError("source manifest lacks complete z identity data")
        selected = np.asarray(source[indices], dtype=np.float64)
    finally:
        closer = getattr(source, "_mmap", None)
        if closer is not None:
            closer.close()
    state = {"delta_n": selected, "vx": np.zeros_like(selected), "vy": np.zeros_like(selected)}
    records = [{
        "ordinal": ordinal, "screen_id": f"source_index_{index:05d}", "source_index": index,
        "z_m": float(z_positions[index]), "source_array_sha256": sha256_array(selected[ordinal]),
    } for ordinal, index in enumerate(indices)]
    destination = Path(args.out_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    serial_state = _initialize_store(destination / "serial_input.npz", state=state, geometry=geometry)
    parallel_state = _initialize_store(destination / "parallel_input.npz", state=state, geometry=geometry)
    _write_json(destination / "serial_state.json", serial_state)
    _write_json(destination / "parallel_state.json", parallel_state)
    prepared = {
        "schema": "khz_filament.hr4e5p.validation_input.v1", "source_manifest": str(Path(args.source_manifest)),
        "source_manifest_sha256": sha256_file(args.source_manifest), "source_state": str(state_file),
        "source_state_file_sha256": sha256_file(state_file), "source_state_array_sha256": source_manifest["hr3b_state_sha256"],
        "geometry": geometry, "dtype": "float64", "screen_records": records,
        "serial_state": serial_state, "parallel_state": parallel_state,
    }
    _write_json(destination / "validation_input.json", prepared)
    return 0


def command_partition(args: argparse.Namespace) -> int:
    prepared = _read_json(args.input)
    partition = build_screen_blocks(prepared["screen_records"], block_size=args.block_size, n_workers=args.n_workers)
    partition.update({"state": prepared["parallel_state"], "validation_input_sha256": sha256_file(args.input)})
    _write_json(args.out, partition)
    return 0


def command_dry_run(args: argparse.Namespace) -> int:
    count = int(args.n_screens)
    if count <= 0:
        raise ValueError("n_screens must be positive")
    records = [{"ordinal": index, "screen_id": f"source_index_{index:05d}"} for index in range(count)]
    partition = build_screen_blocks(records, block_size=args.block_size, n_workers=args.n_workers)
    partition["dry_run"] = {"mode": "metadata_only_no_hr4_computation", "n_screens": count}
    _write_json(args.out, partition)
    return 0


def command_serial(args: argparse.Namespace) -> int:
    state = _read_json(args.state)
    store = open_store_from_spec(state)
    try:
        result = evolve_hr4_full_z(
            store, dt_hydro=args.dt_hydro, n_hydro_steps=args.n_hydro_steps, batch_intervals=args.batch_intervals,
            chi=args.chi, nu=args.nu, n0=args.n0, gravity_x=args.gravity_x, gravity_y=args.gravity_y,
        )
        result["output_state"] = store_spec(store)
    finally:
        store.close()
    _write_json(args.out, result)
    return 0


def command_worker(args: argparse.Namespace) -> int:
    partition = _read_json(args.partition)
    index = int(os.environ["SLURM_PROCID"]) if args.worker_index is None else int(args.worker_index)
    blocks = [block for block in partition["blocks"] if int(block["worker_index"]) == index]
    result = execute_worker(
        worker_index=index, blocks=blocks, state=partition["state"], out_dir=args.out_dir,
        dt_hydro=args.dt_hydro, n_hydro_steps=args.n_hydro_steps, chi=args.chi, nu=args.nu, n0=args.n0,
        gravity_x=args.gravity_x, gravity_y=args.gravity_y,
    )
    print(json.dumps({"worker_index": index, "walltime_s": result["walltime_s"]}, sort_keys=True))
    return 0


def command_gather(args: argparse.Namespace) -> int:
    partition = _read_json(args.partition)
    workers = [_read_json(path) for path in sorted(Path(args.worker_dir).glob("worker_*.json"))]
    result = gather_worker_outputs(
        state=partition["state"], partition=partition, worker_manifests=workers, batch_intervals=args.batch_intervals,
        evolution_metadata={"dt_hydro": args.dt_hydro, "n_hydro_steps": args.n_hydro_steps, "chi": args.chi, "nu": args.nu},
    )
    _write_json(args.out, result)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    prepared = _read_json(args.input)
    serial = _read_json(args.serial)["output_state"]
    parallel = _read_json(args.gather)["output_state"]
    report = compare_store_states(reference_state=serial, candidate_state=parallel, records=prepared["screen_records"])
    _write_json(args.out, report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit"); audit.add_argument("--out", type=Path, required=True); audit.set_defaults(func=command_audit)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-manifest", type=Path, required=True); prepare.add_argument("--source-state", type=Path, required=True)
    prepare.add_argument("--screen-index", action="append", required=True); prepare.add_argument("--out-dir", type=Path, required=True); prepare.set_defaults(func=command_prepare)
    partition = sub.add_parser("partition"); partition.add_argument("--input", type=Path, required=True); partition.add_argument("--block-size", type=int, required=True); partition.add_argument("--n-workers", type=int, required=True); partition.add_argument("--out", type=Path, required=True); partition.set_defaults(func=command_partition)
    dry = sub.add_parser("dry-run"); dry.add_argument("--n-screens", type=int, required=True); dry.add_argument("--block-size", type=int, required=True); dry.add_argument("--n-workers", type=int, required=True); dry.add_argument("--out", type=Path, required=True); dry.set_defaults(func=command_dry_run)
    serial = sub.add_parser("serial"); serial.add_argument("--state", type=Path, required=True); serial.add_argument("--out", type=Path, required=True)
    worker = sub.add_parser("worker"); worker.add_argument("--partition", type=Path, required=True); worker.add_argument("--worker-index", type=int); worker.add_argument("--out-dir", type=Path, required=True)
    gather = sub.add_parser("gather"); gather.add_argument("--partition", type=Path, required=True); gather.add_argument("--worker-dir", type=Path, required=True); gather.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare"); compare.add_argument("--input", type=Path, required=True); compare.add_argument("--serial", type=Path, required=True); compare.add_argument("--gather", type=Path, required=True); compare.add_argument("--out", type=Path, required=True)
    for command in (serial, worker, gather):
        command.add_argument("--dt-hydro", type=float, default=1.0e-6); command.add_argument("--n-hydro-steps", type=int, default=1000)
        command.add_argument("--chi", type=float, default=21.7e-6); command.add_argument("--nu", type=float, default=1.5e-5); command.add_argument("--n0", type=float, default=1.00027)
        command.add_argument("--gravity-x", type=float, default=0.0); command.add_argument("--gravity-y", type=float, default=-9.81)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
