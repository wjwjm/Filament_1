"""HR-3C-C transactional state controller and atomic manifest lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .device import to_cpu
from .slow_state_pingpong import PingPongSlowStateStore, diffuse_current_to_next


SCHEMA = "khz_filament.hr3c.manifest.v1"


def _fsync_memmap(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def state_fingerprint(*, z_edges, shape, dtype, dx: float, dy: float) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(z_edges, dtype=np.float64).tobytes())
    digest.update(json.dumps({"shape": list(shape), "dtype": np.dtype(dtype).name, "dx": float(dx), "dy": float(dy)}, sort_keys=True).encode())
    return digest.hexdigest()


class PulseSlowStateTransaction:
    """A read-only pre-state / write-only post-state adapter for one pulse."""

    def __init__(self, store: PingPongSlowStateStore):
        self.store = store
        self.read_indices: set[int] = set()
        self.written_indices: set[int] = set()
        self.store.begin_next_pass()

    def read_interval(self, interval_index: int):
        index = int(interval_index)
        if index in self.read_indices:
            raise ValueError("HR-3C transaction interval may be read only once")
        self.read_indices.add(index)
        return self.store.read_current_interval(index)

    def update_interval(self, interval_index: int, increment):
        index = int(interval_index)
        if index not in self.read_indices or index in self.written_indices:
            raise ValueError("HR-3C transaction requires one prior read and one post write per interval")
        pre = np.asarray(self.store.read_current_interval(index), dtype=self.store.dtype)
        post = pre + np.asarray(to_cpu(increment), dtype=self.store.dtype)
        self.store.write_next_batch(index, post[None, :, :])
        self.written_indices.add(index)
        return post

    def finalize(self) -> None:
        expected = set(range(self.store.n_intervals))
        if self.read_indices != expected or self.written_indices != expected:
            self.store.mark_next_invalid()
            raise ValueError("HR-3C pulse transaction requires every interval exactly once")
        self.store.flush_next()
        _fsync_memmap(self.store.next_path)
        self.store.mark_next_complete()

    def metadata(self) -> dict[str, object]:
        return {
            "hr3b_state_schema": "khz_filament.hr3c.transactional_delta_n_th.v1",
            "hr3b_state_filename": self.store.next_path.name,
            "hr3b_state_dtype": self.store.dtype.name,
            "hr3b_state_shape": self.store.state_shape,
            "hr3b_state_interval_centered": True,
            "hr3b_state_disk_backed": True,
        }


class HR3CStateController:
    def __init__(self, *, output_path: str, n_intervals: int, shape, dtype, z_edges, dx, dy, D_th, f_rep, edge_threshold, batch_intervals, npulses: int, resume: bool):
        self.output_path = str(output_path)
        self.manifest_path = Path(output_path).with_suffix(".hr3c_state_manifest.json")
        self.expected = {
            "state_shape": [int(n_intervals), *[int(x) for x in shape]], "state_dtype": np.dtype(dtype).name,
            "n_intervals": int(n_intervals), "interval_centered": True, "D_th": float(D_th), "f_rep": float(f_rep),
            "dt_interpulse": 1.0 / float(f_rep), "edge_threshold": float(edge_threshold),
            "batch_intervals": int(batch_intervals), "npulses": int(npulses),
            "fingerprint": state_fingerprint(z_edges=z_edges, shape=shape, dtype=dtype, dx=dx, dy=dy),
        }
        root = Path(output_path).with_suffix("")
        slot_a = root.with_name(root.name + ".hr3c_delta_n_th_current.npy")
        slot_b = root.with_name(root.name + ".hr3c_delta_n_th_next.npy")
        exists = self.manifest_path.exists() or slot_a.exists() or slot_b.exists()
        if resume:
            if not self.manifest_path.is_file():
                raise FileNotFoundError("HR-3C resume requested but manifest is missing")
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._validate_manifest(self.manifest)
            self.store = PingPongSlowStateStore.open_existing(output_path=output_path, n_intervals=n_intervals, shape=shape, dtype=dtype)
            self.store.select_authoritative(self.manifest["authoritative_filename"])
        else:
            if exists:
                raise FileExistsError("HR-3C new run refuses to overwrite existing manifest or state slots")
            self.store = PingPongSlowStateStore(output_path=output_path, n_intervals=n_intervals, shape=shape, dtype=dtype)
            self.manifest = {"schema_version": SCHEMA, "physical_stage": "pre_pulse", "pulse_index": -1, "next_pulse_index": 0, "authoritative_filename": self.store.current_path.name, "scratch_filename": self.store.next_path.name, "run_complete": False, **self.expected}
            self.manifest.update({"n_fresh_pulses_completed_total": 0, "n_hr3b_post_commits_total": 0, "n_hr3c_diffusion_passes_total": 0})
            self._write_manifest()

    def _validate_manifest(self, manifest: dict) -> None:
        for key, value in self.expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"HR-3C resume fingerprint mismatch: {key}")
        if manifest.get("schema_version") != SCHEMA or manifest.get("physical_stage") not in ("pre_pulse", "post_pulse"):
            raise ValueError("HR-3C manifest is invalid")
        slots = {Path(self.output_path).with_suffix("").name + ".hr3c_delta_n_th_current.npy", Path(self.output_path).with_suffix("").name + ".hr3c_delta_n_th_next.npy"}
        if manifest.get("authoritative_filename") not in slots or manifest.get("scratch_filename") not in slots or manifest.get("authoritative_filename") == manifest.get("scratch_filename"):
            raise ValueError("HR-3C manifest slot invariant failed")
        p, nxt, n, complete = int(manifest.get("pulse_index", -99)), int(manifest.get("next_pulse_index", -99)), int(self.expected["npulses"]), manifest.get("run_complete")
        if not isinstance(complete, bool): raise ValueError("HR-3C manifest run_complete invariant failed")
        for key in ("n_fresh_pulses_completed_total", "n_hr3b_post_commits_total", "n_hr3c_diffusion_passes_total"):
            if not isinstance(manifest.get(key), int) or manifest[key] < 0:
                raise ValueError("HR-3C manifest counter invariant failed")
        if manifest["physical_stage"] == "pre_pulse" and (complete or not (0 <= nxt < n and p == nxt - 1)):
            raise ValueError("HR-3C pre_pulse manifest invariant failed")
        if manifest["physical_stage"] == "post_pulse" and (not (0 <= p < n and nxt == p + 1) or (p == n - 1) != complete):
            raise ValueError("HR-3C post_pulse manifest invariant failed")
        fresh = manifest["n_fresh_pulses_completed_total"]
        post = manifest["n_hr3b_post_commits_total"]
        diffusions = manifest["n_hr3c_diffusion_passes_total"]
        if fresh != post:
            raise ValueError("HR-3C manifest pulse/post counter invariant failed")
        if manifest["physical_stage"] == "pre_pulse":
            expected_counts = (nxt, nxt, nxt)
        else:
            expected_counts = (p + 1, p + 1, p)
        if (fresh, post, diffusions) != expected_counts:
            raise ValueError("HR-3C manifest counter/stage/index invariant failed")

    def _write_manifest(self) -> None:
        _atomic_json(self.manifest_path, self.manifest)

    def begin_pulse(self) -> PulseSlowStateTransaction:
        self._validate_manifest(self.manifest)
        if self.manifest["run_complete"] or self.manifest["physical_stage"] != "pre_pulse":
            raise ValueError("HR-3C manifest is not ready for a fresh pulse")
        return PulseSlowStateTransaction(self.store)

    def commit_post_pulse(self, transaction: PulseSlowStateTransaction, pulse_index: int) -> None:
        transaction.finalize()
        final = int(pulse_index) == int(self.expected["npulses"]) - 1
        self.manifest.update({"physical_stage": "post_pulse", "pulse_index": int(pulse_index), "next_pulse_index": int(pulse_index) + 1, "authoritative_filename": self.store.next_path.name, "scratch_filename": self.store.current_path.name, "run_complete": final, "n_fresh_pulses_completed_total": int(self.manifest["n_fresh_pulses_completed_total"]) + 1, "n_hr3b_post_commits_total": int(self.manifest["n_hr3b_post_commits_total"]) + 1})
        self._write_manifest()
        self.store.select_authoritative(self.manifest["authoritative_filename"])

    def diffuse_to_next_pre(self) -> dict:
        self._validate_manifest(self.manifest)
        if self.manifest["physical_stage"] != "post_pulse" or self.manifest["run_complete"]:
            raise ValueError("HR-3C manifest is not ready for diffusion")
        summary = diffuse_current_to_next(self.store, kperp2=self.kperp2, D_th=self.expected["D_th"], f_rep=self.expected["f_rep"], edge_threshold=self.expected["edge_threshold"], batch_intervals=self.expected["batch_intervals"])
        _fsync_memmap(self.store.next_path)
        self.manifest.update({"physical_stage": "pre_pulse", "next_pulse_index": int(self.manifest["next_pulse_index"]), "authoritative_filename": self.store.next_path.name, "scratch_filename": self.store.current_path.name, "n_hr3c_diffusion_passes_total": int(self.manifest["n_hr3c_diffusion_passes_total"]) + 1})
        self._write_manifest()
        self.store.select_authoritative(self.manifest["authoritative_filename"])
        return summary

    def attach_grid(self, kperp2) -> None:
        self.kperp2 = kperp2

    def close(self) -> None:
        self.store.close()
