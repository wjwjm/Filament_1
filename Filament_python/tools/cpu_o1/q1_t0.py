"""Linux-only, single-thread CPU-Q1 tiny science gate.

This command is a pre-allocation software/numerical gate. It runs only the
repository's synthetic tiny eight-interval fixture and never loads the sealed
large optical assets or submits scheduler work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from tools.cpu_o1.cpu_o1 import compare_numeric_bounded, make_joint_propagator_factory
from tools.cpu_o1.numeric_gate import build_numeric_gate, write_numeric_gate


IDENTITIES = ("B", "I", "R", "IR")
SCHEMA = "filament.cpu_q1.linux_single_thread_tiny.v1"


def _write_new_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl_record(stream, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, separators=(",", ":"))
    stream.write(payload + "\n")
    stream.flush()


def _numeric_scalar_diagnostics(diag: dict) -> tuple[dict, dict]:
    """Extract numeric scalar leaves omitted by J1's ndarray-only NPZ filter."""
    import math
    import numpy as np

    payload: dict[str, dict] = {}
    values: dict[str, object] = {}

    def encode_real(value) -> str:
        numeric = float(value)
        if math.isnan(numeric):
            return "NaN"
        if math.isinf(numeric):
            return "+Infinity" if numeric > 0 else "-Infinity"
        return numeric.hex()

    def visit(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                visit(f"{prefix}.{key}" if prefix else str(key), value[key])
            return
        if isinstance(value, (np.ndarray, list, tuple, str, bytes, bytearray)):
            return
        if not np.isscalar(value):
            return
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            return
        name = prefix
        scalar = array[()]
        if array.dtype.kind == "c":
            encoded = {"real": encode_real(np.real(scalar)),
                       "imag": encode_real(np.imag(scalar))}
        elif array.dtype.kind == "f":
            encoded = encode_real(scalar)
        elif array.dtype.kind == "b":
            encoded = bool(scalar)
        else:
            encoded = str(int(scalar))
        payload[name] = {"dtype": str(array.dtype), "shape": [], "value": encoded}
        values[name] = scalar

    for key in sorted(diag, key=lambda item: str(item)):
        visit(str(key), diag[key])
    return payload, values


def _write_scalar_diagnostics(path: Path, identity: str, fields: dict) -> None:
    _write_new_json(path, {
        "schema": "filament.cpu_q1.propagation_scalar_diagnostics.v1",
        "identity": identity,
        "source": "private_propagator_return_diag",
        "fields": fields,
    })


def _write_new_array(path: Path, array) -> None:
    import numpy as np

    value = np.asarray(array)
    if value.dtype.kind not in "biufc":
        raise TypeError(f"public Qacc return has unsupported dtype {value.dtype}")
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _array_comparison_row(field_name: str, reference, candidate, *, scope: str):
    """Build one comparison row, using the bounded comparator for float/complex."""
    import numpy as np
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    left, right = np.asarray(reference), np.asarray(candidate)
    if left.shape != right.shape or left.dtype != right.dtype:
        metrics = {
            "reference_shape": list(left.shape),
            "candidate_shape": list(right.shape),
            "reference_dtype": str(left.dtype),
            "candidate_dtype": str(right.dtype),
            "dtype_equal": bool(left.dtype == right.dtype),
            "shape_mismatch": bool(left.shape != right.shape),
            "canonical_exact": False,
            "bitwise_equal": False,
            "all_finite": False,
        }
    elif left.dtype.kind in "fc":
        metrics = compare_numeric_bounded(left, right, max_working_bytes=64 * 1024 * 1024,
                                          max_examples=128)
    else:
        # Discrete scientific fields have an exact dtype/shape/value contract.
        left_hash, right_hash = j1._canonical_hash(left), j1._canonical_hash(right)
        exact = left_hash == right_hash
        metrics = {
            "reference_shape": list(left.shape),
            "candidate_shape": list(right.shape),
            "reference_dtype": str(left.dtype),
            "candidate_dtype": str(right.dtype),
            "dtype_equal": True,
            "element_count": int(left.size),
            "canonical_exact": bool(exact),
            "bitwise_equal": bool(left.tobytes(order="C") == right.tobytes(order="C")),
            "value_equal": bool(np.array_equal(left, right)),
            "all_finite": True,
            "reference_canonical_sha256": left_hash,
            "candidate_canonical_sha256": right_hash,
        }
    return {"field_name": field_name, "scope": scope, "metrics": metrics}


def _compare_window_outputs(reference_dir: Path, candidate_dir: Path,
                            identity: str, *, comparison_path: Path | None = None,
                            scalar_reference: dict | None = None,
                            scalar_candidate: dict | None = None,
                            rows_sink: list[dict] | None = None) -> list[dict]:
    """Compare outputs and persist each raw bounded metric before gate use."""
    import numpy as np
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    ref_root, got_root = reference_dir / "propagation", candidate_dir / "propagation"
    target = comparison_path or (candidate_dir / "comparison.jsonl")
    rows: list[dict] = []
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        def emit(row: dict) -> None:
            record = {"schema": "filament.cpu_q1.raw_comparison.v1",
                      "record_type": "array_comparison", "identity": identity,
                      **row}
            _write_jsonl_record(stream, record)
            rows.append(record)
            if rows_sink is not None:
                rows_sink.append(record)

        try:
            ref_e = np.load(ref_root / "final_E.npy", mmap_mode="r", allow_pickle=False)
            got_e = np.load(got_root / "final_E.npy", mmap_mode="r", allow_pickle=False)
            emit(_array_comparison_row("final_E", ref_e, got_e, scope="science"))
            ref_qacc_path = reference_dir / "scientific_Qacc.npy"
            got_qacc_path = candidate_dir / "scientific_Qacc.npy"
            if ref_qacc_path.is_file() and got_qacc_path.is_file():
                ref_qacc = np.load(ref_qacc_path, mmap_mode="r", allow_pickle=False)
                got_qacc = np.load(got_qacc_path, mmap_mode="r", allow_pickle=False)
                emit(_array_comparison_row(
                    "public_return_Qacc", ref_qacc, got_qacc, scope="science"))
            else:
                emit({"field_name": "public_return_Qacc", "scope": "science",
                    "metrics": {"reference_shape": [int(ref_qacc_path.is_file())],
                        "candidate_shape": [int(got_qacc_path.is_file())],
                        "reference_dtype": "float64", "candidate_dtype": "float64",
                        "dtype_equal": True, "shape_mismatch": True,
                        "topology_equal": False, "all_finite": False,
                        "canonical_exact": False, "bitwise_equal": False}})
            with np.load(ref_root / "final_POST.npz", allow_pickle=False) as left, \
                    np.load(got_root / "final_POST.npz", allow_pickle=False) as right:
                if set(left.files) != set(right.files):
                    emit({"field_name": "final_POST_fields", "scope": "science",
                        "metrics": {"reference_shape": [len(left.files)],
                            "candidate_shape": [len(right.files)],
                            "reference_dtype": "uint8", "candidate_dtype": "uint8",
                            "dtype_equal": True, "shape_mismatch": True,
                            "all_finite": False, "canonical_exact": False,
                            "bitwise_equal": False}})
                else:
                    for name in sorted(left.files):
                        emit(_array_comparison_row(
                            f"final_POST.{name}", left[name], right[name], scope="science"))
            with np.load(ref_root / "diagnostics.npz", allow_pickle=False) as left, \
                    np.load(got_root / "diagnostics.npz", allow_pickle=False) as right:
                if set(left.files) != set(right.files):
                    emit({"field_name": "diagnostic_fields", "scope": "diagnostics",
                        "metrics": {"reference_shape": [len(left.files)],
                            "candidate_shape": [len(right.files)],
                            "reference_dtype": "uint8", "candidate_dtype": "uint8",
                            "dtype_equal": True, "shape_mismatch": True,
                            "all_finite": False, "canonical_exact": False,
                            "bitwise_equal": False}})
                else:
                    for name in sorted(left.files):
                        scope = ("timing" if name in j1.TIMING_DIAGNOSTIC_FIELDS
                                 else "diagnostics")
                        emit(_array_comparison_row(name, left[name], right[name], scope=scope))

            left_scalars = scalar_reference or {}
            right_scalars = scalar_candidate or {}
            if set(left_scalars) != set(right_scalars):
                emit({"field_name": "python_scalar_diagnostic_fields",
                    "scope": "diagnostics", "metrics": {
                        "reference_shape": [len(left_scalars)],
                        "candidate_shape": [len(right_scalars)],
                        "reference_dtype": "uint8", "candidate_dtype": "uint8",
                        "dtype_equal": True, "shape_mismatch": True,
                        "topology_equal": False, "all_finite": False,
                        "canonical_exact": False, "bitwise_equal": False}})
            else:
                for name in sorted(left_scalars):
                    scope = "timing" if name in j1.TIMING_DIAGNOSTIC_FIELDS else "diagnostics"
                    row = _array_comparison_row(
                        name, np.asarray(left_scalars[name]),
                        np.asarray(right_scalars[name]), scope=scope)
                    row["diagnostic_source"] = "private_propagator_return_diag_scalar"
                    emit(row)
        except BaseException as exc:
            _write_jsonl_record(stream, {
                "schema": "filament.cpu_q1.raw_comparison.v1",
                "record_type": "comparison_error", "identity": identity,
                "error_type": type(exc).__name__, "error": str(exc),
                "rows_written": len(rows),
            })
            raise
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    return rows


def _check_identity_dispatch(identity: str, factory, *, kernel_threads: int,
                             temporal_workers: int, spatial_workers: int) -> dict:
    propagator = getattr(factory, "last_propagator", None)
    dispatch = getattr(factory, "last_dispatch", None)
    if propagator is None or dispatch is None:
        raise RuntimeError("joint factory did not expose the executed private propagator")
    execution = dict(getattr(propagator, "_cpu_o1_execution", {}))
    if execution.get("identity") != identity:
        raise RuntimeError("private propagator identity does not match the requested identity")
    expected = {"temporal_workers": temporal_workers, "spatial_workers": spatial_workers,
                "kernel_threads": kernel_threads, "block_pixels": 32, "layout": "NtB"}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"private dispatch settings differ from the tiny-gate contract: {execution}")
    counts = dict(getattr(dispatch, "counts", {}))
    if not counts or any(int(value) <= 0 for value in counts.values()):
        raise RuntimeError("the scoped SciPy FFT dispatch recorded no calls")
    events = list(getattr(propagator, "_cpu_o1_candidate_events", ()))
    event_kinds = sorted({str(row.get("kind")) for row in events})
    required = set()
    if identity in {"I", "IR"}:
        required.add("ionization")
    if identity in {"R", "IR"}:
        required.update(("raman_stage", "raman_apply"))
    if not required.issubset(event_kinds):
        raise RuntimeError(f"candidate symbols were not exercised: required={sorted(required)} seen={event_kinds}")
    return {"implementation_identity": identity,
        "execution": execution,
        "fft_dispatch": dispatch.report(),
        "candidate_event_kinds": event_kinds,
        "candidate_event_count": len(events)}


def _scalar_capture_factory(factory, capture: dict):
    """Wrap only the returned private callable to retain its diagnostic scalars."""
    def create(frozen_propagator):
        propagator, dispatch = factory(frozen_propagator)

        def captured_propagator(*args, **kwargs):
            result = propagator(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) < 3 or not isinstance(result[2], dict):
                raise RuntimeError("private propagator returned no diagnostic mapping")
            capture["qacc"] = result[1]
            payload, values = _numeric_scalar_diagnostics(result[2])
            capture["payload"] = payload
            capture["values"] = values
            return result

        return captured_propagator, dispatch

    return create


def run_linux_tiny_science_gate(output_root: str | os.PathLike, *,
                                bench_root: str | os.PathLike | None = None) -> dict:
    """Run B/I/R/IR on an isolated one-thread tiny fixture and write a gate."""
    import numpy as np
    import threadpoolctl
    import numba
    from tools.cpu_p1 import cpu_p1_fft as common
    from tools.cpu_p1r import cpu_p1r_optical as p1r
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    from tools.cpu_o1 import linux_q1_preflight
    from tools.cpu_o1.cpu_o1 import _load_numba_kernels

    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)
    repo = (Path(bench_root).resolve() if bench_root is not None
            else Path(__file__).resolve().parents[3])
    started = time.perf_counter()
    environment = linux_q1_preflight.run_linux_q1_preflight()
    _write_new_json(root / "linux_environment.json", environment)
    identity_checks = {
        "linux_environment": bool(environment.get("eligible")),
        "source_binding": False,
        "fixed_single_thread_policy": False,
        "all_four_identities_exercised": False,
    }
    rows: list[dict] = []
    per_identity = []
    error = None
    try:
        if not environment.get("eligible"):
            raise RuntimeError("pinned Linux environment probe did not qualify")
        source_binding = p1r._source_binding(repo)
        identity_checks["source_binding"] = True
        numba.set_num_threads(1)
        thread_limit = threadpoolctl.threadpool_limits(limits=1)
        if any(int(row.get("num_threads", 1)) > 1
               for row in threadpoolctl.threadpool_info()):
            raise RuntimeError("native thread pools are not restricted to one thread")
        identity_checks["fixed_single_thread_policy"] = (
            int(numba.get_num_threads()) == 1
            and str(numba.threading_layer()) == "workqueue")
        if not identity_checks["fixed_single_thread_policy"]:
            raise RuntimeError("actual Numba layer/thread count is not workqueue/1")

        baseline_dir = None
        baseline_source_hash = None
        baseline_pre_hash = None
        baseline_scalar_values = None
        schedule_identity = None
        for identity in IDENTITIES:
            stage = root / identity
            stage.mkdir(exist_ok=False)
            context = j1._tiny_context()
            source_hash = j1._canonical_hash(context["source"])
            pre_hash = j1._canonical_hash(context["pre_fields"]["delta_n"])
            schedule_identity_current = json.dumps(
                j1._json_safe(context["schedule"].as_metadata()),
                sort_keys=True, separators=(",", ":"), allow_nan=False)
            if baseline_source_hash is not None and source_hash != baseline_source_hash:
                raise RuntimeError("fresh tiny source differs across identity cases")
            if baseline_pre_hash is not None and pre_hash != baseline_pre_hash:
                raise RuntimeError("fresh tiny PRE0 differs across identity cases")
            if schedule_identity is not None and schedule_identity_current != schedule_identity:
                raise RuntimeError("tiny schedule differs across identity cases")
            baseline_source_hash = source_hash
            baseline_pre_hash = pre_hash
            schedule_identity = schedule_identity_current
            factory = make_joint_propagator_factory(
                identity, kernel_threads=1, block_pixels=32, layout="NtB",
                temporal_workers=1, spatial_workers=1)
            scalar_capture = {}
            scalar_factory = _scalar_capture_factory(factory, scalar_capture)
            receipt = j1._run_window(
                **{key: context[key] for key in (
                    "source", "pre_fields", "grid", "beam", "prop", "ion", "heat",
                    "raman", "schedule")},
                output_dir=stage / "propagation",
                source_label="synthetic_tiny_PRE0", initial_pre0=True,
                profile=False, source_hash=source_hash, pre0_hash=pre_hash,
                stage_name=f"LINUX_TINY_{identity}",
                execution_scope="SOFTWARE_SMOKE_ONLY", observe=False,
                propagator_factory=scalar_factory)
            if any(key not in scalar_capture for key in ("qacc", "payload", "values")):
                raise RuntimeError("private propagation did not expose all public/scalar diagnostics")
            qacc_path = stage / "scientific_Qacc.npy"
            _write_new_array(qacc_path, scalar_capture["qacc"])
            qacc = np.asarray(scalar_capture["qacc"])
            scalar_path = stage / "scientific_scalar_diagnostics.json"
            _write_scalar_diagnostics(scalar_path, identity, scalar_capture["payload"])
            dispatch = _check_identity_dispatch(identity, factory,
                kernel_threads=1, temporal_workers=1, spatial_workers=1)
            _write_new_json(stage / "tiny_receipt.json", {
                "schema": SCHEMA, "status": "COMPLETE_TINY_DIAGNOSTIC",
                "production_qualification": "NOT_PRODUCTION_QUALIFIED",
                "source_binding": source_binding,
                "environment_fingerprint": environment.get("versions"),
                "source_canonical_sha256": source_hash,
                "pre0_delta_n_canonical_sha256": pre_hash,
                "identity": identity, "dispatch": dispatch,
                "public_return_Qacc": {
                    "path": qacc_path.name, "shape": list(qacc.shape),
                    "dtype": str(qacc.dtype),
                    "canonical_sha256": j1._canonical_hash(qacc),
                    "raw_sha256": j1.raw_sha256_file(qacc_path),
                },
                "scientific_scalar_diagnostics": scalar_path.name,
                "scientific_scalar_diagnostic_count": len(scalar_capture["payload"]),
                "propagation_receipt": receipt,
            })
            if identity == "B":
                baseline_dir = stage
                baseline_scalar_values = scalar_capture["values"]
                with (stage / "comparison.jsonl").open(
                        "x", encoding="utf-8", newline="\n") as stream:
                    _write_jsonl_record(stream, {
                        "schema": "filament.cpu_q1.raw_comparison.v1",
                        "record_type": "baseline_no_comparison", "identity": "B",
                        "reason": "B_is_the_same_node_matrix_reference",
                    })
                    os.fsync(stream.fileno())
            else:
                assert baseline_dir is not None
                assert baseline_scalar_values is not None
                _compare_window_outputs(
                    baseline_dir, stage, identity,
                    comparison_path=stage / "comparison.jsonl",
                    scalar_reference=baseline_scalar_values,
                    scalar_candidate=scalar_capture["values"],
                    rows_sink=rows,
                )
            current_gate = build_numeric_gate(
                identity_checks={
                    "environment": identity_checks["linux_environment"],
                    "source_binding": identity_checks["source_binding"],
                    "source_matches_B": True,
                    "pre0_matches_B": True,
                    "schedule_matches_B": True,
                    "dispatch_exercised": bool(dispatch["fft_dispatch"]["transform_counts"]),
                },
                linux_environment_eligible=True,
                comparisons=([row for row in rows if row.get("identity") == identity]
                             if identity != "B" else []))
            current_gate.update({"identity": identity,
                "elapsed_seconds_through_identity": time.perf_counter() - started,
                "dispatch": dispatch})
            _write_new_json(root / f"NUMERIC_GATE_{identity}.json", current_gate)
            per_identity.append(current_gate)
            if not current_gate["proceed_to_next_identity"]:
                break

        identity_checks["all_four_identities_exercised"] = (
            len(per_identity) == len(IDENTITIES)
            and all(row.get("proceed_to_next_identity") for row in per_identity))
        final_gate = build_numeric_gate(
            identity_checks=identity_checks,
            linux_environment_eligible=bool(environment.get("eligible")),
            comparisons=rows)
        final_gate.update({"schema": "filament.cpu_q1.numeric_gate.v1",
            "gate_scope": "LINUX_SINGLE_THREAD_SYNTHETIC_TINY_ONLY",
            "identities_completed": [row["identity"] for row in per_identity],
            "identity_gates": per_identity,
            "environment_receipt": "linux_environment.json",
            "source_binding": source_binding,
            "fft_workers": {"temporal": 1, "spatial": 1},
            "kernel_threads": 1, "layout": "NtB", "block_pixels": 32,
            "elapsed_seconds": time.perf_counter() - started,
            "real_input_used": False,
            "production_qualification": "NOT_PRODUCTION_QUALIFIED"})
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        final_gate = build_numeric_gate(
            identity_checks=identity_checks,
            linux_environment_eligible=bool(environment.get("eligible")),
            comparisons=rows, tool_error=error)
        final_gate.update({"gate_scope": "LINUX_SINGLE_THREAD_SYNTHETIC_TINY_ONLY",
            "identities_completed": [row["identity"] for row in per_identity],
            "identity_gates": per_identity,
            "environment_receipt": "linux_environment.json",
            "fft_workers": {"temporal": 1, "spatial": 1},
            "kernel_threads": 1, "layout": "NtB", "block_pixels": 32,
            "elapsed_seconds": time.perf_counter() - started,
            "real_input_used": False,
            "production_qualification": "NOT_PRODUCTION_QUALIFIED",
            "error": error, "traceback": traceback.format_exc()})
    write_numeric_gate(root / "NUMERIC_GATE.json", final_gate)
    return final_gate


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux-tiny-gate", action="store_true",
                        help="run the single-thread Linux synthetic science gate")
    parser.add_argument("--bench-root", type=Path,
                        help="checkout containing Filament_python; defaults to this module's repository")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new, non-existing directory for gate evidence")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if not args.linux_tiny_gate:
        raise SystemExit("select --linux-tiny-gate")
    try:
        receipt = run_linux_tiny_science_gate(args.output_root, bench_root=args.bench_root)
        return 0 if receipt.get("proceed_to_next_identity") else 1
    except BaseException as exc:
        root = Path(args.output_root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        receipt = build_numeric_gate(
            identity_checks={"linux_tiny_gate_started": True},
            linux_environment_eligible=False,
            tool_error=f"{type(exc).__name__}: {exc}",
        )
        receipt.update({"gate_scope": "LINUX_SINGLE_THREAD_SYNTHETIC_TINY_ONLY",
            "real_input_used": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "production_qualification": "NOT_PRODUCTION_QUALIFIED"})
        write_numeric_gate(root / "NUMERIC_GATE.json", receipt)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
