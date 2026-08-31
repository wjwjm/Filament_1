"""HR-4D PRE/POST pulse lifecycle over the HR-4C three-field store."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .device import to_cpu
from .hr4c_state import HR4CThreeFieldStore, HR4C_FIELDS, evolve_hr4_full_z


HR4D_SCHEMA = "khz_filament.hr4d.pulse_lifecycle.v1"
HR4D_PHASE_PRE = "PRE"
HR4D_PHASE_POST = "POST"


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"HR-4D {name} must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"HR-4D {name} must be a positive finite real number")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"HR-4D {name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class InterpulseStepSchedule:
    """Exact fixed-step-plus-remainder request; not an adaptive scheduler."""

    duration_s: float
    dt_hydro_s: float
    full_step_count: int
    remainder_s: float

    @property
    def entries(self) -> tuple[tuple[float, int], ...]:
        values = []
        if self.full_step_count:
            values.append((self.dt_hydro_s, self.full_step_count))
        if self.remainder_s > 0.0:
            values.append((self.remainder_s, 1))
        return tuple(values)

    @property
    def total_step_count(self) -> int:
        return self.full_step_count + int(self.remainder_s > 0.0)


def build_interpulse_step_schedule(*, f_rep: float, dt_hydro: float) -> InterpulseStepSchedule:
    """Resolve exactly 1/f_rep into full fixed steps and one optional remainder."""
    repetition_rate = _positive_finite(f_rep, "f_rep")
    dt_value = _positive_finite(dt_hydro, "dt_hydro")
    duration = 1.0 / repetition_rate
    ratio = duration / dt_value
    tolerance = 32.0 * math.ulp(max(abs(duration), abs(dt_value), 1.0)) / dt_value
    full_steps = math.floor(ratio + tolerance)
    remainder = duration - full_steps * dt_value
    if abs(remainder) <= 32.0 * math.ulp(max(abs(duration), abs(dt_value), 1.0)):
        remainder = 0.0
    if remainder < 0.0:
        raise ValueError("HR-4D interpulse schedule became negative after floating-point normalization")
    schedule = InterpulseStepSchedule(
        duration_s=duration,
        dt_hydro_s=dt_value,
        full_step_count=int(full_steps),
        remainder_s=float(remainder),
    )
    if not schedule.entries:
        raise ValueError("HR-4D interpulse schedule contains no evolution step")
    if not math.isclose(
        schedule.full_step_count * schedule.dt_hydro_s + schedule.remainder_s,
        schedule.duration_s,
        rel_tol=0.0,
        abs_tol=64.0 * math.ulp(max(abs(schedule.duration_s), abs(schedule.dt_hydro_s), 1.0)),
    ):
        raise ValueError("HR-4D interpulse schedule does not close to 1/f_rep")
    return schedule


class HR4DPulseTransaction:
    """HR-3B-compatible interval adapter backed by one HR-4C staging generation."""

    def __init__(self, controller: "HR4DPulseController"):
        self.controller = controller
        self.store = controller.store
        self.read_indices: set[int] = set()
        self.written_indices: set[int] = set()
        self.closed = False

    def read_interval(self, interval_index: int):
        index = int(interval_index)
        if index < 0 or index >= self.store.n_intervals:
            raise IndexError("HR-4D pulse interval is outside the slow state")
        if index in self.read_indices:
            raise ValueError("HR-4D pulse transaction may read each interval only once")
        self.read_indices.add(index)
        return self.store.read_authoritative_batch(index, index + 1)["delta_n"][0]

    def update_interval(self, interval_index: int, delta_n_increment):
        index = int(interval_index)
        if index not in self.read_indices or index in self.written_indices:
            raise ValueError("HR-4D pulse transaction requires one read and one POST write per interval")
        increment = np.asarray(to_cpu(delta_n_increment), dtype=self.store.dtype)
        if increment.shape != self.store.shape or not np.all(np.isfinite(increment)):
            raise ValueError("HR-4D HR-3B delta_n increment is invalid")
        pre = self.store.read_authoritative_batch(index, index + 1)
        post_delta_n = np.asarray(pre["delta_n"][0], dtype=self.store.dtype) + increment
        if not np.all(np.isfinite(post_delta_n)):
            raise ValueError("HR-4D POST delta_n is non-finite")
        self.store.write_staging_batch(index, {
            "delta_n": post_delta_n[None, :, :],
            "vx": pre["vx"],
            "vy": pre["vy"],
        })
        self.written_indices.add(index)
        return post_delta_n

    def finalize(self) -> None:
        if self.closed:
            raise ValueError("HR-4D pulse transaction is already closed")
        expected = set(range(self.store.n_intervals))
        if self.read_indices != expected or self.written_indices != expected:
            self.controller.abort_pulse_transition(reason="incomplete_pulse_transaction")
            self.closed = True
            raise ValueError("HR-4D pulse transaction requires every interval exactly once")
        self.controller._commit_post_transition()
        self.closed = True

    def metadata(self) -> dict[str, object]:
        """Expose the existing HR-3B diagnostic metadata shape without new authority."""
        return {
            "hr3b_state_schema": "khz_filament.hr4d.post_transaction.v1",
            "hr3b_state_filename": self.store.manifest["scratch_filenames"]["delta_n"],
            "hr3b_state_dtype": self.store.dtype.name,
            "hr3b_state_shape": self.store.state_shape,
            "hr3b_state_interval_centered": True,
            "hr3b_state_disk_backed": True,
            "hr4d_authoritative_fields": HR4C_FIELDS,
        }


class HR4DPulseController:
    """Restart-safe PRE/POST state machine with HR-4C as the sole authority store."""

    def __init__(
        self, *, output_path: str, n_intervals: int, shape, dtype, z_edges, dx: float, dy: float,
        n_pulses: int, f_rep: float, dt_hydro: float, batch_intervals: int,
        chi: float, nu: float, n0: float, gravity_x: float = 0.0, gravity_y: float = -9.81,
        cfl_limit: float = 1.0, resume: bool = False,
    ):
        self.n_pulses = _positive_integer(n_pulses, "n_pulses")
        self.f_rep = _positive_finite(f_rep, "f_rep")
        self.dt_hydro = _positive_finite(dt_hydro, "dt_hydro")
        self.batch_intervals = _positive_integer(batch_intervals, "batch_intervals")
        self.chi, self.nu, self.n0 = float(chi), float(nu), _positive_finite(n0, "n0")
        self.gravity_x, self.gravity_y, self.cfl_limit = float(gravity_x), float(gravity_y), float(cfl_limit)
        common = dict(
            output_path=output_path, n_intervals=n_intervals, shape=shape, dtype=dtype,
            z_edges=z_edges, dx=dx, dy=dy,
        )
        initial_metadata = self._metadata_for(
            pulse_index=0, phase=HR4D_PHASE_PRE, source_generation=None,
            state_generation=0, transition="initial_state",
        )
        self.store = (
            HR4CThreeFieldStore.open_existing(**common)
            if resume else HR4CThreeFieldStore(**common, authoritative_metadata=initial_metadata)
        )
        if resume:
            self._validate_authoritative_metadata()

    def _flow_parameters(self) -> dict[str, float | int]:
        return {
            "f_rep_hz": self.f_rep,
            "dt_hydro_s": self.dt_hydro,
            "batch_intervals": self.batch_intervals,
            "chi_m2_s": self.chi,
            "nu_m2_s": self.nu,
            "n0": self.n0,
            "gravity_x_m_s2": self.gravity_x,
            "gravity_y_m_s2": self.gravity_y,
            "cfl_limit": self.cfl_limit,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        self._validate_authoritative_metadata()
        return dict(self.store.manifest["authoritative_metadata"])

    @property
    def is_complete(self) -> bool:
        return bool(self.metadata["run_complete"])

    @property
    def next_action(self) -> str:
        metadata = self.metadata
        if metadata["run_complete"]:
            return "complete"
        return "pulse" if metadata["phase"] == HR4D_PHASE_PRE else "interpulse"

    def _metadata_for(
        self, *, pulse_index: int, phase: str, source_generation: int | None,
        state_generation: int, transition: str,
    ) -> dict[str, Any]:
        post = phase == HR4D_PHASE_POST
        return {
            "schema_version": HR4D_SCHEMA,
            "pulse_index": int(pulse_index),
            "phase": phase,
            "n_pulses": self.n_pulses,
            "flow_parameters": self._flow_parameters(),
            "state_generation": int(state_generation),
            "predecessor_generation": source_generation,
            "run_complete": bool(post and pulse_index == self.n_pulses - 1),
            "n_fresh_pulses_completed_total": int(pulse_index + int(post)),
            "n_hr3b_post_commits_total": int(pulse_index + int(post)),
            "n_hr4c_interpulse_evolutions_total": int(pulse_index),
            "last_transition": transition,
        }

    def _validate_authoritative_metadata(self) -> None:
        metadata = self.store.manifest.get("authoritative_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("HR-4D authoritative lifecycle metadata is missing")
        if metadata.get("schema_version") != HR4D_SCHEMA:
            raise ValueError("HR-4D lifecycle metadata schema is invalid")
        phase = metadata.get("phase")
        if phase not in (HR4D_PHASE_PRE, HR4D_PHASE_POST):
            raise ValueError("HR-4D lifecycle phase is invalid")
        pulse = metadata.get("pulse_index")
        if isinstance(pulse, bool) or not isinstance(pulse, int) or not 0 <= pulse < self.n_pulses:
            raise ValueError("HR-4D lifecycle pulse_index is invalid")
        if metadata.get("n_pulses") != self.n_pulses:
            raise ValueError("HR-4D lifecycle n_pulses is incompatible")
        if metadata.get("flow_parameters") != self._flow_parameters():
            raise ValueError("HR-4D lifecycle flow parameters are incompatible")
        if metadata.get("state_generation") != int(self.store.manifest["generation"]):
            raise ValueError("HR-4D lifecycle generation is incompatible with the authoritative state")
        predecessor = metadata.get("predecessor_generation")
        if predecessor is not None and (isinstance(predecessor, bool) or not isinstance(predecessor, int) or predecessor < 0):
            raise ValueError("HR-4D lifecycle predecessor_generation is invalid")
        if int(self.store.manifest["generation"]) == 0:
            if predecessor is not None or phase != HR4D_PHASE_PRE or pulse != 0:
                raise ValueError("HR-4D initial lifecycle invariant failed")
        elif predecessor != int(self.store.manifest["generation"]) - 1:
            raise ValueError("HR-4D lifecycle predecessor_generation invariant failed")
        post = phase == HR4D_PHASE_POST
        expected_counts = (pulse + int(post), pulse + int(post), pulse)
        actual_counts = (
            metadata.get("n_fresh_pulses_completed_total"),
            metadata.get("n_hr3b_post_commits_total"),
            metadata.get("n_hr4c_interpulse_evolutions_total"),
        )
        if actual_counts != expected_counts:
            raise ValueError("HR-4D lifecycle counter invariant failed")
        if metadata.get("run_complete") != bool(post and pulse == self.n_pulses - 1):
            raise ValueError("HR-4D lifecycle completion invariant failed")

    def begin_pulse_transition(self) -> HR4DPulseTransaction:
        if self.next_action != "pulse":
            raise ValueError("HR-4D lifecycle is not ready for a fresh pulse")
        self.store.begin_staging()
        return HR4DPulseTransaction(self)

    def abort_pulse_transition(self, *, reason: str) -> None:
        self.store.abort_staging(reason=reason)

    def _commit_post_transition(self) -> None:
        metadata = self.metadata
        if metadata["phase"] != HR4D_PHASE_PRE:
            raise ValueError("HR-4D POST commit requires a PRE authoritative state")
        source = int(self.store.manifest["generation"])
        target = self._metadata_for(
            pulse_index=int(metadata["pulse_index"]), phase=HR4D_PHASE_POST,
            source_generation=source, state_generation=source + 1, transition="pulse_post_commit",
        )
        self.store.commit_staging({
            "operation": "hr4d_pulse_post",
            "source_generation": source,
            "batch_intervals": self.batch_intervals,
            "pulse_index": int(metadata["pulse_index"]),
            "phase": HR4D_PHASE_POST,
        }, authoritative_metadata=target)

    def run_interpulse_transition(
        self, *, failure_injector: Callable[[int, int], None] | None = None,
    ) -> dict[str, object]:
        if self.next_action != "interpulse":
            raise ValueError("HR-4D lifecycle is not ready for interpulse evolution")
        metadata = self.metadata
        schedule = build_interpulse_step_schedule(f_rep=self.f_rep, dt_hydro=self.dt_hydro)
        source = int(self.store.manifest["generation"])
        target = self._metadata_for(
            pulse_index=int(metadata["pulse_index"]) + 1, phase=HR4D_PHASE_PRE,
            source_generation=source, state_generation=source + 1, transition="interpulse_pre_commit",
        )
        result = evolve_hr4_full_z(
            self.store, dt_hydro=self.dt_hydro, n_hydro_steps=schedule.full_step_count,
            step_schedule=schedule.entries, batch_intervals=self.batch_intervals,
            chi=self.chi, nu=self.nu, n0=self.n0, gravity_x=self.gravity_x,
            gravity_y=self.gravity_y, cfl_limit=self.cfl_limit, failure_injector=failure_injector,
            authoritative_metadata=target,
        )
        result.update({
            "interpulse_duration_s": schedule.duration_s,
            "full_step_count": schedule.full_step_count,
            "remainder_s": schedule.remainder_s,
        })
        return result

    def close(self) -> None:
        self.store.close()


def run_one_pulse_transition(
    controller: HR4DPulseController, source_template, pulse_runner: Callable[[Any, HR4DPulseTransaction], Any],
):
    """Use a fresh source copy, then commit POST only after the full pulse succeeds."""
    fresh_field = source_template.copy()
    transaction = controller.begin_pulse_transition()
    try:
        result = pulse_runner(fresh_field, transaction)
        transaction.finalize()
        return result
    except Exception as error:
        if not transaction.closed:
            controller.abort_pulse_transition(reason=type(error).__name__)
            transaction.closed = True
        raise


def run_hr4_pulse_train(
    controller: HR4DPulseController, source_template, pulse_runner: Callable[[Any, HR4DPulseTransaction], Any],
    *, interpulse_failure_injector: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run or resume the deterministic PRE -> POST -> PRE lifecycle."""
    pulse_calls = interpulse_calls = 0
    last_pulse_result = None
    while not controller.is_complete:
        if controller.next_action == "pulse":
            last_pulse_result = run_one_pulse_transition(controller, source_template, pulse_runner)
            pulse_calls += 1
        else:
            controller.run_interpulse_transition(failure_injector=interpulse_failure_injector)
            interpulse_calls += 1
    return {
        "pulse_calls_this_invocation": pulse_calls,
        "interpulse_calls_this_invocation": interpulse_calls,
        "final_metadata": controller.metadata,
        "last_pulse_result": last_pulse_result,
        "optical_working_field_history_stored": False,
    }


__all__ = [
    "HR4D_PHASE_POST", "HR4D_PHASE_PRE", "HR4D_SCHEMA", "HR4DPulseController",
    "HR4DPulseTransaction", "InterpulseStepSchedule", "build_interpulse_step_schedule",
    "run_hr4_pulse_train", "run_one_pulse_transition",
]
