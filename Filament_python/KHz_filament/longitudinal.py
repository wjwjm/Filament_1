"""Small, immutable metadata contracts for longitudinal propagation.

The HR-2A contract deliberately contains coordinates and descriptions only.  It
does not evaluate deposition and it does not allocate a per-interval field or
deposition array.  In particular, a future ``q[k, c](x, y)`` value is described
by metadata here, but its payload remains owned by a later HR-2 stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


DEPOSITION_CHANNELS: tuple[str, ...] = (
    "ionization",
    "inverse_bremsstrahlung",
    "rotational_raman",
)
# Short aliases keep the channel identity discoverable without introducing a
# second, potentially divergent enumeration.
DEPOSITION_CHANNEL_NAMES = DEPOSITION_CHANNELS
CHANNELS = DEPOSITION_CHANNELS


_DEFAULT_COORD_TOL = 1.0e-12
_LOOP_STOP_TOL = 1.0e-16


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _axis_size(axis: object, name: str) -> int:
    try:
        size = int(axis.size)  # NumPy/CuPy arrays and array-like axes.
    except AttributeError:
        try:
            size = len(axis)  # Lightweight test doubles / Python sequences.
        except TypeError as exc:
            raise ValueError(f"{name} must provide a one-dimensional size") from exc
    if size <= 0:
        raise ValueError(f"{name} must have positive size")
    return size


@dataclass(frozen=True)
class LongitudinalInterval:
    """One fixed optical/deposition interval, containing metadata only."""

    index: int
    z_start: float
    z_end: float
    dz: float

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("interval index must be non-negative")
        z_start = _finite_float(self.z_start, "interval z_start")
        z_end = _finite_float(self.z_end, "interval z_end")
        dz = _finite_float(self.dz, "interval dz")
        if not z_end > z_start:
            raise ValueError("interval z_end must be greater than z_start")
        if dz <= 0.0:
            raise ValueError("interval dz must be positive")
        if not math.isclose(dz, z_end - z_start, rel_tol=_DEFAULT_COORD_TOL,
                            abs_tol=_DEFAULT_COORD_TOL):
            raise ValueError("interval dz does not match its edge difference")

    @property
    def z0(self) -> float:
        return self.z_start

    @property
    def z1(self) -> float:
        return self.z_end

    @property
    def dz_m(self) -> float:
        return self.dz


@dataclass(frozen=True)
class LongitudinalSchedule:
    """Deterministic absolute longitudinal edges and interval metadata.

    ``z_edges`` and ``dz_intervals`` are tuples specifically so a schedule can
    safely be shared by every pulse without a caller mutating a NumPy array in
    place.  The schedule has no field or deposition payload.
    """

    z_edges: tuple[float, ...]
    dz_intervals: tuple[float, ...]
    intervals: tuple[LongitudinalInterval, ...]
    z_start: float
    z_end: float

    def __post_init__(self) -> None:
        edges = tuple(_finite_float(value, "z edge") for value in self.z_edges)
        dz_values = tuple(_finite_float(value, "interval dz") for value in self.dz_intervals)
        intervals = tuple(self.intervals)
        object.__setattr__(self, "z_edges", edges)
        object.__setattr__(self, "dz_intervals", dz_values)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "z_start", _finite_float(self.z_start, "z_start"))
        object.__setattr__(self, "z_end", _finite_float(self.z_end, "z_end"))
        self.validate()

    @property
    def n_intervals(self) -> int:
        return len(self.dz_intervals)

    @property
    def z_max(self) -> float:
        """Compatibility name for the requested absolute endpoint."""
        return self.z_end

    def validate(self, *, tol: float = _DEFAULT_COORD_TOL) -> bool:
        """Validate coordinate, interval-count, and floating-point invariants."""
        tol = _finite_float(tol, "tol")
        if tol < 0.0:
            raise ValueError("tol must be non-negative")
        if len(self.z_edges) != len(self.dz_intervals) + 1:
            raise ValueError("z_edges must contain one more item than dz_intervals")
        if len(self.intervals) != len(self.dz_intervals):
            raise ValueError("interval metadata count does not match dz_intervals")
        if not self.z_edges:
            raise ValueError("longitudinal schedule requires at least one edge")
        if not math.isclose(self.z_edges[0], self.z_start,
                            rel_tol=tol, abs_tol=tol):
            raise ValueError("schedule first edge does not match z_start")
        if not math.isclose(self.z_edges[-1], self.z_end,
                            rel_tol=tol, abs_tol=tol):
            raise ValueError("schedule final edge does not match z_end")
        for index, dz in enumerate(self.dz_intervals):
            edge_dz = self.z_edges[index + 1] - self.z_edges[index]
            if edge_dz <= 0.0:
                raise ValueError("longitudinal edges must be strictly increasing")
            if dz <= 0.0 or not math.isclose(dz, edge_dz,
                                             rel_tol=tol, abs_tol=tol):
                raise ValueError("dz_intervals must match edge differences")
            interval = self.intervals[index]
            if interval.index != index:
                raise ValueError("interval metadata indices must be consecutive")
            if not math.isclose(interval.z_start, self.z_edges[index],
                                rel_tol=tol, abs_tol=tol):
                raise ValueError("interval start does not match schedule edge")
            if not math.isclose(interval.z_end, self.z_edges[index + 1],
                                rel_tol=tol, abs_tol=tol):
                raise ValueError("interval end does not match schedule edge")
        return True

    def as_metadata(self) -> dict[str, object]:
        """Return only scalar/tuple metadata suitable for diagnostics."""
        return {
            "z_edges": self.z_edges,
            "dz_intervals": self.dz_intervals,
            "n_intervals": self.n_intervals,
            "z_start": self.z_start,
            "z_end": self.z_end,
        }


def build_longitudinal_schedule(
    dz: float,
    z_max: float,
    *,
    z_start: float = 0.0,
    focus_window_step: bool = False,
    focus_center_m: float | None = None,
    focus_halfwidth_m: float = 0.0,
    dz_focus: float | None = None,
) -> LongitudinalSchedule:
    """Build the current base/focus/final stepping sequence once.

    ``z_start`` and ``z_max`` are absolute coordinates.  The midpoint test is
    intentionally the existing production rule, evaluated in that same
    coordinate frame.  The final edge is produced by the same floating-point
    addition as the legacy loop; it is validated within tolerance rather than
    snapped to the requested endpoint.
    """
    dz_base = _finite_float(dz, "dz")
    z0 = _finite_float(z_start, "z_start")
    z1 = _finite_float(z_max, "z_max")
    if dz_base <= 0.0:
        raise ValueError("dz must be positive")
    if z1 < z0:
        raise ValueError("z_max must be greater than or equal to z_start")

    use_focus = bool(focus_window_step)
    z_half = _finite_float(focus_halfwidth_m, "focus_halfwidth_m")
    z_center = None if focus_center_m is None else _finite_float(
        focus_center_m, "focus_center_m")
    dz_focus_value = dz_base if dz_focus is None else _finite_float(dz_focus, "dz_focus")
    if use_focus and z_center is not None and z_half > 0.0 and dz_focus_value <= 0.0:
        raise ValueError("dz_focus must be positive when focus stepping is enabled")

    # This is deliberately a plain Python list of scalar metadata, not a
    # [K,Ny,Nx] or [K,C,Ny,Nx] allocation.
    edges: list[float] = [z0]
    dz_values: list[float] = []
    interval_values: list[LongitudinalInterval] = []
    z = z0
    while z < z1 - _LOOP_STOP_TOL:
        z_mid = z + 0.5 * dz_base
        if use_focus and z_center is not None and z_half > 0.0:
            candidate = (
                dz_focus_value
                if abs(z_mid - z_center) <= z_half
                else dz_base
            )
        else:
            candidate = dz_base
        dz_try = min(candidate, z1 - z)
        if dz_try <= 0.0:
            raise ValueError("longitudinal stepping produced a non-positive interval")
        next_z = z + dz_try
        index = len(dz_values)
        edges.append(next_z)
        dz_values.append(dz_try)
        interval_values.append(LongitudinalInterval(index, z, next_z, dz_try))
        z = next_z

    # A zero-length propagation is useful as metadata, while a positive span
    # that falls entirely below the legacy loop tolerance cannot satisfy the
    # endpoint contract and is rejected explicitly.
    if z1 > z0 and not dz_values:
        raise ValueError("z span is too small for the longitudinal endpoint tolerance")

    schedule = LongitudinalSchedule(
        z_edges=tuple(edges),
        dz_intervals=tuple(dz_values),
        intervals=tuple(interval_values),
        z_start=z0,
        z_end=z1,
    )
    schedule.validate()
    return schedule


def validate_longitudinal_schedule(
    schedule: LongitudinalSchedule, *, tol: float = _DEFAULT_COORD_TOL
) -> bool:
    """Standalone validation helper for callers and focused tests."""
    if not isinstance(schedule, LongitudinalSchedule):
        raise TypeError("schedule must be a LongitudinalSchedule")
    return schedule.validate(tol=tol)


@dataclass(frozen=True)
class GridMetadata:
    """CPU-only transverse grid geometry; no coordinate arrays are retained."""

    Nx: int
    Ny: int
    dx: float
    dy: float
    Lx: float
    Ly: float

    def __post_init__(self) -> None:
        Nx = int(self.Nx)
        Ny = int(self.Ny)
        dx = _finite_float(self.dx, "dx")
        dy = _finite_float(self.dy, "dy")
        Lx = _finite_float(self.Lx, "Lx")
        Ly = _finite_float(self.Ly, "Ly")
        if Nx <= 0 or Ny <= 0:
            raise ValueError("grid dimensions must be positive")
        if dx <= 0.0 or dy <= 0.0 or Lx <= 0.0 or Ly <= 0.0:
            raise ValueError("grid spacings and extents must be positive")
        object.__setattr__(self, "Nx", Nx)
        object.__setattr__(self, "Ny", Ny)
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)
        object.__setattr__(self, "Lx", Lx)
        object.__setattr__(self, "Ly", Ly)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.Ny, self.Nx)

    def as_metadata(self, *, prefix: str = "") -> dict[str, object]:
        return {
            f"{prefix}Nx": self.Nx,
            f"{prefix}Ny": self.Ny,
            f"{prefix}dx": self.dx,
            f"{prefix}dy": self.dy,
            f"{prefix}Lx": self.Lx,
            f"{prefix}Ly": self.Ly,
        }


@dataclass(frozen=True)
class TransverseGridMetadata:
    """Explicit optical and thermal grid descriptors.

    HR-2A sets both fields to the same descriptor.  Keeping them as separate
    fields leaves a future remapping stage free to supply another descriptor.
    """

    optical_grid: GridMetadata
    thermal_grid: GridMetadata

    @property
    def optical(self) -> GridMetadata:
        return self.optical_grid

    @property
    def thermal(self) -> GridMetadata:
        return self.thermal_grid

    @property
    def same_grid(self) -> bool:
        return self.optical_grid == self.thermal_grid

    @property
    def remapping_required(self) -> bool:
        return not self.same_grid

    def as_metadata(self) -> dict[str, object]:
        result = {}
        result.update(self.optical_grid.as_metadata(prefix="optical_grid_"))
        result.update(self.thermal_grid.as_metadata(prefix="thermal_grid_"))
        result["thermal_grid_matches_optical"] = self.same_grid
        result["transverse_remapping"] = "none" if self.same_grid else "external"
        return result


def build_transverse_grid_metadata(
    axes: object, *, thermal_grid: GridMetadata | None = None
) -> TransverseGridMetadata:
    """Derive geometry from the actual optical axes and make the HR-2A pairing."""
    optical = GridMetadata(
        Nx=_axis_size(getattr(axes, "x"), "axes.x"),
        Ny=_axis_size(getattr(axes, "y"), "axes.y"),
        dx=_finite_float(getattr(axes, "dx"), "axes.dx"),
        dy=_finite_float(getattr(axes, "dy"), "axes.dy"),
        # Derive extents from sampled axes, rather than retaining a config
        # value or assuming a repository-wide production shape.
        Lx=_axis_size(getattr(axes, "x"), "axes.x") * _finite_float(
            getattr(axes, "dx"), "axes.dx"),
        Ly=_axis_size(getattr(axes, "y"), "axes.y") * _finite_float(
            getattr(axes, "dy"), "axes.dy"),
    )
    if thermal_grid is None:
        thermal = optical
    elif isinstance(thermal_grid, GridMetadata):
        thermal = thermal_grid
    else:
        raise TypeError("thermal_grid must be GridMetadata or None")
    return TransverseGridMetadata(optical_grid=optical, thermal_grid=thermal)


@dataclass(frozen=True)
class DepositionIntervalMetadata:
    """Description of one future ``q[k,c](x,y)`` payload, with no payload."""

    index: int
    z_start: float
    z_end: float
    dz: float
    channels: tuple[str, ...] = DEPOSITION_CHANNELS
    value_name: str = "q[k,c](x,y)"
    value_unit: str = "J/m^3"
    representation: str = "interval_average_volumetric_deposition"

    @property
    def payload_shape_semantics(self) -> str:
        return "one transverse (Ny,Nx) map per interval and channel; not allocated"


@dataclass(frozen=True)
class DepositionContract:
    """Shared metadata contract for future mechanism-resolved deposition."""

    schedule: LongitudinalSchedule
    transverse_grid: TransverseGridMetadata
    channels: tuple[str, ...] = DEPOSITION_CHANNELS
    intervals: tuple[DepositionIntervalMetadata, ...] = ()
    value_name: str = "q[k,c](x,y)"
    value_unit: str = "J/m^3"
    representation: str = "interval_average_volumetric_deposition"
    payload_allocated: bool = False

    def __post_init__(self) -> None:
        channels = tuple(str(channel) for channel in self.channels)
        if channels != DEPOSITION_CHANNELS:
            raise ValueError(
                "deposition channels must be exactly "
                f"{DEPOSITION_CHANNELS!r}"
            )
        if not isinstance(self.schedule, LongitudinalSchedule):
            raise TypeError("deposition contract requires a LongitudinalSchedule")
        if not isinstance(self.transverse_grid, TransverseGridMetadata):
            raise TypeError("deposition contract requires transverse grid metadata")
        intervals = tuple(self.intervals)
        if intervals and len(intervals) != self.schedule.n_intervals:
            raise ValueError("deposition interval metadata count does not match schedule")
        if any(not isinstance(interval, DepositionIntervalMetadata)
               for interval in intervals):
            raise TypeError("deposition intervals must be metadata records")
        if self.payload_allocated:
            raise ValueError("HR-2A deposition contract cannot own payload arrays")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "intervals", intervals)

    @property
    def n_intervals(self) -> int:
        return self.schedule.n_intervals

    @property
    def q_shape(self) -> tuple[int, int]:
        """Per-interval map shape metadata; no array is allocated."""
        return self.transverse_grid.thermal_grid.shape

    def as_metadata(self) -> dict[str, object]:
        result = {
            "deposition_channels": self.channels,
            "deposition_channel_count": len(self.channels),
            "deposition_value_name": self.value_name,
            "deposition_value_unit": self.value_unit,
            "deposition_representation": self.representation,
            "deposition_payload_allocated": self.payload_allocated,
            "deposition_q_shape": self.q_shape,
            "deposition_n_intervals": self.n_intervals,
            "deposition_contract_schema": "khz_filament.deposition_contract.v1",
        }
        result.update(self.transverse_grid.as_metadata())
        return result


def build_deposition_contract(
    schedule: LongitudinalSchedule,
    axes: object | GridMetadata | TransverseGridMetadata | None = None,
    *,
    transverse_grid: TransverseGridMetadata | None = None,
) -> DepositionContract:
    """Build a no-payload deposition contract for a shared schedule."""
    if not isinstance(schedule, LongitudinalSchedule):
        raise TypeError("schedule must be a LongitudinalSchedule")
    grid = transverse_grid
    if grid is None and isinstance(axes, TransverseGridMetadata):
        grid = axes
    elif grid is None and isinstance(axes, GridMetadata):
        grid = TransverseGridMetadata(optical_grid=axes, thermal_grid=axes)
    elif grid is None and axes is not None:
        grid = build_transverse_grid_metadata(axes)
    if grid is None:
        raise ValueError("axes or transverse_grid metadata is required")

    intervals = tuple(
        DepositionIntervalMetadata(
            index=interval.index,
            z_start=interval.z_start,
            z_end=interval.z_end,
            dz=interval.dz,
        )
        for interval in schedule.intervals
    )
    return DepositionContract(
        schedule=schedule,
        transverse_grid=grid,
        channels=DEPOSITION_CHANNELS,
        intervals=intervals,
    )


# A descriptive alias for callers that prefer the interface terminology.
build_deposition_metadata = build_deposition_contract


__all__ = [
    "CHANNELS",
    "DEPOSITION_CHANNELS",
    "DEPOSITION_CHANNEL_NAMES",
    "DepositionContract",
    "DepositionIntervalMetadata",
    "GridMetadata",
    "LongitudinalInterval",
    "LongitudinalSchedule",
    "TransverseGridMetadata",
    "build_deposition_contract",
    "build_deposition_metadata",
    "build_longitudinal_schedule",
    "build_transverse_grid_metadata",
    "validate_longitudinal_schedule",
]
