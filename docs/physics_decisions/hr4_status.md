# HR-4 status: isobaric transverse slow flow

**Program status:** HR-4 branch active; HR-4A **IN PROGRESS** (2026-08-31).

**Branch provenance:** HR-4A started from local `main`
`543499daf6058e1782b125bb3a16c84155a11e05`; its verified upstream
`origin/main` ancestor is `8b7b975580f0b7665a48758853fa2ae9ad3db7bc`.
The one local-parent commit preserves only user-authorized workspace artifacts;
it changes no production physics, configuration, or baseline result. This
document is written before any HR-4A code change.

## Scope and model level

**FROZEN — `REFERENCE_EXPLICIT` / `REFERENCE_DERIVED`.** HR-4 reproduces the
post-acoustic, isobaric, 2-D transverse slow-flow level used for pulse trains
by Isaacs *et al.* (2022); it is not a general CFD project. Each longitudinal
interval (screen) owns an independent transverse slow state, with no
longitudinal fluid coupling.

HR-4A is limited to data/parameter contracts, state interfaces, boundary
policy, and validation scaffolding. It must not implement the Eq. (32)–(33)
advance, z-batched evolution, pulse-train runner integration, production-scale
allocation, or an HPC/Slurm case. Those belong to HR-4B or later.

## Reference provenance

| Source | Label | HR-4 use |
| --- | --- | --- |
| Isaacs *et al.* (2022), Secs. 3–5, Eqs. (20), (23), (31)–(33) | `REFERENCE_EXPLICIT` | isobaric relation, transverse transport model, buoyancy term, STP thermal diffusivity |
| Algebra from the isobaric index–density relation | `REFERENCE_DERIVED` | derived density and temperature diagnostics |
| Existing HR-3B/HR-3C contracts | `IMPLEMENTATION_CHOICE` | interval-centred compatibility and the unique `D_th` authority |
| Upwind/collocated/open-FD realization | `IMPLEMENTATION_CHOICE` | paper does not specify an interpulse flow algorithm |
| `dx`, `dy`, `dt_hydro`, production CFL threshold | `PROVISIONAL` | development values only |

The paper specifies the model-level equations, not an exact interpulse CFD
scheme. Local discretization choices must never be reported as paper-explicit.

## Frozen physical contract

### HR4-D1 — Authoritative persistent slow state

**FROZEN — `REFERENCE_DERIVED` / `IMPLEMENTATION_CHOICE`.** The authoritative
persistent state for every screen `k` is exactly:

```text
delta_n[k, y, x]
vx[k, y, x]
vy[k, y, x]
```

`delta_T`, `delta_rho`, and pressure are not authoritative persistent fields.
They are derived diagnostics only:

```text
delta_rho / rho0 = delta_n / (n0 - 1)
delta_T / T0     = -delta_n / (n0 - 1)
```

Pressure is not an independent slow variable in the post-acoustic isobaric
approximation.

### HR4-D2 — Pulse PRE/POST semantics

**FROZEN — `REFERENCE_DERIVED` / `IMPLEMENTATION_CHOICE`.** Each pulse maps
one screen by:

```text
delta_n_post = delta_n_pre + delta_n_HR3B
vx_post      = vx_pre
vy_post      = vy_pre
```

The ambient initial condition has `vx = vy = 0`. A phenomenological
pulse-induced velocity kick is prohibited.

### HR4-D3 — Model equations / model level

**FROZEN — `REFERENCE_EXPLICIT`.** Every screen independently uses:

```text
d(delta_n)/dt + v · grad_perp(delta_n) = chi · laplacian_perp(delta_n)
d(v)/dt + v · grad_perp(v) = nu · laplacian_perp(v)
                             + delta_n/(n0 - 1) · g
```

The baseline excludes explicit acoustic waves, independent pressure, full
compressibility, shocks, 3-D Navier–Stokes, longitudinal fluid coupling,
turbulence, and extra persistent plasma chemistry.

## Frozen persistent-state contract

**FROZEN — `IMPLEMENTATION_CHOICE`.** `delta_n`, `vx`, and `vy` are
collocated real floating arrays with the same `[K, Ny, Nx]` shape and a dtype
compatible with existing slow-state storage. Creation initializes `vx/vy`
exactly to zero; non-floating, non-finite, or shape-inconsistent data fail
closed. Metadata records geometry, SI units, authority, PRE/POST stage, and
schema version. No persistent `delta_T`, `delta_rho`, or pressure storage is
allowed. HR-4A tests must use `K <= 4` and `Ny, Nx <= 32`.

## Pulse PRE/POST semantics

**FROZEN — `IMPLEMENTATION_CHOICE`.** The optical pulse reads PRE state, then
adds the HR-3B increment only to `delta_n` to form POST state; velocity is
exactly unchanged. Only the later HR-4B interpulse advance may evolve all three
fields. HR-4A's pure helper must not recompute heat/density/temperature or call
a future flow solver.

## Geometry and coordinates

### HR4-D4 — Domain and gravity

**FROZEN except spacing — `REFERENCE_DERIVED` / `IMPLEMENTATION_CHOICE`.**

```text
x in [-1.5, +1.5] mm
y in [-1.0, +2.5] mm
+y is upward
g = (0, -9.81) m/s^2
```

If the window derives from an Isaacs figure, it is a figure-derived modelling
domain, not a paper-explicit solver-box dimension. `dx = dy approximately
10 um` is `PROVISIONAL`, never a production-resolution freeze.

## Transport parameters

### HR4-D5 — Diffusivity and viscosity

**FROZEN — `REFERENCE_EXPLICIT` / `IMPLEMENTATION_CHOICE`.**

```text
chi = 21.7e-6 m^2/s
nu  = 1.5e-5 m^2/s
```

`chi` must equal the existing authoritative HR-3C `HeatConfig.D_th`; HR-4
must not create a second thermal-diffusivity authority. `D_gas` is legacy-only
and not an HR-4 fallback. Baseline excludes `chi(T)` and `nu(T)`.

## Boundary conditions

### HR4-D8 — Open/free-space boundary

**FROZEN — `IMPLEMENTATION_CHOICE`.** The intended physical boundary is open
free space:

- `delta_n` outer boundary is ambient Dirichlet `0`, representing far-field
  ambient recovery rather than a solid cold wall.
- Velocity has neither solid wall nor no-slip: outflow has zero normal
  gradient, inflow has ambient velocity `0`.
- Corner rule: if either incident face is inflow, set both velocity components
  to `0`; otherwise copy the diagonally adjacent interior velocity. Tests must
  make this deterministic rule explicit.
- Periodic wrap-around is forbidden. Later work must perform domain-size and
  edge-contamination testing.

## Spatial discretization

### HR4-D6 — Advection

**FROZEN — `IMPLEMENTATION_CHOICE`.** Use first-order local upwind derivatives
for `v dot grad(delta_n)`, `v dot grad(vx)`, and `v dot grad(vy)`, selecting
the upstream derivative from local velocity sign. Do not silently replace the
material-advection PDE with a physically different conservative PDE. MUSCL/TVD,
WENO, higher-order finite volume, and semi-Lagrangian schemes are `DEFERRED`.

### HR4-D7 — Diffusion and viscosity

**FROZEN — `IMPLEMENTATION_CHOICE`.** Use explicit second-order central finite
differences for `chi laplacian(delta_n)` and `nu laplacian(v)`. HR-4 must not
reuse HR-3C periodic spectral diffusion: buoyancy transports toward `+y`,
periodic wrap-around is not free-space flow, enlarging a periodic domain only
delays contamination, all three fields can wrap, and over-expansion raises
pulse-train cost. HR-3C spectral diffusion remains a reduced-limit comparison
only, not the HR-4 baseline.

### HR4-D9 — Grid arrangement

**FROZEN — `IMPLEMENTATION_CHOICE`.** All three fields use a collocated
`[K, Ny, Nx]` grid. There is no staggered pressure–velocity architecture.

## Slow-time integration

### HR4-D10 — Architecture

**FROZEN — `IMPLEMENTATION_CHOICE`.** The future HR-4B baseline is explicit,
fixed-step, unsplit Forward Euler. Every RHS is computed from the same old
tuple `(delta_n^m, vx^m, vy^m)`, then all fields commit simultaneously. No
adaptive timestep, implicit solve, or sequential operator ordering is allowed.

## Stability requirements

### HR4-D11 — `dt_hydro`

**PROVISIONAL / NOT NUMERICALLY FROZEN — `REFERENCE_DERIVED`.** The fixed-step
architecture is frozen, not the number `dt_hydro`. A diagnostic audit must
independently check both diffusivities:

```text
D * dt * (1/dx^2 + 1/dy^2) <= 1/2, for D in {chi, nu}
```

and the upwind CFL condition:

```text
max_abs_vx * dt/dx + max_abs_vy * dt/dy <= C_CFL
```

The HR-4A audit returns both diffusion numbers, advection CFL, individual pass
flags, and `overall_pass`. Invalid/non-finite inputs fail closed. It never
changes timestep or provides an adaptive fallback. The production `C_CFL` is
`PROVISIONAL`; HR-4E must compare `dt`, `dt/2`, and `dt/4` before a
production `dt_hydro` freeze.

## Validation observables and reduced limits

**FROZEN — `REFERENCE_DERIVED` / `IMPLEMENTATION_CHOICE`.** HR-4E must record
`min/max delta_n`, thermal-channel centroid `y_c`, thermal-channel width,
`max |vx|`, `max |vy|`, `max |v|`, and boundary/edge-contamination metrics.
Expected future trends are thermal broadening, buoyant upward (`+y`) motion of
a heated low-index channel, and optical deflection toward `-y`; HR-4A makes no
claim they have been reproduced.

Required reduced limits: a zero state remains zero; `g=0` gives no systematic
buoyant rise; and disabled velocity evolution reduces Eq. (32) to pure thermal
diffusion. HR-4 finite-difference diffusion and HR-3C spectral diffusion need
only agree physically under convergence, never bitwise.

## Provisional / deferred items

**PROVISIONAL:** `dx`, `dy`, `dt_hydro`, and production `C_CFL`; validation
may report failure but must not alter them automatically.

**DEFERRED:** MUSCL/TVD, WENO, semi-Lagrangian and higher-order schemes,
implicit/adaptive integration, acoustic/pressure/compressible/longitudinal
flow, turbulence, full HR-4B single-screen PDE evolution, z batching, runner
integration, production allocation/benchmarks, beam-deflection benchmark, and
all HPC/Slurm work.

## Upstream preserved status

HR-3 is **CLOSED / MERGED TO MAIN**. HR-3B's post-acoustic `delta_n_th`
mapping and HR-3C's transactional periodic-spectral lifecycle remain preserved.
HR-4A changes neither their physics nor production configurations/frozen
results. HR-2E remains **DEFERRED** and the longitudinal production schedule
remains **NOT FROZEN**.

## HR-4A implementation status

**IN PROGRESS.** This document and the status update are the required first
artifact. After their self-check and documentation commit, permitted code is
only config validation, state contract, derived diagnostics, PRE/POST helper,
boundary helpers, stability audit, and tiny targeted tests. Complete PDE
advance remains outside HR-4A.

## Change log

| Date | Stage | Change |
| --- | --- | --- |
| 2026-08-31 | HR-4A | Wrote this authority document before code; froze D1–D10, recorded D11 provisional, and separated HR-4B+ work. |
