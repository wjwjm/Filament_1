# HR-4 status: isobaric transverse slow flow

**Program status:** HR-4 branch active; HR-4A, HR-4B, HR-4C, and HR-4D are
**CLOSED** (2026-08-31).

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

HR-4A limited itself to contracts and validation scaffolding. HR-4B closes the
single-screen Eq. (32)–(33) operator only: it has no z-batched evolution,
pulse-train runner integration, persistent storage lifecycle, production-scale
allocation, or an HPC/Slurm case. Those remain HR-4C or later.

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
flow, turbulence, production allocation/benchmarks, HR-4E convergence and
domain-size studies, HR-4F beam-deflection benchmark, and all HPC/Slurm work.

## Upstream preserved status

HR-3 is **CLOSED / MERGED TO MAIN**. HR-3B's post-acoustic `delta_n_th`
mapping and HR-3C's transactional periodic-spectral lifecycle remain preserved.
HR-4A changes neither their physics nor production configurations/frozen
results. HR-2E remains **DEFERRED** and the longitudinal production schedule
remains **NOT FROZEN**.

## HR-4A implementation status

**CLOSED.** HR-4A added only config validation, the three-field state contract,
derived diagnostics, PRE/POST helper, boundary helper, stability audit, and
tiny-array targeted tests. The documented D1–D11 contract is preserved; no
PDE advance, runner integration, production allocation, production physics
configuration change, or HPC/Slurm work occurred.

Local validation used the required explicit Windows test environment:
compileall and the backend probe passed. The controlled targeted run reported
150 passed and 3 skipped. Its only failure was the pre-existing HR-2E
strict-float baseline (3.0000000000000004 != 3.0), explicitly outside HR-4A;
there were zero new HR-4A failures. Full pytest was not run because the
repository-approved local entrypoint exposes only the bounded targeted set.

## HR-4B implementation status

**CLOSED.** HR-4B implements a direct, single-screen [Ny, Nx] operator in the
HR-4 module; it neither accepts nor constructs a full-z persistent state. One
or more fixed slow-time steps use exactly:

BC(old) -> all RHS(old) -> synchronous Forward Euler -> BC(new).

The advection term is local first-order upwind in the original material form.
The scalar and both velocity components use second-order central
finite-difference Laplacians; no periodic FFT, roll, artificial viscosity, or
conservative-flux rewrite is used. Buoyancy is explicitly restricted to the
vy RHS; nonzero gravity_x fails closed.

The velocity boundary now uses the nearest interior cell to evaluate each face
normal velocity. u_n < 0 sets both velocity components to ambient zero;
u_n >= 0 copies the nearest interior pair. A corner checks its diagonal
nearest-interior velocity pair for the two incident-face tests; if either is
inflow it is zero, otherwise it copies that diagonal pair. This is a
deterministic HR-4B implementation choice.

dt_hydro = 1.0 us is the development default, not a production value. The
stability audit retains independent chi/nu diffusion and advection-CFL checks,
and adds the conservative unsplit combined check for both chi and nu. Reduced
operator tests may set chi, nu, or gravity_y to zero; this does not relax the
frozen production-config authority that chi equals HR-3C D_th and nu is the
STP baseline value.

Thermal-channel diagnostics use max(-delta_n, 0) weighting with
y_j = y_min + j*dy. The reported width is radial RMS width
sqrt(<(x-xc)^2 + (y-yc)^2>/2). A screen with no negative-index channel has
undefined centroid/width represented by NaN and
thermal_channel_defined=false.

Validation covers zero invariance, buoyancy sign, all four upwind directions,
central Laplacians, Gaussian thermal diffusion, constant-velocity advection,
viscous velocity diffusion, unsplit-Euler ordering, all faces/corners,
no-wrap topward transport, coupled rise plus broadening, and the 1.0 us versus
0.5 us short comparison. The bounded local gate reports 180 passed, 3 skipped,
and one pre-existing HR-2E strict-float failure
(3.0000000000000004 != 3.0); there are zero new HR-4B failures.

The operator returns per-call shape, dtype, backend, total/per-step wall-time,
and a conservative temporary working-set estimate of
12 * Ny * Nx * dtype.itemsize; it stores no slow-time history. This estimate
is development-only, not a measured peak-memory or production-performance
claim. CPU test results do not establish CuPy/GPU numerical equivalence.

One non-test development benchmark used a centered negative Gaussian on an
81 by 81 float64 NumPy screen for 400 steps at 1.0 us. It measured
0.37595 s total and 0.000940 s per step; the temporary estimate was 629856
bytes. The channel centroid rose from -5.9994e-5 m after the first step to
-5.1450e-5 m, radial RMS width broadened from 8.0264e-5 m to 1.4392e-4 m,
and max absolute vy was 4.4411e-3 m/s. The same 400 us case at 0.5 us gave
centroid -5.1449e-5 m, width 1.4390e-4 m, and max absolute vy
4.4399e-3 m/s. This is a short development comparison, not an HR-4E
convergence or production-performance result.

## HR-4C implementation status

**CLOSED.** HR-4C extends the established HR-3C disk-backed lifecycle without
changing its `.npy` memmap format, flush/fsync discipline, geometry
fingerprint, or atomic JSON-manifest authority selection. It creates six
field-specific slots: `delta_n`, `vx`, and `vy` each have one authoritative
and one scratch slot. A committed state is exactly the manifest-selected
three-field tuple, with every field shape `[K, Ny, Nx]`; there is no persistent
temperature, density, or pressure field.

Each full-z interpulse request begins a staging generation. The implementation
reads one `[B, Ny, Nx]` batch for each of the three fields, advances each
2-D screen by direct invocation of the HR-4B single-screen operator for all
requested fixed hydro steps, and writes its three output screens together to
staging. The fixed order is:

```text
z batch -> screen -> all HR-4B hydro steps -> three-field staging write
```

Only after every field and every screen is present, has the declared layout,
and is finite are all scratch memmaps flushed/fsynced and the manifest
atomically switched to the new generation. A failed operator call, incomplete
field write, invalid value, or failed staging validation leaves the former
committed generation authoritative and records an abort reason. Reopen
discards a manifest-marked incomplete staging generation; HR-4C intentionally
has no per-batch resume, partial promotion, repetition-rate logic, pulse
PRE/POST orchestration, or runner wiring.

The explicit legacy path reads an existing HR-3C-like `delta_n` memmap in
z batches, creates a new HR-4C generation with numerically preserved
`delta_n`, and initializes `vx = vy = 0`; it never modifies the source file.
`batch_intervals` reuses the existing HR-3C batch authority. Its development
value in HR-4C tests was `B = 1`, `2`, or `4`; any production batch size,
production `dx/dy`, and production `dt_hydro` remain **PROVISIONAL**.

Local HR-4C validation added nine lifecycle tests and passed all nine. They
cover three-field create/reopen, legacy migration, batch-size equivalence,
full-memory screen-by-screen reference equivalence, `4 + 4 + 2` partial-batch
handling, z-screen independence, mid-transaction failure and restart-safe
reopen, incomplete/non-finite staging rejection, manifest/grid/z-ordering
mismatch rejection, and an instrumented `[B, Ny, Nx]` read/write trace. With
fixed `B=2`, the working-set estimate was identical for `K=3` and `K=17`, and
it is independent of hydro-step count because no hydro history is retained.
The reported I/O accounting is three full state reads plus three full state
writes per complete evolution; it is storage volume, not a claim of
production I/O performance.

Required local gates used the explicit Windows test environment: `compileall`,
backend, and `sanity` all passed; HR-4C-specific tests were `9 passed`.
The full repository bounded targeted gate reported `189 passed, 3 skipped,
1 failed`. The sole failure is the pre-existing HR-2E strict-float assertion
in `test_hr2e_error_localization.py`
(`3.0000000000000004 != 3.0`), outside HR-4C. New HR-4C failures are zero.
These Windows CPU checks do not establish GPU/CuPy equivalence or production
performance. No HPC/Slurm work, push, merge to `main`, HR-4D runner work,
HR-4E convergence study, or HR-4F benchmark was performed.

## HR-4D implementation status

**CLOSED.** HR-4D adds a thin, restart-safe orchestration layer over the
existing authorities. It uses `HR4CThreeFieldStore` as the only persistent
state authority and places the lifecycle metadata in the same atomically
replaced manifest as the field-slot promotion. The metadata records
`pulse_index`, `phase` (`PRE` or `POST`), `n_pulses`, authoritative and
predecessor generations, completion state, and completed-pulse, POST-commit,
and interpulse counters. Initial lifecycle metadata is written in the same
initial HR-4C manifest creation; resume additionally binds repetition rate,
hydro step, batch count, transport/gravity parameters, `n0`, and CFL limit.
Invalid phase, index, generation, predecessor, parameter, or counter
combinations fail closed on reopen.

The actual lifecycle is:

```text
PRE_p -> fresh source copy -> propagate_one_pulse / HR-2 / HR-3A / HR-3B
      -> atomic POST_p(delta_n + increment, vx unchanged, vy unchanged)
      -> HR-4C full-z flow -> PRE_(p+1)
```

The pulse transaction implements the existing `read_interval` / `update_interval`
contract used by `propagate_one_pulse`; therefore HR-3B continues to call its
authoritative mapping from HR-3A `q_thermal` rather than HR-4D duplicating a
deposition or density conversion. Each pulse receives `source_template.copy()`;
the working field is pulse-local and no optical-field history is retained.

HR-4D computes the interpulse duration exactly as `1 / f_rep`, then sends one
transactional HR-4C request containing `N_full = floor(duration / dt_hydro)`
fixed steps and, when needed, one final remainder step. Both types call the
unchanged HR-4B stability audit. This avoids `round()` spacing drift while
keeping all full and remainder steps in one HR-4C staging/commit transaction.
Consequently, an evolution failure retains `POST_p`, rather than committing a
partial PRE state. The final pulse commits `POST_final` and stops; it never
runs an extra interpulse advance.

Local lifecycle tests cover fresh source content, POST increment and velocity
continuity, one/two/five-pulse counts, exact and remainder schedules,
remainder stability rejection, restart from PRE, restart from intermediate
POST, restart from final POST, optical/conversion/commit/interpulse failures,
metadata tamper rejection, and bounded optical working-state behavior. A
small repository-level connection test also exercised the real
`propagate_one_pulse -> HR-2/HR-3A/HR-3B -> HR-4C` chain for two pulses and one
interpulse evolution. The HR-4D test file reported `12 passed`.

Required local gates used the explicit Windows test environment: `compileall`,
backend, and `sanity` passed. The full bounded targeted gate reported
`201 passed, 3 skipped, 1 failed`; the sole failure remains the pre-existing
HR-2E strict-float assertion in `test_hr2e_error_localization.py`
(`3.0000000000000004 != 3.0`). New HR-4D failures are zero. These are CPU
software/configuration checks only; production `dt_hydro`, `dx/dy`, domain,
and z batch size remain **PROVISIONAL**. HR-4E convergence, HR-4F benchmark,
HPC/Slurm work, push, and merge to `main` remain out of scope.

## Change log

| Date | Stage | Change |
| --- | --- | --- |
| 2026-08-31 | HR-4A | Wrote this authority document before code; froze D1–D10, recorded D11 provisional, and separated HR-4B+ work. |
| 2026-08-31 | HR-4A | Added and validated contract-only scaffolding; HR-4A closed with no solver, runner, or HPC action. |
| 2026-08-31 | HR-4B | Closed the bounded single-screen operator and its local validation; HR-4C/4D/4E/4F remain deferred. |
| 2026-08-31 | HR-4C | Closed three-field transactional disk-backed full-z evolution and local storage-lifecycle validation; HR-4D/4E/4F remain deferred. |
| 2026-08-31 | HR-4D | Closed PRE/POST fresh-pulse orchestration, exact interpulse scheduling, and restart-safe lifecycle validation; HR-4E/4F remain deferred. |
