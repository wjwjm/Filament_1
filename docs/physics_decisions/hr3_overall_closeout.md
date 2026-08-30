# HR-3 overall closeout and merge-readiness ledger

**Date:** 2026-08-30
**Branch:** `HR-3`
**Classification:** **HR-3 BLOCKED**. This is not a merge to `main`, a
production-performance certification, or an HR-4 authorization.

## Scope and frozen model boundary

HR-3 closes the reduced high-repetition-rate chain from one pulse's deposited
energy to the next pulse's persistent transverse thermal-index state:

```text
fresh optical pulse
  -> q_dep -> q_thermal -> Delta delta_n_th^(p,+)
  -> delta_n_th^(p,+) -> delta_n_th^(p+1,-)
```

It is not a full hydrodynamic model. Longitudinal thermal diffusion/z mixing,
acoustic transients, pressure, velocity, advection, buoyancy, viscosity,
gravity, and production performance tuning are outside HR-3.

## Frozen authoritative chain

| Stage | Authoritative quantity | Units / centering | Producer -> consumer | Persistence |
| --- | --- | --- | --- | --- |
| HR-2 | `q_ion`, `q_IB`, `q_Raman`; `q_dep=q_ion+q_IB+q_Raman` | J m^-3; current longitudinal interval | deposition contract -> HR-3A | transient per interval |
| HR-3A | `q_thermal=q_dep` | J m^-3; current longitudinal interval | `thermalize_interval` -> HR-3B | transient per interval; scalar/sparse diagnostics are non-authoritative |
| HR-3B | `Delta delta_n_th=-beta_th*q_thermal` | dimensionless; current interval | HR-3B mapping -> transactional post slot | transient increment |
| HR-3B/3C | `delta_n_th^(p,+)` | dimensionless, interval-centered `[K,Ny,Nx]` | post commit -> HR-3C | authoritative post state |
| HR-3C | `delta_n_th^(p+1,-)` | dimensionless, interval-centered `[K,Ny,Nx]` | transverse diffusion -> next pulse | authoritative pre state |

`beta_th=(n0-1)/(rho0*C_V*T0)`. Positive thermal deposition therefore creates
a non-positive refractive-index increment. The HR-3C operator is exactly one
transverse spectral step per interval, `exp(-D_th*kperp2/f_rep)`, with
`D_th=21.7e-6 m^2/s` and `dt_interpulse=1/f_rep`. It has no `D_gas` fallback.
The periodic FFT boundary is guarded by the existing edge fail-closed check.

Legacy `Qacc`, `Q2D`, `gamma_heat`, `dn_gas`, optical field loss, and Raman
field diagnostics are not an authoritative HR-3 source. The legacy
`Q2D -> gamma_heat -> diffuse_dn_gas -> dn_gas` path remains isolated from the
HR-3C runner path; standalone legacy and standalone HR-3B modes remain
compatibility modes only.

## Pulse timeline and exact count contract

```text
PRE(p): authoritative delta_n_th^(p,-)
  -> fresh copy of the optical source for pulse p
  -> HR-3A q_thermal^p
  -> HR-3B POST(p): delta_n_th^(p,+)
  -> if p is not final: HR-3C transverse diffusion -> PRE(p+1)
```

Pulse `p` reads only its pre-state and cannot see heat it creates itself. The
optical field is never persistent: every propagation receives `E_source.copy()`.
For `N` pulses, frozen counts are exactly `fresh=N`, `post=N`, and
`diffusion=N-1`. The final state is `delta_n_th^(N,+)`; no implicit final
diffusion or cooling step is allowed.

## Persistence and restart contract

HR-3C owns exactly two full-volume disk-backed slots,
`<run>.hr3c_delta_n_th_current.npy` and
`<run>.hr3c_delta_n_th_next.npy`. The manifest is the authority selector:
the pre-state is read-only during a pulse, and the scratch post slot is not
promoted until its complete flush/fsync and atomic manifest replacement succeed.

The manifest binds state shape/dtype, interval convention, `D_th`, `f_rep`,
edge threshold, batch size, schedule/transverse-grid fingerprint, stage/index,
slot identities, completion flag, and persistent counters. Its exact counter
invariants are:

| State | `F` fresh | `B` post | `C` diffusion |
| --- | ---: | ---: | ---: |
| `pre_pulse`, `next_pulse_index=q` | `q` | `q` | `q` |
| `post_pulse`, `pulse_index=p` | `p+1` | `p+1` | `p` |
| final completed `post_pulse`, `p=N-1` | `N` | `N` | `N-1` |

Invalid/tampered manifests fail closed. A pulse interruption retains the
authoritative pre-state; an interrupted diffusion retains the authoritative
post-state and recomputes scratch on resume. Final post and `run_complete=true`
are committed atomically, and resuming a completed run executes neither a pulse
nor a diffusion pass. However, the runner currently still writes a new primary
NPZ after that no-op resume; it does not preserve or reload the completed run's
original final optical diagnostics. This is the overall-closeout blocker below.

## Audit gates

| Gate | Result | Evidence |
| --- | --- | --- |
| H3-1 branch integrity | PASS | `HR-3` is synchronized with its remote; `main` remains unchanged. |
| H3-2 manifest integrity | PASS | exact counter/stage/index validator plus tampered-manifest tests. |
| H3-3 authoritative chain | PASS | only `q_ion/q_IB/q_Raman -> q_thermal -> delta_n_th` reaches the HR-3 slow state. |
| H3-4 temporal ordering | PASS | old pre-state read occurs before interval thermalization/update; final post does not diffuse. |
| H3-5 persistence/restart | BLOCKED | State restart is correct, but completed-run resume rewrites the primary NPZ from a fresh source rather than preserving/loading completed diagnostics. |
| H3-6 exact counts | PASS | runner N=1/2/3 confirms `N/N/(N-1)` and manifest totals. |
| H3-7 memory/storage architecture | PASS | disk-backed ping-pong, bounded z batches, one kernel per volume pass; no full-volume device materialization. |
| H3-8 regression | PASS | all HR-3-relevant tests pass; only the known HR-2E strict-float baseline fails. |
| H3-9 deferred boundary | PASS | HR-2E deferred, production schedule not frozen, HR-4 not started. |
| H3-10 merge-readiness report | BLOCKED | This ledger records the blocker; merge readiness cannot be granted until H3-5 is repaired and revalidated. |

## Deferred and future work

- **HR-2E:** longitudinal convergence is **DEFERRED**. The strict-float test
  baseline `3.0000000000000004 != 3.0` remains recorded and was not changed.
- **Production longitudinal schedule:** **NOT FROZEN**.
- **HR-4:** **NOT STARTED**; no acoustic/hydrodynamic physics is implied here.
- **Performance engineering:** production disk footprint, large-state I/O,
  host/device transfer cost, production batch size, GPU throughput, and long-time
  periodic-boundary/domain validity remain future engineering/numerical work.

No HPC or Slurm job was submitted for this closeout.

## Required follow-up before merge readiness

Perform one bounded runner-level repair: a completed `resume_hr3c=true` must
either preserve the existing final NPZ without rewriting it, or load a validated
completed diagnostic artifact before returning. It must then be covered by an
idempotence regression that compares the NPZ byte/schema-relevant content before
and after completed resume. This is software/provenance repair only; it must not
change HR-3A/B/C physics, configurations, or the frozen `D_th` contract.
