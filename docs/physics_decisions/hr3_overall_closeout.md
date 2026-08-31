# HR-3 overall closeout and merged-status ledger

**Date:** 2026-08-30
**Source branch:** `HR-3` at `7d74370ab56049591b69f01ceede3e14a0e0ecec`
**Merged main:** `654fb0236b9c119ab7d89524c08cf0b84fe9181e`
**Classification:** **HR-3 CLOSED / MERGED TO MAIN**. This is not a
production-performance certification or an HR-4 authorization.

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

## HR-3 reference-based approximations and reduced-model boundary

HR-3 is a reference-compatible reduced baseline, not a complete fluid-
hydrodynamics model. In the modeling boundary supported by Isaacs *et al.*
(2022) and Zeng Qingwei (2022), it makes the following explicit
approximations:

- **HR-3A:** deposited energy is assumed eventually to thermalize completely,
  so `q_thermal=q_dep`. No empirical `eta_ion`, `eta_Raman`, or `eta_IB`
  thermalization efficiency is introduced.
- **HR-3B:** the ps--us acoustic transient is not explicitly solved. The
  Isaacs-style post-acoustic/isobaric reduced mapping is
  `Delta delta_n_th=-beta_th*q_thermal`; only `delta_n_th` is persistent, not
  full `Delta T`, `delta rho`, pressure, or other thermodynamic fields.
- **HR-3C:** interpulse transport is purely transverse,
  `partial_t delta_n_th=D_th nabla_perp^2 delta_n_th`. Every longitudinal
  interval evolves independently: there is no longitudinal diffusion or
  z mixing.

Velocity, advection, buoyancy, viscosity, gravity, and complete hydrodynamics
are deliberately omitted. Those flow effects are future HR-4 scope; this
ledger does not freeze any HR-4 equation or implementation choice.

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
nor a diffusion pass. A completed resume validates and loads the existing primary
NPZ; it does not recreate final diagnostics from a fresh source or rewrite the
NPZ, diagnostic report, manifest, or state slots.

## Audit gates

| Gate | Result | Evidence |
| --- | --- | --- |
| H3-1 branch and merge integrity | PASS | frozen `HR-3` source `7d74370ab56049591b69f01ceede3e14a0e0ecec` was merged to `main` by `654fb0236b9c119ab7d89524c08cf0b84fe9181e`. |
| H3-2 manifest integrity | PASS | exact counter/stage/index validator plus tampered-manifest tests. |
| H3-3 authoritative chain | PASS | only `q_ion/q_IB/q_Raman -> q_thermal -> delta_n_th` reaches the HR-3 slow state. |
| H3-4 temporal ordering | PASS | old pre-state read occurs before interval thermalization/update; final post does not diffuse. |
| H3-5 persistence/restart | PASS | Completed artifact validation/load is byte-idempotent for NPZ/report/manifest/state, while incomplete pulse and diffusion resumes retain their transactional semantics. |
| H3-6 exact counts | PASS | runner N=1/2/3 confirms `N/N/(N-1)` and manifest totals. |
| H3-7 memory/storage architecture | PASS | disk-backed ping-pong, bounded z batches, one kernel per volume pass; no full-volume device materialization. |
| H3-8 regression | PASS | all HR-3-relevant tests pass; only the known HR-2E strict-float baseline fails. |
| H3-9 deferred boundary | PASS | HR-2E deferred, production schedule not frozen, HR-4 not started. |
| H3-10 merge-readiness report | PASS | This ledger records the frozen contract, tested completed-resume provenance, and future boundaries. |

## Deferred and future work

- **HR-2E:** longitudinal convergence is **DEFERRED**. The strict-float test
  baseline `3.0000000000000004 != 3.0` remains recorded and was not changed.
- **Production longitudinal schedule:** **NOT FROZEN**.
- **HR-4:** **NOT STARTED**; no acoustic/hydrodynamic physics is implied here.
- **Performance engineering:** production disk footprint, large-state I/O,
  host/device transfer cost, production batch size, GPU throughput, and long-time
  periodic-boundary/domain validity remain future engineering/numerical work.

No HPC or Slurm job was submitted for this closeout.

## Completed-resume provenance repair

The bounded runner repair is complete. A completed `resume_hr3c=true` validates
the primary NPZ against the manifest before returning it. Missing, unreadable,
or provenance-mismatched artifacts fail closed; no pulse, diffusion, NPZ/report
write, or synthetic final field is permitted. The returned completed artifact
preserves the persisted propagation diagnostics and pulse history. Full-field
`E_final`/`I_final` are intentionally `None` on this path because the primary
NPZ does not persist them and a fresh-field reconstruction would be false.

Runner-level coverage verifies no physics reexecution, NPZ byte idempotence,
schema/content and diagnostic preservation, `pulse_index=[1,2,3]`,
`return_results=True`, missing/mismatched artifact fail-closed behavior, report
idempotence, and manifest/two-slot state immutability. This is
software/provenance repair only; it does not change HR-3A/B/C physics,
configurations, or the frozen `D_th` contract.
