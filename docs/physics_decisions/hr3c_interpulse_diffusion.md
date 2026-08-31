# HR-3C-A interpulse transverse thermal diffusion

**Status:** HR-3C-A CLOSED; HR-3C-B CLOSED; HR-3C-C CLOSED; HR-3C CLOSED
(2026-08-30, source branch `HR-3`). HR-3 is CLOSED / MERGED TO MAIN via
`654fb0236b9c119ab7d89524c08cf0b84fe9181e`.

## HR-3C-C transactional integration

HR-3C-C uses two slots only. During a pulse, authoritative A is read-only and
the transaction writes `B[k]=A[k]+Delta_delta_n_th[k]` exactly once per
interval. After target flush/fsync, an atomically replaced manifest promotes
the physical stage to `post_pulse`. For non-final pulses, B is diffused to A,
durably flushed, then atomically promoted to `pre_pulse` for the next fresh
optical pulse. The manifest binds shape/dtype, interval-centered state,
`D_th`, `f_rep`, edge threshold, batch size, and a schedule/transverse-grid
fingerprint. Resume is explicit and opens existing memmaps without truncation.
Persistent fresh/post/diffusion counters are exact functions of the manifest
stage and pulse indices; any mismatch is rejected before resume can open the
state slots.
For a completed resume, the existing primary NPZ is validated and loaded; it is
not rewritten from a fresh source field, and the report, manifest, and state
slots remain unchanged.

## Scope and frozen state

The only authoritative persistent slow state remains HR-3B's interval-centered
`delta_n_th[K, Ny, Nx]`. HR-3C-A consumes one post-pulse 2-D slice
`delta_n_th^(p,+)[k]` and defines its theoretical next-pulse predecessor
`delta_n_th^(p+1,-)[k]`. It does not add persistent `Delta_T`, `delta_rho`,
pressure, velocity, or any other full-volume state.

For every longitudinal interval, the frozen transport model is

`partial_t delta_n_th = D_th (partial_x^2 + partial_y^2) delta_n_th`.

Transverse diffusion is ON. Longitudinal diffusion, acoustics, advection,
pressure/velocity/hydrodynamics, buoyancy, viscosity, gravity, and thermal-grid
coarsening are OFF. The thermal grid is the existing optical grid.

## Parameter and time authority

`D_th = 21.7e-6 m^2/s` is the authoritative HR-3C thermal diffusivity. The
repository copy of Isaacs *et al.* (2022), Sec. 3 Eq. (20), defines thermal
diffusivity as `chi = kappa_T/(rho0 C_p)`; Sec. 4 gives dry air at STP as
`chi = 21.7 mm^2/s`. This converts exactly to `2.17e-5 m^2/s`.

`HeatConfig.D_gas = 2.0e-5 m^2/s` remains legacy compatibility only. The
HR-3C-A API takes `D_th` explicitly and has no `D_gas` argument or fallback.
Both `D_th` and `f_rep` must be finite and positive. The interpulse duration is
frozen as `dt_interpulse = 1/f_rep`; acoustic relaxation time is not subtracted.

## Spectral operator and periodic-boundary gate

Each call performs one exact transverse spectral step on the repository's
existing `kperp2` grid:

`delta_n' = IFFT2(exp(-D_th*kperp2/f_rep) * FFT2(delta_n))`.

The Fourier implementation therefore has periodic computational boundaries.
For the authoritative default path, the numerical-validity metric is the
one-cell boundary-band ratio

`R_edge = max_boundary(abs(delta_n_th)) / max_domain(abs(delta_n_th))`.

The frozen numerical threshold is `R_edge <= 1e-3`. It is not a universal
physical constant. An exceedance raises a `ValueError`; the evolution is
fail-closed rather than merely logged. Analytical zero/uniform tests may
explicitly disable this boundary gate because a nonzero uniform field fills the
periodic domain by construction.

## Validation gates

- C1 zero-state invariance: PASS.
- C2 uniform-state invariance: PASS.
- C3 Gaussian analytical broadening: PASS for `w^2(t)=w0^2+4 D_th t`, peak,
  and full-map analytical closure.
- C4 non-positive thermal-index channel: PASS within dtype roundoff tolerance.
- C5 transverse integral conservation: PASS.
- C6 periodic edge contamination: PASS; deliberate boundary-filled state fails
  closed at the `1e-3` threshold.
- C7 authority and time: PASS; `D_th` and `f_rep` validate, and no `D_gas`
  fallback exists.
- C8 dtype/finite: PASS; float32 input returns finite float32 output, while
  float64 is used for analytical validation.

## HR-3C-B storage and streaming contract

`PingPongSlowStateStore` creates two distinct disk-backed `.npy` memmaps named
`<run>.hr3c_delta_n_th_current.npy` and
`<run>.hr3c_delta_n_th_next.npy`, each shaped `[K, Ny, Nx]` at the real
propagation dtype. `current` remains authoritative input and is never modified
by `diffuse_current_to_next`; `next` is non-authoritative scratch. The
following is the **historical HR-3C-B substage boundary**: the storage helper
itself had no role swap, generation record, checkpoint, restart, or runner
integration. HR-3C-C subsequently supplied those lifecycle responsibilities
through the manifest-governed two-slot transaction described above, without
altering the HR-3C-A/B diffusion contract.

The pass reads a bounded `[B, Ny, Nx]` host batch, transfers it once to the
configured `xp` backend, applies batched `FFT2/IFFT2` over `(-2,-1)` using a
single kernel built once per volume pass, transfers one output batch to host,
and writes it to `next`. Working state is therefore `O(B*Ny*Nx)`, not a full
`[K,Ny,Nx]` materialization. Each evolved slice receives the same HR-3C-A
sign and edge gates. A failure leaves `current` intact and marks `next` invalid;
partial `next` contents are never promoted. Only successful full writes followed
by `flush_next()` report `complete=true`, which still does not confer authority.

For the nominal `K=16000`, `Ny=Nx=512`, float32 case, one raw state payload is
`16,777,216,000` bytes = `15.625 GiB`; the two-file ping-pong payload is
`33,554,432,000` bytes = `31.25 GiB`. A disk-space preflight runs before either
file is created and fails closed if capacity is insufficient. The local benchmark
uses only a temporary tiny state and must not be interpreted as a production
batch-size freeze.

### HR-3C-B validation gates

- CB1 separate disk-backed layout and nominal-byte estimators: PASS.
- CB2 current immutability: PASS.
- CB3 batch/slice HR-3C-A equivalence in float32 and float64: PASS.
- CB4 tiny-volume streaming closure with partial final batch: PASS.
- CB5 edge failure includes the global interval index, leaves current intact,
  and leaves next unpromoted: PASS.
- CB6 bounded batch reads and a single kernel build per volume pass: PASS.
- CB7 successful flush, close, and read-only reopen persistence: PASS.
- CB8 estimator and local microbenchmark harness: PASS; observed local values
  are CPU/NumPy `K=32`, `64x64`, float32 only: B=1/2/4/8/16/32 gave
  73.30/76.07/54.64/48.10/56.70/58.65 MiB/s, respectively. B=2 was the local
  best observed value (76.07 MiB/s); it is not a production batch-size freeze
  or GPU/HPC certification.

## HR-3C-C transactional lifecycle and final closeout

HR-3C-C integrates the existing HR-3B post-acoustic state and the frozen HR-3C-A
diffusion operator without changing either physical contract. A pulse reads
authoritative pre-state A and writes B exactly once per interval; a successful
post commit atomically promotes B. For non-final pulses, B is diffused into A;
the final post commit atomically records `run_complete=true` and is never
diffused.

Closeout gates CC1–CC12 are PASS. The evidence includes exact N=1/2/3
`fresh/post/diffusion = N/N/(N-1)` totals, fresh optical-source copies, pulse and
diffusion interruption recovery equivalent to uninterrupted references,
completed-run resume without re-execution, HR-3C legacy-path isolation,
standalone HR-3B regression, and exactly two HR-3C full-volume state slots.

This closeout performs no production propagation, HPC/Slurm work, HR-3D/HR-4
gas dynamics, or HR-2E convergence work. HR-2E remains DEFERRED and the
production longitudinal schedule remains NOT FROZEN.
