# HR-3C-A interpulse transverse thermal diffusion

**Status:** CLOSED (2026-08-30, branch `HR-3`). This is the HR-3C-A physics
operator and validation baseline only; it does not create the HR-3C-B storage
lifecycle or the HR-3C-C pulse orchestration.

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

## Deferred

HR-3C-B owns current/next disk-backed ping-pong files, z chunking, batched FFT,
host/device streaming, flushing, and performance work. HR-3C-C owns atomic
generation metadata, checkpoint/restart, crash consistency, role swaps, and
the final `Npulses=N -> N-1` diffusion orchestration. No production propagation,
HPC/Slurm work, HR-3D/HR-4 gas dynamics, or HR-2E convergence work is included.
