# HR-3B post-acoustic slow-state decision

**Status:** HR-3B CLOSED; incorporated in HR-3 CLOSED / MERGED TO MAIN by
`654fb0236b9c119ab7d89524c08cf0b84fe9181e`.

## Evidence gate: PASS

The two repository reference PDFs establish the intended reduced model and
its boundary.

- Isaacs *et al.* (2022), Sec. 3, Eq. (23), derives the single-pulse,
  post-acoustic refractive-index imprint after pressure has relaxed to the
  isobaric regime and before thermal transport matters:
  `delta_n = -(n0 - 1) u / (rho0 C_V T0)`. Sec. 5, Eq. (31), applies the
  same relation as `delta_n_plus = delta_n_minus - (n0 - 1)/(rho0 C_V T0)
  * dF_L/dz`. The paper defines `delta_n = (n0 - 1) delta_rho/rho0`, uses
  `C_V` rather than `C_p` for the impulsive heating term, and gives the
  isobaric relation `delta_rho/rho0 = -delta_T/T0` after the acoustic stage.
- Isaacs Sec. 4 gives the intended scale: approximately 1 kHz pulse
  separation (`tau_s about 1 ms`) versus acoustic transit time
  (`tau_a about 0.6 us`) at STP. HR-3B therefore collapses microscopic
  thermalization plus acoustic pressure release into a post-acoustic jump;
  it does not solve the acoustic transient.
- Zeng Qingwei (2022), Sec. 7.3, identifies filament heat deposition, the
  formation/propagation of high-pressure shock or acoustic waves, and heat
  conduction as distinct stages. Its model treats only initial deposited
  energy and initial temperature/pressure perturbations, leaving subsequent
  gas motion and heat conduction for later work. This supports the explicit
  HR-3B / HR-3C boundary.

The authoritative input is the current-interval HR-3A `q_thermal` map, not
legacy `Q2D`, `gamma_heat`, field loss, or the HR-3A sparse diagnostic archive.

## Frozen mapping and state

For each interval-centered `q_thermal[k, y, x]` in `J/m^3`, HR-3B applies

`Delta delta_n_th = -beta_th q_thermal`,

where

`beta_th = (n0 - 1)/(rho0 C_V T0)`.

Positive deposited heat therefore produces a non-positive thermal index
increment. The sole authoritative persistent variable is the interval-centered
`delta_n_th[k, y, x]`; `Delta_T_impulse`, `Delta_T_post`, and `delta_rho` are
derived diagnostics only.

## Parameter audit

| Symbol | Runtime source | Unit | HR-3B value / reason |
| --- | --- | --- | --- |
| `n0` | `BeamConfig.n0` | 1 | Reuses the optical propagation convention (`1.00027`). The existing `air_dispersion.n_of_omega` ambient-value discrepancy is recorded but not changed in HR-3B. |
| `T0` | `PropagationConfig.air_T` | K | Reuses the single ambient-air temperature. |
| `rho0` | `HeatConfig.rho0` | kg/m^3 | New explicit dry-air mass density; Isaacs STP value is 1.23. |
| `C_V` | `HeatConfig.Cv` | J/(kg K) | New explicit dry-air constant-volume heat capacity. Isaacs gives `C_p=1 kJ/(kg K)` and `gamma=1.4`, hence `C_V=C_p/gamma=714.2857 J/(kg K)` for this frozen reference convention. |

`gamma_heat`, `D_gas`, and `C_p` are not authoritative HR-3B mapping inputs.
`hr3b_enabled` is explicit opt-in because the repository's legacy default
Raman path is non-authoritative by the HR-2/HR-3 contract. When enabled,
non-authoritative HR-3A input fails closed rather than falling back to legacy
heat plumbing.

## Data lifecycle and ordering

The storage file is a single disk-backed NumPy `.npy` memmap with shape
`[K, Ny, Nx]` and the propagation real dtype. It is initialised exactly to
zero and updated in place, forward in interval index:

1. read the old `delta_n_th[k]` slice;
2. apply that old slice in the current optical nonlinear phase;
3. finalize authoritative HR-3A `q_thermal[k]`;
4. form `Delta delta_n_th[k]` and add it in place;
5. release transient maps and advance.

This creates `delta_n_th^(p,+)` after a pulse. HR-3C alone may evolve it to
`delta_n_th^(p+1,-)` through interpulse transport. The following is the
**historical HR-3B substage boundary**, not a statement about final HR-3:
no acoustic solver, checkpoint transaction, double buffer, node state, grid
resampling, or full-volume history of derived thermodynamic variables belongs
to HR-3B. HR-3C-C subsequently supplied the transactional two-slot,
runner/restart lifecycle without changing the HR-3B mapping; explicit acoustic
and full hydrodynamic evolution remain outside HR-3.

For `K=16000`, `Nx=Ny=512`, float32, the persistent state file is about
15.625 GiB. It is disk-backed, not an in-memory array; normal per-interval
working data are bounded to current host/device slices and current transient
maps.

HR-3A remains **CLOSED**. HR-2E remains **DEFERRED** and the production
longitudinal schedule remains **NOT FROZEN**.

HR-3B is **CLOSED** after local mapping, storage, ordering, phase-law, legacy
isolation, and targeted-regression checks. The known HR-2E strict-float
baseline failure remains outside this closure scope.
