# Production linear-half-step call graph

## Scope and selected production path

The completed replacement Job 1 (`179988`) used the locked configuration
`120fs_talebpour_isaacs_full_operator_on_energy_audit.json`.  Its
`propagation.linear_model` is `bk_nee`, so the actual production path is the
BK-NEE branch below, not the UPPE or paraxial fallback branches.

```text
runner.run_demo / propagate_one_pulse
  |
  +-- z step start: energy_step_start_J
  |
  +-- first linear half step, dz_try / 2
  |     propagate.py:408--432
  |       -> linear.step_linear_bk_nee_factorized
  |          linear.py:33--69
  |       -> energy_after_linear_half1_J
  |
  +-- nonlinear Strang section
  |     Raman half -> non-Raman operator -> Raman half
  |
  +-- second linear half step, dz_try / 2
  |     propagate.py:711--743
  |       -> linear.step_linear_bk_nee_factorized
  |          linear.py:33--69
  |       -> energy_after_linear_half2_J (= U_z)
  ```

`runner._linear_advance` has a similar branch implementation, but is not the
per-step production call used by `propagate_one_pulse`; it is therefore not a
substitute for this trace.

## Operations executed in each BK-NEE half step

| Order | File and function | Operation | Input/output dtype in Job 1 | Theoretical energy property | Explicit energy-budget channel |
|---|---|---|---|---|---|
| 1 | `linear.py:step_linear_bk_nee_factorized` | `fft(E, axis=0)` | complex64 -> complex64 (CuPy) | Unitary under the repository FFT convention in exact arithmetic | none |
| 2 | same | Build `denom=1+Omega/omega0`, clamp its *real phase coefficient* away from zero | float32 | Changes diffraction phase coefficient only; no amplitude multiplier | none |
| 3 | same, per temporal-frequency slice | `fft2(Ew[i])` | complex64 -> complex64 | Unitary in exact arithmetic | none |
| 4 | same | `prop2d=exp(i*phase_xy*dz_eff).astype(complex64)` then `S *= prop2d` | complex64 | Intended pure phase.  Finite-precision modulus and multiply rounding require audit. | none |
| 5 | same | `ifft2(S)` | complex64 -> complex64 | Unitary in exact arithmetic | none |
| 6 | same | `ifft(Ew, axis=0).astype(complex64, copy=False)` | complex64 -> complex64 | Unitary in exact arithmetic | none |

## Operations not present on this selected path

The BK-NEE production half step contains no explicit spectral amplitude mask,
high-k cutoff, evanescent-bin deletion, spatial/temporal absorbing mask,
padding, crop, guard-cell removal, interpolation, resampling, or post-step
NaN/Inf sanitization.  Those operations therefore cannot be silently charged
to this selected linear path.  The UPPE paths in `linear_full.py` are separate
branches and are not executed by Job 179988.

## Consequences for the current energy budget

The implemented BK-NEE multiplier is intended to be a pure phase multiplier,
not physical linear absorption.  It has no explicit deposition channel in
`E_dep_z`, `E_dep_rot_z`, or `E_dep_cumulative_z`; those channels track
nonlinear ionization/IB and Raman quantities.  Consequently, a measurable
linear field-energy decrease can only be accepted after R4.2--R4.4 establish
an explicit non-unitary operation and its accounting semantics.  It must not
be labelled physical absorption by default.

The Job 179988 checkpoint audit already reports a combined first-plus-second
linear-half-step change of `-3.094249404966831e-5 J`, while its unaccounted
final total-energy discrepancy is `3.0784489354118705e-5 J`.  R4.2 therefore
replays the actual operator components rather than inferring the cause from
the checkpoint sums alone.
