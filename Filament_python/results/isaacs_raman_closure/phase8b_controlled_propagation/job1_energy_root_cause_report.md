# Phase 8B-R Job 1 total-energy root-cause decision

## Classification

`inconclusive_insufficient_diagnostics`

The final `1.4176516743495496%` closure is real **within the saved runtime
energy histories**: a float64 reconstruction produces exactly the same value
as the corrected audit.  It is not a float32 coordinate artifact, and the
archive coordinate checks now pass independently.

It cannot, however, be independently localized further.  The archive contains
no full `E/I(z,t,x,y)` history and no energy deltas for the linear half-steps,
the non-Raman nonlinear update, or each Raman Strang half-step.  The final
centerline waveform cannot be integrated over the transverse plane.  Hence the
archive cannot distinguish a missing accounting channel from production
numerical/boundary/filter dissipation.

## Raman is not the demonstrated source

- Raman per-step closure p99: `1.990885304985568e-4` (contract `<1e-3`).
- Raman cumulative closure: `2.245331188532873e-6` (contract `<5e-3`).
- Cumulative Eq.(10) target minus actual field loss: `8.731149137035942e-11 J`.
- Legacy `alpha_R` is exactly zero.

The signed final total residual is `30.784489354118705 µJ`.  Raman's actual
loss is accounted in the total budget and matches its independent target; it
therefore cannot be named as the primary *unaccounted* contribution.

## Near-focus versus final closure

The near-focus maximum is `1.2044126022869913%` at `z=1.0499500036 m`.
The final value is larger by `0.2132390721` percentage points.  From that
near-focus maximum to `z=1.300 m`, field loss increases by `13.37612048 µJ`,
recorded deposition increases by `8.74560646 µJ`, and the signed unaccounted
increment is `4.63051101 µJ`.  Of the recorded post-focus deposition,
`8.00330236 µJ` is Raman and `0.74230896 µJ` is ionization/IB; this is an
accounted split, not a proof of Raman causality for the residual.

## Required consequence

Re-auditing Job 179706 can correct the coordinate semantics but cannot make the
total-energy contract pass.  A minimal production-observability change is
required, followed by a short smoke test.  A new full Job 1 is then required to
prove the `<1%` total-energy contract.  Job 2 remains neither prepared nor
submitted.
