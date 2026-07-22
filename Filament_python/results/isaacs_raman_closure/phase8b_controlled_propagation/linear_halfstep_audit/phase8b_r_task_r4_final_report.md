# Phase 8B-R Task R4 — linear-half-step energy audit

## Decision

**Status: `job1_rerun_required`.** The replacement Job 1 result `179988`
cannot be made admissible by a post-processing-only audit.  Its final
total-energy closure is `1.4176516743495496%`, above the unchanged `1%`
contract, and the deficit is quantitatively consistent with unintended
complex64 BK-NEE linear-operator dissipation.

No Job 2 file was prepared or submitted.  No new full Job 1 was submitted.
The only new remote execution was controlled GPU smoke job `180046`, a
20-step pure-linear CuPy run.

## Evidence chain

1. The production call graph is `FFT(t) -> FFT2(x,y) -> pure-phase transfer
   -> IFFT2(x,y) -> IFFT(t)` for each of the two BK-NEE half steps.  It has no
   mask, filter, padding, crop, guard-cell removal, or physical linear
   absorption.  See `linear_halfstep_call_graph.md`.
2. The actual Job 179988 operator-energy checkpoints show combined linear
   half-step field loss of `-3.094249404966831e-05 J`; the independently
   reconstructed total-budget discrepancy is `3.0784489354118705e-05 J`.
   These agree in magnitude to within `0.00728` percentage points of initial
   field energy.
3. The kernel audit finds no zeroed bins and no designed amplitude attenuation:
   complex64 `max(abs(abs(H)-1)) = 1.1920928955078125e-07`, while the float64
   reference is `2.220446049250313e-16`.
4. Smoke job `180046` isolates the same full-size grid and CuPy BK-NEE path
   with all nonlinear physics off.  Each half step has p99 relative residual
   `5.249573123657346e-07` (< `1e-6`), but twenty steps accumulate an
   unaccounted loss of `-4.499202038580133e-08 J`, or
   `2.071920612813116e-05` relative to input (> `1e-5`).  Linear extrapolation
   to 15,000 steps is `1.5539404596098372%`, comparable to the observed
   `1.4176516743495496%`; this extrapolation is classification evidence only,
   not a substitute for a stepwise Job 1 ledger.
5. The first measurable loss occurs after the forward temporal FFT.  Further
   losses occur after the spatial pure-phase multiply and both inverse FFTs.
   This identifies accumulated complex64 FFT/multiply roundoff, not a single
   physical absorption or omitted boundary/filter channel.

## Energy-domain audit

The runtime energy diagnostic and operator checkpoints use the entire
Cartesian `[Nt, Ny, Nx] = [384, 512, 512]` field with `sum(I) * dt * dx * dy`.
There is no padding/crop domain or cylindrical Jacobian on the selected
BK-NEE route.  The archive retains scalar histories but no complex field
checkpoints, so the six stored checkpoints cannot be replayed internally;
they are retained as an explicitly inconclusive replay audit rather than
being interpolated into a fictitious 15,000-step loss history.

## Required action

The primary classification is **`linear_operator_numerical_dissipation`**.
The production BK-NEE implementation must be corrected (for example by a
validated precision-preserving linear execution strategy) and must then pass
operator tests, a pure-linear full-size CuPy smoke, and a full-physics
short smoke before a new full Job 1 can be requested.  The Raman Eq. (27)
operator itself remains within its local and cumulative closure contract in
Job 179988; legacy `alpha_R` is zero.  The present Job 179988 cannot be
re-audited into passing merely by adding its unintended linear loss to a
physical-loss budget.

GitHub push was attempted exactly twice during R4: the permitted normal push
and one explicit retry.  Both failed because the GitHub HTTPS connection was
reset, so no further retry is made in this task phase.  The remote tracking
reference therefore remains `e321223b6ee2c79407192ff4be3087fd83ec5ef8`.
