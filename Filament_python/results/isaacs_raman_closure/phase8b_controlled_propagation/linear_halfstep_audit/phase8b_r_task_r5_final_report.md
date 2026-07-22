# Phase 8B-R Task R5 — BK-NEE precision repair and short-run admission

## Result

The numerical repair is effective, but R5 is **not admitted to a new full Job
1**.  The final classification is `bk_nee_fix_failed_performance_gate`.

`mixed_precision` is the selected numerical strategy: the propagated field
remains complex64 between operators; each BK-NEE half step casts once to
complex128, performs its time FFT, spatial FFTs, pure-phase multiply and
inverse FFTs in complex128, then casts once back to complex64.  This changes
no Raman, Kerr, ionization, plasma, or linear physical coefficient.  It is
opt-in; historical default behavior remains `baseline_complex64`.

## Numerical evidence

| Gate | Result |
|---|---:|
| complex128-reference max 200-step relative L2 | `5.1303e-7` |
| 20-step full-grid pure-linear residual | `1.1364e-8` |
| 200-step full-grid pure-linear residual | `1.1023e-8` |
| 15,000-step pure-linear extrapolation | `8.2672e-7` |
| 20-step full-physics linear residual | `1.0128e-8` |
| Raman step closure p99 | `1.5019e-4` |
| Raman cumulative closure | `2.8315e-5` |
| legacy `alpha_R` | `0` |

The baseline and matched orthonormal FFT candidates fail the 20-step
cumulative linear-residual limit.  The unitary-projection candidate satisfies
its scalar gates but is not selected: it has a larger reference error and is a
corrective projection rather than the lower-error arithmetic solution.

## Performance gate

The 200-step pure-linear run used 193 s, which extrapolates to 4.02 h.
However, the full-physics smoke used 58.1 s for 20 steps.  Its direct
15,000-step linear extrapolation is 12.10 h, above the locked 6.4 h limit.
The R5 smoke also did not capture peak reserved GPU memory, although no OOM
occurred.  The performance gate therefore fails; this report does not claim
that a full mixed-precision production run is ready.

## Scope and next action

No full Job 1 was submitted.  Job 2 was neither prepared nor submitted.
The required next action is to profile/optimize the full-physics
mixed-precision execution path and capture peak GPU memory, then repeat only
the necessary short smoke gates.  A new 15,000-step Job 1 still requires a
separate explicit authorization.

## GitHub synchronization

R4's original two push attempts failed with HTTPS connection resets.  A later
retry succeeded and synchronized `47aee9f` to remote `main`; R5 commits are
pushed separately after this report is committed.  GitHub Actions CI evidence
is unavailable because this repository has no workflow evidence.
