# HR-2 closeout

Date: 2026-08-29

## Frozen HR-2 status

| Work package | Status | Frozen conclusion |
| --- | --- | --- |
| HR-2A | PASS | Longitudinal schedule/deposition metadata contract is complete. |
| HR-2B | PASS | Plasma deposition ledger is complete. |
| HR-2C / HR-2C-R | PASS | Full-Raman local deposition contract and its reduction/operator-energy closures are complete. |
| HR-2D | PASS | Unified authoritative deposition ledger is complete. |
| HR-2E | DEFERRED | Longitudinal schedule convergence remains unresolved numerical convergence debt. |

The authoritative Ion, Raman, and Total deposition contracts are valid.  For
the corrected 120 fs reprocessing, Raman deposition-reduction and
operator-energy closure both pass.  The remaining HR-2E blocker is solely:

```text
longitudinal schedule convergence unresolved
```

## HR-2E convergence evidence and interpretation

The 120 fs candidate-to-fine pulse/deposition differences are approximately
2.220% (Ion), 2.367% (Raman), and 2.275% (Total).  The longitudinal
localization shows that Ion and Total differences are primarily formed in the
0.75--1.05 m focus window, while Raman has material focus-window formation and
a dominant post-focus tail.  The candidate schedule is therefore a
**provisional schedule for subsequent development only**; it is not a
converged production schedule.

The production longitudinal schedule is **NOT FROZEN**.  Production config is
unchanged and must not be changed merely to adopt the temporary candidate or
fine schedule.

## Deferred HR-2E work

```text
refine both base dz and focus dz
→ revalidate 120 fs convergence
→ perform 40 fs cross-check only if necessary
→ freeze the production longitudinal schedule
```

The 40 fs candidate case is **not executed** and the 40 fs fine case is **not
executed**.  Both are deferred work, not current required work: 120 fs has not
passed its final schedule-convergence gate.  This closeout does not authorize a
new schedule, a 40 fs job, a new 120 fs job, or any production propagation.

## Closeout boundary

HR-2 core deposition interface work is complete.  HR-2E longitudinal schedule
convergence is intentionally retained as numerical convergence debt.  This
closeout does not represent production longitudinal-schedule convergence
certification.
