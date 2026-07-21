# Phase 8B-R Task R3.5 — minimal observability fix

`propagation.diag_operator_energy` is a new opt-in diagnostic switch, default
`false`.  It does not alter the field update, Raman parameters, non-Raman
physics, step controller, or default production configuration.  With the switch
enabled and `record_every_z=1`, the archive records float64 field energy at:

1. accepted-step start;
2. after linear half-step 1;
3. after the Raman pre-substep;
4. after the non-Raman nonlinear update;
5. after the Raman post-substep;
6. after linear half-step 2.

The local one-step CPU Strang smoke passed: the telescoping sum of these split
energy deltas differs from the net field change by `0 J`; total closure is
`4.504480257793818e-17`; Raman p99 is `2.3358172561609283e-6`; Raman cumulative
closure is `2.3358172561609283e-6`; legacy `alpha_R=0`; and the expected two
Raman substeps/four convolutions were observed.

This does **not** retroactively repair Job 179706.  Its coordinate audit and
Raman closure now pass, but its true runtime-history total-energy closure remains
`1.4176516743495496%`, above the unchanged `<1%` contract.  The R3 acceptance
criterion requiring a real full-job budget below 1% is therefore **not met**.

## Required authorization request

Authorize, only after reviewing this R3 audit, one replacement full Job 1 with
the same full-operator-ON physics and `propagation.diag_operator_energy=true`.
The replacement must use the same controlled execution constraints and must
complete its own audit before any discussion of Job 2.  Job 2 remains neither
prepared nor submitted.
