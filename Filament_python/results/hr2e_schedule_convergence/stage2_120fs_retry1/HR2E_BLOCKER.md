# HR-2E Stage 2 120 fs blocker

Date: 2026-08-28

All three replacement public-GPU jobs completed successfully at execution SHA
`a11e1f1b3ad2fc2f95f5effbe231fbec4558a3b7`:

- `224730` coarse: `COMPLETED/0:0`;
- `224731` candidate: `COMPLETED/0:0`;
- `224732` fine: `COMPLETED/0:0`.

The existing `postprocess_hr2e_schedule_convergence.sh` stopped before creating
the comparison JSON, CSV, or figures because the mandatory Level-1 canonical
deposition closure prerequisite failed. A focused read-only audit classified
the failure as category D: authoritative-channel closure.

Ion Level-1 closure passed and IB was correctly classified as not applicable in
all three runs. Raman Level-1 closure failed in all three schedules, even though
the metadata reports `actual_field_fluence_loss`, authoritative Raman and total
ledgers, passing Level-2 closure, and available authoritative field-energy
bookkeeping. Candidate and fine Raman interval comparisons exceeded the frozen
`2e-5` relative Level-1 tolerance on every interval; their largest relative
differences were approximately `1.66e-3` and `1.23e-2`, respectively.

Consequences:

- authoritative candidate-versus-fine convergence gates were not evaluated;
- coarse-to-candidate convergence was not evaluated;
- Stage 3 40 fs jobs were not submitted;
- no successful job was repeated;
- the production schedule and production Raman configuration were not changed.

This is a scientific/numerical prerequisite blocker, not a scheduler failure.
The next authorized step must diagnose and repair the full-Raman interval
Level-1 closure contract before producing a scientifically valid schedule
comparison. Existing raw NPZ files remain remote and unchanged.
