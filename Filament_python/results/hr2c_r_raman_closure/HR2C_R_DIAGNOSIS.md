# HR-2C-R diagnosis: full-Raman local deposition closure

## Root cause

The former Raman Level-1 gate compared two different reductions of the
complete field-difference map:

`integral(max(actual_local_fluence_loss, 0))` versus
`integral(actual_local_fluence_loss)`.

For every inspected 120 fs interval, the stored residual equals their
difference and is nonnegative.  All three jobs also report negative local
field-difference values.  The failure is therefore a deposition-contract
mismatch, not schedule non-convergence and not evidence that the Raman field
update is physically invalid.

## Four Raman quantities

| Quantity | Role after HR-2C-R | Authoritative local deposition? |
|---|---|---|
| Eq.10/Heun positive rotational target | Canonical local medium gain and interval source | Yes, when the full operator is applied |
| Complete Eq.27 signed local field difference | Field diagnostic; may include conservative/local redistribution | No |
| Positive-clipped signed local field difference | Historical diagnostic only | No |
| Signed global field-energy loss | Independent operator-energy closure reference | No |

The complete Eq.27 electronic derivative contribution remains outside the
Raman medium-deposition source.  No HR-2C-R value is thermalized heat.

## Repaired closures

- Deposition-reduction closure: local Eq.10/Heun positive gain -> interval
  energy from the same map.  This remains strict (`rtol=2e-5`).
- Operator-energy closure: target interval energy versus signed actual global
  field loss.  This reuses the Phase-8B operator criteria: per-interval p99
  `<=1e-3` and cumulative `<=5e-3`.  A single coarse tail interval has a
  maximum relative residual of one because its target is effectively zero;
  its p99 is `1.9894498109351844e-4`.

Feedback OFF remains zero applied deposition, despite a nonzero target
diagnostic.  Legacy Raman remains `legacy_unavailable` and has no fallback.

## Existing 120 fs reuse classification

**Class A — fully reusable for the corrected HR-2E scalar convergence
analysis.**  In jobs 224730/224731/224732, `raman_target_loss_step_J` is
finite, nonnegative, and exactly one-to-one with the canonical interval count.
The revised postprocessor reconstructs the corrected scalar Raman and total
ledgers in memory; it neither overwrites nor downloads the raw NPZ files.

This classification did not itself run HR-2E convergence or freeze a production
schedule.  Corrected HR-2E reprocessing has since validated the authoritative
Raman contract; the remaining deferred work is longitudinal schedule
convergence, not Raman closure.
