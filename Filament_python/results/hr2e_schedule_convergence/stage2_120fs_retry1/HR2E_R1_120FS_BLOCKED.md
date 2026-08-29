# HR-2E R1: corrected 120 fs reprocessing result

## Verdict

**HR-2E DEFERRED.**  The corrected Class-A reconstruction completed and the
authoritative prerequisite gate passed, but the 120 fs candidate-to-fine
longitudinal convergence gate did not pass.  This is unresolved longitudinal
schedule convergence debt, not a Raman closure or provenance failure.

## Provenance and reuse

- Existing jobs reused without rerun: coarse `224730`, candidate `224731`,
  fine `224732`.
- Propagation execution SHA: `a11e1f1b3ad2fc2f95f5effbe231fbec4558a3b7`.
- Corrected analysis SHA: `485790961fa57da9cde6a316b0128000d5279431`.
- The analysis checkout was acquired as a separate strict-remote-verified
  staging checkout.  It did not modify the historical `a11e1f1` source or raw
  NPZ files.
- Raman reconstruction uses `raman_target_loss_step_J` and the authoritative
  `eq10_heun_positive_rotational_energy` contract.  Ion passed, IB was zero,
  Raman deposition-reduction passed, Raman operator-energy passed, and total
  deposition was authoritative.

## Candidate versus fine (120 fs)

The primary thresholds are pulse-energy error <= 1%, cumulative shape <= 0.02,
and location shifts <= local candidate resolution allowance.  All three active
channels pass shape but fail pulse energy and locations.

| Channel | Pulse error | Shape error | Largest location shift | Primary result |
|---|---:|---:|---:|---|
| Ion | 2.220% | 0.00234 | 0.642 mm centroid | FAIL |
| Raman | 2.367% | 0.00761 | 5.181 mm z90 | FAIL |
| Total | 2.275% | 0.00426 | 10.504 mm z90 | FAIL |
| IB | zero | 0 | not limiting | PASS |

Peak metrics are also above their secondary threshold for all active channels;
they do not change the primary failure decision.

The coarse-to-candidate comparison shows the same qualitative direction:
pulse errors are about 1.14--1.22%, and Raman/total location shifts remain
above the candidate-resolution allowance.  The error did not decrease from
coarse-to-candidate to candidate-to-fine, so no monotonicity claim is made.

## Deferred work boundary

- 40 fs candidate/fine submitted: **0**.
- New 120 fs jobs: **0**.
- Production longitudinal schedule: **not frozen**.
- Production config: **unchanged**.
- HR-3 and HR-5: **not executed**.

The scientifically relevant debt is insufficient convergence of the existing
0.10 mm / 0.05 mm candidate schedule relative to the 0.05 mm / 0.025 mm fine
schedule under the frozen primary pulse-energy and location gates.  Future
work is to refine both base and focus dz, revalidate 120 fs, use 40 fs only as
a necessary cross-check, and then freeze the production schedule.  This task
does not create a finer schedule or submit another run.
