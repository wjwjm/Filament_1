# HR-2E 120 fs Longitudinal Error Localization

## Scope

This is a zero-HPC, postprocess-only localization of the completed immutable
120 fs results: coarse (224730), candidate (224731), and fine (224732).  It
uses the canonical scalar interval ledgers with conservative overlap remapping
and splits intervals exactly at 0.75 m and 1.05 m.  No NPZ was downloaded,
overwritten, or rerun.

The signed convention is left schedule minus right schedule.  Percentages are
each region's share of the corresponding full signed deposition difference.

## Candidate to fine

| Channel | Pre-focus (<0.75 m) | Focus (0.75-1.05 m) | Post-focus (>1.05 m) |
| --- | ---: | ---: | ---: |
| Ion | 1.785e-11 J (0.001%) | 1.285e-6 J (89.22%) | 1.552e-7 J (10.78%) |
| Raman | 5.694e-8 J (6.22%) | 3.822e-7 J (41.77%) | 4.758e-7 J (52.00%) |
| Total | 5.696e-8 J (2.42%) | 1.667e-6 J (70.78%) | 6.310e-7 J (26.80%) |

Total cumulative `candidate - fine` is 5.696e-8 J at 0.75 m, 1.724e-6 J at
1.05 m, and 2.355e-6 J at 1.30 m.  Thus the observed pulse-energy/deposition
difference is not materially present before focus, is mainly established in
the focus window, and then continues to grow materially after focus.  The
post-focus Raman tail is larger than its focus contribution (52.00% versus
41.77%), so it substantially drives the Raman and Total late-position (z90)
offsets.

## Coarse to candidate consistency

The same spatial pattern occurs for coarse to candidate:

| Channel | Pre-focus share | Focus share | Post-focus share |
| --- | ---: | ---: | ---: |
| Ion | 0.001% | 87.43% | 12.57% |
| Raman | 6.08% | 40.77% | 53.15% |
| Total | 2.36% | 69.32% | 28.32% |

Therefore the focus-dominant Total/Ion formation and persistent post-focus
Raman accumulation are not unique to the candidate-to-fine comparison.

## Unique next-step recommendation

`refine both`

Focus resolution is the main limiter for Ion and Total deposition formation,
but base-region resolution remains material after the focus window (26.80% of
Total and 52.00% of Raman candidate-to-fine difference).  This finding does
not authorize creation of a new schedule, an extra-fine case, or any HPC job.
It also does not change the existing HR-2E Raman Level-1 closure blocker.

## Artifacts

- `error_localization_summary.json`: machine-readable canonical-ledger results.
- `error_localization_segments.csv`: per-channel, per-region deposition values.
- `cumulative_delta_energy_candidate_vs_fine.png`: cumulative Ion, Raman, and
  Total difference curves.
