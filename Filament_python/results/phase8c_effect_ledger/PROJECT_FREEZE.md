# Phase 8C-0 project freeze

Audit timestamp: 2026-07-22 (Asia/Shanghai). This directory contains only read-only post-processing, provenance indexes, figures, and audit documents. It must not be used to modify an archived propagation result.

## Frozen repository state

| Item | Recorded value |
| --- | --- |
| Repository | `wjwjm/Filament_1` |
| Frozen `main` SHA | `e11d13f103c484953c0f733aa9b410bff385b2b5` |
| `origin/main` SHA at freeze | `e11d13f103c484953c0f733aa9b410bff385b2b5` |
| Local branch | `main` |
| Local-only commits at freeze | none (`HEAD == origin/main`) |
| Working tree | not clean; only pre-existing untracked items listed below |

Pre-existing untracked paths (recorded, deliberately neither deleted nor staged):

- `Filament_python/results/isaacs_raman_closure/phase8b_controlled_propagation/job1_179706.err`
- `Filament_python/results/isaacs_raman_closure/phase8b_controlled_propagation/job1_179706.out`
- `phase8b_r_job1_audit_e724cd66.bundle`
- `tmp/`

## Fixed production reference

```text
lambda0 = 800 nm
P0_peak = 17 GW
focal_length = 0.95 m
FT90 radius = 1.979 mm
tau_fwhm = 120 fs
z_max = 1.300 m
x_focus_cm = 100 * (z_m - 0.95)
```

The focal-coordinate zero is permanently `z = 0.95 m`, `x_focus_cm = 0`. No vacuum-intensity maximum, density maximum, curve fit, or PyCAP shift may redefine it.

## Phase status

```text
Phase 6 = historical Raman split causality evidence
Phase 8A/8A.1 = full Isaacs Eq.27 static/operator validation completed
Phase 8B-P/R1 = production wiring and Raman admission completed
Phase 8B-R Job 1 = energy admission failed
R4 = complex64 BK-NEE numerical dissipation identified
R5 = mixed-precision numerical repair passed, performance gate failed
R6 = suspended
```

## Required suspension state

```text
R6 performance profiling = suspended
mixed-precision performance optimization = suspended
new full Job 1 = not authorized
Job 2 preparation = not authorized
Job 2 submission = not authorized
high-repetition thermal accumulation = out of scope
40 fs new production rerun = not authorized
120 fs new production rerun = not authorized
```

No Slurm submission, full-Job configuration generation, production-physics modification, Raman/BK-NEE modification, energy-contract modification, filament-criterion modification, or PyCAP re-digitization is authorized by Phase 8C-0.

## Frozen job ledger

| Job | Purpose | Execution SHA | Configuration / result evidence | Status | Production propagation executed? | Physics comparison use | Numerical/audit-only? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 176915 | Phase 6 legacy Raman phase-off 120 fs control | `8dcd01ee38adf2167a2fd6083ae4785e94de89a0` | `configs/raman_phase_causality/120fs_talebpour_full_model_raman_phase_off.json`; local axial CSV | COMPLETED | yes | historical causal evidence; strict phase switch pair with the recorded baseline | no |
| 179288 | Phase 8B preflight 20-step full-Eq.27 ON smoke | not separately archived | preflight smoke log; 2 mm smoke | COMPLETED | no (smoke only) | no | yes |
| 179311 | Phase 8B preflight 20-step full-Eq.27 OFF smoke | not separately archived | `phase8b_full_size_smoke_off_metrics.json` | COMPLETED | no (smoke only) | no | yes |
| 179619 | R1 20-step full-Eq.27 ON smoke (earlier closure attempt) | not separately archived | `r1_smoke_evidence/job_179619` | COMPLETED | no (smoke only) | no | yes |
| 179623 | R1 corrected 20-step full-Eq.27 ON smoke | not separately archived | `r1_smoke_evidence/job_179623` | COMPLETED | no (smoke only) | no | yes |
| 179706 | Phase 8B-R Job 1, full Eq.27 ON | `aad67a100fba612789dba2e41e39fadf04219d63` | `120fs_talebpour_isaacs_full_operator_on.json`; 15,000 records | COMPLETED, admission failed | yes | invalid for physics: final energy contract failed | no |
| 179983 | replacement Job 1 submission attempt | not applicable | failed launcher metadata; no GPU execution | FAILED before execution | no | no | invalid |
| 179988 | replacement Job 1 full Eq.27 ON energy audit | `e321223b6ee2c79407192ff4be3087fd83ec5ef8` | `120fs_talebpour_isaacs_full_operator_on_energy_audit.json`; 15,000 records | COMPLETED, admission failed | yes | invalid for physics: 1.41765% final energy closure failure | root-cause evidence only |
| 180046 | R4 20-step pure-linear complex64 audit | `35f42cb9a756019ba1216b29eb96e4c0d7de00c0` | `pure_linear_20step_smoke.json` | COMPLETED | no | no | yes |
| 180068 | R5 200-step pure-linear mixed-precision audit | R5 smoke source | `r5_4_mixed_precision_long_smoke.json` | COMPLETED | no | no | yes |
| 180076 | R5 20-step full-physics mixed-precision audit | R5 smoke source | `r5_5_full_physics_smoke.json` | COMPLETED | no (smoke only) | no | yes |

`179983` is retained as a failed pre-execution record; it did not produce GPU propagation evidence. `180055`–`180058` are candidate pure-linear screens and are indexed in the result inventory although they are not part of the minimum freeze-job list.

## Admission and authorization

```text
new full Job 1 authorized = false
Job 2 prepared = false
Job 2 submitted = false
```

R5 selected `mixed_precision` solely as a numerical repair: it removes the measured complex64 linear dissipation in controlled smoke tests, but its 20-step full-physics runtime extrapolates to 12.10 h against the locked `< 6.4 h` performance gate. This does not authorize R6 or any new propagation.

GitHub Actions CI = unavailable; no workflow present.
