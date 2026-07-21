# Phase 8B-P controlled-propagation preflight report

## Decision

- `full_job_submission_gate`: **passed**
- Phase 8B-R Task R1 prepared: **true**
- Phase 8B-R Task R2 executed: **false**
- Full 1.3 m Slurm jobs submitted: **0**
- GitHub Actions CI evidence: **unavailable**
- Required next action: obtain explicit user approval for Task R2 before submitting full Job 1.

## Full-grid smoke evidence

- Short smoke Job IDs: 179623, 179311 (both `COMPLETED 0:0`).
- GPU: NVIDIA GeForce RTX 5090; grid: 512x512x384; 20 z steps per case.
- Peak reserved GPU memory: 83.783% (threshold <85%).
- Mean ON step time: 0.974969 s; projected 15000-step runtime: 4.062 h.
- Requested 8 h fraction: 50.780%; slowdown vs legacy: 1.953x.
- ON convolution count: 2 per Heun application and 4 per Strang z step.
- OFF raw diagnostic convolution count: 1 per z step.
- ON Raman per-step closure p99: 0.000133499 (threshold <1e-3).
- ON Raman cumulative closure residual: 2.70299e-05 (threshold <5e-3).
- Legacy Raman alpha maximum: 0.

## Combined nonlinear split

- Selected order: `strang`.
- Refined estimated order: 1.581025 (threshold >=1.5).
- Production dz vs dz/2 field L2 difference: 5.98557e-07.

## Gate summary

| Gate | Status |
|---|---|
| `baseline_config_lock_gate` | passed |
| `on_off_single_factor_gate` | passed |
| `explicit_operator_switch_gate` | passed |
| `legacy_absorption_rejection_gate` | passed |
| `full_operator_diagnostic_wiring_gate` | passed |
| `raman_energy_accounting_gate` | passed |
| `convolution_reuse_gate` | passed |
| `combined_split_convergence_gate` | passed |
| `combined_split_production_step_gate` | passed |
| `full_size_smoke_gate` | passed |
| `memory_gate` | passed |
| `runtime_gate` | passed |
| `expected_diagnostic_contract_gate` | passed |
| `full_job_submission_gate` | passed |

The completed full jobs, if later authorized, must still satisfy the stricter diagnostic contract, including Raman per-step closure p99 <1e-3, cumulative closure <5e-3, and total-energy closure limits.
