# Phase 8B-P controlled-propagation preflight report

## Decision

- `full_job_submission_gate`: **passed**
- Phase 8B-R executed: **false**
- Full 1.3 m Slurm jobs submitted: **0**
- GitHub Actions CI evidence: **unavailable**
- Required next action: merge this preflight, then obtain explicit user approval before preparing Job 1.

## Full-grid smoke evidence

- Short smoke Job IDs: 179288, 179311 (both `COMPLETED 0:0`).
- GPU: NVIDIA GeForce RTX 5090; grid: 512x512x384; 20 z steps per case.
- Peak reserved GPU memory: 83.783% (threshold <85%).
- Mean ON step time: 1.022286 s; projected 15000-step runtime: 4.260 h.
- Requested 8 h fraction: 53.244%; slowdown vs legacy: 2.048x.
- ON convolution count: 2 per Heun application and 4 per Strang z step.
- OFF raw diagnostic convolution count: 1 per z step.
- ON Raman cumulative closure residual: 0.000360067.

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
