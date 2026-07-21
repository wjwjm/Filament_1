# Phase 8A.1 final report

## Decision

- Selected Raman architecture: `ready_full_operator`
- Propagation admission gate: `passed`
- Failed/inconclusive gates: production_split_comparison_gate
- Phase 8B executed: false
- New Slurm jobs submitted: 0
- Full 40/120 fs three-dimensional propagation rerun: false
- Production non-Raman physics changed: false

The failed legacy production split comparison is not an admission prerequisite after selection of the independently verified full Eq. (27) architecture. Phase 8B still requires separate user approval.

## Numerical highlights

- FFT float64 criterion passed: `True`
- FFT float32 criterion passed: `True`
- Full/reference RHS relative error: `4.743516664105255e-10`
- Full/reference Heun-step relative error: `2.430969602114687e-13`
- Local energy closure status: `passed`
- Global energy closure status: `passed`
- Full local pytest status: `passed`

## Gates

| Gate | Status | Comparison |
| --- | --- | --- |
| `source_equation_mapping_gate` | `passed` | `all` |
| `parameter_boundary_gate` | `passed` | `all` |
| `configuration_ambiguity_gate` | `passed` | `all` |
| `gate_generator_integrity_gate` | `passed` | `all` |
| `time_derivative_sign_gate` | `passed` | `all` |
| `tdiff_fft_consistency_gate` | `passed` | `all` |
| `kernel_normalization_gate` | `passed` | `le` |
| `fft_linear_convolution_gate` | `passed` | `all` |
| `iir_convergence_gate` | `passed` | `all` |
| `eq10_signed_energy_gate` | `passed` | `all` |
| `eq11_analytic_recovery_gate` | `passed` | `lt` |
| `operator_prefactor_gate` | `passed` | `all` |
| `production_split_comparison_gate` | `failed` | `all` |
| `full_operator_reference_gate` | `passed` | `all` |
| `no_double_counting_gate` | `passed` | `all` |
| `local_energy_closure_gate` | `passed` | `lt` |
| `global_energy_closure_gate` | `passed` | `lt` |
| `dz_convergence_gate` | `passed` | `all` |
| `full_pytest_gate` | `passed` | `all` |
| `propagation_admission_gate` | `passed` | `all` |

The `production_split_comparison_gate` remains failed because the real split source exceeds the locked thresholds for 40 fs TL and both chirped pulses. The candidate does not use that architecture; it uses the independently validated, opt-in full operator.
