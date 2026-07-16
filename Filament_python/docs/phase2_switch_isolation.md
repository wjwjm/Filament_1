# Phase 2 nonlinear-switch isolation smoke report

Run the following CPU-only check after changing a nonlinear switch, its diagnostics, or its configuration compatibility layer:

```powershell
python Filament_python/tools/validate_nonlinear_switch_isolation.py --out Filament_python/results/nonlinear_switch_isolation_report.json
```

The command creates only the JSON report. It uses temporary small-grid NPZ files internally and removes them before returning; it does not run Slurm, submit a job, or write a production result.

The report passes only if all of the following hold:

- Legacy/default full-model configuration agrees with explicit all-ON switches.
- Electronic Kerr OFF leaves raw `delta_n_elec_max_z` nonzero while `*_applied_*` is zero.
- Raman phase and Raman absorption can be disabled independently while their required raw diagnostics remain available.
- Plasma phase OFF retains nonzero `rho` and raw plasma phase while the applied phase is zero.
- Ionization loss OFF retains `rho` and `alpha_ion_raw_max_z` while `alpha_ion_applied_max_z` is zero.

This is a switch-isolation and regression test, not an ablation-production result. The 8×8×16, 0.2 mm setup uses the MPA smoke species solely to make each diagnostic channel observable.
