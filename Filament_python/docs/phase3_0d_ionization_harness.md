# Phase 3 Task 1: 0D ionization harness

`validate_ionization_time_integrator.py` validates only the temporal ionization chain. It does not construct a transverse field, call `propagate_one_pulse`, submit Slurm work, or alter the production integrator.

```powershell
python Filament_python/tools/validate_ionization_time_integrator.py --out-dir Filament_python/results/ionization_integrator_validation/task1_run
```

The default input configurations are the current FT90 40 fs and 120 fs production configurations. The default peak-intensity scan is `1e15` through `1e18 W/m^2`; the unit is always **W/m²**.

For every duration/intensity case, the harness writes these time series to `ionization_integrator_timeseries.npz`:

- `t_s`, `I_W_m2`
- `W_N2_s-1`, `W_O2_s-1`
- `rho_N2_m3`, `rho_O2_m3`, `rho_total_m3`

`ionization_integrator_cases.csv` additionally records `max(W_N2*dt)`, `max(W_O2*dt)`, final ionization fraction, and peak/final density. The evaluator and integrator are called through production `make_Wfunc` and `evolve_rho_time`; no rate formula is copied into the validation tool.

## Temporal convention

`gaussian_pulse_t(t, tau_fwhm)` returns the field envelope. The harness supplies the rate evaluator with

```text
I(t) = I_peak * |gaussian_pulse_t(t, tau_fwhm)|^2.
```

In the current implementation, `tau_fwhm` is the **intensity FWHM**. This follows directly from the field-envelope exponent used in `gaussian_pulse_t`; it is not a separate validation convention.

## RK4/reference comparison mode

```powershell
python Filament_python/tools/validate_ionization_time_integrator.py --compare-refinements --out-dir Filament_python/results/ionization_integrator_validation/comparison_run
```

Comparison mode recomputes the production Gaussian envelope and production `W(t)` evaluator on `dt`, `dt/2`, `dt/4`, and `dt/8` grids. It compares the production `evolve_rho_time(..., integrator='rk4')` result with a no-recombination per-species trapezoid reference. The optional exponential-average update is reported only as a candidate; it does not replace production RK4.

The resulting error CSV records final/peak/time-history density errors, fixed-threshold rise-time error, `max(W*dt)`, pre-clip RK4 extrema, intermediate-stage violations, and actual clip counts. Pre-clip statistics are returned only when `diagnose_integrator_stability=True`; ordinary production calls retain the original return shape and numerical result.
