# Ionization time-integrator validation report

- Code SHA: `e1023da2445a9ee975dcb5eb9b1a8ee382738ca2`
- Generated (UTC): 2026-07-16T17:21:43.036846+00:00
- Production path: `make_Wfunc` → `evolve_rho_time` with the supplied configuration evaluator (LUT/reference as configured).
- Temporal refinements: 1, 2, 4, 8
- Fixed rise threshold: 1.000e+20 m^-3
- `tau_fwhm` is the intensity FWHM; `gaussian_pulse_t` produces the field envelope and the rate receives `I(t)` in W/m².

## Test configuration

- 40 fs: `D:\Filament_1\.codex_worktrees\phase2-phase3\Filament_python\configs\profile_validation\flat_top_90_40fs.json`; Nt=384, Twin=960.0 fs, dt=2.500 fs; species=[{"name": "N2", "rate": "popruzhenko_atom_i_lut", "reference_model": "popruzhenko_atom_i_full_reference", "Ip_eV": 15.6, "Z": 1, "l": 0, "m": 0, "fraction": 0.8}, {"name": "O2", "rate": "popruzhenko_atom_i_lut", "reference_model": "popruzhenko_atom_i_full_reference", "Ip_eV": 12.1, "Z": 1, "l": 0, "m": 0, "fraction": 0.2}]
- 120 fs: `D:\Filament_1\.codex_worktrees\phase2-phase3\Filament_python\configs\profile_validation\flat_top_90_120fs.json`; Nt=384, Twin=960.0 fs, dt=2.500 fs; species=[{"name": "N2", "rate": "popruzhenko_atom_i_lut", "reference_model": "popruzhenko_atom_i_full_reference", "Ip_eV": 15.6, "Z": 1, "l": 0, "m": 0, "fraction": 0.8}, {"name": "O2", "rate": "popruzhenko_atom_i_lut", "reference_model": "popruzhenko_atom_i_full_reference", "Ip_eV": 12.1, "Z": 1, "l": 0, "m": 0, "fraction": 0.2}]

## Production-dt density errors

| tau (fs) | Ipeak (W/m²) | species | final rho error | time-max rho error | rise error (fs) | max(Wdt) | step clips | intermediate violations |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 1.000e+15 | N2 | 1.268e-26 | 1.949e-20 | nan | 3.989e-34 | 0 | 0 |
| 40 | 1.000e+15 | O2 | 9.662e-17 | 2.396e-10 | nan | 2.378e-23 | 0 | 0 |
| 40 | 1.000e+15 | total | 9.662e-17 | 2.396e-10 | nan | 2.378e-23 | 0 | 0 |
| 40 | 3.000e+15 | N2 | 1.622e-20 | 3.247e-15 | nan | 6.657e-29 | 0 | 0 |
| 40 | 3.000e+15 | O2 | 1.059e-11 | 1.439e-06 | nan | 1.434e-19 | 0 | 0 |
| 40 | 3.000e+15 | total | 1.059e-11 | 1.439e-06 | nan | 1.434e-19 | 0 | 0 |
| 40 | 1.000e+16 | N2 | 4.079e-14 | 1.600e-09 | nan | 3.293e-23 | 0 | 0 |
| 40 | 1.000e+16 | O2 | 2.276e-07 | 3.244e-03 | nan | 1.788e-15 | 0 | 0 |
| 40 | 1.000e+16 | total | 2.276e-07 | 3.244e-03 | nan | 1.788e-15 | 0 | 0 |
| 40 | 3.000e+16 | N2 | 1.875e-08 | 2.176e-04 | nan | 4.515e-18 | 0 | 0 |
| 40 | 3.000e+16 | O2 | 2.076e-06 | 3.049e-03 | nan | 7.078e-12 | 0 | 0 |
| 40 | 3.000e+16 | total | 2.076e-06 | 3.049e-03 | nan | 7.078e-12 | 0 | 0 |
| 40 | 1.000e+17 | N2 | 1.229e-06 | 4.394e-03 | nan | 1.311e-12 | 0 | 0 |
| 40 | 1.000e+17 | O2 | 5.926e-05 | 3.422e-03 | nan | 1.028e-08 | 0 | 0 |
| 40 | 1.000e+17 | total | 5.924e-05 | 3.423e-03 | nan | 1.028e-08 | 0 | 0 |
| 40 | 3.000e+17 | N2 | 2.183e-03 | 4.747e-03 | nan | 1.355e-08 | 0 | 0 |
| 40 | 3.000e+17 | O2 | 3.097e-05 | 2.750e-03 | -9.276e-02 | 3.080e-05 | 0 | 0 |
| 40 | 3.000e+17 | total | 3.429e-05 | 2.752e-03 | -9.267e-02 | 3.080e-05 | 0 | 0 |
| 40 | 1.000e+18 | N2 | 7.603e-04 | 3.398e-03 | -2.463e-01 | 1.074e-04 | 0 | 0 |
| 40 | 1.000e+18 | O2 | 2.450e-03 | 4.011e-03 | -1.726e-01 | 1.314e-02 | 0 | 0 |
| 40 | 1.000e+18 | total | 2.359e-03 | 3.891e-03 | -1.729e-01 | 1.314e-02 | 0 | 0 |
| 120 | 1.000e+15 | N2 | 5.317e-26 | 6.450e-21 | nan | 3.989e-34 | 0 | 0 |
| 120 | 1.000e+15 | O2 | 1.048e-15 | 8.178e-11 | nan | 2.378e-23 | 0 | 0 |
| 120 | 1.000e+15 | total | 1.048e-15 | 8.178e-11 | nan | 2.378e-23 | 0 | 0 |
| 120 | 3.000e+15 | N2 | 2.476e-20 | 1.074e-15 | nan | 6.657e-29 | 0 | 0 |
| 120 | 3.000e+15 | O2 | 1.801e-11 | 4.917e-07 | nan | 1.434e-19 | 0 | 0 |
| 120 | 3.000e+15 | total | 1.801e-11 | 4.917e-07 | nan | 1.434e-19 | 0 | 0 |
| 120 | 1.000e+16 | N2 | 5.321e-14 | 5.296e-10 | nan | 3.293e-23 | 0 | 0 |
| 120 | 1.000e+16 | O2 | 1.217e-07 | 3.704e-04 | nan | 1.788e-15 | 0 | 0 |
| 120 | 1.000e+16 | total | 1.217e-07 | 3.704e-04 | nan | 1.788e-15 | 0 | 0 |
| 120 | 3.000e+16 | N2 | 3.435e-08 | 7.201e-05 | nan | 4.515e-18 | 0 | 0 |
| 120 | 3.000e+16 | O2 | 1.331e-06 | 3.471e-04 | nan | 7.078e-12 | 0 | 0 |
| 120 | 3.000e+16 | total | 1.331e-06 | 3.471e-04 | nan | 7.078e-12 | 0 | 0 |
| 120 | 1.000e+17 | N2 | 8.308e-07 | 4.853e-04 | nan | 1.311e-12 | 0 | 0 |
| 120 | 1.000e+17 | O2 | 4.724e-05 | 4.097e-04 | nan | 1.028e-08 | 0 | 0 |
| 120 | 1.000e+17 | total | 4.722e-05 | 4.097e-04 | nan | 1.028e-08 | 0 | 0 |
| 120 | 3.000e+17 | N2 | 4.819e-04 | 6.996e-04 | nan | 1.355e-08 | 0 | 0 |
| 120 | 3.000e+17 | O2 | 3.186e-05 | 3.323e-04 | -4.646e-02 | 3.080e-05 | 0 | 0 |
| 120 | 3.000e+17 | total | 3.255e-05 | 3.325e-04 | -4.641e-02 | 3.080e-05 | 0 | 0 |
| 120 | 1.000e+18 | N2 | 2.136e-05 | 3.796e-04 | -1.183e-01 | 1.074e-04 | 0 | 0 |
| 120 | 1.000e+18 | O2 | 2.939e-04 | 4.964e-04 | -9.055e-02 | 1.314e-02 | 0 | 0 |
| 120 | 1.000e+18 | total | 2.854e-04 | 4.857e-04 | -9.033e-02 | 1.314e-02 | 0 | 0 |

## Decision gates

- `not_supported`: every meaningful non-saturated production-dt case has final and time-history errors below 1%, fixed-threshold rise-time error below 0.5 fs (when crossed), and no pre-clip violation.
- `inconclusive`: no severe failure, but a meaningful case is in the 1%–5% warning band or lacks a decisive threshold crossing.
- `supported`: a meaningful non-saturated case exceeds 5%, exceeds 0.5 fs in rise time, or shows a pre-clip violation.

## Causal conclusion

**not_supported** — All meaningful non-saturated production-dt cases satisfy the 1%/0.5 fs gates without clipping.

Automatic meaningful intensity interval: 3.000e+17 to 1.000e+18 W/m²; non-saturated cases: 4; clip cases: 0.

This conclusion concerns whether the current fixed-step RK4 can be the principal cause of the observed filament electron-density onset/peak/tail discrepancy. It does not change the production integrator.
