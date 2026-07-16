# FT90 profile-definition cause closure: converged vacuum result

All axial coordinates are fixed as `x_focus = 100*(z-0.95) cm`.  The zero is the 0.95 m geometric thin-lens focus; no intensity or electron-density peak is used to recenter a curve.

## Locked density reference

The Fig. 5(b) source-locked digitization gives fixed-density rising-edge shifts of **−2.5889 cm** (120 fs) and **−3.2696 cm** (40 fs).  Both curves retain separately reported post-peak broadening; that broadening cannot by itself cause an earlier rising crossing.

## Numerical status

All 25 linear-vacuum cases completed on GPU Slurm arrays `172780`, `172791`, and `172819`.

- **Spatial resolution:** P1 512²/8 mm versus 1024²/8 mm differs by −0.0789 cm, passing the 0.1 cm gate.
- **Window convergence:** 10→12 mm did not close for every profile, so 14 mm was used as specified.  For every P1--P6 case, 12→14 mm absolute-focus and relative-to-P1 differential changes are below 0.1 cm.
- **Independent algorithm:** at the final 14 mm window, every FFT on-axis focus agrees with the continuous axisymmetric Fresnel result within 0.026 cm; every relative-to-P1 FFT/Fresnel differential agrees within 0.026 cm.
- **Axisymmetry diagnostic:** FFT `I_max` and FFT on-axis focus agree within 0.1 cm for every final-window case.

Thus `resolution_convergence_ok`, `window_convergence_ok`, and `independent_fresnel_crosscheck_ok` are all true.

## Final differential focus evidence

| Candidate | Δx_vac vs P1 (cm) | epsilon_120 (cm) | epsilon_40 (cm) |
| --- | ---: | ---: | ---: |
| P2 | −2.089 | −4.678 | −5.359 |
| P3 | −2.754 | −5.343 | −6.023 |
| P4 | +0.487 | −2.101 | −2.782 |
| P5 | −1.334 | −3.923 | −4.603 |
| P6 | +0.038 | −2.551 | −3.232 |

P4 has the largest downstream shift, but it is only +0.487 cm and cannot close either pulse-width offset.  P6 is positive by only +0.038 cm.  The remaining candidates move in the wrong (upstream) direction.

## Final classification

**not_supported**.

After window convergence and independent continuous-Fresnel verification, the tested P1--P6 FT90 mathematical definitions do not generate a downstream differential vacuum focus large enough to compensate the approximately 2.6--3.3 cm electron-density rising-edge advance.  This does not claim to reproduce PyCAP or infer PyCAP's unknown vacuum-focus baseline.

## Consequence

Do not further tune the FT90 edge definition to explain this offset.  If a subsequent controlled nonlinear test is authorized, its purpose should be to isolate nonlinear self-focusing and temporal nonlinear-response contributions; no complete nonlinear filamentation job was launched in this task.
