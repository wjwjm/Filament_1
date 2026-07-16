# FT90 profile-definition cause closure

All coordinates are fixed as `x_focus = 100*(z-0.95) cm`.  The origin is the 0.95 m geometric thin-lens focus; neither the vacuum-intensity maximum nor either plasma-density maximum is used to recenter a curve.

## Repaired PyCAP comparison

- 120 fs: the mean fixed-density rising-edge offset is **−2.589 cm**.  The peak-interval-centre offset is −2.875 cm.  The shape is classified as **translation plus post-peak broadening**: FWHM ratio 1.571 and a 69.9% residual-RMSE improvement when scale is added.
- 40 fs: the mean fixed-density rising-edge offset is **−3.270 cm**.  The peak-interval-centre offset is −2.978 cm.  The shape is classified as **translation plus post-peak broadening**: FWHM ratio 1.754 and a 72.0% residual-RMSE improvement when scale is added.
- Therefore, the repaired data retain an approximately rigid early rising-edge shift while also showing a separately reported post-peak tail broadening.  The latter cannot by itself cause an earlier rising crossing.

## Differential linear-vacuum scan

P1 (the production FT90 definition) gives `x_vac = −4.0740 cm` with direct 2-D angular-spectrum propagation.  Relative to P1, all tested alternative definitions move the focus in the wrong direction:

| Profile | Δx_vac versus P1 (cm) |
| --- | ---: |
| P2: zero at 0.9R, narrow cosine | −2.914 |
| P3: zero at 0.9R, wide cosine | −2.929 |
| P4: hard top to R | −0.561 |
| P5: hard top to 0.9R | −1.115 |
| P6: P2 cosine with P1-matched second moment | −0.610 |

No candidate produces the positive 2.5–3.1 cm vacuum-focus displacement needed to compensate the repaired 120 fs and 40 fs rising-edge shifts.

## Numerical gate and final classification

- 512²/8 mm versus 1024²/8 mm: **0.0035 cm**, passing the 0.1 cm high-resolution check.
- 512²/8 mm versus 640²/10 mm (matched transverse spacing): **−0.4286 cm**, failing the required 0.1 cm window check.

The final classification is therefore **inconclusive**, not a physical endorsement or rejection of FT90 edge physics: the prescribed baseline-window convergence gate fails.  The nine distinct cases were completed as GPU Slurm array **172588** on `scvi806@nc-n50r5` (submitted 2026-07-16 02:05:52 UTC); only CSV/JSON diagnostics were downloaded and committed.  No full nonlinear filamentation job was submitted.

## Next action

Do not change the FT90 edge, nonlinear coefficients, or production filament parameters on the basis of this scan.  First resolve the 8 mm versus 10 mm linear-window discrepancy.  Only after that quality gate passes should a single controlled nonlinear validation be considered; the only candidate parameter allowed to change would be the documented transverse-profile mathematical definition.
