# FT90 density curve: translation versus broadening

All curves use the permanent coordinate `x_focus = 100 * (z - 0.95) cm`; neither peak nor onset was aligned.

## Evidence sources

- Paper: Isaacs et al. (2022), Fig. 5(b), digitized PyCAP traces with the saved pixel calibration and colour-selection metadata.
- Simulation: downloaded FT90 `rho_max_z` NPZ files, converted as `rho_e / 1e22` to `10^16 cm^-3`.

## 120 fs

- Classification: **translation_plus_broadening** (confidence: **medium**).
- Mean absolute rising-edge shift: -2.494 cm; spread across available levels: 0.064 cm.
- Peak shift (current FT90 minus paper PyCAP): -5.880 cm.
- FWHM ratio (current/paper): 1.3860718645002377; 10-90% rising-width ratio: 1.067087110629821.
- Translation-only fit: Δx = -2.224 cm, RMSE = 0.933.
- Translation+scale fit: x_c = 3.355 cm, s = 1.443, RMSE improvement = 67.6%.
- Bootstrap 95% CI for translation Δx: [-3.002, -2.333] cm.

The paper 120 fs trace has a visibly flat high-density plateau. Its single `argmax` is therefore digitization-sensitive; the fixed absolute rising-edge shifts are the more stable translation evidence.

## 40 fs

- Classification: **translation_plus_broadening** (confidence: **medium**).
- Mean absolute rising-edge shift: -3.059 cm; spread across available levels: 0.231 cm.
- Peak shift (current FT90 minus paper PyCAP): -3.027 cm.
- FWHM ratio (current/paper): 1.751302894554255; 10-90% rising-width ratio: 0.8617954020761212.
- Translation-only fit: Δx = -2.578 cm, RMSE = 0.591.
- Translation+scale fit: x_c = 2.743 cm, s = 1.592, RMSE improvement = 71.1%.
- Bootstrap 95% CI for translation Δx: [-3.009, -2.850] cm.

For the translation+scale model, `x_c` is a model coordinate parameter. When `s != 1`, it is not directly interchangeable with a feature-by-feature translation; use the reported fixed-threshold shifts for that comparison.

## Provisional physical implication

This report classifies the existing nonlinear curves only. The vacuum-focus job is the independent optical test: its result decides whether a measured translation can be attributed primarily to FT90 finite-aperture/edge diffraction or instead leaves nonlinear self-focusing and ionization-tail mechanisms as the leading candidates.

## Digitization caveat

The paper curves are raster-digitized, not author-supplied data. The stored bootstrap includes ±0.15 cm horizontal and ±0.05 in `10^16 cm^-3` vertical reading uncertainty; conclusions should therefore be treated as quantitative with this image-resolution caveat.
