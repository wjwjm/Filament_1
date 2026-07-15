# FT90 focus-shift diagnosis: merged vacuum and nonlinear-curve evidence

All coordinates are permanently `x_focus = 100 * (z - 0.95) cm`; no peak, intensity maximum, or fitted translation defines zero.

## Independent vacuum test

- Direct-from-lens angular-spectrum FT90 vacuum focus: **-4.0740 cm** relative to the 0.95 m geometric focus.
- Parabolic interpolation sampling uncertainty: 0.0125 cm; maximum transverse-power drift: 2.583e-07.
- This satisfies the predeclared `x_vac,peak <= -2 cm` criterion: finite aperture / FT90 edge diffraction is a strong candidate for the early shift.

## Existing nonlinear FT90 curves versus paper PyCAP

- 120 fs: fixed-absolute-density rising edge is -2.494 cm early; classification is **translation_plus_broadening** (medium confidence).
  The vacuum value is -1.580 cm more forward than that leading-edge shift; FWHM ratio is 1.386 and adding a scale parameter improves RMSE by 67.6%.
- 40 fs: fixed-absolute-density rising edge is -3.059 cm early; classification is **translation_plus_broadening** (medium confidence).
  The vacuum value is -1.015 cm more forward than that leading-edge shift; FWHM ratio is 1.751 and adding a scale parameter improves RMSE by 71.1%.

## Final interpretation

The vacuum focus is strongly and independently shifted forward in the same direction as both nonlinear leading edges. The 40 fs and 120 fs leading-edge shifts differ by only about 0.57 cm, which supports a shared transverse-optical contribution rather than a solely pulse-duration-specific temporal mechanism.

However, the full profiles are not pure translations: both have materially broader FWHM/tails, and the translation-plus-scale fit reduces the residual by roughly 68-71%. The paper 120 fs trace also has a flat peak plateau, so its single peak coordinate is less reliable than its threshold crossings. Thus the verified statement is: **FT90 finite-aperture/edge diffraction is the most reliable primary explanation for the common 2.5-3.1 cm early leading edge, while a residual nonlinear-shape difference remains.**

## Is another full nonlinear run needed?

Not to establish the vacuum offset: task one already does that. It is needed only for causal closure of the residual width/tail: run one FT90 nonlinear control in which only the lens/wavefront geometry is corrected against the measured vacuum focus, then reuse the same fixed coordinate and compare the entire density curves. Keep the current 512² grid, 8 mm window, 17 GW, FT90 profile, n2, Raman, and ionization settings unchanged.
