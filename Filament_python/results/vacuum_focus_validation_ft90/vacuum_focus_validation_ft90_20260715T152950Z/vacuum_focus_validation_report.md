# FT90 vacuum-focus validation

Coordinate convention is fixed for every axial quantity:

`x_focus = 100 * (z - 0.95) cm`.

No intensity maximum, density maximum, or post-hoc translation defines zero.

## Result

- Parabolically interpolated `I_max` focus: `0.909260292 m`
- Relative to the 0.95 m geometric focus: `-4.0740 cm`
- Axial sampling half-step uncertainty: `0.0125 cm`
- Input sampled peak power: `1.7e+10 W`
- Maximum relative transverse-power drift: `2.583e-07`

## Interpretation

FT90 finite aperture / edge diffraction is a strong candidate for a material forward focus shift.

The propagated field is direct-from-lens angular-spectrum propagation, so the result has no axial stepping accumulation. All nonlinear, plasma, gas-dispersion, Raman, ionization, collision, recombination, absorption, and self-steepening terms are explicitly absent from this driver.
