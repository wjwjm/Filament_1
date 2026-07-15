# Transverse profile validation: Gaussian vs FT90

## Status

- Technical: completed
- Quality gates: passed
- Interpretation: controlled comparison only; no experimental reference curve was supplied.

## Input normalization

| Case | Profile | Peak power (W) | Peak intensity (W/m²) | Effective area (m²) | r50 (m) | r90 (m) | Boundary I fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian, 120 fs | gaussian | 17000000000.000006 | 2763652403162880.0 | 6.1512800888216775e-06 | 0.0011644556802429193 | 0.0021236783573848938 | 0.00030142939569711367 |
| FT90, 120 fs | flat_top_cosine | 17000000000.000002 | 1530144103581916.0 | 1.1110064705804298e-05 | 0.0013295029616364154 | 0.001783715399103792 | 0.0 |

## Filament metrics

| Case | Status | z_on (m) | z_peak (m) | rho_peak (m⁻³) | z_end (m) | Length (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian, 120 fs | detected | 0.7789000272750854 | 0.8140000104904175 | 5.17310249617558e+22 | 0.9873499870300293 | 0.20844995975494385 |
| FT90, 120 fs | detected | 0.7738000154495239 | 0.8008000254631042 | 5.9098260930187965e+22 | 1.2963500022888184 | 0.5225499868392944 |

## Outputs

- reports/input_profiles.png
- comparison/comparison_overview.png
- comparison/rho_onaxis_max_z.png
- comparison/rho_max_z.png
- comparison/I_max_z.png
- comparison/fwhm_plasma_z.png

## Interpretation limit

No experimental reference curve is supplied; the report quantifies differences only and does not assign causal importance.
