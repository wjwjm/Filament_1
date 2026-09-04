# HR-4E-2C Mapping-vs-Evolution Adjudication

- Status: `WARNING`
- Decision: `A3_SUPPLEMENTARY_SPATIAL_LEVEL_REQUIRED`

## front

| Observable | Initial status | Evolution trend | Category | Fine tolerance fraction |
|---|---|---|---|---:|
| xc_m | INIT_NEAR_ZERO_NA | N/A_NEAR_ZERO | NO_FINAL_WARNING | 3.491e-16 |
| yc_m | INIT_MONOTONIC | N/A_NEAR_ZERO | NO_FINAL_WARNING | 9.15165e-08 |
| sigma_x_m | INIT_NONMONOTONIC_WITHIN_TOLERANCE | WARNING | WARNING_MIXED_OR_AMBIGUOUS | 0.0945418 |
| sigma_y_m | INIT_NONMONOTONIC_WITHIN_TOLERANCE | WARNING | WARNING_MIXED_OR_AMBIGUOUS | 0.0678405 |
| min_delta_n | INIT_MONOTONIC | PASS | WARNING_MIXED_OR_AMBIGUOUS | 0.26327 |
| max_abs_vx_m_s | INIT_NEAR_ZERO_NA | N/A_NEAR_ZERO | NO_FINAL_WARNING | 0 |
| max_abs_vy_m_s | INIT_NEAR_ZERO_NA | WARNING | WARNING_HYDRO_EVOLUTION_DOMINATED | 0.307686 |
| max_abs_v_m_s | INIT_NEAR_ZERO_NA | WARNING | WARNING_HYDRO_EVOLUTION_DOMINATED | 0.307686 |
| M0_negative_index_m2 | INIT_MONOTONIC | PASS | NO_FINAL_WARNING | 1.72128e-05 |

## peak

| Observable | Initial status | Evolution trend | Category | Fine tolerance fraction |
|---|---|---|---|---:|
| xc_m | INIT_NEAR_ZERO_NA | N/A_NEAR_ZERO | NO_FINAL_WARNING | 3.16825e-16 |
| yc_m | INIT_MONOTONIC | N/A_NEAR_ZERO | NO_FINAL_WARNING | 1.11648e-08 |
| sigma_x_m | INIT_NONMONOTONIC_WITHIN_TOLERANCE | WARNING | WARNING_MIXED_OR_AMBIGUOUS | 0.12175 |
| sigma_y_m | INIT_NONMONOTONIC_WITHIN_TOLERANCE | WARNING | WARNING_MIXED_OR_AMBIGUOUS | 0.114318 |
| min_delta_n | INIT_MONOTONIC | PASS | WARNING_MIXED_OR_AMBIGUOUS | 0.293913 |
| max_abs_vx_m_s | INIT_NEAR_ZERO_NA | N/A_NEAR_ZERO | NO_FINAL_WARNING | 0 |
| max_abs_vy_m_s | INIT_NEAR_ZERO_NA | WARNING | WARNING_HYDRO_EVOLUTION_DOMINATED | 0.341335 |
| max_abs_v_m_s | INIT_NEAR_ZERO_NA | WARNING | WARNING_HYDRO_EVOLUTION_DOMINATED | 0.341335 |
| M0_negative_index_m2 | INIT_MONOTONIC | PASS | NO_FINAL_WARNING | 0.000168237 |

## rear

| Observable | Initial status | Evolution trend | Category | Fine tolerance fraction |
|---|---|---|---|---:|
| xc_m | INIT_NEAR_ZERO_NA | N/A_NEAR_ZERO | NO_FINAL_WARNING | 5.71887e-16 |
| yc_m | INIT_NONMONOTONIC_WITHIN_TOLERANCE | N/A_NEAR_ZERO | NO_FINAL_WARNING | 2.57937e-08 |
| sigma_x_m | INIT_NONMONOTONIC_WITHIN_TOLERANCE | WARNING | WARNING_MIXED_OR_AMBIGUOUS | 0.0762651 |
| sigma_y_m | INIT_NONMONOTONIC_WITHIN_TOLERANCE | WARNING | WARNING_MIXED_OR_AMBIGUOUS | 0.0763484 |
| min_delta_n | INIT_MONOTONIC | PASS | WARNING_MIXED_OR_AMBIGUOUS | 0.281852 |
| max_abs_vx_m_s | INIT_NEAR_ZERO_NA | N/A_NEAR_ZERO | NO_FINAL_WARNING | 0 |
| max_abs_vy_m_s | INIT_NEAR_ZERO_NA | WARNING | WARNING_HYDRO_EVOLUTION_DOMINATED | 0.328848 |
| max_abs_v_m_s | INIT_NEAR_ZERO_NA | WARNING | WARNING_HYDRO_EVOLUTION_DOMINATED | 0.328848 |
| M0_negative_index_m2 | INIT_MONOTONIC | PASS | NO_FINAL_WARNING | 2.21517e-05 |

## Scope

This is a report-only adjudication using stored t=0 and 100 us scalar snapshots. It does not alter HR-4 physics, raw E2-C artifacts, the validation adapter, HR-5, or HR-4F.
