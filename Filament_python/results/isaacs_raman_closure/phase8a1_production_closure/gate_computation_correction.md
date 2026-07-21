# Gate computation correction

Phase 8A incorrectly took a `relative_error` value from `eq10_eq11_validation.csv` and labeled it as an FFT/direct error. It also allowed literal `passed` states that were not derived from a threshold comparison.

Phase 8A.1 uses independent contracts:

- FFT/direct: `raman_fft_direct_comparison.csv::relative_linf_error`.
- Eq. (10)/(11): `eq10_eq11_validation_v2.csv::direct_vs_eq11_error`.
- IIR/direct: `raman_iir_direct_convergence.csv::iir_vs_direct_error`.
- Production operator: `production_split_vs_full_operator.csv` with waveform-specific thresholds.

Every numerical status is derived by `threshold_gate` or a named boolean contract. Values and thresholds must be finite. Missing files, missing fields, NaN, and Inf are `inconclusive`, never passing. The float32 impulse wrap-around flag is evaluated relative to the response peak using the float32 acceptance tolerance; this distinguishes roundoff from causal wrap-around.
