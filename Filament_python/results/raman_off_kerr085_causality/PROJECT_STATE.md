# Raman phase OFF + 0.85 electronic Kerr causal test

- Direct parent: the archived configuration snapshot from completed job `176915`,
  `results/raman_phase_causality/raman_phase_off_120fs_20260718T201000Z/120fs_talebpour_full_model_raman_phase_off.json`.
- Single physical variable relative to that parent: `beam.n2_air`, from `7.8e-24` to `6.63e-24 m^2/W`.
- Raman phase remains OFF; Raman convolution and absorption remain active and unchanged.
- Production defaults, existing configurations, and existing result artifacts are not modified.
- Exactly one complete 120 fs propagation job was submitted: Slurm `220822`.

## Execution

- Final state: `COMPLETED`, exit code `0:0`, node `m4gn1401`, elapsed `02:44:56`.
- Execution Git SHA: `ba730a28568eb46a811c418a040a09a302c60662`.
- Configuration SHA-256: `da9cbdecf0e231ab55e4f66d2ba0e20df9f76642bc0bca0f608ec84d754d5088`.
- NPZ SHA-256: `7719ccf9600049ff918f2167b8213e0e002be4172880e22ac953b56c4a9aa55b`; the NPZ remains on HPC and is not committed.
- The diagnostic gate passed with 15,000 finite z records, zero adaptive rejections, and zero safety triggers.

## Result

- The `1e22 m^-3` onset is `-13.393972 cm`, shifted `+0.619664 cm` from Raman phase OFF.
- The residual relative to historical f_R mixture is only `+0.001036 cm`, within the fixed `0.1 cm` criterion.
- This supports the causal statement that the historical mixture's additional onset delay relative to Raman phase OFF is primarily caused by its 15% reduction of instantaneous electronic Kerr, not by the delayed Raman phase response.
- This does not improve agreement with PyCAP: the candidate is `+0.633238 cm` later than PyCAP, while Raman phase OFF is only `+0.013574 cm` later.
- Candidate peak electron density is `2.03446e22 m^-3` at `-9.850 cm`; peak `I_max` is `5.43277e17 W/m^2` at `-7.830 cm`.
- Final pulse-energy change is `-3.40805%`. Full-axis density RMSE versus PyCAP is `2.44374e22 m^-3`, essentially identical to historical mixture and worse than Raman phase OFF.

Small postprocess and comparison artifacts are under `postprocess_220822/` and `comparison_220822/`.
