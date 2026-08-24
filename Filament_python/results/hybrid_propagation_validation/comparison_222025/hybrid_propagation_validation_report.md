# Hybrid Propagation 0.60 m comparison

Classification: **hybrid_0p60_not_supported**

- Execution SHA: `5ce3be1e4a74eff71dee219116e9a2f29aa3b34b`
- GPU: `NVIDIA GeForce RTX 5090`
- Failed gates: `G1_onset_1e22, G2_peak_rho, G3_rho_topology, G3_intensity_topology, G3_curve, G3_low_threshold_risk, numerical_health`
- G1 onset shift: `0.30534815679126304` cm
- G2 peak relative difference: `0.0411356`
- Peak rho reference/hybrid: `6.36147e+22` / `6.09979e+22` m^-3
- Peak rho x_focus reference/hybrid/delta: `-14.3` / `-13.95` / `0.350004` cm
- I curve NRMSE/correlation: `0.0592682` / `0.986491`
- rho curve NRMSE/correlation: `0.0274074` / `0.993973`
- I/rho peak-count reference->hybrid: `2->2` / `1->2`
- Step/case wall-time reduction: `0.352402` / `0.349115`
- Step/case speedup: `1.54417` / `1.53637`
- Nonlinear/ionization/Raman-substep/Raman-convolution call reduction: `6000` / `6000` / `12000` / `24000`
- GPU peak allocated reference/hybrid: `15321817600` / `15321817600` bytes
- Visual veto: `False`

No coordinate shift, smoothing, or case-specific renormalization was applied; normalized-curve metrics use the fixed reference peak only.
Raw NPZ files remain in the HPC run directory and are not copied into this report.
