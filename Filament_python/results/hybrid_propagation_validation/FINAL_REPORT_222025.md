# Hybrid Propagation 0.60 m validation — final result (job 222025)

## Outcome

- Mechanical classification: `hybrid_0p60_not_supported`.
- Human final classification: **`hybrid_0p60_partially_supported_for_acceleration`**.
- Human verdict: **partial pass** (visual review approved, no visual veto).

The strict G1/G2/G3 and numerical-health thresholds failed, but the single
forward effect is modest and well understood. This configuration is accepted
for **low-accuracy acceleration use**, not as a strict-equivalence replacement
for the full `z=0` nonlinear reference.

Reviewer's decision (verbatim):

> 人工视觉我赞成部分通过。0.6 m 开始非线性计算会造成一定的影响，会让电子密度峰值
> 降低以及成丝峰值位置后延，但是在精度要求不高的情况下，可以通过此做法来加速计算。

## Provenance

- Execution SHA: `5ce3be1e4a74eff71dee219116e9a2f29aa3b34b`.
- Branch: `codex/hybrid-propagation-validation`.
- Job: `222025` on `m4gn1401`, partition `gpu`, 1 GPU (RTX 5090), 8 CPU.
- Scheduler: `COMPLETED / 0:0`, elapsed `07:35:00`.
- Run directory:
  `/data/run01/scvi806/user_Wangjimin/hybrid_propagation_validation_0p60/run_5ce3be1_20260823T143958Z`.
- Postprocess gate: `complete_evidence`; 15,000 finite z records per case;
  raw NPZ remains in the HPC run directory (not committed).

## Mechanical gates

- Failed: `G1_onset_1e22, G2_peak_rho, G3_rho_topology, G3_intensity_topology,
  G3_curve, G3_low_threshold_risk, numerical_health`.
- Passed: `performance, visual_veto`.

## Accepted physical effect

Activating nonlinear propagation at `0.60 m` causes:

- Electron-density peak `6.36147e+22 -> 6.09979e+22 m^-3` (`-4.11%`),
  peak position `z=0.8070 -> 0.8105 m` (`+0.35 cm` downstream).
- `rho=1e22 m^-3` onset shift `+0.3053 cm` (delayed).
- Intensity peak position `z=0.8349 -> 0.8390 m` (`+0.41 cm` downstream).
- A secondary density feature appears near `z ~1.19 m` (`rho` peak count
  `1 -> 2`), the direct cause of the G3 topology failures.

These are consistent with the reviewer's interpretation: starting nonlinearity
at `0.60 m` lowers the density peak and postpones the filamentation peak.

## Numerical health note

The comparison `numerical_health` gate failed only because the hybrid energy
diagnostic drift `0.0632302` exceeds the reference `0.0630072` by `2.2e-4`
(beyond the strict `1e-12` relative tolerance). All raw diagnostics are finite
(postprocess returned `complete_evidence`), adaptive-rejection and
safety-trigger counters are zero in both cases, and the linear preamble
deposits zero energy. This is a threshold exceedance, not a NaN/Inf or
divergence condition.

## Measured acceleration

- Case wall time `16500.02 -> 10739.61 s` (`-34.91%`, speed-up `1.536x`).
- Step time reduction `35.24%` (step speed-up `1.544x`).
- Nonlinear/ionization call count `15000 -> 9000` each (`-6000` each).
- Raman substeps `30000 -> 18000` (`-12000`).
- Raman convolutions `60000 -> 36000` (`-24000`).
- GPU peak allocated/reserved memory unchanged
  (`15321817600` / `28612113408` bytes).

## Conclusion

This result is not a strict-equivalence pass: the predefined G1/G2/G3 and
numerical-health thresholds remain violated. It is, however, accepted as a
final partial pass for low-accuracy acceleration use, where the documented
`~4%` peak-density reduction, `~0.35 cm` filamentation-peak delay, and
secondary downstream density feature are acceptable trade-offs for the
`~1.54x` wall-time speed-up.
