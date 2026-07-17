# N₂/O₂ ionization-rate model validation

## Decision

**Classification: `supported`.** Both pulse widths show a same-direction >=10% fixed-threshold intensity shift in the 1e20–1e21 m^-3 onset band.

This is a local CPU/0D comparison. It does not convert the result into a centimetre-scale axial shift and does not identify Talebpour as PyCAP's internal model. It only prioritizes (or deprioritizes) the present Popruzhenko-versus-Talebpour model difference for a later controlled propagation check.

## Scope and reproducibility

- Code commit evaluated: `609c4f56bc5f3554079ace02aff7e103bd72f7cc`.
- FT90 configuration: `Filament_python/configs/profile_validation/flat_top_90_40fs.json` (SHA256 `dcc9761c71a3ae3f1da3898a4d2070f89a04dfd4c393b31d83676729d6eec20d`).
- Intensity scan: 1.0e+14–1.0e+19 W/m², 501 log-spaced points.
- Local-density pulse grids: 40 fs and 120 fs, both with production Nt=384/Twin=960 fs; the primary solution is the no-recombination trapezoid cumulative reference. Production `evolve_rho_time` RK4 is a consistency check.
- Paths in this archive are repository relative. No 3D propagation or Slurm submission was performed.

## Rate parameters and LUT accuracy

The production Popruzhenko species are N₂ (`Ip_eV=15.6`, `Z=1`) and O₂ (`Ip_eV=12.1`, `Z=1`). The runtime Talebpour comparator resolves N₂ to `Ip_eV_eff=15.6`, `Zeff=0.9`, and O₂ to `Ip_eV_eff=12.55`, `Zeff=0.53`; both retain `l=0`, `m=0`. The production phase sampling is 32, while LUT tables use their configured reference sampling of 64; `W_cap=1e19 s^-1`.

LUT accuracy is tested against each table's actual 64-sample reference evaluator, separately from the physical Popruzhenko-versus-Talebpour comparison. The relevant range is 1e+16–1e+18 W/m²; the acceptance rule is max relative error <= 3%.

- N2_popruzhenko: pass; relevant-window max relative error = 1.6566%.
- N2_talebpour: pass; relevant-window max relative error = 0.9359%.
- O2_popruzhenko: pass; relevant-window max relative error = 0.5784%.
- O2_talebpour: pass; relevant-window max relative error = 0.3953%.

## Physical-model difference

- N₂ maximum absolute rate difference in the relevant interval: 94.9% (1.294 decades).
- O₂ maximum absolute rate difference in the relevant interval: 340.6% (0.644 decades).
- The local low-density onset is O₂-dominated: across 10^19–10^21 m^-3, O₂ contributes 97.79%–99.92% of the total density at each model's own threshold intensity.

## Fixed-density threshold map

- 40 fs, 1e+19 m^-3: Pop/Tal threshold-intensity ratio = 0.9960, Δlog10 I = -0.0018.
- 40 fs, 1e+20 m^-3: Pop/Tal threshold-intensity ratio = 0.9410, Δlog10 I = -0.0264.
- 40 fs, 1e+21 m^-3: Pop/Tal threshold-intensity ratio = 0.8593, Δlog10 I = -0.0658.
- 40 fs, 1e+22 m^-3: Pop/Tal threshold-intensity ratio = 0.9005, Δlog10 I = -0.0455.
- 120 fs, 1e+19 m^-3: Pop/Tal threshold-intensity ratio = 1.0166, Δlog10 I = 0.0072.
- 120 fs, 1e+20 m^-3: Pop/Tal threshold-intensity ratio = 0.9711, Δlog10 I = -0.0127.
- 120 fs, 1e+21 m^-3: Pop/Tal threshold-intensity ratio = 0.8703, Δlog10 I = -0.0603.
- 120 fs, 1e+22 m^-3: Pop/Tal threshold-intensity ratio = 0.9260, Δlog10 I = -0.0334.

At 10^21 m^-3, the Popruzhenko/Talebpour intensity ratios are below one for both pulse widths (40 fs: 0.8593; 120 fs: 0.8703). This is a same-direction local response difference above the 10% high-priority screen.

## Numerical consistency

- Maximum production-RK4 versus cumulative-reference final-density error over the direct relevant probes: 1.936e-10.
- Maximum 1× versus 8× cumulative-reference final-density difference over the stated probes: 0.278%.

## Causal interpretation and next action

The result supports the ionization-rate-model difference as a high-priority candidate for the observed common rising-front/peak shift, because both 40 fs and 120 fs show a consistent >10% threshold shift at 10^21 m^-3 and the onset is overwhelmingly O₂-controlled. This is not proof of the reported approximately -3.270 cm (40 fs) or -2.589 cm (120 fs) rising-front advance, nor of the approximately -2.9 cm common peak-centre shift: propagation feedback remains untested in this phase.

Recommended new full-propagation cases: 2.

- 120 fs: current Popruzhenko full model versus Talebpour full-model control
- 40 fs: Talebpour full-model control only after the 120 fs comparison confirms a material propagation effect

Production physics defaults were unchanged; no Slurm jobs were submitted.
