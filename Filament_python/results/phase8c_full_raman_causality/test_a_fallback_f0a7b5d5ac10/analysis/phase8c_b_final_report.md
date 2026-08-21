# Phase 8C-B final result report

## Execution

- Raman ON Slurm job: `180748`, `COMPLETED`, exit `0:0`, elapsed `04:21:05`.
- Raman OFF Slurm job: `180749`, `COMPLETED`, exit `0:0`, elapsed `02:59:13`.
- Staging source SHA: `f0a7b5d5ac103546bd693378e8f8efb4f07c6c27`.
- ON/OFF configuration hashes matched the locked Test A configurations; the
  intended physical difference remained `propagation.use_raman_full_operator`.

## Numerical and diagnostic admission

Both output files passed the required finite-diagnostic audit.  The ON result
has full Eq. (27) feedback applied with nonzero RHS; the OFF result retains the
raw Raman response with the feedback RHS disabled.  Legacy Raman absorption is
zero in both cases.

## Causal result

Coordinates use `x_focus_cm = 100 * (z_m - 0.95)` with no shifting,
smoothing, or renormalization.  At the `1e22 m^-3` density threshold, full
Eq. (27) Raman ON crosses at `-16.380 cm` and OFF at `-14.088 cm`: ON minus
OFF is `-2.292 cm`.  The analysis therefore classifies full Eq. (27) Raman
feedback as a **major contributor** to the high-density onset under this fixed
configuration.

The peak density is `6.387e22 m^-3` for ON at `-14.350 cm`, versus
`2.524e22 m^-3` for OFF at `-10.370 cm`.

## Provenance limitation

The upstream GitHub fetch/clone remained unavailable through the HPC proxy.
The production staging was therefore loaded via a locally verified Git bundle
fallback, not a direct GitHub-SHA checkout.  The numerical result is complete
and the ON/OFF diagnostic contract passes, but this provenance deviation means
the result must be labelled **fallback-provenance evidence** rather than the
strict GitHub-SHA-locked Phase 8C-B admission record until a direct GitHub
checkout rerun reproduces it.

## Assets

- `test_a_effect_summary.json`
- `test_a_crossing_shifts.csv`
- `test_a_metrics.csv`
- `rho_max_on_off_pycap.png`
- `i_max_on_off.png`
- `crossing_shift_vs_threshold.png`
- `raman_intensity_density_effect_chain.png`

