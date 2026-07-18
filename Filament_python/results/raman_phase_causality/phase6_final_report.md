# Phase 6 final report — corrected postprocessing closure

## Formal status

The propagation experiment is **accepted** and the feedback analysis, classification, and reporting are **corrected and complete**. This correction uses only archived CSV/JSON/Markdown/PyCAP data; it submitted no Slurm jobs, reran no propagation, changed no production physics, and committed no NPZ/MAT/LUT files.

All curves retain the fixed coordinate `x_focus_cm = 100 * (z_m - 0.95)`. The original Phase-6 classification remains **`raman_phase_partially_supported`**, but the rationale below supersedes the earlier postprocessing.

## Corrected data and validity gates

- Input audit passed: full and Raman-off axes are identical, each has 15,000 finite records, and the Raman raw/applied/absorption diagnostics are complete.
- The U0 source is each case's own `diagnostic_summary.json.metrics.U0_J`: `2.1715106e-3 J` for full and Raman-off. `U_rel_change_z` is not used as energy.
- Raman-off raw response is finite and nonzero, applied phase is zero, Raman absorption remains on, and rejection/safety counters are equal and zero.
- PyCAP has no valid first rising crossing for `1e19`, `1e20`, or `1e21 m^-3`; those rows are explicitly marked `not_available_in_pycap`, never as a failed improvement.

## Answers to the required physical questions

1. **Pop/Tal intensity before significant ionization:** no resolved separation. The maximum pre-ionization intensity-threshold separation is `1.90e-05 cm`, below `epsilon_x = 0.10 cm`.
2. **Correct energy-deposition order:** `f_dep=E_dep,cumulative/U0` crosses `1e-6`, `1e-5`, and `1e-4` at essentially the same locations for Pop/Tal (`-94.096`, `-86.670`, `-48.687 cm`). The Tal `1e-3` and `1e-2` events occur at `-15.717` and `-12.451 cm`, versus Pop at `-15.926` and `-12.734 cm`.
3. **Where the ionization-rate-model difference emerges:** the corrected event-chain classification is `feedback_after_ionization`; the material divergence appears after the density feedback becomes appreciable rather than in pre-ionization intensity growth.
4. **Raman phase density causal shifts, full minus off:** `1e19: +0.054 cm` (unresolved), `1e20: -0.418 cm`, `1e21: -1.253 cm`, `1e22: -2.398 cm`. Negative values mean Raman-on develops earlier.
5. **Peak-top-center shift:** full minus Raman-off is `-4.135 cm`; Raman phase advances the peak-top position.
6. **Peak collapse on Raman removal:** yes. Full/off peaks are `6.4609e22`/`2.4978e22 m^-3`; off/full is `0.3866`, below the specified `0.5` collapse threshold (a 61.3% reduction).
7. **Peak density relative to PyCAP:** PyCAP is `6.4546e22 m^-3`; full differs by `6.26e19 m^-3` (~0.10%), Raman-off by `3.96e22 m^-3` (~61.3%).
8. **FWHM:** full/off/PyCAP are `12.076/12.842/8.738 cm`; full is closer to PyCAP.
9. **Tail:** full/off/PyCAP areas are `1.666e23/6.596e22/1.353e23 m^-3 cm`. Relative errors are `0.232/0.512`; full is closer. The obsolete full/off tail ratio is retained only descriptively (`2.525`), not used for classification.
10. **Global density RMSE:** full is lower (`1.872e22 m^-3`) than Raman-off (`2.144e22 m^-3`).
11. **PyCAP peak-top bracketing:** yes: full `-14.405 cm <` PyCAP `-12.045 cm <` Raman-off `-10.270 cm`.
12. **Raman endpoint diagnostic:** `R_Raman=(x_off-x_full)/(x_PyCAP-x_full)=1.752`. Raman-off crosses past PyCAP. This is an on/off endpoint diagnostic only; nonlinear propagation means it cannot be used to rescale `n_R` or `f_R`.
13. **Why Raman phase cannot simply be removed:** removal gives a severe peak-density collapse, worsens RMSE and FWHM, and makes the tail farther from PyCAP even though it improves the available `1e22` crossing and peak-center errors.
14. **Why the conclusion is 120 fs only:** no matched 40 fs Raman-off propagation was authorized or run.
15. **Next priorities:** validate Raman response normalization first, then check possible `n_R/f_R` double weighting, the Raman temporal-response model, and finally electronic Kerr under separately authorized single-factor controls.

## Corrected classification

`raman_phase_partially_supported` is justified by resolved medium/high-density and peak-top causal shifts together with a severe Raman-off peak collapse and mixed PyCAP metrics. The classification is not based on unavailable low-density PyCAP crossings, the old reverse collapse check, or the old full/off-only tail ratio.

## Traceability and validation

- Input main SHA: `0a5264cbc07cea48e4fc579964dbc7dc0dcbca05`; the correction worktree began clean on `codex/phase6-postprocess-correction`.
- Local checks: `python -m compileall Filament_python/tools` and `pytest -q Filament_python/tests`.
- Result: `84 passed`, `0 failed`; Python `3.13.0`, NumPy `2.4.4`.
- Local pytest passed; **no GitHub Actions CI run was available**.
- Detailed corrected artifacts are under `phase6_postprocess_correction/`; the prior outputs remain for traceability and are superseded only by the changelog below.
