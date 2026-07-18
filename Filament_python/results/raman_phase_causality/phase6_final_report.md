# Phase 6 final report: 120 fs Raman-phase causality

## Outcome

The controlled 120 fs Raman-phase ablation is classified as **`raman_phase_partially_supported`**. Raman phase has a resolved causal effect on the Talebpour propagation history, but its removal does not improve every absolute-coordinate comparison with the digitized PyCAP curve. The conclusion is deliberately limited to 120 fs.

## Controls and gates

- Coordinate for every comparison: `x_focus_cm = 100 * (z_m - 0.95)`; no formal curve shift was applied.
- Execution-parity gate: **accepted**. The propagation-critical files at the formal Talebpour execution revision `8dcd01e` and the Phase 6 branch were identical before non-core Phase 6 tooling changes.
- Existing Popruzhenko/Talebpour feedback analysis: **`feedback_after_ionization`**. The pre-ionization intensity-threshold separation was `1.90e-05 cm`, below `epsilon_x = 0.10 cm`.
- Raman-off configuration changed only `propagation.use_raman_phase: true -> false`; Raman absorption remained enabled.

## Raman-phase-off propagation audit

The only Phase 6 full propagation, Slurm job `176915`, completed on July 18, 2026 with `COMPLETED` and exit code `0:0`. It reached `z = 1.300 m` with 15,000 aligned records.

- Execution SHA: `8dcd01ee38adf2167a2fd6083ae4785e94de89a0`
- Configuration SHA256: `d57aadda4c75999722f63919ac92d6a7a42c743d9c3ae2837d502e98176a49b5`
- Raman raw phase maximum: `6.645e-03 rad`; raw response remained finite and nonzero.
- Raman applied phase maximum: `0.0 rad`; the intended ablation was realized.
- Raman absorption remained applied, with maximum `alpha_R = 1.809e-02 m^-1`.
- Species bounds, energy diagnostics, actual step size, rejection counters and safety counters all passed. The latter two counters were zero.

The raw NPZ remains only on the remote execution directory; this repository contains CSV/JSON/Markdown/PNG/log summaries only.

## Causal comparison

`epsilon_x = 0.10 cm`. With Raman phase enabled, the first `rho = 1e21 m^-3` crossing is `1.253 cm` earlier than in the Raman-phase-off propagation, far above the positional resolution. This establishes a resolved Raman-phase contribution to the 120 fs axial feedback chain.

The comparison is mixed rather than uniformly corrective relative to the digitized PyCAP trace:

- The unshifted density RMSE is lower for Talebpour full (`1.872e22 m^-3`) than Raman-phase-off (`2.144e22 m^-3`).
- The full-model peak density (`6.461e22 m^-3`) closely matches PyCAP (`6.455e22 m^-3`), whereas Raman-off peaks at `2.498e22 m^-3`.
- The peak-top center shifts from `-10.270 cm` (Raman off) to `-14.405 cm` (full), while PyCAP is `-12.045 cm`; therefore the global-position criterion is not uniformly improved.
- The full-model half-maximum tail area is larger than Raman-off. This prevents an unconditional “supported” classification even though the full-model tail is closer to the PyCAP tail than the Raman-off tail.

Accordingly, Raman phase is a material 120 fs causal contributor, but this single-factor ablation does not show that it alone resolves the full axial-position discrepancy.

## Scope and exclusions

- `40fs_raman_phase_off`: not submitted.
- `O2-off`: not submitted.
- `electronic-Kerr-only` and all other ablations: not submitted.
- Production default physics: unchanged.

No Phase 6 conclusion should be extended to a 40 fs common-advance explanation without a separately authorized matched 40 fs study.
