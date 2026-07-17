# Phase 5 final ionization-model propagation report

## Controlled 120 fs result

The current-observability Popruzhenko baseline (job `175050`) and Talebpour
case (job `175354`) both completed to `z=1.3 m` with the same execution SHA
`8dcd01ee38adf2167a2fd6083ae4785e94de89a0`, the same full-model propagation
settings, and the same diagnostic schema.  The only physical change was the
N2/O2 ionization-rate family and the explicit Talebpour effective parameters.

The fixed-coordinate comparison used `x_focus_cm = 100 * (z_m - 0.95)` with
no formal curve shift.  Talebpour moved the `1e21 m^-3` rising edge later by
`0.332 cm`, exceeding `epsilon_x = 0.100 cm`; its peak-top centre moved toward
the PyCAP digitization and its tail was not classified as unacceptably worse.
The 120 fs classification is therefore **propagation_supported**.

## 40 fs validation

The Talebpour 40 fs job `175950` completed successfully to `z=1.3 m` with
the same execution SHA and observability schema.  It passed the remote
re-audit (15,000 aligned finite records, bounded species densities, valid
energy/step/safety histories).  Its peak density was `3.426e22 m^-3` at
`-11.020 cm`; the PyCAP digitization peak was `3.506e22 m^-3` at `-8.067 cm`.

No new current-observability Popruzhenko 40 fs job was authorized.  Therefore
the 40 fs result is a completed Talebpour validation, not a second formally
matched Popruzhenko-versus-Talebpour causal comparison.

## Gate outcome

- Phase-5 120 fs causal gate: passed (`propagation_supported`).
- Talebpour 40 fs validation: completed and diagnostic gate passed.
- O2-off: not submitted.
- Raw NPZ/MAT/LUT cache: retained remotely only; the repository contains only
  lightweight JSON/CSV/Markdown provenance artifacts.
