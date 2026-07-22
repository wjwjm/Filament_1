# Phase 8C-0 final effect-chain report

## Decision

Final classification: `insufficient_historical_causal_pairs`.

The archive contains one useful strict physical onset pair—the Phase 6 legacy Raman-phase ON/OFF comparison—but it does not contain valid strict pairs for full Eq.27 feedback, electronic Kerr, ionization model, plasma phase, ionization loss, or self-steepening. Numerical BK-NEE evidence is deliberately not counted as physical absorption or a physical onset contribution.

## Answers to the required questions

1. **How early is the current production curve relative to PyCAP?** At the first non-left-censored PyCAP threshold, `rho=1e22 m^-3`, the current curve crosses at `-16.4119 cm` and PyCAP at `-14.0272 cm`: `-2.3847 cm` (negative = earlier). The displayed differences at `1e19`, `1e20`, and `1e21 m^-3` are `-3.9393`, `-3.0724`, and `-2.0432 cm`, respectively, but the PyCAP curve is left-censored there and those are not complete onset comparisons.
2. **Which density threshold is defensible?** `1e22 m^-3` for a complete current-versus-PyCAP crossing comparison. Lower thresholds remain reported with a censoring flag, not silently discarded.
3. **What is the known Raman effect?** In the strict legacy split Raman phase pair, phase ON shifts crossings by `+0.0544 cm` at `1e19`, `-0.4180 cm` at `1e20`, `-1.2531 cm` at `1e21`, and `-2.3983 cm` at `1e22 m^-3`, versus phase OFF.
4. **What fraction of the total PyCAP offset can Raman explain?** At `1e22 m^-3`, the legacy phase contribution divided by the current-PyCAP signed offset is `1.0057`. This is a threshold-specific legacy-operator result, not a statement about full Eq.27 Raman feedback. Fractions at lower thresholds are `null` because PyCAP is left-censored.
5. **Which fixes changed onset materially?** The legacy Raman phase switch changes `1e22` onset by about `2.40 cm`. Coordinate, FT90, and normalization corrections have no locally archived strict before/after curve, so no centimeter claim is permitted. The full Eq.27 Job 1 outputs are invalid for physics due to energy admission failure.
6. **Which work only repaired numerical stability?** R4/R5 BK-NEE precision work: mixed precision removes measured complex64 linear numerical dissipation in short controlled tests. It does not supply a valid physical onset curve and therefore changes no physical attribution in this report.
7. **Most likely main nonlinear source?** No unique main source is supportable. Legacy Raman phase is a demonstrated material contributor in its legacy formulation; electronic Kerr and plasma/ionization feedback remain unisolated candidates.
8. **Is the evidence enough to confirm the main source?** No. The required classification is `insufficient_historical_causal_pairs`.
9. **Next strict causal pair worth running?** Test A: full Eq.27 Raman feedback ON/OFF, with identical admitted numerical path and all other settings locked. It requires new authorization and complete archives.
10. **Should R6 mixed-precision performance optimization resume?** No—not as a standalone stage. It remains suspended. A narrowly scoped performance prerequisite could be separately authorized only if needed to execute a high-priority causal pair after its physical question and acceptance criteria are fixed.

## Evidence boundaries

- Job 179988 establishes a numerical root cause (`linear_operator_numerical_dissipation`) for its 1.41765% energy failure; it is not physics-comparison evidence.
- Job 180076 verifies the mixed-precision short full-physics numerical/closure gate, but its 58.1 s for 20 steps extrapolates to 12.10 h, above the 6.4 h gate. Peak reserved GPU memory was not captured. This is why no new full Job 1 is authorized.
- The result inventory explicitly records curves that are missing or invalid. Their metrics are `null`; no curve was reconstructed from Markdown or scalar reports.

## Validation

- `python -m compileall Filament_python/KHz_filament`: passed.
- Targeted effect-ledger tests: `7 passed`.
- Complete test directory, run in two file groups to stay below the 60 s command window: `80 passed` and `108 passed` (`188 passed` total).
- No GPU smoke, Slurm submission, or propagation job was run for Phase 8C-0.

## Final frozen status

```text
frozen main SHA = e11d13f103c484953c0f733aa9b410bff385b2b5
new propagation jobs submitted = 0
new full Job 1 submitted = false
Job 2 prepared = false
Job 2 submitted = false
R6 resumed = false
effect-chain ledger completed = true
next recommended causal test = Test A — full Eq.27 Raman feedback ON/OFF
new authorization required = true
```

GitHub Actions CI = unavailable; no workflow present.
