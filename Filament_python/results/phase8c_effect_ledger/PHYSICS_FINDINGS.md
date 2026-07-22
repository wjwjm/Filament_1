# Phase 8C-0 physics findings

These findings distinguish an archived strict pair from a numerical audit and from an unsupported inference. Positions use the fixed focus coordinate only.

## A. Confirmed

- The coordinate origin is fixed at `z=0.95 m`; no curve in this ledger was shifted.
- The Phase 6 legacy Raman pair is a strict single-delta pair: the archived configuration difference is `propagation.use_raman_phase` only, while Raman absorption remains enabled. At `rho = 1e22 m^-3`, enabling the legacy Raman phase moves first crossing from `-14.0136 cm` to `-16.4119 cm`: `-2.3983 cm` (earlier). At `1e21 m^-3` the corresponding shift is `-1.2531 cm`, but PyCAP is left-censored at that threshold.
- Job 179988's 1.41765% total-energy closure failure is numerical rather than a Raman physical-loss result: R4 identified BK-NEE complex64 linear-operator numerical dissipation, and the R5 controlled mixed-precision tests suppressed that residual to about `1e-8` in the stated smoke contracts.
- The mixed-precision change is a numerical repair, not evidence that a physical onset mechanism changed. No archived full-length mixed-precision onset curve exists.

## B. Partially supported

The legacy split Raman phase can explain a material portion of the historical early onset, but only within its own operator formulation.

- At the first PyCAP threshold that is not left-censored, `1e22 m^-3`, the current legacy baseline is `-2.3847 cm` earlier than PyCAP. The strict legacy Raman phase ON/OFF shift is `-2.3983 cm`, or `1.0057` of that signed offset. The phase-off curve lies `+0.0136 cm` later than PyCAP at this one threshold.
- This is not evidence that full Isaacs Eq.27 Raman feedback has the same net contribution: Job 179988 is invalid for physics comparison and lacks a local `rho_max_vs_z` curve.
- At `1e19`–`1e21 m^-3`, PyCAP's digitized curve is already above threshold at its first x sample. The current offsets (`-3.9393`, `-3.0724`, and `-2.0432 cm`) may describe the displayed-domain difference, but their total-offset fractions are intentionally `null`.

## C. Not confirmed

- Whether electronic Kerr is the leading residual early-onset cause.
- The net full Eq.27 Raman-feedback contribution.
- The independent contributions of the ionization-rate model, plasma phase, ionization loss, and self-steepening.
- Coupled multi-nonlinear contributions.

No existing strict production-length curve pair in the local archive isolates these factors. They must not be ranked from physical intuition or from a numerical smoke.

## D. Not currently answerable

- The residual discrepancy caused by PyCAP's non-public numerical implementation.
- Centimeter contributions from historical coordinate, FT90, and discrete peak-power corrections: their strict before/after 120 fs curves are absent.
- Full Eq.27-versus-legacy onset differences: the only full Eq.27 Job 1 curves are unavailable locally and their energy admission failed.

The correct current classification is therefore `insufficient_historical_causal_pairs`, not a forced claim that Raman, Kerr, or plasma is uniquely primary.
